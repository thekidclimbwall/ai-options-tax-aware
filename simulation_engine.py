import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
import warnings
import logging
import pickle
import os
from typing import Dict, List, Tuple, Optional
import openai
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress yfinance warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class TaxAwareOptionsSimulator:
    def __init__(self):
        # Investor persona - Sarah from Palo Alto
        self.initial_capital = 500000.0
        self.cash_reserve_minimum = 0.30  # 30% minimum cash
        self.max_position_size = 0.05  # 5% max per position
        
        # Tax rates
        self.federal_ltcg = 0.15
        self.federal_stcg = 0.35
        self.state_cg = 0.093  # California
        self.niit = 0.038  # Net Investment Income Tax
        
        # Trading parameters
        self.options_commission = 0.65
        self.stock_slippage = 0.02
        self.vix_threshold = 18
        self.ivp_threshold = 65
        self.dte_min = 21
        self.dte_max = 45
        self.delta_min = 0.18
        self.delta_max = 0.30
        self.profit_take_threshold = 0.70
        self.roll_delta_threshold = 0.38
        
        # Simulation period
        self.start_date = "2020-01-01"
        self.end_date = "2024-12-31"
        
        # Data containers
        self.portfolio_value = []
        self.cash = []
        self.positions = []
        self.trade_log = []
        self.daily_returns = []
        
        # OpenAI client
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY')) if os.getenv('OPENAI_API_KEY') else None
        
        # S&P 500 symbols (top liquid names for simulation)
        self.sp500_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK-B',
            'UNH', 'JNJ', 'JPM', 'V', 'PG', 'XOM', 'HD', 'CVX', 'MA', 'PFE',
            'BAC', 'ABBV', 'KO', 'AVGO', 'PEP', 'TMO', 'COST', 'WMT', 'DIS',
            'ABT', 'MCD', 'VZ', 'ADBE', 'ACN', 'NKE', 'CRM', 'NFLX', 'DHR',
            'LIN', 'NEE', 'PM', 'TXN', 'RTX', 'SPGI', 'HON', 'QCOM', 'UNP',
            'LLY', 'AMD', 'LOW', 'IBM', 'CAT', 'AMGN', 'GS', 'SBUX', 'INTU',
            'BLK', 'BA', 'AXP', 'DE', 'MDT', 'GILD', 'TGT', 'AMT', 'BKNG',
            'MU', 'ADP', 'LRCX', 'ADI', 'VRTX', 'SYK', 'TMUS', 'REGN', 'ZTS',
            'MMC', 'C', 'MO', 'PLD', 'NOW', 'EOG', 'EQIX', 'CSX', 'DUK',
            'CCI', 'WM', 'FCX', 'SLB', 'ICE', 'NSC', 'GD', 'EMR', 'SO',
            'APD', 'BSX', 'CL', 'ITW', 'AON', 'EL', 'FDX', 'CME', 'USB',
            'PNC', 'COF', 'GE', 'NOC', 'D', 'SHW', 'MCO', 'TJX', 'HUM'
        ]

    def fetch_market_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical stock data with fallback to simulated data"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            if data.empty:
                logger.warning(f"No data for {symbol}, generating simulated data")
                return self.generate_simulated_data(symbol, start_date, end_date)
            return data
        except Exception as e:
            logger.warning(f"Error fetching {symbol}: {e}, generating simulated data")
            return self.generate_simulated_data(symbol, start_date, end_date)

    def generate_simulated_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate simulated stock data using Geometric Brownian Motion"""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n_days = len(dates)
        
        # Base parameters for simulation
        initial_price = 100.0
        mu = 0.08  # Annual return
        sigma = 0.25  # Annual volatility
        
        # Generate GBM
        dt = 1/252  # Daily time step
        random_returns = np.random.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), n_days)
        prices = initial_price * np.exp(np.cumsum(random_returns))
        
        # Create OHLCV data
        data = pd.DataFrame(index=dates)
        data['Close'] = prices
        data['Open'] = prices * np.random.uniform(0.98, 1.02, n_days)
        data['High'] = np.maximum(data['Open'], data['Close']) * np.random.uniform(1.0, 1.05, n_days)
        data['Low'] = np.minimum(data['Open'], data['Close']) * np.random.uniform(0.95, 1.0, n_days)
        data['Volume'] = np.random.randint(10000000, 50000000, n_days)
        
        logger.info(f"Generated simulated data for {symbol}")
        return data

    def fetch_vix_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch VIX data"""
        try:
            vix = yf.Ticker("^VIX")
            data = vix.history(start=start_date, end=end_date)
            if data.empty:
                # Generate simulated VIX data
                dates = pd.date_range(start=start_date, end=end_date, freq='D')
                vix_values = 15 + 10 * np.random.beta(2, 5, len(dates))  # VIX-like distribution
                data = pd.DataFrame({'Close': vix_values}, index=dates)
                logger.info("Generated simulated VIX data")
            return data
        except:
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            vix_values = 15 + 10 * np.random.beta(2, 5, len(dates))
            data = pd.DataFrame({'Close': vix_values}, index=dates)
            logger.info("Generated simulated VIX data")
            return data

    def calculate_black_scholes_greeks(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'put') -> Dict:
        """Calculate Black-Scholes option price and Greeks"""
        if T <= 0:
            if option_type == 'put':
                return {
                    'price': max(K - S, 0),
                    'delta': -1.0 if S < K else 0.0,
                    'gamma': 0.0,
                    'theta': 0.0,
                    'vega': 0.0
                }
            else:
                return {
                    'price': max(S - K, 0),
                    'delta': 1.0 if S > K else 0.0,
                    'gamma': 0.0,
                    'theta': 0.0,
                    'vega': 0.0
                }
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'put':
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            delta = -norm.cdf(-d1)
        else:
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            delta = norm.cdf(d1)
        
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - 
                r * K * np.exp(-r * T) * norm.cdf(-d2 if option_type == 'put' else d2))
        if option_type == 'put':
            theta += r * K * np.exp(-r * T) * norm.cdf(-d2)
        vega = S * norm.pdf(d1) * np.sqrt(T)
        
        return {
            'price': price,
            'delta': abs(delta),
            'gamma': gamma,
            'theta': theta,
            'vega': vega
        }

    def calculate_implied_volatility_percentile(self, symbol: str, current_iv: float, lookback_days: int = 252) -> float:
        """Calculate implied volatility percentile (simulated)"""
        # Since we don't have real IV data, simulate IVP based on VIX and randomness
        base_ivp = min(100, max(0, (current_iv - 0.15) * 200))  # Scale to 0-100
        noise = np.random.normal(0, 15)  # Add some noise
        return min(100, max(0, base_ivp + noise))

    def estimate_implied_volatility(self, symbol: str, price_data: pd.DataFrame, window: int = 30) -> float:
        """Estimate implied volatility from historical volatility"""
        returns = price_data['Close'].pct_change().dropna()
        if len(returns) < window:
            return 0.25  # Default IV
        
        historical_vol = returns.tail(window).std() * np.sqrt(252)
        # IV is typically higher than HV
        implied_vol = historical_vol * np.random.uniform(1.1, 1.5)
        return min(1.0, max(0.1, implied_vol))

    def calculate_after_tax_ev(self, premium: float, prob_profit: float, expected_loss: float, days_to_exp: int) -> float:
        """Calculate after-tax annualized expected value"""
        # Determine tax rate (assuming short-term for options premium)
        tax_rate = self.federal_stcg + self.state_cg + self.niit
        
        gross_ev = (prob_profit * premium) - ((1 - prob_profit) * expected_loss)
        after_tax_ev = gross_ev * (1 - tax_rate)
        
        # Annualize
        annualized_ev = after_tax_ev * (365 / days_to_exp)
        
        return annualized_ev

    def scan_for_opportunities(self, current_date: pd.Timestamp, vix_value: float, available_cash: float) -> List[Dict]:
        """Scan for trading opportunities based on strategy criteria"""
        opportunities = []
        
        if vix_value < self.vix_threshold:
            return opportunities
        
        # Sample a subset of symbols for simulation efficiency
        symbols_to_scan = np.random.choice(self.sp500_symbols, min(20, len(self.sp500_symbols)), replace=False)
        
        for symbol in symbols_to_scan:
            try:
                # Get price data
                end_date = current_date + timedelta(days=1)
                start_date = current_date - timedelta(days=60)
                price_data = self.fetch_market_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                
                if price_data.empty:
                    continue
                
                current_price = price_data['Close'].iloc[-1]
                
                # Check liquidity filters (simulated)
                avg_volume = price_data['Volume'].tail(20).mean()
                if avg_volume < 5000000:  # 5M ADV requirement
                    continue
                
                # Estimate market cap (simplified)
                estimated_market_cap = current_price * avg_volume * 50  # Rough estimate
                if estimated_market_cap < 20e9:  # $20B requirement
                    continue
                
                # Calculate IV and IVP
                current_iv = self.estimate_implied_volatility(symbol, price_data)
                ivp = self.calculate_implied_volatility_percentile(symbol, current_iv)
                
                if ivp < self.ivp_threshold:
                    continue
                
                # Find suitable strike and expiration
                for dte in range(self.dte_min, self.dte_max + 1):
                    expiration_date = current_date + timedelta(days=dte)
                    
                    # Find strike price with target delta
                    for delta_target in np.arange(self.delta_min, self.delta_max + 0.02, 0.02):
                        # Estimate strike price for target delta
                        strike_price = current_price * (1 - delta_target)  # Rough approximation for puts
                        
                        # Calculate option price and Greeks
                        option_data = self.calculate_black_scholes_greeks(
                            S=current_price,
                            K=strike_price,
                            T=dte/365.0,
                            r=0.02,  # Risk-free rate
                            sigma=current_iv,
                            option_type='put'
                        )
                        
                        if abs(option_data['delta'] - delta_target) > 0.05:
                            continue
                        
                        premium = option_data['price']
                        
                        # Check bid-ask spread (simulated as % of premium)
                        bid_ask_spread = premium * 0.02  # Assume 2% spread
                        if bid_ask_spread > 0.10:
                            continue
                        
                        # Calculate cash requirement
                        cash_required = strike_price * 100  # Cash-secured put
                        if cash_required > available_cash * self.max_position_size:
                            continue
                        
                        # Calculate EV
                        prob_profit = norm.cdf(np.log(current_price / strike_price) / (current_iv * np.sqrt(dte/365.0)))
                        expected_loss = (strike_price - current_price) * 100  # Loss if assigned
                        
                        after_tax_ev = self.calculate_after_tax_ev(
                            premium * 100, prob_profit, expected_loss, dte
                        )
                        
                        opportunities.append({
                            'symbol': symbol,
                            'current_price': current_price,
                            'strike_price': strike_price,
                            'premium': premium,
                            'delta': option_data['delta'],
                            'dte': dte,
                            'expiration_date': expiration_date,
                            'iv': current_iv,
                            'ivp': ivp,
                            'cash_required': cash_required,
                            'after_tax_ev': after_tax_ev,
                            'prob_profit': prob_profit
                        })
            
            except Exception as e:
                logger.warning(f"Error scanning {symbol}: {e}")
                continue
        
        # Sort by after-tax EV
        opportunities.sort(key=lambda x: x['after_tax_ev'], reverse=True)
        return opportunities[:5]  # Return top 5

    def generate_trade_rationale(self, trade_data: Dict, action: str) -> str:
        """Generate natural language rationale using OpenAI API"""
        if not self.openai_client:
            return f"{action} on {trade_data['symbol']} - API key not configured"
        
        try:
            prompt = f"""
            Generate a concise trading rationale for this options trade:
            Action: {action}
            Symbol: {trade_data['symbol']}
            Strike: ${trade_data.get('strike_price', 0):.2f}
            Premium: ${trade_data.get('premium', 0):.2f}
            IVP: {trade_data.get('ivp', 0):.1f}%
            After-tax EV: {trade_data.get('after_tax_ev', 0):.1f}%
            
            Write one sentence explaining the trade rationale from a tax-aware perspective.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            logger.warning(f"OpenAI API error: {e}")
            return f"{action} on {trade_data['symbol']} due to attractive risk-adjusted returns"

    def execute_simulation(self):
        """Run the complete simulation"""
        logger.info("Starting tax-aware options simulation...")
        
        # Initialize portfolio
        current_cash = self.initial_capital
        open_positions = []
        trade_id = 0
        
        # Fetch VIX data
        vix_data = self.fetch_vix_data(self.start_date, self.end_date)
        
        # Create date range for simulation
        simulation_dates = pd.date_range(start=self.start_date, end=self.end_date, freq='B')  # Business days
        
        # Benchmark data (50% SPY, 50% BIL)
        spy_data = self.fetch_market_data('SPY', self.start_date, self.end_date)
        bil_data = self.fetch_market_data('BIL', self.start_date, self.end_date)
        
        benchmark_portfolio = self.initial_capital
        spy_shares = (benchmark_portfolio * 0.5) / spy_data['Close'].iloc[0]
        bil_shares = (benchmark_portfolio * 0.5) / bil_data['Close'].iloc[0]
        
        for current_date in simulation_dates:
            if current_date not in vix_data.index:
                continue
            
            vix_value = vix_data.loc[current_date, 'Close']
            
            # Check existing positions for management
            positions_to_remove = []
            for i, position in enumerate(open_positions):
                # Check for expiration or profit taking
                days_to_exp = (position['expiration_date'] - current_date).days
                
                if days_to_exp <= 0:
                    # Handle expiration
                    current_stock_price = self.fetch_market_data(
                        position['symbol'], 
                        current_date.strftime('%Y-%m-%d'), 
                        (current_date + timedelta(days=1)).strftime('%Y-%m-%d')
                    )['Close'].iloc[-1]
                    
                    if current_stock_price < position['strike_price']:
                        # Assignment
                        assignment_cost = position['strike_price'] * 100
                        current_cash -= assignment_cost
                        
                        # Add to trade log
                        pnl = position['premium_received'] * 100 - self.options_commission
                        self.trade_log.append({
                            'date': current_date,
                            'trade_id': trade_id,
                            'symbol': position['symbol'],
                            'action': 'Assignment',
                            'strike': position['strike_price'],
                            'premium': 0,
                            'pnl': pnl,
                            'rationale': f"Assigned shares of {position['symbol']} at ${position['strike_price']}"
                        })
                        trade_id += 1
                    else:
                        # Expired worthless
                        pnl = position['premium_received'] * 100 - self.options_commission
                        current_cash += pnl
                        
                        self.trade_log.append({
                            'date': current_date,
                            'trade_id': trade_id,
                            'symbol': position['symbol'],
                            'action': 'Expire Worthless',
                            'strike': position['strike_price'],
                            'premium': position['premium_received'],
                            'pnl': pnl,
                            'rationale': f"Put on {position['symbol']} expired worthless - kept full premium"
                        })
                        trade_id += 1
                    
                    positions_to_remove.append(i)
            
            # Remove closed positions
            for i in sorted(positions_to_remove, reverse=True):
                del open_positions[i]
            
            # Scan for new opportunities
            available_cash = current_cash - (current_cash * self.cash_reserve_minimum)
            if available_cash > 10000:  # Minimum trade size
                opportunities = self.scan_for_opportunities(current_date, vix_value, available_cash)
                
                for opp in opportunities[:2]:  # Limit to 2 new trades per day
                    if len(open_positions) >= 10:  # Position limit
                        break
                    
                    if opp['cash_required'] <= available_cash:
                        # Open new position
                        open_positions.append({
                            'symbol': opp['symbol'],
                            'strike_price': opp['strike_price'],
                            'expiration_date': opp['expiration_date'],
                            'premium_received': opp['premium'],
                            'entry_date': current_date,
                            'delta': opp['delta']
                        })
                        
                        current_cash -= self.options_commission
                        available_cash -= opp['cash_required']
                        
                        # Generate rationale and log trade
                        rationale = self.generate_trade_rationale(opp, 'Sell Put')
                        
                        self.trade_log.append({
                            'date': current_date,
                            'trade_id': trade_id,
                            'symbol': opp['symbol'],
                            'action': 'Sell Put',
                            'strike': opp['strike_price'],
                            'premium': opp['premium'],
                            'pnl': opp['premium'] * 100 - self.options_commission,
                            'rationale': rationale
                        })
                        trade_id += 1
            
            # Calculate portfolio value
            portfolio_value = current_cash
            for position in open_positions:
                portfolio_value += position['strike_price'] * 100  # Cash secured amount
            
            self.portfolio_value.append(portfolio_value)
            self.cash.append(current_cash)
            
            # Calculate benchmark value
            if current_date in spy_data.index and current_date in bil_data.index:
                spy_price = spy_data.loc[current_date, 'Close']
                bil_price = bil_data.loc[current_date, 'Close']
                benchmark_value = (spy_shares * spy_price) + (bil_shares * bil_price)
            else:
                benchmark_value = benchmark_portfolio
            
            # Store daily data
            daily_return = (portfolio_value - self.initial_capital) / self.initial_capital
            self.daily_returns.append({
                'date': current_date,
                'portfolio_value': portfolio_value,
                'benchmark_value': benchmark_value,
                'strategy_return': daily_return,
                'benchmark_return': (benchmark_value - self.initial_capital) / self.initial_capital
            })
        
        logger.info(f"Simulation completed. Executed {len(self.trade_log)} trades.")
        
        # Save results
        results = {
            'daily_returns': pd.DataFrame(self.daily_returns),
            'trade_log': pd.DataFrame(self.trade_log),
            'final_portfolio_value': self.portfolio_value[-1] if self.portfolio_value else self.initial_capital,
            'total_trades': len(self.trade_log)
        }
        
        # Create results directory if it doesn't exist
        os.makedirs('simulation_results', exist_ok=True)
        
        with open('simulation_results/simulation_data.pkl', 'wb') as f:
            pickle.dump(results, f)
        
        logger.info("Simulation results saved to simulation_results/simulation_data.pkl")
        return results

if __name__ == "__main__":
    simulator = TaxAwareOptionsSimulator()
    results = simulator.execute_simulation()