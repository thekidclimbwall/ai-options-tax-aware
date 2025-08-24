import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm
import re
import requests
import time
import os
import openai
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Options Strategy Assistant",
    page_icon="🎯",
    layout="wide"
)

class OptionsAnalyzer:
    def __init__(self):
        self.risk_free_rate = 0.02
        self.polygon_api_key = os.getenv('POLYGON_API_KEY', 'EXN4qKcPBbCoHboaMnSMbMxmysSeZow4')
        
    def calculate_black_scholes_put(self, S, K, T, r, sigma):
        """Calculate Black-Scholes put option price and Greeks"""
        if T <= 0:
            return {
                'price': max(K - S, 0),
                'delta': -1.0 if S < K else 0.0,
                'prob_profit': 0.0 if S < K else 1.0
            }
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = abs(-norm.cdf(-d1))
        prob_profit = norm.cdf(d2)
        
        return {
            'price': price,
            'delta': delta,
            'prob_profit': prob_profit
        }
    
    def estimate_implied_volatility(self, symbol):
        """Estimate IV from historical volatility"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="3mo")
            if hist.empty:
                return 0.25
            
            returns = hist['Close'].pct_change().dropna()
            historical_vol = returns.std() * np.sqrt(252)
            implied_vol = historical_vol * 1.3  # IV premium
            return min(0.8, max(0.15, implied_vol))
        except:
            return 0.25
    
    def get_stock_info_polygon(self, symbol):
        """Get stock info from Polygon.io API"""
        try:
            # Get current price
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev"
            params = {"apikey": self.polygon_api_key}
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('results'):
                    result = data['results'][0]
                    current_price = result['c']  # Close price
                    previous_close = result['o']  # Open price
                    volume = result['v']
                    
                    return {
                        'symbol': symbol,
                        'current_price': current_price,
                        'previous_close': previous_close,
                        'volume': volume,
                        'company_name': symbol,
                        'market_cap': 0  # Not available in this endpoint
                    }
            return None
        except Exception as e:
            print(f"Polygon API error for {symbol}: {e}")
            return None

    def get_stock_info(self, symbol):
        """Get current stock information with fallback to Polygon"""
        try:
            # Try yfinance first
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="5d")
            
            if hist.empty:
                raise Exception("No yfinance data")
                
            current_price = hist['Close'].iloc[-1]
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'market_cap': info.get('marketCap', 0),
                'volume': hist['Volume'].mean() if len(hist) > 0 else 0,
                'company_name': info.get('longName', symbol),
                'previous_close': info.get('previousClose', current_price)
            }
        except Exception as e:
            print(f"yfinance error for {symbol}: {e}, trying Polygon...")
            # Fallback to Polygon
            polygon_data = self.get_stock_info_polygon(symbol)
            if polygon_data:
                return polygon_data
            
            st.error(f"Error fetching {symbol} from both yfinance and Polygon: {e}")
            return None
    
    def find_optimal_put_strikes(self, symbol, target_delta_range=(0.15, 0.35)):
        """Find optimal put strike prices"""
        stock_info = self.get_stock_info(symbol)
        if not stock_info:
            return []
        
        current_price = stock_info['current_price']
        iv = self.estimate_implied_volatility(symbol)
        
        opportunities = []
        
        # Check different expiration periods (21-45 DTE)
        for dte in [21, 30, 45]:
            T = dte / 365.0
            
            # Check different strike prices (5-25% OTM puts)
            for otm_pct in np.arange(0.05, 0.30, 0.02):
                strike = current_price * (1 - otm_pct)
                
                option_data = self.calculate_black_scholes_put(
                    current_price, strike, T, self.risk_free_rate, iv
                )
                
                if target_delta_range[0] <= option_data['delta'] <= target_delta_range[1]:
                    cash_required = strike * 100
                    premium_income = option_data['price'] * 100
                    
                    # Calculate annualized return
                    max_return = (premium_income / cash_required) * (365 / dte) * 100
                    
                    opportunities.append({
                        'strike': strike,
                        'premium': option_data['price'],
                        'delta': option_data['delta'],
                        'prob_profit': option_data['prob_profit'],
                        'dte': dte,
                        'cash_required': cash_required,
                        'premium_income': premium_income,
                        'max_return': max_return,
                        'otm_percent': otm_pct * 100,
                        'iv_used': iv
                    })
        
        opportunities.sort(key=lambda x: x['max_return'], reverse=True)
        return opportunities[:5]

@st.cache_data(ttl=300)  # Cache for 5 minutes
def parse_user_input_with_openai(user_input):
    """Use OpenAI to parse user's natural language input"""
    openai_api_key = os.getenv('OPENAI_API_KEY')
    
    if not openai_api_key or openai_api_key == 'your_openai_api_key_here':
        st.error("🔑 OpenAI API key required for natural language processing!")
        st.info("Please add your OpenAI API key to the .env file to use this feature.")
        return None
    
    try:
        client = openai.OpenAI(api_key=openai_api_key)
        
        prompt = f"""
        You are a financial assistant that extracts information from user investment descriptions.
        Parse this user input and extract the following information in JSON format:

        User input: "{user_input}"

        Extract and return ONLY a valid JSON object with these exact fields:
        {{
            "annual_income": <number or null>,
            "investment_amount": <number or null>,
            "stock_symbols": ["SYMBOL1", "SYMBOL2"],
            "risk_tolerance": "conservative/moderate/aggressive" or null,
            "tax_status": "single/married" or null,
            "state": "state_name" or null,
            "investment_goal": "brief description" or null,
            "timeline": "short/medium/long term" or null
        }}

        Rules for extraction:
        1. For stock_symbols: Only return valid US stock ticker symbols (2-5 uppercase letters)
        2. Look for company names and convert them to tickers:
           - Tesla/TSLA -> "TSLA" 
           - Nvidia/NVDA -> "NVDA"
           - Apple -> "AAPL"
           - Microsoft -> "MSFT" 
           - Google -> "GOOGL"
           - Amazon -> "AMZN"
           - Meta/Facebook -> "META"
        3. For amounts: Extract actual numbers (convert "200k" to 200000, "1M" to 1000000)
        4. For income: Look for salary, income, make, earn keywords
        5. For investment: Look for invest, capital, money, funds keywords
        6. For state: US states only
        7. If unsure about any field, use null

        Return ONLY the JSON object, no explanation or additional text.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a precise financial data extraction assistant. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.1
        )
        
        json_response = response.choices[0].message.content.strip()
        
        # Clean up response (remove code block markers if present)
        json_response = re.sub(r'```json\s*|\s*```', '', json_response)
        json_response = json_response.strip()
        
        # Parse JSON
        parsed_data = json.loads(json_response)
        
        # Validate and clean data
        result = {
            'stocks': parsed_data.get('stock_symbols', []),
            'income': parsed_data.get('annual_income'),
            'investment': parsed_data.get('investment_amount'),
            'state': parsed_data.get('state'),
            'tax_status': parsed_data.get('tax_status'),
            'risk_tolerance': parsed_data.get('risk_tolerance'),
            'goal': parsed_data.get('investment_goal'),
            'timeline': parsed_data.get('timeline')
        }
        
        return result
        
    except json.JSONDecodeError as e:
        st.error(f"❌ Failed to parse OpenAI response: {e}")
        st.error(f"Response was: {json_response}")
        return None
    except Exception as e:
        st.error(f"❌ OpenAI API error: {e}")
        return None

def fallback_simple_parsing(user_input):
    """Simple fallback parsing when OpenAI is not available"""
    st.warning("⚠️ Using basic parsing (OpenAI API not configured)")
    
    # Basic number extraction
    numbers = re.findall(r'\b\d+(?:,\d{3})*(?:k|K|thousand|million|M)?\b', user_input)
    large_numbers = re.findall(r'\b[1-9]\d{4,6}\b', user_input)
    all_numbers = numbers + large_numbers
    
    # Convert to actual values
    income = None
    investment = None
    
    for num_str in all_numbers:
        try:
            if 'k' in num_str.lower() or 'K' in num_str:
                value = float(num_str.replace('k', '').replace('K', '')) * 1000
            elif 'million' in num_str.lower() or 'M' in num_str:
                value = float(num_str.replace('million', '').replace('M', '')) * 1000000
            else:
                value = float(num_str.replace(',', ''))
            
            # Better logic: look for context keywords around the number
            if value > 10000:  # Large numbers
                if ('invest' in user_input.lower() or 'have' in user_input.lower()) and 'invest' in user_input.lower():
                    investment = value
                elif value > 50000 and not income:  # Likely salary range
                    income = value
                elif not investment:
                    investment = value
            else:
                # Small numbers - likely years, etc.
                continue
                
        except ValueError:
            continue
    
    # Basic stock symbol extraction (very limited)
    basic_stocks = []
    if 'TSLA' in user_input.upper() or 'Tesla' in user_input:
        basic_stocks.append('TSLA')
    if 'NVDA' in user_input.upper() or 'Nvidia' in user_input:
        basic_stocks.append('NVDA')
    if 'AAPL' in user_input.upper() or 'Apple' in user_input:
        basic_stocks.append('AAPL')
    
    # Basic state detection
    states = ['california', 'texas', 'florida', 'new york', 'washington']
    state = None
    for s in states:
        if s in user_input.lower():
            state = s
            break
    
    return {
        'stocks': basic_stocks,
        'income': income,
        'investment': investment,
        'state': state,
        'tax_status': None,
        'risk_tolerance': None,
        'goal': None,
        'timeline': None
    }

def parse_user_input(user_input):
    """Parse user input using OpenAI API with fallback"""
    # Try OpenAI first
    openai_result = parse_user_input_with_openai(user_input)
    
    if openai_result:
        return openai_result
    else:
        # Fall back to simple parsing
        return fallback_simple_parsing(user_input)

def calculate_tax_rates(income, state):
    """Calculate tax rates based on income and state"""
    if not income:
        return None
    
    # Federal rates (simplified)
    if income <= 44625:
        fed_ltcg, fed_stcg = 0.0, 0.22
    elif income <= 492300:
        fed_ltcg, fed_stcg = 0.15, 0.35
    else:
        fed_ltcg, fed_stcg = 0.20, 0.37
    
    # State rates
    state_rates = {
        'california': 0.133, 'new york': 0.109, 'new jersey': 0.105,
        'oregon': 0.099, 'minnesota': 0.099, 'washington': 0.0,
        'nevada': 0.0, 'texas': 0.0, 'florida': 0.0
    }
    
    state_cg = state_rates.get(state.lower() if state else '', 0.05)
    niit = 0.038 if income > 200000 else 0.0
    
    return {
        'total_ltcg': fed_ltcg + state_cg + niit,
        'total_stcg': fed_stcg + state_cg + niit
    }

def main():
    st.title("🎯 Options Strategy Assistant")
    st.markdown("Get personalized cash-secured put recommendations using AI-powered natural language processing")
    
    # Check API key setup
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key or openai_api_key == 'your_openai_api_key_here':
        st.warning("🔑 **OpenAI API key not configured!** Add your API key to .env file for best experience.")
        
        with st.expander("🔧 How to set up OpenAI API Key"):
            st.markdown("""
            1. Get an API key from https://platform.openai.com/api-keys
            2. Copy `.env.example` to `.env` 
            3. Add your key: `OPENAI_API_KEY=your_actual_key_here`
            4. Restart the app
            
            **Without API key:** Basic parsing will be used (limited accuracy)  
            **With API key:** Advanced AI parsing (much better results)
            """)
    else:
        st.success("✅ OpenAI API configured - Using advanced AI parsing")
    
    # Input section
    st.header("💬 Describe Your Situation")
    
    user_input = st.text_area(
        "Tell me about your financial situation and target stocks:",
        value="I'm international student, chinese, green card, have been in california for the past 5 years. I have 200000 to invest, want to invest into TSLA and NVDA, but they seems to be pricey right now.",
        height=100
    )
    
    if st.button("🚀 Analyze", type="primary"):
        if not user_input:
            st.warning("Please describe your situation first!")
            return
        
        # Parse input using OpenAI
        with st.spinner("🤖 Analyzing your input with AI..."):
            parsed = parse_user_input(user_input)
        
        if not parsed:
            st.error("Failed to parse your input. Please try rephrasing.")
            return
        
        # Display parsed info
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Detected Information")
            if parsed['income']:
                st.metric("Annual Income", f"${parsed['income']:,.0f}")
            if parsed['investment']:
                st.metric("Investment Amount", f"${parsed['investment']:,.0f}")
            if parsed['state']:
                st.write(f"**State:** {parsed['state'].title()}")
            if parsed['tax_status']:
                st.write(f"**Tax Status:** {parsed['tax_status'].title()}")
            if parsed['risk_tolerance']:
                st.write(f"**Risk Tolerance:** {parsed['risk_tolerance'].title()}")
        
        with col2:
            st.subheader("🎯 Target Stocks")
            if parsed['stocks']:
                for stock in parsed['stocks']:
                    st.write(f"• {stock}")
            else:
                st.write("No stock symbols detected")
            
            if parsed['goal']:
                st.write(f"**Goal:** {parsed['goal']}")
            if parsed['timeline']:
                st.write(f"**Timeline:** {parsed['timeline']}")
        
        # Tax analysis
        if parsed['income'] and parsed['state']:
            tax_rates = calculate_tax_rates(parsed['income'], parsed['state'])
            if tax_rates:
                st.subheader("💰 Tax Analysis")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Long-term CG", f"{tax_rates['total_ltcg']*100:.1f}%")
                with col2:
                    st.metric("Options/Short-term", f"{tax_rates['total_stcg']*100:.1f}%")
                with col3:
                    st.metric("Tax Savings Opportunity", f"{(tax_rates['total_stcg']-tax_rates['total_ltcg'])*100:.1f}%")
        
        # Analyze stocks
        if parsed['stocks']:
            st.header("📈 Options Analysis")
            
            analyzer = OptionsAnalyzer()
            
            for symbol in parsed['stocks'][:3]:  # Limit to 3 stocks
                with st.expander(f"🔍 {symbol} Analysis", expanded=True):
                    with st.spinner(f"Fetching {symbol} data..."):
                        stock_info = analyzer.get_stock_info(symbol)
                    
                    if not stock_info:
                        st.error(f"Could not fetch data for {symbol}")
                        continue
                    
                    # Display stock info
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Current Price", f"${stock_info['current_price']:.2f}")
                    with col2:
                        change = stock_info['current_price'] - stock_info['previous_close']
                        st.metric("Change", f"${change:.2f}", delta=f"{change/stock_info['previous_close']*100:.1f}%")
                    with col3:
                        st.metric("Market Cap", f"${stock_info['market_cap']/1e9:.1f}B" if stock_info['market_cap'] else "N/A")
                    with col4:
                        st.metric("Avg Volume", f"{stock_info['volume']:,.0f}" if stock_info['volume'] else "N/A")
                    
                    # Find opportunities
                    with st.spinner("Calculating optimal put options..."):
                        opportunities = analyzer.find_optimal_put_strikes(symbol)
                    
                    if opportunities:
                        st.subheader("💡 Best Put Options")
                        
                        for i, opp in enumerate(opportunities[:3]):
                            st.markdown(f"**Option {i+1}:** ${opp['strike']:.2f} Strike - {opp['dte']} Days")
                            
                            col1, col2, col3, col4, col5 = st.columns(5)
                            with col1:
                                st.metric("Premium", f"${opp['premium']:.2f}")
                            with col2:
                                st.metric("Delta", f"{opp['delta']:.3f}")
                            with col3:
                                st.metric("Win Rate", f"{opp['prob_profit']*100:.1f}%")
                            with col4:
                                st.metric("Cash Required", f"${opp['cash_required']:,.0f}")
                            with col5:
                                st.metric("Max Return", f"{opp['max_return']:.1f}%")
                            
                            # Position sizing advice
                            if parsed['investment']:
                                max_pos_size = parsed['investment'] * 0.10  # 10% max
                                if opp['cash_required'] <= max_pos_size:
                                    st.success(f"✅ Position fits your budget (${max_pos_size:,.0f} max per position)")
                                else:
                                    st.warning(f"⚠️ Position exceeds 10% limit (${max_pos_size:,.0f})")
                            
                            st.markdown("---")
                    else:
                        st.warning("No suitable options found for current market conditions")
        
        # Strategy recommendation
        if parsed['investment']:
            st.header("🎯 Strategy Recommendations")
            
            max_positions = max(1, int(parsed['investment'] / 5000))  # $5k min per position
            total_allocation = min(parsed['investment'] * 0.8, 50000)  # Max 80% allocated
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Position Sizing")
                st.write(f"• **Max positions:** {max_positions}")
                st.write(f"• **Total allocation:** ${total_allocation:,.0f}")
                st.write(f"• **Cash reserve:** ${parsed['investment'] - total_allocation:,.0f}")
                st.write(f"• **Max per position:** ${total_allocation / max_positions:,.0f}")
            
            with col2:
                st.subheader("Strategy Guidelines")
                st.write("• Target 15-25% annualized returns")
                st.write("• Focus on 70%+ win probability")
                st.write("• Use 21-45 day expirations")
                st.write("• Keep cash for potential assignments")
        
        # Next steps
        with st.expander("📚 Next Steps & Education"):
            st.markdown("""
            **Cash-Secured Put Strategy:**
            1. **Sell put options** on stocks you want to own at lower prices
            2. **Collect premium** immediately as income
            3. **Keep cash** to buy 100 shares if assigned
            4. **Manage positions** by closing at 50-70% profit
            
            **Immediate Next Steps:**
            1. Open options trading account with your broker
            2. Get Level 2 options approval (cash-secured puts)
            3. Start with paper trading to practice
            4. Monitor VIX levels - enter when VIX > 18
            5. Keep detailed records for tax purposes
            
            **Risk Management:**
            - Never risk more than 10% on one position
            - Only sell puts on stocks you want to own
            - Have exit strategy before entering
            - Understand assignment risk
            """)

if __name__ == "__main__":
    main()