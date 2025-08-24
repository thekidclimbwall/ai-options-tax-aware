#!/usr/bin/env python3
"""
Demo script showing the AI Options Strategy Assistant functionality
"""

import yfinance as yf
import numpy as np
from scipy.stats import norm

class OptionsDemo:
    def __init__(self):
        self.risk_free_rate = 0.02
        
    def get_stock_price(self, symbol):
        """Get current stock price"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            return hist['Close'].iloc[-1] if not hist.empty else None
        except:
            return None
    
    def calculate_put_option(self, stock_price, strike, days, iv=0.25):
        """Calculate put option price and metrics"""
        T = days / 365.0
        
        if T <= 0:
            return None
        
        d1 = (np.log(stock_price / strike) + (self.risk_free_rate + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
        d2 = d1 - iv * np.sqrt(T)
        
        put_price = strike * np.exp(-self.risk_free_rate * T) * norm.cdf(-d2) - stock_price * norm.cdf(-d1)
        delta = abs(-norm.cdf(-d1))
        prob_profit = norm.cdf(d2)
        
        return {
            'premium': put_price,
            'delta': delta,
            'prob_profit': prob_profit,
            'cash_required': strike * 100,
            'premium_income': put_price * 100,
            'max_return': (put_price * 100) / (strike * 100) * (365 / days) * 100
        }
    
    def analyze_stock(self, symbol):
        """Analyze options opportunities for a stock"""
        print(f"\n🔍 Analyzing {symbol}")
        print("=" * 40)
        
        price = self.get_stock_price(symbol)
        if not price:
            print(f"❌ Could not fetch data for {symbol}")
            return
        
        print(f"Current Price: ${price:.2f}")
        
        print(f"\n💡 Cash-Secured Put Opportunities:")
        print("-" * 50)
        
        # Analyze different strikes and expirations
        opportunities = []
        
        for dte in [21, 30, 45]:
            for otm_pct in [0.05, 0.10, 0.15, 0.20]:
                strike = price * (1 - otm_pct)
                option = self.calculate_put_option(price, strike, dte)
                
                if option and 0.15 <= option['delta'] <= 0.35:
                    opportunities.append({
                        'strike': strike,
                        'dte': dte,
                        'otm_pct': otm_pct * 100,
                        **option
                    })
        
        # Sort by max return and show top 3
        opportunities.sort(key=lambda x: x['max_return'], reverse=True)
        
        for i, opp in enumerate(opportunities[:3], 1):
            print(f"\n{i}. Strike ${opp['strike']:.2f} ({opp['otm_pct']:.0f}% OTM) - {opp['dte']} days")
            print(f"   Premium: ${opp['premium']:.2f} per share")
            print(f"   Cash Required: ${opp['cash_required']:,}")
            print(f"   Premium Income: ${opp['premium_income']:.0f}")
            print(f"   Win Probability: {opp['prob_profit']*100:.1f}%")
            print(f"   Max Annualized Return: {opp['max_return']:.1f}%")
            print(f"   Delta: {opp['delta']:.3f}")

def demo_scenario():
    """Demo scenario matching user's request"""
    print("🎯 AI Options Strategy Assistant - Demo")
    print("=" * 50)
    
    # Simulate user input
    print("\n💬 User Input:")
    print("\"I make $150k a year, live in California, and want to invest $50k.")
    print("I'm interested in getting exposure to NVDA and TSLA but they're too expensive.\"")
    
    # Parse scenario
    print("\n📊 Parsed Profile:")
    print("- Annual Income: $150,000")
    print("- Investment Amount: $50,000") 
    print("- Location: California")
    print("- Target Stocks: NVDA, TSLA")
    
    # Calculate tax rates
    print("\n💰 Tax Analysis:")
    print("- Federal Short-term: 22%")
    print("- California State: 13.3%") 
    print("- Total Options Tax Rate: ~35.3%")
    print("- Recommended Strategy: Cash-secured puts for tax efficiency")
    
    # Analyze target stocks
    analyzer = OptionsDemo()
    
    for symbol in ['NVDA', 'TSLA']:
        analyzer.analyze_stock(symbol)
    
    print("\n🎯 Strategy Recommendation:")
    print("-" * 30)
    print("Based on your $50k investment amount and tax situation:")
    print("• Maximum position size: $5,000 (10% per position)")
    print("• Focus on higher-probability, shorter-term puts")
    print("• Target 15-25% annualized returns after taxes")
    print("• Keep cash ready for potential assignments")
    print("• Use HIFO accounting if assigned shares later")
    
    print("\n✅ Next Steps:")
    print("1. Open options-approved brokerage account")
    print("2. Set aside cash for selected positions")
    print("3. Monitor VIX levels (enter when VIX > 18)")
    print("4. Start with paper trading to practice")
    
    print("\n🚀 Launch the full app with: streamlit run interactive_app.py")

if __name__ == "__main__":
    demo_scenario()