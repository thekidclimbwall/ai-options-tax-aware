import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import openai
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm
import json
import re
from dotenv import load_dotenv

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Options Strategy Assistant",
    page_icon="🎯",
    layout="wide"
)

# Initialize OpenAI client
@st.cache_resource
def get_openai_client():
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        st.error("⚠️ OpenAI API key not found. Please set OPENAI_API_KEY in your .env file")
        return None
    return openai.OpenAI(api_key=api_key)

class TaxAwareOptionsAnalyzer:
    def __init__(self):
        self.risk_free_rate = 0.02
        self.options_commission = 0.65
        
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
        prob_profit = norm.cdf(d2)  # Probability option expires worthless
        
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
            # IV typically trades at premium to HV
            implied_vol = historical_vol * 1.3
            return min(0.8, max(0.15, implied_vol))
        except:
            return 0.25
    
    def get_stock_info(self, symbol):
        """Get current stock information"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="1d")
            
            if hist.empty:
                return None
                
            current_price = hist['Close'].iloc[-1]
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'market_cap': info.get('marketCap', 0),
                'volume': hist['Volume'].iloc[-1] if len(hist) > 0 else 0,
                'company_name': info.get('longName', symbol)
            }
        except:
            return None
    
    def find_optimal_put_strikes(self, symbol, target_delta_range=(0.15, 0.35)):
        """Find optimal put strike prices for the given symbol"""
        stock_info = self.get_stock_info(symbol)
        if not stock_info:
            return []
        
        current_price = stock_info['current_price']
        iv = self.estimate_implied_volatility(symbol)
        
        opportunities = []
        
        # Check different expiration periods (21-45 DTE)
        for dte in [21, 30, 45]:
            T = dte / 365.0
            
            # Check different strike prices (5-20% OTM puts)
            for otm_pct in np.arange(0.05, 0.25, 0.02):
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
                        'otm_percent': otm_pct * 100
                    })
        
        # Sort by max return
        opportunities.sort(key=lambda x: x['max_return'], reverse=True)
        return opportunities[:5]

def parse_user_input(user_input, client):
    """Use OpenAI to parse user's natural language input"""
    if not client:
        return None
    
    prompt = f"""
    Parse this user input for options trading strategy:
    "{user_input}"
    
    Extract and return JSON with these fields:
    {{
        "income": annual_income_number,
        "investment_amount": amount_to_invest,
        "stocks_interested": ["SYMBOL1", "SYMBOL2"],
        "risk_tolerance": "low/medium/high",
        "tax_status": "single/married",
        "state": "state_name",
        "investment_goal": "brief_description"
    }}
    
    If information is missing, use null. For stocks, extract ticker symbols mentioned.
    Return only the JSON, no other text.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        
        json_str = response.choices[0].message.content.strip()
        # Remove code block markers if present
        json_str = re.sub(r'```json\s*|\s*```', '', json_str)
        
        return json.loads(json_str)
    except Exception as e:
        st.error(f"Error parsing input: {e}")
        return None

def calculate_tax_rates(income, tax_status, state):
    """Calculate applicable tax rates"""
    # Federal rates (2024)
    if tax_status.lower() == 'single':
        if income <= 44625:
            fed_ltcg = 0.0
            fed_stcg = 0.22
        elif income <= 492300:
            fed_ltcg = 0.15
            fed_stcg = 0.35
        else:
            fed_ltcg = 0.20
            fed_stcg = 0.37
    else:  # married
        if income <= 89250:
            fed_ltcg = 0.0
            fed_stcg = 0.22
        elif income <= 553850:
            fed_ltcg = 0.15
            fed_stcg = 0.35
        else:
            fed_ltcg = 0.20
            fed_stcg = 0.37
    
    # State rates (simplified)
    state_rates = {
        'california': 0.133, 'new york': 0.109, 'new jersey': 0.105,
        'hawaii': 0.11, 'oregon': 0.099, 'minnesota': 0.099,
        'washington': 0.0, 'nevada': 0.0, 'texas': 0.0, 'florida': 0.0
    }
    
    state_cg = state_rates.get(state.lower(), 0.05)  # Default 5%
    
    # NIIT for high earners
    niit = 0.038 if income > 200000 else 0.0
    
    return {
        'federal_ltcg': fed_ltcg,
        'federal_stcg': fed_stcg,
        'state_cg': state_cg,
        'niit': niit,
        'total_ltcg': fed_ltcg + state_cg + niit,
        'total_stcg': fed_stcg + state_cg + niit
    }

def generate_strategy_recommendation(user_data, opportunities, client):
    """Generate personalized strategy recommendation"""
    if not client:
        return "Strategy analysis available with OpenAI API key."
    
    prompt = f"""
    Based on this user profile and options opportunities, provide a personalized strategy recommendation:
    
    User Profile:
    - Income: ${user_data.get('income', 0):,}
    - Investment Amount: ${user_data.get('investment_amount', 0):,}
    - Risk Tolerance: {user_data.get('risk_tolerance', 'medium')}
    - Tax Status: {user_data.get('tax_status', 'single')}
    - State: {user_data.get('state', 'unknown')}
    - Goals: {user_data.get('investment_goal', 'general investing')}
    
    Top Options Opportunities:
    {opportunities[:3] if opportunities else 'None found'}
    
    Provide a 2-3 paragraph recommendation covering:
    1. Suitability of cash-secured put strategy for their profile
    2. Specific position sizing and risk management advice
    3. Tax considerations and optimization strategies
    
    Write in a helpful, professional tone.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating recommendation: {e}"

def main():
    st.title("🎯 AI-Powered Options Strategy Assistant")
    st.markdown("Get personalized cash-secured put recommendations based on your profile and target stocks")
    
    # Initialize OpenAI client
    client = get_openai_client()
    
    # User input section
    st.header("💬 Tell me about your situation")
    
    user_input = st.text_area(
        "Describe your situation in natural language:",
        placeholder="Example: I make $150k a year, live in California, and want to invest $50k. I'm interested in getting exposure to NVDA and TSLA but they're too expensive right now. I'm married and have medium risk tolerance.",
        height=100
    )
    
    if st.button("🚀 Analyze My Situation", type="primary"):
        if not user_input:
            st.warning("Please describe your situation first!")
            return
        
        if not client:
            st.error("OpenAI API key required for natural language processing")
            return
        
        with st.spinner("🤖 Analyzing your input..."):
            # Parse user input
            user_data = parse_user_input(user_input, client)
            
            if not user_data:
                st.error("Could not parse your input. Please try rephrasing.")
                return
            
            # Display parsed information
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Your Profile")
                if user_data.get('income'):
                    st.metric("Annual Income", f"${user_data['income']:,}")
                if user_data.get('investment_amount'):
                    st.metric("Investment Amount", f"${user_data['investment_amount']:,}")
                if user_data.get('risk_tolerance'):
                    st.write(f"**Risk Tolerance:** {user_data['risk_tolerance'].title()}")
                if user_data.get('tax_status'):
                    st.write(f"**Tax Status:** {user_data['tax_status'].title()}")
                if user_data.get('state'):
                    st.write(f"**State:** {user_data['state'].title()}")
            
            with col2:
                st.subheader("🎯 Target Stocks")
                stocks = user_data.get('stocks_interested', [])
                if stocks:
                    for stock in stocks:
                        st.write(f"• {stock}")
                else:
                    st.write("No specific stocks mentioned")
        
        # Calculate tax rates
        if user_data.get('income') and user_data.get('state'):
            tax_rates = calculate_tax_rates(
                user_data['income'], 
                user_data.get('tax_status', 'single'),
                user_data['state']
            )
            
            st.subheader("💰 Your Tax Situation")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Long-term Capital Gains", f"{tax_rates['total_ltcg']*100:.1f}%")
            with col2:
                st.metric("Short-term/Options", f"{tax_rates['total_stcg']*100:.1f}%")
            with col3:
                st.metric("Tax Savings Opportunity", f"{(tax_rates['total_stcg']-tax_rates['total_ltcg'])*100:.1f}%")
        
        # Analyze target stocks
        stocks = user_data.get('stocks_interested', [])
        if stocks:
            st.header("📈 Options Analysis")
            
            analyzer = TaxAwareOptionsAnalyzer()
            all_opportunities = []
            
            for stock in stocks:
                with st.expander(f"🔍 {stock} Analysis", expanded=True):
                    stock_info = analyzer.get_stock_info(stock)
                    
                    if not stock_info:
                        st.error(f"Could not fetch data for {stock}")
                        continue
                    
                    # Display stock info
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Price", f"${stock_info['current_price']:.2f}")
                    with col2:
                        st.metric("Market Cap", f"${stock_info['market_cap']/1e9:.1f}B")
                    with col3:
                        st.metric("Volume", f"{stock_info['volume']:,}")
                    
                    # Find opportunities
                    opportunities = analyzer.find_optimal_put_strikes(stock)
                    
                    if opportunities:
                        st.subheader("💡 Best Put Options")
                        
                        for i, opp in enumerate(opportunities[:3]):
                            with st.container():
                                col1, col2, col3, col4, col5 = st.columns(5)
                                with col1:
                                    st.metric("Strike", f"${opp['strike']:.2f}")
                                with col2:
                                    st.metric("Premium", f"${opp['premium']:.2f}")
                                with col3:
                                    st.metric("Days", f"{opp['dte']}")
                                with col4:
                                    st.metric("Win Rate", f"{opp['prob_profit']*100:.1f}%")
                                with col5:
                                    st.metric("Max Return", f"{opp['max_return']:.1f}%")
                                
                                st.write(f"💰 **Cash Required:** ${opp['cash_required']:,} | **Premium Income:** ${opp['premium_income']:.0f}")
                                st.markdown("---")
                        
                        all_opportunities.extend([(stock, opp) for opp in opportunities[:3]])
                    else:
                        st.warning(f"No suitable options found for {stock}")
            
            # Generate recommendation
            if all_opportunities:
                st.header("🎯 Personalized Strategy Recommendation")
                
                with st.spinner("🤖 Generating personalized advice..."):
                    recommendation = generate_strategy_recommendation(
                        user_data, 
                        [opp[1] for opp in all_opportunities], 
                        client
                    )
                    st.write(recommendation)
                
                # Position sizing calculator
                if user_data.get('investment_amount'):
                    st.subheader("📊 Position Sizing Guide")
                    
                    investment_amount = user_data['investment_amount']
                    max_position_pct = 0.10 if user_data.get('risk_tolerance') == 'high' else 0.05
                    max_position_size = investment_amount * max_position_pct
                    
                    st.info(f"💡 **Recommended max position size:** ${max_position_size:,.0f} ({max_position_pct*100:.0f}% of portfolio)")
                    
                    suitable_positions = []
                    for stock, opp in all_opportunities:
                        if opp['cash_required'] <= max_position_size:
                            suitable_positions.append((stock, opp))
                    
                    if suitable_positions:
                        st.write("✅ **Positions within your size limit:**")
                        for stock, opp in suitable_positions[:5]:
                            st.write(f"• **{stock}** ${opp['strike']:.2f} put: ${opp['cash_required']:,} required, {opp['max_return']:.1f}% max return")
                    else:
                        st.warning("⚠️ All positions exceed your recommended size limit. Consider increasing investment amount or targeting lower-priced stocks.")

    # Educational section
    with st.expander("📚 How This Strategy Works"):
        st.markdown("""
        **Cash-Secured Put Strategy:**
        1. **Sell put options** on stocks you'd like to own at lower prices
        2. **Collect premium** immediately (your income)
        3. **Keep cash** to buy shares if assigned
        4. **Tax advantages** - manage timing of gains/losses
        
        **Benefits:**
        - Generate income while waiting to buy stocks
        - Get stocks at lower prices if assigned
        - Tax-efficient compared to dividends
        - Defined risk (premium received reduces cost basis)
        
        **Risks:**
        - Must own stock if assigned
        - Limited upside (premium only)
        - Requires significant cash
        """)

if __name__ == "__main__":
    main()