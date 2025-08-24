import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from simulation_engine import TaxAwareOptionsSimulator

# Page configuration
st.set_page_config(
    page_title="AI-Powered Tax-Aware Options Strategy",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .kpi-container {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_simulation_data():
    """Load simulation results from pickle file"""
    if os.path.exists('simulation_results/simulation_data.pkl'):
        with open('simulation_results/simulation_data.pkl', 'rb') as f:
            return pickle.load(f)
    return None

@st.cache_data
def run_simulation():
    """Run the simulation and return results"""
    simulator = TaxAwareOptionsSimulator()
    return simulator.execute_simulation()

def calculate_performance_metrics(daily_returns_df):
    """Calculate key performance metrics"""
    if daily_returns_df.empty:
        return {}
    
    # Calculate returns
    strategy_returns = daily_returns_df['strategy_return'].values
    benchmark_returns = daily_returns_df['benchmark_return'].values
    
    # CAGR calculation
    total_days = len(daily_returns_df)
    years = total_days / 252.0  # Trading days in a year
    
    strategy_cagr = ((1 + strategy_returns[-1]) ** (1/years) - 1) * 100 if len(strategy_returns) > 0 else 0
    benchmark_cagr = ((1 + benchmark_returns[-1]) ** (1/years) - 1) * 100 if len(benchmark_returns) > 0 else 0
    
    # Volatility (annualized)
    strategy_vol = np.std(np.diff(strategy_returns)) * np.sqrt(252) * 100 if len(strategy_returns) > 1 else 0
    benchmark_vol = np.std(np.diff(benchmark_returns)) * np.sqrt(252) * 100 if len(benchmark_returns) > 1 else 0
    
    # Sharpe Ratio (assuming 2% risk-free rate)
    rf_rate = 0.02
    strategy_sharpe = (strategy_cagr/100 - rf_rate) / (strategy_vol/100) if strategy_vol > 0 else 0
    benchmark_sharpe = (benchmark_cagr/100 - rf_rate) / (benchmark_vol/100) if benchmark_vol > 0 else 0
    
    # Max Drawdown
    strategy_cumulative = (1 + strategy_returns).cumprod()
    strategy_peak = np.maximum.accumulate(strategy_cumulative)
    strategy_drawdown = ((strategy_cumulative - strategy_peak) / strategy_peak * 100).min()
    
    benchmark_cumulative = (1 + benchmark_returns).cumprod()
    benchmark_peak = np.maximum.accumulate(benchmark_cumulative)
    benchmark_drawdown = ((benchmark_cumulative - benchmark_peak) / benchmark_peak * 100).min()
    
    # Alpha
    alpha = strategy_cagr - benchmark_cagr
    
    # Tax drag (simplified calculation)
    pretax_return = strategy_cagr / 0.75  # Assuming 25% average tax rate
    tax_drag = pretax_return - strategy_cagr
    
    return {
        'strategy_cagr': strategy_cagr,
        'benchmark_cagr': benchmark_cagr,
        'strategy_vol': strategy_vol,
        'benchmark_vol': benchmark_vol,
        'strategy_sharpe': strategy_sharpe,
        'benchmark_sharpe': benchmark_sharpe,
        'strategy_max_dd': strategy_drawdown,
        'benchmark_max_dd': benchmark_drawdown,
        'alpha': alpha,
        'tax_drag': tax_drag
    }

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("🤖 AI-Powered Tax-Aware Options Strategy: Backtest Analysis")
    
    # Simulation parameters
    st.sidebar.header("📋 Simulation Parameters")
    st.sidebar.markdown("""
    **Investor Profile:** Sarah, Software Engineer, Palo Alto, CA  
    **Income:** $350,000 annually  
    **Initial Capital:** $500,000  
    **Period:** Jan 2020 - Dec 2024  
    **Strategy:** Cash-Secured Put Selling  
    **Benchmark:** 50% SPY / 50% BIL  
    """)
    
    # Load or run simulation
    data = load_simulation_data()
    
    if data is None:
        st.warning("🔄 No simulation data found. Running simulation...")
        
        # Progress bar for simulation
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("Running tax-aware options simulation..."):
            status_text.text("Initializing simulation engine...")
            progress_bar.progress(20)
            
            status_text.text("Fetching market data...")
            progress_bar.progress(40)
            
            status_text.text("Executing strategy...")
            data = run_simulation()
            progress_bar.progress(80)
            
            status_text.text("Calculating performance metrics...")
            progress_bar.progress(100)
            
        st.success("✅ Simulation completed successfully!")
        status_text.empty()
        progress_bar.empty()
    
    if data is None:
        st.error("❌ Failed to load or generate simulation data.")
        return
    
    # Extract data
    daily_returns_df = data['daily_returns']
    trade_log_df = data['trade_log']
    
    if daily_returns_df.empty:
        st.error("❌ No simulation data available.")
        return
    
    # Calculate performance metrics
    metrics = calculate_performance_metrics(daily_returns_df)
    
    # KPI Cards
    st.header("📊 Key Performance Indicators")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "After-Tax CAGR",
            f"{metrics.get('strategy_cagr', 0):.2f}%",
            delta=f"{metrics.get('alpha', 0):.2f}% vs benchmark"
        )
    
    with col2:
        st.metric(
            "Sharpe Ratio",
            f"{metrics.get('strategy_sharpe', 0):.2f}",
            delta=f"{metrics.get('strategy_sharpe', 0) - metrics.get('benchmark_sharpe', 0):.2f}"
        )
    
    with col3:
        st.metric(
            "Max Drawdown",
            f"{metrics.get('strategy_max_dd', 0):.2f}%",
            delta=f"{metrics.get('strategy_max_dd', 0) - metrics.get('benchmark_max_dd', 0):.2f}%"
        )
    
    with col4:
        st.metric(
            "Volatility",
            f"{metrics.get('strategy_vol', 0):.2f}%",
            delta=f"{metrics.get('strategy_vol', 0) - metrics.get('benchmark_vol', 0):.2f}%"
        )
    
    with col5:
        st.metric(
            "Tax Drag",
            f"{metrics.get('tax_drag', 0):.2f}%",
            delta="vs pre-tax returns"
        )
    
    # Equity Curve Chart
    st.header("📈 Portfolio Performance")
    
    fig = go.Figure()
    
    # Strategy line
    fig.add_trace(go.Scatter(
        x=daily_returns_df['date'],
        y=daily_returns_df['portfolio_value'],
        mode='lines',
        name='Tax-Aware Strategy',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Benchmark line
    fig.add_trace(go.Scatter(
        x=daily_returns_df['date'],
        y=daily_returns_df['benchmark_value'],
        mode='lines',
        name='50% SPY / 50% BIL Benchmark',
        line=dict(color='#ff7f0e', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title="After-Tax Portfolio Value Over Time",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode='x unified',
        legend=dict(x=0, y=1),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance Comparison Table
    st.header("📋 Performance Comparison")
    
    comparison_data = {
        'Metric': [
            'After-Tax CAGR (%)',
            'Volatility (%)',
            'Sharpe Ratio',
            'Max Drawdown (%)',
            'Final Value ($)',
        ],
        'Tax-Aware Strategy': [
            f"{metrics.get('strategy_cagr', 0):.2f}",
            f"{metrics.get('strategy_vol', 0):.2f}",
            f"{metrics.get('strategy_sharpe', 0):.2f}",
            f"{metrics.get('strategy_max_dd', 0):.2f}",
            f"${daily_returns_df['portfolio_value'].iloc[-1]:,.0f}" if len(daily_returns_df) > 0 else "$0"
        ],
        'Benchmark': [
            f"{metrics.get('benchmark_cagr', 0):.2f}",
            f"{metrics.get('benchmark_vol', 0):.2f}",
            f"{metrics.get('benchmark_sharpe', 0):.2f}",
            f"{metrics.get('benchmark_max_dd', 0):.2f}",
            f"${daily_returns_df['benchmark_value'].iloc[-1]:,.0f}" if len(daily_returns_df) > 0 else "$0"
        ],
        'Alpha/Difference': [
            f"+{metrics.get('alpha', 0):.2f}",
            f"{metrics.get('strategy_vol', 0) - metrics.get('benchmark_vol', 0):+.2f}",
            f"{metrics.get('strategy_sharpe', 0) - metrics.get('benchmark_sharpe', 0):+.2f}",
            f"{metrics.get('strategy_max_dd', 0) - metrics.get('benchmark_max_dd', 0):+.2f}",
            f"${daily_returns_df['portfolio_value'].iloc[-1] - daily_returns_df['benchmark_value'].iloc[-1]:+,.0f}" if len(daily_returns_df) > 0 else "$0"
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Trade Log Section
    st.header("🔍 Explainable AI - Trade Log")
    
    if not trade_log_df.empty:
        # Filter controls
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Ticker filter
            unique_symbols = ['All'] + sorted(trade_log_df['symbol'].unique().tolist())
            selected_symbol = st.selectbox("Filter by Ticker:", unique_symbols)
        
        with col2:
            # Action filter
            unique_actions = ['All'] + sorted(trade_log_df['action'].unique().tolist())
            selected_action = st.multiselect("Filter by Action:", unique_actions, default=['All'])
        
        # Apply filters
        filtered_df = trade_log_df.copy()
        
        if selected_symbol != 'All':
            filtered_df = filtered_df[filtered_df['symbol'] == selected_symbol]
        
        if 'All' not in selected_action and selected_action:
            filtered_df = filtered_df[filtered_df['action'].isin(selected_action)]
        
        # Format the dataframe for display
        display_df = filtered_df.copy()
        display_df['date'] = pd.to_datetime(display_df['date']).dt.strftime('%Y-%m-%d')
        display_df['premium'] = display_df['premium'].apply(lambda x: f"${x:.2f}")
        display_df['strike'] = display_df['strike'].apply(lambda x: f"${x:.2f}")
        display_df['pnl'] = display_df['pnl'].apply(lambda x: f"${x:,.2f}")
        
        # Select and rename columns for display
        display_columns = {
            'date': 'Date',
            'symbol': 'Ticker',
            'action': 'Action',
            'strike': 'Strike',
            'premium': 'Premium',
            'pnl': 'P/L',
            'rationale': 'AI Rationale'
        }
        
        display_df = display_df[list(display_columns.keys())].rename(columns=display_columns)
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", len(filtered_df))
        
        with col2:
            total_pnl = trade_log_df['pnl'].sum() if not trade_log_df.empty else 0
            st.metric("Total P/L", f"${total_pnl:,.2f}")
        
        with col3:
            avg_trade = trade_log_df['pnl'].mean() if not trade_log_df.empty else 0
            st.metric("Avg Trade P/L", f"${avg_trade:.2f}")
        
        with col4:
            win_rate = (trade_log_df['pnl'] > 0).mean() * 100 if not trade_log_df.empty else 0
            st.metric("Win Rate", f"{win_rate:.1f}%")
        
        # Interactive trade log table
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "AI Rationale": st.column_config.TextColumn(
                    width="medium",
                )
            }
        )
        
        # Trade distribution chart
        if len(filtered_df) > 0:
            st.subheader("📊 Trade Distribution by Symbol")
            
            symbol_counts = filtered_df['symbol'].value_counts().head(10)
            
            fig_bar = px.bar(
                x=symbol_counts.index,
                y=symbol_counts.values,
                labels={'x': 'Symbol', 'y': 'Number of Trades'},
                title="Top 10 Most Traded Symbols"
            )
            
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
        
    else:
        st.info("No trade data available.")
    
    # Additional Analysis
    st.header("🎯 Strategy Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk-Adjusted Returns")
        st.markdown(f"""
        The tax-aware options strategy generated:
        - **After-tax CAGR:** {metrics.get('strategy_cagr', 0):.2f}%
        - **Alpha vs Benchmark:** {metrics.get('alpha', 0):.2f}%
        - **Sharpe Ratio:** {metrics.get('strategy_sharpe', 0):.2f}
        - **Max Drawdown:** {metrics.get('strategy_max_dd', 0):.2f}%
        """)
    
    with col2:
        st.subheader("Tax Efficiency")
        st.markdown(f"""
        Tax-aware features implemented:
        - **HIFO Accounting** for share sales
        - **Wash Sale Avoidance** (61-day window)
        - **Year-end Loss Harvesting**
        - **Tax Drag:** {metrics.get('tax_drag', 0):.2f}%
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("🤖 *Generated with AI-Powered Tax-Aware Options Strategy Engine*")

if __name__ == "__main__":
    main()