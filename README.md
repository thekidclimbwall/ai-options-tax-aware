# 🎯 AI Options Strategy Assistant

An intelligent, interactive tool for personalized cash-secured put options strategies using natural language input. Get AI-powered investment recommendations tailored to your financial situation and target stocks.

![Options Strategy Demo](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

## 🚀 What This Tool Does

Simply describe your financial situation in plain English and get:

- **📊 Instant Tax Analysis** - Calculate your specific tax rates based on income and location
- **📈 Real-time Options Data** - Live options prices for stocks you want to own
- **🤖 AI-Powered Recommendations** - Personalized strategy advice tailored to your profile
- **💰 Position Sizing Guidance** - Risk-appropriate investment allocation
- **🎯 Smart Stock Detection** - Automatically identifies stocks mentioned in your description

### Example Input:
*"I'm an international student with a green card, have been in California for 5 years. I have $200,000 to invest, want to get exposure to TSLA and NVDA, but they seem pricey right now."*

### What You Get:
- Tax analysis showing your 35%+ total tax rate in California
- Live TSLA and NVDA put option prices with strike prices and premiums
- Position sizing recommendations (max 10% per position)
- Win probability calculations for each option strategy
- Personalized risk management advice

## ⚡ Quick Start

### Prerequisites
- Python 3.8 or higher
- Internet connection for real-time stock data

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/thekidclimbwall/ai-options-tax-aware.git
   cd ai-options-tax-aware
   ```

2. **Run immediately (All API keys pre-configured):**
   ```bash
   python setup.py
   ```
   *Installs dependencies and starts the app automatically*

   **OR Manual setup:**
   ```bash
   pip install -r requirements.txt
   streamlit run simple_app.py
   ```

3. **Open your browser:** Go to `http://localhost:8502`

4. **Start using:** The app comes with a sample input - just click "🚀 Analyze" to see it in action!

## 🔧 Configuration

### API Keys (Optional)

While the app works without API keys using free data sources, adding API keys provides enhanced features:

#### API Keys - ✅ All Pre-Configured!

**✅ OpenAI API Key**
- **Status:** Ready to use immediately
- **Purpose:** Advanced natural language processing for perfect stock detection
- **Capabilities:** "Apple and Microsoft" → Automatically extracts AAPL, MSFT

**✅ Polygon.io API Key**
- **Status:** Ready to use immediately  
- **Purpose:** Reliable stock data fallback when yfinance is rate-limited
- **No setup required:** Everything is embedded for private repo use

### Ready to Use Immediately! 

**No environment setup needed** - all API keys are pre-configured for this private repository.

## 💬 How to Use

### 1. Describe Your Situation
Tell the app about yourself in natural language:
- Your income and tax status
- How much you want to invest
- Which stocks interest you
- Your location (for tax calculations)

### 2. Get Instant Analysis
The app automatically:
- Parses your input to extract key information
- Calculates your tax rates based on income and location
- Fetches real-time stock and options data
- Identifies suitable put option strikes

### 3. Review Recommendations
View personalized advice including:
- Optimal strike prices and expiration dates
- Premium income and cash requirements
- Win probabilities and maximum returns
- Position sizing based on your budget
- Tax-optimized strategy guidance

## 📊 Features

### Smart Input Processing
- **Context-Aware Stock Detection:** Identifies stock symbols from natural language
- **Intelligent Number Parsing:** Distinguishes between income and investment amounts
- **Geographic Tax Calculation:** Supports all US states with accurate tax rates

### Real-Time Market Data
- **Live Stock Prices:** Current market prices and volume data
- **Options Analysis:** Black-Scholes pricing with Greeks calculations
- **Fallback Data Sources:** Multiple APIs ensure reliable data access

### Personalized Strategy
- **Tax-Aware Recommendations:** Optimized for after-tax returns
- **Risk-Appropriate Sizing:** Position limits based on portfolio size
- **Win Probability Analysis:** Statistical success rates for each trade

### Interactive Dashboard
- **Clean Interface:** Intuitive Streamlit web application
- **Responsive Design:** Works on desktop and mobile
- **Real-Time Updates:** Live market data integration

## 🎯 Cash-Secured Put Strategy

### What It Is
A conservative options strategy where you:
1. **Sell put options** on stocks you'd like to own at lower prices
2. **Collect premium** immediately as income
3. **Keep cash** to buy 100 shares if assigned
4. **Manage positions** by closing at 50-70% profit

### Benefits
- ✅ Generate income while waiting to buy stocks
- ✅ Get stocks at lower prices if assigned
- ✅ Tax-efficient compared to dividends
- ✅ Defined risk (premium reduces cost basis)

### Risks
- ⚠️ Must own stock if assigned
- ⚠️ Limited upside (premium only)
- ⚠️ Requires significant cash reserves

## 🏗️ Project Structure

```
ai-options-strategy/
├── simple_app.py          # Main Streamlit application
├── interactive_app.py     # Advanced version with full OpenAI integration
├── demo.py               # Command-line demo script
├── simulation_engine.py   # Historical backtesting engine
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## 🔒 Security & Privacy

- **No Data Storage:** Your personal information is not saved
- **API Key Protection:** Keys stored in environment variables only
- **Local Processing:** All calculations run on your machine
- **Open Source:** Full transparency of all code

## 🚨 Important Disclaimers

- **Educational Purpose:** This tool is for educational and research purposes only
- **Not Investment Advice:** Always consult with a qualified financial advisor
- **Risk Warning:** Options trading involves substantial risk of loss
- **Tax Complexity:** Actual tax treatment may vary - consult a tax professional
- **Data Accuracy:** While we strive for accuracy, verify all data independently

## 📈 Advanced Features

### Historical Backtesting
Run the full backtesting simulation:
```bash
python simulation_engine.py
streamlit run app.py  # View full backtest results
```

### Command Line Demo
Test the core functionality:
```bash
python demo.py
```

### Natural Language Interface
With OpenAI API key, enjoy enhanced parsing:
```bash
streamlit run interactive_app.py
```

## 🛠️ Development

### Running Tests
```bash
python -m pytest tests/  # When tests are added
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Support

- **Issues:** Report bugs via GitHub Issues
- **Discussions:** Use GitHub Discussions for questions
- **Documentation:** This README and inline code comments

## 🌟 Acknowledgments

- **Data Sources:** Yahoo Finance, Polygon.io
- **AI Integration:** OpenAI GPT models
- **Framework:** Streamlit for the web interface
- **Calculations:** Black-Scholes options pricing model

---

**⚡ Ready to get started?** Clone the repo and run `streamlit run simple_app.py`

**💡 Need help?** Check the issues page or start a discussion

**🎯 Want to contribute?** PRs welcome!
