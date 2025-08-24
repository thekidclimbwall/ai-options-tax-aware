#!/usr/bin/env python3
"""
Quick verification script to check if the setup is correct
"""

import sys

def check_setup():
    """Check if everything is set up correctly"""
    print("🔍 Checking AI Options Strategy Assistant setup...")
    
    # Check Python version
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Need 3.8+")
        return False
    
    # Check core dependencies
    required_packages = [
        'streamlit',
        'pandas', 
        'numpy',
        'yfinance',
        'plotly',
        'scipy',
        'requests',
        'dotenv'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - OK")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    # Test core functionality
    try:
        from simple_app import OptionsAnalyzer, parse_user_input
        print("✅ Core modules - OK")
        
        # Test options analyzer
        analyzer = OptionsAnalyzer()
        if hasattr(analyzer, 'polygon_api_key') and analyzer.polygon_api_key:
            print("✅ Polygon API key - OK") 
        else:
            print("⚠️ Polygon API key - Not configured")
        
        # Test parsing function exists
        test_text = "I want to invest $50000 in TSLA and NVDA"
        result = parse_user_input(test_text)
        if result and isinstance(result, dict):
            print("✅ AI parsing function - OK")
        else:
            print("⚠️ AI parsing function - May need API key")
            
    except Exception as e:
        print(f"❌ Core functionality test failed: {e}")
        return False
    
    print("\n🎉 Setup verification completed!")
    print("🚀 Ready to run: streamlit run simple_app.py")
    return True

if __name__ == "__main__":
    success = check_setup()
    sys.exit(0 if success else 1)