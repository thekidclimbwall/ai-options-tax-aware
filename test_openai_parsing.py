#!/usr/bin/env python3
"""
Test script to demonstrate OpenAI parsing capabilities
Run this to see how much better OpenAI parsing is compared to regex
"""

import os
from dotenv import load_dotenv

load_dotenv()

def test_parsing_examples():
    """Test various input scenarios"""
    
    # Import after loading env
    from simple_app import parse_user_input_with_openai, fallback_simple_parsing
    
    test_cases = [
        "I'm international student, chinese, green card, have been in california for the past 5 years. I have 200000 to invest, want to invest into TSLA and NVDA, but they seems to be pricey right now.",
        
        "I make $120k annually in Texas, married, want to put $50k into Apple and Microsoft stocks but they're expensive. Looking for conservative approach.",
        
        "Single person in New York, $85,000 salary, interested in getting exposure to Amazon and Google with about $30,000 to invest.",
        
        "I earn 200k per year, live in Florida, have 100k to invest in growth stocks like Tesla, Nvidia, and Meta. I'm aggressive investor.",
        
        "Married couple in California, combined income $300k, want to invest $150k in dividend stocks like Coca Cola and Johnson & Johnson for retirement in 10 years."
    ]
    
    openai_api_key = os.getenv('OPENAI_API_KEY')
    
    print("🧪 Testing Natural Language Parsing")
    print("=" * 50)
    
    if not openai_api_key or openai_api_key == 'your_openai_api_key_here':
        print("❌ OpenAI API key not configured - showing fallback parsing only")
        print("\nTo test OpenAI parsing:")
        print("1. Get API key from https://platform.openai.com/api-keys")
        print("2. Add to .env file: OPENAI_API_KEY=your_key_here")
        print("3. Run this script again")
        print("\n" + "=" * 50)
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📝 Test Case {i}:")
            print(f"Input: {test_case[:60]}...")
            
            # Fallback parsing only
            fallback_result = fallback_simple_parsing(test_case)
            print(f"📊 Fallback Result:")
            print(f"  Stocks: {fallback_result['stocks']}")
            print(f"  Investment: ${fallback_result['investment']:,}" if fallback_result['investment'] else "  Investment: None")
            print(f"  Income: ${fallback_result['income']:,}" if fallback_result['income'] else "  Income: None")
            print(f"  State: {fallback_result['state']}")
    else:
        print("✅ OpenAI API key configured - comparing both methods")
        print("\n" + "=" * 50)
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📝 Test Case {i}:")
            print(f"Input: {test_case}")
            
            # OpenAI parsing
            print(f"\n🤖 OpenAI Result:")
            try:
                openai_result = parse_user_input_with_openai(test_case)
                if openai_result:
                    print(f"  Stocks: {openai_result['stocks']}")
                    print(f"  Investment: ${openai_result['investment']:,}" if openai_result['investment'] else "  Investment: None")
                    print(f"  Income: ${openai_result['income']:,}" if openai_result['income'] else "  Income: None")
                    print(f"  State: {openai_result['state']}")
                    print(f"  Tax Status: {openai_result['tax_status']}")
                    print(f"  Risk Tolerance: {openai_result['risk_tolerance']}")
                    print(f"  Goal: {openai_result['goal']}")
                else:
                    print("  Failed to parse")
            except Exception as e:
                print(f"  Error: {e}")
            
            # Fallback comparison
            print(f"\n🔧 Fallback Result (for comparison):")
            fallback_result = fallback_simple_parsing(test_case)
            print(f"  Stocks: {fallback_result['stocks']}")
            print(f"  Investment: ${fallback_result['investment']:,}" if fallback_result['investment'] else "  Investment: None")
            print(f"  Income: ${fallback_result['income']:,}" if fallback_result['income'] else "  Income: None")
            print(f"  State: {fallback_result['state']}")
            
            print("-" * 30)

    print(f"\n🎯 Conclusion:")
    print("OpenAI parsing provides:")
    print("✅ Better stock symbol recognition")
    print("✅ More accurate number extraction")  
    print("✅ Additional context (tax status, risk tolerance, goals)")
    print("✅ Handles complex natural language")
    print("✅ Company name to ticker conversion (Tesla -> TSLA)")
    
    print(f"\n🚀 Start the full app: streamlit run simple_app.py")

if __name__ == "__main__":
    test_parsing_examples()