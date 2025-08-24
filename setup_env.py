import os

def setup_environment():
    """Setup the environment for the AI Options Strategy Assistant"""
    
    print("🎯 AI Options Strategy Assistant Setup")
    print("=====================================")
    
    # Check if .env exists
    if os.path.exists('.env'):
        print("✅ .env file already exists")
    else:
        print("📝 Creating .env file...")
        
        # Get API key from user
        api_key = input("\n🔑 Enter your OpenAI API key (or press Enter to skip): ").strip()
        
        if api_key:
            with open('.env', 'w') as f:
                f.write(f"OPENAI_API_KEY={api_key}\n")
            print("✅ .env file created successfully!")
        else:
            print("⚠️  No API key provided. You can add it later to the .env file.")
            with open('.env', 'w') as f:
                f.write("OPENAI_API_KEY=your_openai_api_key_here\n")
    
    print("\n🚀 Setup Complete!")
    print("\nTo run the application:")
    print("  streamlit run interactive_app.py")
    print("\nTo get an OpenAI API key:")
    print("  1. Go to https://platform.openai.com/api-keys")
    print("  2. Create a new API key")
    print("  3. Add it to your .env file")

if __name__ == "__main__":
    setup_environment()