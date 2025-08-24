#!/usr/bin/env python3
"""
Setup script for AI Options Strategy Assistant
Automates the installation and setup process
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {description}: {e}")
        print(f"Output: {e.output}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def setup_environment():
    """Set up the environment and install dependencies"""
    print("🎯 AI Options Strategy Assistant Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        return False
    
    # Create .env file if it doesn't exist
    if not os.path.exists('.env'):
        print("📝 Creating .env file...")
        try:
            with open('.env.example', 'r') as source:
                content = source.read()
            with open('.env', 'w') as target:
                target.write(content)
            print("✅ .env file created from template")
        except Exception as e:
            print(f"⚠️ Could not create .env file: {e}")
    else:
        print("✅ .env file already exists")
    
    # Test import of main dependencies
    print("🧪 Testing core dependencies...")
    try:
        import streamlit
        import pandas
        import numpy
        import yfinance
        import plotly
        print("✅ All core dependencies imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("\n🚀 To start the application:")
    print("   streamlit run simple_app.py")
    print("\n🌐 The app will open in your browser at http://localhost:8502")
    print("\n💡 Features:")
    print("   • Natural language input processing")
    print("   • Real-time stock and options data") 
    print("   • Tax-aware strategy recommendations")
    print("   • Interactive web dashboard")
    
    return True

def main():
    """Main setup function"""
    try:
        success = setup_environment()
        if success:
            try:
                # Ask user if they want to start the app immediately
                start_app = input("\n❓ Start the application now? (y/n): ").lower().strip()
                if start_app in ['y', 'yes']:
                    print("🚀 Starting application...")
                    subprocess.run("streamlit run simple_app.py", shell=True)
            except EOFError:
                print("\n✅ Setup completed - you can start the app with: streamlit run simple_app.py")
        return success
    except KeyboardInterrupt:
        print("\n\n⏹️ Setup interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)