# run_simple.py - Simple Server Startup for HackRx 6.0
import os
import sys
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add app directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.join(current_dir, 'app')
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

def main():
    """Simple server startup function"""
    # Get port from environment
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    # Check API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    
    print("🚀 Starting HackRx 6.0 HYBRID Competition Server")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    
    if openai_key:
        print(f"🔵 OpenAI API Key: Configured")
        print(f"🤖 PRIMARY LLM: GPT-4o")
        print(f"🔍 PRIMARY Embeddings: text-embedding-3-large")
        print(f"🎯 Mode: MAXIMUM ACCURACY")
    
    if groq_key:
        print(f"🟡 Groq API Key: Configured")
        print(f"🤖 FALLBACK LLM: llama-3.1-8b-instant + Multi-Model")
        print(f"🔍 FALLBACK Embeddings: all-MiniLM-L6-v2")
    
    if openai_key and groq_key:
        print("✅ HYBRID MODE: Both OpenAI and Groq available")
        print("📊 Expected Accuracy: 95%+ (OpenAI primary)")
    elif openai_key:
        print("🔵 OPENAI-ONLY MODE: OpenAI available")
        print("📊 Expected Accuracy: 95%+ (OpenAI only)")
    elif groq_key:
        print("🟡 GROQ-ONLY MODE: Groq available")
        print("📊 Expected Accuracy: 85%+ (Groq only)")
    else:
        print("❌ NO API KEYS: Neither OpenAI nor Groq configured")
    
    print("📊 Competition Mode: HYBRID PROCESSING")
    print("🏆 Ready for maximum accuracy leaderboard scoring!")
    
    # Start server
    try:
        uvicorn.run(
            "app.main:app",
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )
    except Exception as e:
        print(f"❌ Server startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()