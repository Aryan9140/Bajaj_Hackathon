# run.py - Simple Server Startup for HackRx 6.0 - FIXED
import os
import sys
import logging
import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add app directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.join(current_dir, 'app')
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

def main():
    """Main server startup function"""
    # Get port from environment
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    # Log startup information
    logger.info("🚀 Starting HackRx 6.0 Competition Server")
    logger.info(f"   Host: {host}")
    logger.info(f"   Port: {port}")
    logger.info(f"   Environment: {'Production' if port != 8000 else 'Development'}")
    
    # Check Groq API key - FIXED
    groq_key = os.getenv("GROQ_API_KEY")
    print(f"🔍 DEBUG - Groq Key Value: {groq_key}")  # Debug print - FIXED
    
    if groq_key:
        logger.info(f"🔑 Groq API Key: Configured ({groq_key[:10]}...)")
        logger.info(f"🤖 LLM Model: llama-3.1-70b-versatile")
        logger.info(f"🎯 Optimization: Maximum Accuracy")
    else:
        logger.warning(f"⚠️ Groq API Key: NOT SET")
        logger.warning(f"   Please set GROQ_API_KEY environment variable")
        
        # Additional debugging
        print("🔍 DEBUG - All environment variables containing 'GROQ':")
        for key, value in os.environ.items():
            if 'GROQ' in key.upper():
                print(f"   {key} = {value}")
        
        print("🔍 DEBUG - Checking common env var names:")
        alternatives = ["GROQ_API_KEY", "groq_api_key", "GROQ_KEY", "groq_key"]
        for alt in alternatives:
            val = os.getenv(alt)
            if val:
                print(f"   Found: {alt} = {val}")
    
    logger.info(f"📊 Competition Mode: ACTIVE")
    logger.info(f"🏆 Ready for leaderboard scoring!")
    
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
        logger.error(f"❌ Server startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()