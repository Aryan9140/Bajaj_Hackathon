# run.py - Simple Server Startup for HackRx 6.0
import os
import sys
import logging
import uvicorn

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
    
    # Check Groq API key
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        logger.info(f"🔑 Groq API Key: Configured")
        logger.info(f"🤖 LLM Model: gemma2-9b-it")
        logger.info(f"🎯 Optimization: Maximum Accuracy")
    else:
        logger.warning(f"⚠️ Groq API Key: NOT SET")
        logger.warning(f"   Please set GROQ_API_KEY environment variable")
    
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