# run.py
"""
HackRx 6.0 - Application Entry Point
Run this file to start the server
"""

import uvicorn
import sys
import os

# Optional: Add app directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ✅ Define your config here instead of using .env
DEBUG = False
HOST = "0.0.0.0"
PORT = 8000
LOG_LEVEL = "info" if DEBUG else "warning"

# Optional: Keys for Claude, Groq, etc.
API_KEY = "your_hackrx_api_key_here"
CLAUDE_API_KEY = "your_claude_key_here"
GROQ_API_KEY = "your_groq_key_here"

def main():
    """Main function to start the server"""
    print("🚀 Starting HackRx 6.0 Server...")
    print(f"📍 Host: {HOST}")
    print(f"📍 Port: {PORT}")
    print(f"🔧 Debug: {DEBUG}")
    print(f"🔑 API Key: {'✅' if API_KEY else '❌'}")
    print(f"🔑 Claude: {'✅' if CLAUDE_API_KEY else '❌'}")
    print(f"🔑 Groq: {'✅' if GROQ_API_KEY else '❌'}")
    
    print("\n🎯 Server will be accessible at:")
    print(f"   Local:  http://localhost:{PORT}")
    print(f"   Network: http://{HOST}:{PORT}")
    print(f"   Health: http://localhost:{PORT}/health")
    print(f"   API:    http://localhost:{PORT}/hackrx/run")
    print(f"   Docs:   http://localhost:{PORT}/docs")
    
    print("\n🚀 Starting server now...")
    
    uvicorn.run(
        "app.main:app",
        host=HOST,
        port=PORT,
        reload=DEBUG,
        log_level=LOG_LEVEL
    )

if __name__ == "__main__":
    main()
