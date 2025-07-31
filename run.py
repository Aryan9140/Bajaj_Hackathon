"""
HackRx 6.0 - Render Compatible Entry Point
Handles port binding correctly for Render deployment
"""

import os
import uvicorn
from app.core.config import settings

def main():
    """Main entry point for Render deployment"""
    
    # Get port from environment (Render sets this automatically)
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"🚀 Starting HackRx 6.0 Server...")
    print(f"📍 Host: {host}")
    print(f"📍 Port: {port}")
    print(f"🎯 Server will be accessible at:")
    print(f"   Local:  http://localhost:{port}")
    print(f"   Network: http://{host}:{port}")
    print(f"   Health: http://{host}:{port}/health")
    print(f"   API:    http://{host}:{port}/hackrx/run")
    print(f"   Docs:   http://{host}:{port}/docs")
    print(f"🚀 Starting server now...")
    
    # Start server with proper configuration
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=False,  # Disable reload in production
        log_level="info",
        access_log=True,
        timeout_keep_alive=30,
        timeout_graceful_shutdown=10
    )

if __name__ == "__main__":
    main()