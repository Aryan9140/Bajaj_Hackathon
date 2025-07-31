#!/usr/bin/env python3
"""
HackRx 6.0 - Fixed Entry Point for Render
Handles port binding correctly
"""

import os
import sys

def main():
    """Entry point for the application"""
    try:
        # Import uvicorn here to handle any import issues
        import uvicorn
        
        # Get port from environment - Render sets this automatically
        port = int(os.environ.get("PORT", 8000))
        host = os.environ.get("HOST", "0.0.0.0")
        
        print(f"🚀 Starting HackRx 6.0...")
        print(f"📍 Port: {port}")
        print(f"📍 Host: {host}")
        
        # Start the application
        uvicorn.run(
            "app.main:app",
            host=host,
            port=port,
            reload=False,
            log_level="info"
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("🔧 Trying alternative import...")
        
        # Fallback: try importing from app
        try:
            from app.main import app
            import uvicorn
            
            port = int(os.environ.get("PORT", 8000))
            host = os.environ.get("HOST", "0.0.0.0")
            
            uvicorn.run(app, host=host, port=port)
            
        except Exception as fallback_error:
            print(f"❌ Fallback failed: {fallback_error}")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Startup error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()