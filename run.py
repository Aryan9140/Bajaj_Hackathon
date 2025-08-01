
# ================================
# run.py - Render-Safe Version
# ================================

"""
run.py - Server startup and API endpoints
Enhanced for HackRx 6.0 Competition with Groq LLM primary
RENDER-SAFE: Handles missing dependencies gracefully
"""

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
import sys
import logging
import time

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Safe dependency imports for Render
def safe_import(module_name, fallback_name=None):
    try:
        return __import__(module_name)
    except ImportError as e:
        logger.warning(f"⚠️ {module_name} not available: {e}")
        return None

# Import required dependencies safely
numpy = safe_import('numpy')
aiohttp = safe_import('aiohttp')

# Fix import path for app/main.py structure
COMPETITION_FEATURES_AVAILABLE = False
doc_processor = None
answer_engine = None

try:
    # Add current directory and app directory to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_dir = os.path.join(current_dir, 'app')
    
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Strategy 1: Import from app.main - PRIMARY STRATEGY
    try:
        from app.main import (
            DocumentProcessor,
            CompetitionAnswerEngine,
            doc_processor,
            answer_engine
        )
        COMPETITION_FEATURES_AVAILABLE = True
        logger.info("✅ Competition features imported from app.main")
    except ImportError as e1:
        logger.warning(f"⚠️ Strategy 1 failed (app.main): {e1}")
        
        # Strategy 2: Try simplified import without numpy dependencies
        try:
            # Create fallback processor without numpy
            logger.info("🔄 Attempting fallback import strategy...")
            
            # Import basic modules needed
            import sys
            import os
            import re
            import hashlib
            import time
            import asyncio
            
            # Check if we can import the basic modules from main.py
            from app import main
            
            # Try to get classes even if numpy failed
            if hasattr(main, 'DocumentProcessor') and hasattr(main, 'CompetitionAnswerEngine'):
                DocumentProcessor = main.DocumentProcessor
                CompetitionAnswerEngine = main.CompetitionAnswerEngine
                doc_processor = main.doc_processor if hasattr(main, 'doc_processor') else DocumentProcessor()
                answer_engine = main.answer_engine if hasattr(main, 'answer_engine') else CompetitionAnswerEngine()
                COMPETITION_FEATURES_AVAILABLE = True
                logger.info("✅ Competition features imported via fallback strategy")
            else:
                raise ImportError("Required classes not found in main.py")
                
        except ImportError as e2:
            logger.error(f"❌ All import strategies failed:")
            logger.error(f"   - app.main: {e1}")
            logger.error(f"   - fallback: {e2}")
            COMPETITION_FEATURES_AVAILABLE = False

except Exception as e:
    logger.error(f"❌ Critical import error: {e}")
    COMPETITION_FEATURES_AVAILABLE = False

app = FastAPI(
    title="HackRx 6.0 - Competition Server",
    description="Competition server with Groq LLM and advanced document processing",
    version="6.0.0"
)

# Add CORS for competition
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)
HACKRX_API_KEY = "6d2683f80eca9847d20948e1e5508885d08fdc65d943182f85de250687859193"

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key authentication"""
    if credentials and credentials.credentials != HACKRX_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials if credentials else None

# Pydantic Models for simple data processing
class QueryRequest(BaseModel):
    data: list

class QueryResponse(BaseModel):
    is_success: bool
    user_id: str
    email: str
    roll_number: str
    numbers: list
    alphabets: list
    highest_lowercase_alphabet: list

@app.get("/")
async def root():
    """Root endpoint with system status"""
    return {
        "message": "HackRx 6.0 - Competition Server Ready",
        "status": "ready", 
        "version": "6.0.0",
        "competition_features": "available" if COMPETITION_FEATURES_AVAILABLE else "not available",
        "import_status": "✅ Imported successfully" if COMPETITION_FEATURES_AVAILABLE else "❌ Import failed",
        "primary_llm": "Groq (Free tier optimized)",
        "dependencies": {
            "numpy": "available" if numpy else "missing",
            "aiohttp": "available" if aiohttp else "missing"
        },
        "endpoints": {
            "/hackrx/run": "Main competition endpoint (handles both formats)",
            "/health": "Health check",
            "/test-processing": "Test document processing",
            "/competition-status": "Competition readiness check"
        },
        "features": [
            "Advanced PDF extraction",
            "Groq LLM integration", 
            "Semantic search with fallbacks",
            "Competition scoring optimization",
            "Multi-format document support"
        ]
    }

@app.post("/hackrx/run")
async def competition_endpoint(
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    """
    MAIN COMPETITION ENDPOINT
    Handles both:
    1. Simple data processing: {"data": [...]}
    2. Document processing: {"documents": "...", "questions": [...]}
    """
    start_time = time.time()
    
    try:
        request_data = await request.json()
        
        # SIMPLE DATA PROCESSING (Original format)
        if "data" in request_data:
            logger.info("Processing simple data format")
            
            data = request_data["data"]
            numbers = [item for item in data if item.isdigit()]
            alphabets = [item for item in data if item.isalpha()]
            lowercase_alphabets = [item for item in alphabets if item.islower()]
            highest_lowercase = [max(lowercase_alphabets)] if lowercase_alphabets else []
            
            response = QueryResponse(
                is_success=True,
                user_id="patel",
                email="aryanpatel77462@gmail.com",
                roll_number="1047",
                numbers=numbers,
                alphabets=alphabets,
                highest_lowercase_alphabet=highest_lowercase
            )
            
            processing_time = time.time() - start_time
            logger.info(f"Simple data processed in {processing_time:.2f}s")
            
            return response
        
        # DOCUMENT PROCESSING (Competition format)
        elif "documents" in request_data and "questions" in request_data:
            logger.info("🏆 Processing document format for HackRx competition")
            
            if not COMPETITION_FEATURES_AVAILABLE:
                logger.error("❌ Competition features not available")
                questions = request_data.get("questions", [])
                return {
                    "answers": ["Competition features not available due to missing dependencies. Please check server configuration and ensure all required packages are installed."] * len(questions)
                }
            
            if not doc_processor or not answer_engine:
                logger.error("❌ Processors not initialized")
                questions = request_data.get("questions", [])
                return {
                    "answers": ["Document processors not initialized. Please check server configuration."] * len(questions)
                }
            
            documents = request_data.get("documents", "")
            questions = request_data.get("questions", [])
            
            logger.info(f"🏆 HackRx Competition Request:")
            logger.info(f"   - Questions: {len(questions)}")
            logger.info(f"   - Document URL: {documents[:100]}...")
            
            try:
                # Stage 1: Advanced Document Processing
                logger.info("📄 Stage 1: Advanced Document Processing")
                document_text = await doc_processor.process_document(documents)
                
                if not document_text or len(document_text.strip()) < 50:
                    logger.error("❌ Document processing failed or insufficient content")
                    return {
                        "answers": ["Unable to extract sufficient information from the provided document. The document may be empty, corrupted, or inaccessible."] * len(questions)
                    }
                
                logger.info(f"✅ Document processed successfully: {len(document_text)} characters extracted")
                
                # Stage 2: Prepare Competition Answer Engine
                logger.info("🤖 Stage 2: Preparing Competition Answer Engine")
                answer_engine.prepare_document(document_text)
                
                # Stage 3: Generate Competition Answers
                logger.info("🎯 Stage 3: Generating Competition-Grade Answers")
                answers = []
                
                for i, question in enumerate(questions):
                    q_start = time.time()
                    logger.info(f"🤔 Processing Q{i+1}/{len(questions)}: {question[:80]}...")
                    
                    try:
                        answer = await answer_engine.generate_competition_answer(question)
                        answers.append(answer)
                        
                        q_time = time.time() - q_start
                        logger.info(f"✅ Q{i+1} completed in {q_time:.2f}s - Answer length: {len(answer)} chars")
                        
                    except Exception as e:
                        logger.error(f"❌ Failed to process Q{i+1}: {e}")
                        answers.append("Unable to process this question due to a processing error. Please try again.")
                
                # Competition Performance Metrics
                total_time = time.time() - start_time
                
                logger.info(f"🏆 HACKRX COMPETITION COMPLETED:")
                logger.info(f"   ⏱️  Total Processing Time: {total_time:.2f}s")
                logger.info(f"   ✅ Answers Generated: {len(answers)}")
                
                # Return competition-formatted response
                return {
                    "answers": answers,
                    "processing_time": round(total_time, 2),
                    "request_id": f"hackrx_{int(time.time())}",
                    "cached": False
                }
                
            except Exception as e:
                logger.error(f"❌ Document processing pipeline failed: {e}")
                questions = request_data.get("questions", [])
                return {
                    "answers": [f"Document processing pipeline failed: {str(e)}. Please check the document URL and try again."] * len(questions)
                }
        
        else:
            raise HTTPException(
                status_code=400, 
                detail="Invalid request format. Expected either 'data' array or 'documents' + 'questions' fields."
            )
            
    except Exception as e:
        logger.error(f"❌ Competition endpoint critical failure: {e}")
        
        # Emergency fallback response
        try:
            if 'request_data' in locals():
                if "questions" in request_data:
                    question_count = len(request_data.get("questions", []))
                    return {
                        "answers": [f"Critical system error occurred: {str(e)}. Please try again later."] * question_count
                    }
                elif "data" in request_data:
                    return QueryResponse(
                        is_success=False,
                        user_id="patel",
                        email="aryanpatel77462@gmail.com",  
                        roll_number="1047",
                        numbers=[],
                        alphabets=[],
                        highest_lowercase_alphabet=[]
                    )
        except:
            pass
        
        raise HTTPException(status_code=500, detail=f"Critical system error: {str(e)}")

@app.get("/health")
async def health():
    """Health check with competition readiness assessment"""
    health_status = {
        "status": "healthy", 
        "version": "6.0.0",
        "competition_features": COMPETITION_FEATURES_AVAILABLE,
        "import_status": "success" if COMPETITION_FEATURES_AVAILABLE else "failed",
        "dependencies": {
            "numpy": "available" if numpy else "missing",
            "aiohttp": "available" if aiohttp else "missing"
        },
        "timestamp": time.time(),
        "ready_for_hackrx": COMPETITION_FEATURES_AVAILABLE
    }
    
    if COMPETITION_FEATURES_AVAILABLE:
        try:
            health_status["processors"] = {
                "document_processor": bool(doc_processor),
                "answer_engine": bool(answer_engine),
                "groq_client": bool(hasattr(answer_engine, 'groq_client') and answer_engine.groq_client) if answer_engine else False
            }
            
            groq_api_key = os.getenv("GROQ_API_KEY")
            health_status["groq_api_key"] = "configured" if groq_api_key else "missing"
            
        except Exception as e:
            health_status["processors"] = f"error_checking_processors: {str(e)}"
    
    return health_status

@app.get("/test-processing")
async def test_processing():
    """Test endpoint for competition processing"""
    if not COMPETITION_FEATURES_AVAILABLE:
        return {
            "status": "❌ Competition features not available",
            "error": "Failed to import competition classes",
            "likely_cause": "Missing dependencies (numpy, aiohttp, etc.)",
            "suggestion": "Check requirements.txt and ensure all packages are installed"
        }
    
    return {
        "status": "✅ Competition features available",
        "processors_initialized": {
            "document_processor": bool(doc_processor),
            "answer_engine": bool(answer_engine)
        },
        "ready_for_testing": True
    }

@app.get("/competition-status")
async def competition_status():
    """Detailed HackRx competition readiness status"""
    return {
        "competition_ready": COMPETITION_FEATURES_AVAILABLE,
        "status": "✅ Ready for Competition" if COMPETITION_FEATURES_AVAILABLE else "❌ Not Ready - Missing Dependencies",
        "dependencies": {
            "numpy": "available" if numpy else "missing - required for semantic search",
            "aiohttp": "available" if aiohttp else "missing - required for document download"
        },
        "groq_api_key": "configured" if os.getenv("GROQ_API_KEY") else "missing",
        "recommendation": "Install missing dependencies and set GROQ_API_KEY" if not COMPETITION_FEATURES_AVAILABLE else "System ready!"
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return {"error": f"HTTP {exc.status_code}: {exc.detail}"}

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return {"error": f"Internal server error: {str(exc)}"}

# Server startup
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting HackRx 6.0 Competition Server on port {port}")
    logger.info(f"📊 Competition features: {'✅ Available' if COMPETITION_FEATURES_AVAILABLE else '❌ Not Available'}")
    logger.info(f"🤖 Primary LLM: Groq (Free tier optimized)")
    logger.info(f"🏆 Competition mode: {'ACTIVE' if COMPETITION_FEATURES_AVAILABLE else 'DISABLED - CHECK DEPENDENCIES'}")
    
    # Dependency status
    if not numpy:
        logger.warning("⚠️ numpy not available - semantic search disabled")
    if not aiohttp:
        logger.warning("⚠️ aiohttp not available - document download may fail")
    
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        logger.info(f"🔑 Groq API Key: Configured")
    else:
        logger.warning(f"⚠️ Groq API Key: NOT SET")
    
    uvicorn.run(app, host="0.0.0.0", port=port)