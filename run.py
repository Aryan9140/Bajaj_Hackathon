"""
run.py - Server startup and API endpoints
Enhanced for HackRx 6.0 Competition with Groq LLM primary
Fixed imports for app/main.py structure
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
    
    # Try multiple import strategies
    try:
        # Strategy 1: Import from app.main
        from app.main import (
            DocumentProcessor,
            CompetitionAnswerEngine,
            doc_processor,
            answer_engine
        )
        COMPETITION_FEATURES_AVAILABLE = True
        logger.info("✅ Competition features imported from app.main")
    except ImportError:
        try:
            # Strategy 2: Import from main (after adding app to path)
            import main
            DocumentProcessor = main.DocumentProcessor
            CompetitionAnswerEngine = main.CompetitionAnswerEngine
            doc_processor = main.doc_processor
            answer_engine = main.answer_engine
            COMPETITION_FEATURES_AVAILABLE = True
            logger.info("✅ Competition features imported from main")
        except ImportError:
            try:
                # Strategy 3: Manual import and instantiation
                from app import main
                DocumentProcessor = main.DocumentProcessor
                CompetitionAnswerEngine = main.CompetitionAnswerEngine
                doc_processor = main.DocumentProcessor()
                answer_engine = main.CompetitionAnswerEngine()
                COMPETITION_FEATURES_AVAILABLE = True
                logger.info("✅ Competition features imported and instantiated manually")
            except ImportError as e:
                logger.error(f"❌ Failed to import competition features: {e}")
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
            logger.info("Processing document format for competition")
            
            if not COMPETITION_FEATURES_AVAILABLE:
                logger.error("❌ Competition features not available")
                questions = request_data.get("questions", [])
                return {
                    "answers": ["Competition features import failed. Check app/main.py file exists and is properly configured."] * len(questions)
                }
            
            if not doc_processor or not answer_engine:
                logger.error("❌ Processors not initialized")
                questions = request_data.get("questions", [])
                return {
                    "answers": ["Document processors not initialized. Check app/main.py configuration."] * len(questions)
                }
            
            documents = request_data.get("documents", "")
            questions = request_data.get("questions", [])
            
            logger.info(f"🏆 Competition request: {len(questions)} questions")
            logger.info(f"📄 Document URL: {documents[:100]}...")
            
            try:
                # Use advanced document processor from app/main.py
                logger.info("Stage 1: Document Processing")
                document_text = await doc_processor.process_document(documents)
                
                if not document_text or len(document_text.strip()) < 50:
                    logger.error("❌ Document processing failed or insufficient content")
                    return {
                        "answers": ["Unable to extract sufficient information from the provided document. The document may be empty, corrupted, or inaccessible."] * len(questions)
                    }
                
                logger.info(f"✅ Document processed: {len(document_text)} characters")
                
                # Prepare answer engine
                logger.info("Stage 2: Preparing Answer Engine with Groq LLM")
                answer_engine.prepare_document(document_text)
                
                # Generate competition answers
                logger.info("Stage 3: Generating Competition Answers")
                answers = []
                
                for i, question in enumerate(questions):
                    q_start = time.time()
                    logger.info(f"🤔 Processing Q{i+1}/{len(questions)}: {question[:60]}...")
                    
                    try:
                        # Use competition answer engine with Groq LLM
                        answer = await answer_engine.generate_competition_answer(question)
                        answers.append(answer)
                        
                        q_time = time.time() - q_start
                        logger.info(f"✅ Q{i+1} completed in {q_time:.2f}s - Length: {len(answer)} chars")
                        
                    except Exception as e:
                        logger.error(f"❌ Failed to process Q{i+1}: {e}")
                        answers.append("Unable to process this question due to processing error. Please try again.")
                
                # Competition metrics
                total_time = time.time() - start_time
                avg_time = total_time / len(questions) if questions else 0
                
                logger.info(f"🏆 COMPETITION COMPLETED:")
                logger.info(f"   - Total Time: {total_time:.2f}s")
                logger.info(f"   - Average per Question: {avg_time:.2f}s")
                logger.info(f"   - Answers Generated: {len(answers)}")
                
                return {"answers": answers}
                
            except Exception as e:
                logger.error(f"❌ Document processing pipeline failed: {e}")
                questions = request_data.get("questions", [])
                return {
                    "answers": ["Document processing pipeline failed. Please check the document URL and try again."] * len(questions)
                }
        
        else:
            raise HTTPException(status_code=400, detail="Invalid request format. Expected 'data' array or 'documents' + 'questions' fields.")
            
    except Exception as e:
        logger.error(f"❌ Competition endpoint critical failure: {e}")
        
        # Emergency fallback
        try:
            if 'request_data' in locals():
                if "questions" in request_data:
                    question_count = len(request_data.get("questions", []))
                    return {"answers": ["Critical system error occurred. Please try again later."] * question_count}
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
        
        raise HTTPException(status_code=500, detail="Critical system error occurred")

@app.get("/health")
async def health():
    """Health check with competition readiness"""
    health_status = {
        "status": "healthy", 
        "version": "6.0.0",
        "competition_features": COMPETITION_FEATURES_AVAILABLE,
        "import_status": "success" if COMPETITION_FEATURES_AVAILABLE else "failed",
        "timestamp": time.time()
    }
    
    if COMPETITION_FEATURES_AVAILABLE:
        try:
            health_status["processors"] = {
                "document_processor": bool(doc_processor),
                "answer_engine": bool(answer_engine),
                "groq_client": bool(hasattr(answer_engine, 'groq_client') and answer_engine.groq_client) if answer_engine else False
            }
        except Exception as e:
            health_status["processors"] = f"error_checking_processors: {str(e)}"
    
    return health_status

@app.get("/test-processing")
async def test_processing():
    """Test endpoint for competition processing"""
    if not COMPETITION_FEATURES_AVAILABLE:
        return {
            "status": "❌ Competition features not available",
            "error": "Failed to import from app/main.py",
            "suggestion": "Check if app/main.py exists and contains required classes"
        }
    
    if not doc_processor or not answer_engine:
        return {
            "status": "❌ Processors not initialized",
            "error": "doc_processor or answer_engine is None"
        }
    
    try:
        # Test document processing
        test_text = """
        SAMPLE POLICY DOCUMENT
        
        Grace Period: A grace period of thirty (30) days is provided for premium payment after the due date.
        
        Waiting Period: Pre-existing diseases have a waiting period of thirty-six (36) months of continuous coverage.
        
        Maternity Coverage: Maternity expenses are covered after twenty-four (24) months of continuous coverage. Limited to two deliveries per policy period.
        """
        
        logger.info("🧪 Testing competition pipeline")
        
        # Test the competition pipeline
        answer_engine.prepare_document(test_text)
        
        test_questions = [
            "What is the grace period for premium payment?",
            "What is the waiting period for pre-existing diseases?",
            "Does this policy cover maternity expenses?"
        ]
        
        test_results = []
        for question in test_questions:
            start_time = time.time()
            answer = await answer_engine.generate_competition_answer(question)
            process_time = time.time() - start_time
            
            test_results.append({
                "question": question,
                "answer": answer,
                "processing_time": f"{process_time:.2f}s",
                "answer_length": len(answer)
            })
        
        return {
            "status": "✅ competition_processing_successful",
            "groq_llm_status": "active",
            "test_document_length": len(test_text),
            "questions_processed": len(test_questions),
            "results": test_results
        }
        
    except Exception as e:
        logger.error(f"❌ Test processing failed: {e}")
        return {
            "status": "❌ competition_processing_failed", 
            "error": str(e),
            "suggestion": "Check Groq API key configuration and app/main.py implementation"
        }

@app.get("/competition-status")
async def competition_status():
    """Detailed competition readiness status"""
    if not COMPETITION_FEATURES_AVAILABLE:
        return {
            "competition_ready": False,
            "status": "❌ Import Failed",
            "error": "Competition features not imported from app/main.py",
            "suggestions": [
                "Check if app/main.py file exists in your repository",
                "Verify app/main.py contains DocumentProcessor and CompetitionAnswerEngine classes",
                "Check for syntax errors in app/main.py",
                "Ensure all required dependencies are installed"
            ]
        }
    
    try:
        status = {
            "competition_ready": True,
            "status": "✅ Ready for Competition",
            "primary_llm": "Groq (Free tier)",
            "document_processing": "Advanced PDF/DOCX extraction",
            "semantic_search": "Available with fallbacks",
            "answer_generation": "Multi-strategy with Groq LLM",
            "scoring_optimization": "Maximum accuracy mode",
            "endpoints": {
                "simple_data": "/hackrx/run with {'data': [...]}",
                "document_processing": "/hackrx/run with {'documents': '...', 'questions': [...]}"
            }
        }
        
        # Check system components
        if answer_engine and hasattr(answer_engine, 'groq_client') and answer_engine.groq_client:
            status["groq_status"] = "✅ configured_and_ready"
        else:
            status["groq_status"] = "⚠️ not_configured - check GROQ_API_KEY environment variable"
            
        return status
        
    except Exception as e:
        return {
            "competition_ready": False,
            "status": "❌ System Check Failed",
            "error": str(e)
        }

@app.get("/debug-imports")
async def debug_imports():
    """Debug endpoint to check import status"""
    import_info = {
        "current_directory": os.getcwd(),
        "python_path": sys.path[:5],  # First 5 entries
        "competition_features_available": COMPETITION_FEATURES_AVAILABLE,
        "doc_processor_exists": doc_processor is not None,
        "answer_engine_exists": answer_engine is not None
    }
    
    # Check if app directory exists
    app_dir = os.path.join(os.getcwd(), 'app')
    main_file = os.path.join(app_dir, 'main.py')
    
    import_info["file_checks"] = {
        "app_directory_exists": os.path.exists(app_dir),
        "app_main_py_exists": os.path.exists(main_file),
        "app_directory_contents": os.listdir(app_dir) if os.path.exists(app_dir) else "directory_not_found"
    }
    
    return import_info

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return {"error": f"HTTP {exc.status_code}: {exc.detail}"}

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return {"error": "Internal server error occurred during request processing"}

# Server startup
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting HackRx 6.0 Competition Server on port {port}")
    logger.info(f"📊 Competition features: {'✅ Available' if COMPETITION_FEATURES_AVAILABLE else '❌ Not Available'}")
    logger.info(f"🤖 Primary LLM: Groq (Free tier optimized)")
    logger.info(f"🏆 Competition mode: ACTIVE")
    
    uvicorn.run(app, host="0.0.0.0", port=port)