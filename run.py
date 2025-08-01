"""
run.py - Server startup and API endpoints
Enhanced for HackRx 6.0 Competition with Groq LLM primary
Imports advanced features from app/main.py
"""

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
import logging
import time

# Import competition features from app/main.py
try:
    from app.main import (
        DocumentProcessor,
        CompetitionAnswerEngine,
        doc_processor,
        answer_engine
    )
    COMPETITION_FEATURES_AVAILABLE = True
    logging.info("Competition features imported from app/main.py")
except ImportError as e:
    COMPETITION_FEATURES_AVAILABLE = False
    logging.warning(f"Competition features not available: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        "primary_llm": "Groq (Free tier optimized)",
        "endpoints": {
            "/hackrx/run": "Main competition endpoint (handles both formats)",
            "/health": "Health check",
            "/test-processing": "Test document processing"
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
                logger.error("Competition features not available")
                questions = request_data.get("questions", [])
                return {
                    "answers": ["Competition features not available. Please check app/main.py configuration."] * len(questions)
                }
            
            documents = request_data.get("documents", "")
            questions = request_data.get("questions", [])
            
            logger.info(f"Competition request: {len(questions)} questions")
            logger.info(f"Document URL: {documents[:100]}...")
            
            # Use advanced document processor from app/main.py
            logger.info("Stage 1: Document Processing")
            document_text = await doc_processor.process_document(documents)
            
            if not document_text or len(document_text.strip()) < 100:
                logger.error("Document processing failed or insufficient content")
                return {
                    "answers": ["Unable to extract sufficient information from the provided document."] * len(questions)
                }
            
            logger.info(f"Document processed: {len(document_text)} characters")
            
            # Prepare answer engine
            logger.info("Stage 2: Preparing Answer Engine with Groq LLM")
            answer_engine.prepare_document(document_text)
            
            # Generate competition answers
            logger.info("Stage 3: Generating Competition Answers")
            answers = []
            
            for i, question in enumerate(questions):
                q_start = time.time()
                logger.info(f"Processing Q{i+1}/{len(questions)}: {question[:60]}...")
                
                try:
                    # Use competition answer engine with Groq LLM
                    answer = await answer_engine.generate_competition_answer(question)
                    answers.append(answer)
                    
                    q_time = time.time() - q_start
                    logger.info(f"Q{i+1} completed in {q_time:.2f}s - Length: {len(answer)} chars")
                    
                except Exception as e:
                    logger.error(f"Failed to process Q{i+1}: {e}")
                    answers.append("Unable to process this question due to processing error.")
            
            # Competition metrics
            total_time = time.time() - start_time
            avg_time = total_time / len(questions)
            
            logger.info(f"COMPETITION COMPLETED:")
            logger.info(f"- Total Time: {total_time:.2f}s")
            logger.info(f"- Average per Question: {avg_time:.2f}s")
            logger.info(f"- Answers Generated: {len(answers)}")
            
            return {"answers": answers}
        
        else:
            raise HTTPException(status_code=400, detail="Invalid request format. Expected 'data' or 'documents+questions'")
            
    except Exception as e:
        logger.error(f"Competition endpoint failed: {e}")
        
        # Emergency fallback
        try:
            if 'request_data' in locals():
                if "questions" in request_data:
                    question_count = len(request_data.get("questions", []))
                    return {"answers": ["Unable to process this question due to system error."] * question_count}
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
        "timestamp": time.time()
    }
    
    if COMPETITION_FEATURES_AVAILABLE:
        try:
            # Check if processors are working
            health_status["processors"] = {
                "document_processor": bool(doc_processor),
                "answer_engine": bool(answer_engine),
                "groq_client": bool(answer_engine.groq_client) if answer_engine else False
            }
        except:
            health_status["processors"] = "error_checking_processors"
    
    return health_status

@app.get("/test-processing")
async def test_processing():
    """Test endpoint for competition processing"""
    if not COMPETITION_FEATURES_AVAILABLE:
        return {"status": "Competition features not available"}
    
    try:
        # Test document processing
        test_text = """
        SAMPLE POLICY DOCUMENT
        
        Grace Period: A grace period of thirty (30) days is provided for premium payment after the due date.
        
        Waiting Period: Pre-existing diseases have a waiting period of thirty-six (36) months of continuous coverage.
        
        Maternity Coverage: Maternity expenses are covered after twenty-four (24) months of continuous coverage. Limited to two deliveries per policy period.
        """
        
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
            "status": "competition_processing_successful",
            "groq_llm_status": "active",
            "test_document_length": len(test_text),
            "questions_processed": len(test_questions),
            "results": test_results
        }
        
    except Exception as e:
        return {
            "status": "competition_processing_failed", 
            "error": str(e)
        }

@app.get("/competition-status")
async def competition_status():
    """Detailed competition readiness status"""
    if not COMPETITION_FEATURES_AVAILABLE:
        return {
            "competition_ready": False,
            "error": "Competition features not imported from app/main.py"
        }
    
    try:
        status = {
            "competition_ready": True,
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
        if hasattr(answer_engine, 'groq_client') and answer_engine.groq_client:
            status["groq_status"] = "configured_and_ready"
        else:
            status["groq_status"] = "not_configured"
            
        return status
        
    except Exception as e:
        return {
            "competition_ready": False,
            "error": str(e)
        }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return {"error": f"HTTP {exc.status_code}: {exc.detail}"}

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return {"error": "Internal server error occurred"}

# Server startup
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting HackRx 6.0 Competition Server on port {port}")
    logger.info(f"📊 Competition features: {'Available' if COMPETITION_FEATURES_AVAILABLE else 'Not Available'}")
    logger.info(f"🤖 Primary LLM: Groq (Free tier optimized)")
    logger.info(f"🏆 Competition mode: ACTIVE")
    
    uvicorn.run(app, host="0.0.0.0", port=port)