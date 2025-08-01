# run.py - Simple Server Startup for HackRx 6.0
import os
import sys
import logging
import time
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add app directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.join(current_dir, 'app')
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

# Import competition features
COMPETITION_AVAILABLE = False
try:
    from app.main import doc_processor, answer_engine
    COMPETITION_AVAILABLE = True
    logger.info("✅ Competition features loaded successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import competition features: {e}")
    COMPETITION_AVAILABLE = False

# FastAPI app
app = FastAPI(
    title="HackRx 6.0 - Competition Server",
    version="6.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)
API_KEY = "6d2683f80eca9847d20948e1e5508885d08fdc65d943182f85de250687859193"

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials and credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials if credentials else None

# Request/Response models
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

class DocumentRequest(BaseModel):
    documents: str
    questions: List[str]

class DocumentResponse(BaseModel):
    answers: List[str]

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "HackRx 6.0 - Competition Server Ready",
        "status": "ready",
        "version": "6.0.0",
        "competition_features": "available" if COMPETITION_AVAILABLE else "not available",
        "groq_model": "gemma2-9b-it",
        "endpoints": {
            "/hackrx/run": "Main competition endpoint",
            "/health": "Health check"
        }
    }

@app.post("/hackrx/run")
async def hackrx_run(
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    """Main HackRx competition endpoint"""
    start_time = time.time()
    
    try:
        request_data = await request.json()
        
        # Handle simple data processing
        if "data" in request_data:
            logger.info("Processing simple data format")
            
            data = request_data["data"]
            numbers = [item for item in data if item.isdigit()]
            alphabets = [item for item in data if item.isalpha()]
            lowercase_alphabets = [item for item in alphabets if item.islower()]
            highest_lowercase = [max(lowercase_alphabets)] if lowercase_alphabets else []
            
            return QueryResponse(
                is_success=True,
                user_id="patel",
                email="aryanpatel77462@gmail.com",
                roll_number="1047",
                numbers=numbers,
                alphabets=alphabets,
                highest_lowercase_alphabet=highest_lowercase
            )
        
        # Handle document processing
        elif "documents" in request_data and "questions" in request_data:
            logger.info("🏆 Processing HackRx competition request")
            
            if not COMPETITION_AVAILABLE:
                questions = request_data.get("questions", [])
                return {
                    "answers": ["Competition features not available. Please check server configuration."] * len(questions)
                }
            
            documents = request_data.get("documents", "")
            questions = request_data.get("questions", [])
            
            logger.info(f"Document: {documents[:100]}...")
            logger.info(f"Questions: {len(questions)}")
            
            try:
                # Process document
                logger.info("📄 Processing document...")
                document_text = await doc_processor.process_document(documents)
                
                if not document_text or len(document_text.strip()) < 50:
                    logger.error("Document processing failed")
                    return {
                        "answers": ["Unable to extract information from the document. Please check the document URL."] * len(questions)
                    }
                
                logger.info(f"✅ Document processed: {len(document_text)} characters")
                
                # Prepare answer engine
                answer_engine.prepare_document(document_text)
                
                # Generate answers
                logger.info("🤖 Generating answers with Groq gemma2-9b-it...")
                answers = []
                
                for i, question in enumerate(questions):
                    logger.info(f"Processing Q{i+1}: {question[:50]}...")
                    answer = await answer_engine.generate_competition_answer(question)
                    answers.append(answer)
                    logger.info(f"✅ Q{i+1} completed: {len(answer)} chars")
                
                total_time = time.time() - start_time
                logger.info(f"🏆 Competition completed in {total_time:.2f}s")
                
                return {"answers": answers}
                
            except Exception as e:
                logger.error(f"Processing error: {e}")
                questions = request_data.get("questions", [])
                return {
                    "answers": [f"Processing error: {str(e)}"] * len(questions)
                }
        
        else:
            raise HTTPException(
                status_code=400, 
                detail="Invalid request format"
            )
            
    except Exception as e:
        logger.error(f"Endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "competition_features": COMPETITION_AVAILABLE,
        "groq_api_key": "configured" if os.getenv("GROQ_API_KEY") else "missing"
    }

# Start server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting HackRx 6.0 Server on port {port}")
    logger.info(f"📊 Competition features: {'✅ Available' if COMPETITION_AVAILABLE else '❌ Not Available'}")
    logger.info(f"🤖 Using Groq gemma2-9b-it model")
    
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        logger.info(f"🔑 Groq API Key: Configured")
    else:
        logger.warning(f"⚠️ Groq API Key: NOT SET")
    
    uvicorn.run(app, host="0.0.0.0", port=port)