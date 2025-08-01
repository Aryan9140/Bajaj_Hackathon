"""
run.py - Server startup and API endpoints
Imports LLM features from main.py
"""

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
import logging

# Import your LLM features from main.py
try:
    from main import (
        UniversalDocumentProcessor, 
        HybridAnswerGenerator,
        doc_processor,
        answer_generator
    )
    LLM_FEATURES_AVAILABLE = True
    logging.info("LLM features imported from main.py")
except ImportError as e:
    LLM_FEATURES_AVAILABLE = False
    logging.warning(f"LLM features not available: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="HackRx 6.0 - Server",
    description="Main server with LLM integration",
    version="6.0.0"
)

# Pydantic Models
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
    return {
        "message": "HackRx 6.0 - Server Running",
        "status": "ready", 
        "version": "6.0.0",
        "llm_features": "available" if LLM_FEATURES_AVAILABLE else "not available",
        "endpoints": {
            "/hackrx/run": "Main competition endpoint",
            "/health": "Health check"
        }
    }

@app.post("/hackrx/run")
async def unified_hackrx_endpoint(request: Request):
    """
    Main competition endpoint that handles:
    1. Simple data processing: {"data": [...]}
    2. Document processing: {"documents": "...", "questions": [...]}
    """
    try:
        request_data = await request.json()
        
        # Simple data processing (always available)
        if "data" in request_data:
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
        
        # Document processing (requires LLM features from main.py)
        elif "documents" in request_data and "questions" in request_data:
            if not LLM_FEATURES_AVAILABLE:
                logger.error("LLM features not available for document processing")
                return {
                    "answers": ["LLM features not available. Please check main.py configuration."] * len(request_data.get("questions", []))
                }
            
            documents = request_data.get("documents", "")
            questions = request_data.get("questions", [])
            
            logger.info(f"Processing {len(questions)} questions with LLM")
            
            # Use document processor from main.py
            document_text = await doc_processor.process_document(documents)
            
            if not document_text:
                logger.warning("No text extracted from document")
                return {
                    "answers": ["Unable to extract information from the provided document."] * len(questions)
                }
            
            logger.info(f"Document processed: {len(document_text)} characters")
            
            # Use LLM answer generator from main.py
            answers = []
            for i, question in enumerate(questions):
                logger.info(f"Processing question {i+1}: {question[:50]}...")
                
                answer = await answer_generator.generate_answer(question, document_text)
                answers.append(answer)
                
                logger.info(f"Answer {i+1} generated: {answer[:100]}...")
            
            logger.info(f"All {len(questions)} questions processed successfully")
            return {"answers": answers}
        
        else:
            raise HTTPException(status_code=400, detail="Invalid request format")
            
    except Exception as e:
        logger.error(f"Request processing failed: {e}")
        
        # Error fallback
        if 'request_data' in locals() and "questions" in request_data:
            question_count = len(request_data.get("questions", []))
            return {"answers": ["Unable to process this question at the moment."] * question_count}
        
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {
        "status": "healthy", 
        "version": "6.0.0",
        "llm_features": LLM_FEATURES_AVAILABLE
    }

@app.get("/test-llm")
async def test_llm():
    """Test endpoint to check LLM features"""
    if not LLM_FEATURES_AVAILABLE:
        return {"status": "LLM features not available"}
    
    try:
        # Simple test
        test_text = "This is a test document. The grace period is 30 days."
        test_question = "What is the grace period?"
        
        answer = await answer_generator.generate_answer(test_question, test_text)
        
        return {
            "status": "LLM features working",
            "test_question": test_question,
            "test_answer": answer
        }
    except Exception as e:
        return {"status": "LLM test failed", "error": str(e)}

# Server startup code (keep this in run.py)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)