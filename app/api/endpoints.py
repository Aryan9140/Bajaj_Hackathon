# app/api/endpoints.py
"""
API endpoints for HackRx 6.0 hackathon
Optimized for sub-5 second latency with proper error handling
"""

from fastapi import APIRouter, HTTPException, Depends, Request, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List, Dict, Any
import asyncio
import time
from contextlib import asynccontextmanager

from app.core.response_models import QueryRequest, QueryResponse
from app.core.security import verify_api_key
from app.services.query_processor import QueryProcessor
from app.utils.logger import get_logger
from app.utils.cache import CacheManager

logger = get_logger(__name__)
security = HTTPBearer()
router = APIRouter()

# Initialize cache for performance
cache_manager = CacheManager()

async def get_services(request: Request):
    """Dependency injection for services"""
    return {
        "vector_service": request.app.state.vector_service,
        "embedding_service": request.app.state.embedding_service
    }

@router.post("/hackrx/run", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    services: Dict = Depends(get_services),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Main endpoint for processing document queries
    Target: < 5 seconds response time
    """
    start_time = time.time()
    request_id = f"req_{int(start_time * 1000)}"
    
    logger.info(f"🚀 [{request_id}] Processing query with {len(request.questions)} questions")
    
    try:
        # 1. Verify API key (< 0.1s)
        if not verify_api_key(credentials.credentials):
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        # 2. Initialize query processor
        processor = QueryProcessor(
            vector_service=services["vector_service"],
            embedding_service=services["embedding_service"],
            cache_manager=cache_manager
        )
        
        # 3. Check cache first for performance
        cache_key = f"doc_{hash(request.documents)}_{hash(tuple(request.questions))}"
        cached_result = await cache_manager.get(cache_key)
        
        if cached_result:
            logger.info(f"📦 [{request_id}] Cache hit - returning cached result")
            processing_time = time.time() - start_time
            return QueryResponse(
                answers=cached_result["answers"],
                processing_time=processing_time,
                request_id=request_id,
                cached=True
            )
        
        # 4. Process documents and questions (main processing)
        answers = await processor.process_batch_queries(
            document_url=request.documents,
            questions=request.questions,
            request_id=request_id
        )
        
        # 5. Cache successful results
        await cache_manager.set(
            cache_key, 
            {"answers": answers},
            ttl=3600  # 1 hour cache
        )
        
        processing_time = time.time() - start_time
        
        logger.info(f"✅ [{request_id}] Completed in {processing_time:.2f}s")
        
        return QueryResponse(
            answers=answers,
            processing_time=processing_time,
            request_id=request_id,
            cached=False
        )
        
    except asyncio.TimeoutError:
        logger.error(f"⏰ [{request_id}] Request timeout after 30s")
        raise HTTPException(
            status_code=408, 
            detail="Request timeout - processing took too long"
        )
    
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"❌ [{request_id}] Error after {processing_time:.2f}s: {str(e)}")
        
        # Return graceful error with empty answers
        return QueryResponse(
            answers=["Error processing query. Please try again." for _ in request.questions],
            processing_time=processing_time,
            request_id=request_id,
            cached=False,
            error=str(e)
        )

@router.get("/status")
async def api_status():
    """API status and performance metrics"""
    return {
        "status": "operational",
        "version": "1.0.0",
        "cache_stats": await cache_manager.get_stats(),
        "endpoints": {
            "/hackrx/run": "Main query processing endpoint"
        }
    }

@router.post("/cache/clear")
async def clear_cache(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Clear cache - useful for testing"""
    if not verify_api_key(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    await cache_manager.clear()
    return {"message": "Cache cleared successfully"}