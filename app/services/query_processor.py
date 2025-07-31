# app/services/query_processor.py
"""
Main query processing pipeline for HackRx 6.0
Optimized for sub-5 second latency with high accuracy
"""

import asyncio
import aiohttp
import time
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import tempfile
import os

from app.services.document_processor import DocumentProcessor
from app.services.vector_store import VectorStoreService
from app.services.embedding_service import EmbeddingService
from app.services.llm_handler import LLMHandler
from app.utils.logger import get_logger
from app.utils.cache import CacheManager
from app.core.config import settings

logger = get_logger(__name__)

class QueryProcessor:
    """Main orchestrator for document query processing"""
    
    def __init__(
        self,
        vector_service: VectorStoreService,
        embedding_service: EmbeddingService,
        cache_manager: CacheManager
    ):
        self.vector_service = vector_service
        self.embedding_service = embedding_service
        self.cache_manager = cache_manager
        self.llm_handler = LLMHandler()
        self.doc_processor = DocumentProcessor()
        
        # Thread pool for CPU-bound operations
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def process_batch_queries(
        self,
        document_url: str,
        questions: List[str],
        request_id: str
    ) -> List[str]:
        """
        Process multiple questions against a document
        Target: < 5 seconds total processing time
        """
        start_time = time.time()
        
        try:
            # Step 1: Download and process document (parallel with embedding prep)
            logger.info(f"📄 [{request_id}] Starting document processing...")
            
            # Create tasks for parallel execution
            tasks = [
                self._download_and_process_document(document_url, request_id),
                self._prepare_embeddings(questions, request_id)
            ]
            
            # Execute document processing and question embedding in parallel
            doc_chunks, question_embeddings = await asyncio.gather(*tasks)
            
            if not doc_chunks:
                logger.warning(f"⚠️ [{request_id}] No content extracted from document")
                return ["Unable to process document content." for _ in questions]
            
            logger.info(f"📝 [{request_id}] Document processed: {len(doc_chunks)} chunks")
            
            # Step 2: Store document chunks in vector database
            await self._store_document_chunks(doc_chunks, request_id)
            
            # Step 3: Process all questions in parallel
            logger.info(f"🔍 [{request_id}] Processing {len(questions)} questions...")
            
            # Create answer tasks
            answer_tasks = [
                self._process_single_question(question, question_embeddings[i], request_id, i)
                for i, question in enumerate(questions)
            ]
            
            # Execute all questions in parallel with timeout
            answers = await asyncio.wait_for(
                asyncio.gather(*answer_tasks, return_exceptions=True),
                timeout=settings.REQUEST_TIMEOUT - 5  # Leave 5s buffer
            )
            
            # Handle any exceptions in parallel processing
            final_answers = []
            for i, answer in enumerate(answers):
                if isinstance(answer, Exception):
                    logger.error(f"❌ [{request_id}] Question {i+1} failed: {answer}")
                    final_answers.append(f"Error processing question: {str(answer)}")
                else:
                    final_answers.append(answer)
            
            processing_time = time.time() - start_time
            logger.info(f"✅ [{request_id}] Batch processing completed in {processing_time:.2f}s")
            
            return final_answers
            
        except asyncio.TimeoutError:
            logger.error(f"⏰ [{request_id}] Batch processing timeout")
            return ["Processing timeout - please try again." for _ in questions]
        
        except Exception as e:
            logger.error(f"❌ [{request_id}] Batch processing error: {e}")
            return [f"Processing error: {str(e)}" for _ in questions]
    
    async def _download_and_process_document(self, document_url: str, request_id: str) -> List[Dict]:
        """Download and process document into chunks"""
        try:
            # Download document
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(document_url) as response:
                    if response.status != 200:
                        raise Exception(f"Failed to download document: HTTP {response.status}")
                    
                    content = await response.read()
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(content)
                tmp_path = tmp_file.name
            
            try:
                # Process document in thread pool (CPU-bound)
                loop = asyncio.get_event_loop()
                chunks = await loop.run_in_executor(
                    self.executor,
                    self.doc_processor.process_document,
                    tmp_path
                )
                return chunks
            finally:
                # Clean up temporary file
                os.unlink(tmp_path)
                
        except Exception as e:
            logger.error(f"❌ [{request_id}] Document processing error: {e}")
            return []
    
    async def _prepare_embeddings(self, questions: List[str], request_id: str) -> List[List[float]]:
        """Generate embeddings for questions in parallel"""
        try:
            # Generate embeddings for all questions at once (batch processing)
            embeddings = await self.embedding_service.embed_texts(questions)
            logger.info(f"🔢 [{request_id}] Generated {len(embeddings)} question embeddings")
            return embeddings
        except Exception as e:
            logger.error(f"❌ [{request_id}] Embedding generation error: {e}")
            return [[] for _ in questions]
    
    async def _store_document_chunks(self, chunks: List[Dict], request_id: str):
        """Store document chunks in vector database"""
        try:
            # Add chunks to vector store with batch processing
            await self.vector_service.add_chunks(chunks, request_id)
            logger.info(f"💾 [{request_id}] Stored {len(chunks)} chunks in vector DB")
        except Exception as e:
            logger.error(f"❌ [{request_id}] Vector storage error: {e}")
            raise
    
    async def _process_single_question(
        self,
        question: str,
        question_embedding: List[float],
        request_id: str,
        question_idx: int
    ) -> str:
        """Process a single question with retrieval and LLM generation"""
        try:
            # Retrieve relevant chunks
            relevant_chunks = await self.vector_service.similarity_search(
                embedding=question_embedding,
                top_k=settings.MAX_RETRIEVE_DOCS,
                request_id=request_id
            )
            
            if not relevant_chunks:
                logger.warning(f"⚠️ [{request_id}] No relevant chunks found for question {question_idx+1}")
                return "No relevant information found in the document."
            
            # Prepare context for LLM
            context = self._build_context(relevant_chunks)
            
            # Generate answer using LLM
            answer = await self.llm_handler.generate_answer(
                question=question,
                context=context,
                request_id=request_id
            )
            
            logger.info(f"✅ [{request_id}] Question {question_idx+1} processed successfully")
            return answer
            
        except Exception as e:
            logger.error(f"❌ [{request_id}] Single question processing error: {e}")
            return f"Error processing question: {str(e)}"
    
    def _build_context(self, chunks: List[Dict]) -> str:
        """Build context string from retrieved chunks"""
        context_parts = []
        
        for i, chunk in enumerate(chunks):
            content = chunk.get('content', '').strip()
            if content:
                # Add chunk with source information for explainability
                context_parts.append(f"[Chunk {i+1}]\n{content}")
        
        return "\n\n".join(context_parts)
    
    async def close(self):
        """Clean up resources"""
        self.executor.shutdown(wait=True)
        await self.llm_handler.close()
        logger.info("🔄 QueryProcessor cleanup completed")