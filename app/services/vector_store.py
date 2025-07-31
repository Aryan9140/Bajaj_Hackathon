# app/services/vector_store.py
"""
AstraDB Vector Store Service optimized for high-performance retrieval
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
from astrapy.db import AstraDB
from astrapy.ops import AstraDBOps
import numpy as np

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

class VectorStoreService:
    """High-performance vector store service using AstraDB"""
    
    def __init__(self, embedding_service):
        self.embedding_service = embedding_service
        self.db = None
        self.collection = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize AstraDB connection"""
        try:
            logger.info("🔗 Initializing AstraDB connection...")
            
            # Initialize AstraDB client
            self.db = AstraDB(
                api_endpoint=settings.ASTRA_DB_API_ENDPOINT,
                token=settings.ASTRA_DB_APPLICATION_TOKEN,
                namespace=settings.ASTRA_DB_KEYSPACE
            )
            
            # Get or create collection
            self.collection = await self._get_or_create_collection()
            
            self.initialized = True
            logger.info("✅ AstraDB initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize AstraDB: {e}")
            raise
    
    async def _get_or_create_collection(self):
        """Get existing collection or create new one"""
        try:
            # Try to get existing collection
            collection = self.db.collection(settings.ASTRA_DB_COLLECTION)
            
            # Test collection access
            await self._test_collection_access(collection)
            
            return collection
            
        except Exception:
            # Create new collection if it doesn't exist
            logger.info(f"📝 Creating collection: {settings.ASTRA_DB_COLLECTION}")
            
            collection = self.db.create_collection(
                settings.ASTRA_DB_COLLECTION,
                dimension=settings.EMBEDDING_DIMENSION,
                metric="cosine"  # Optimal for sentence embeddings
            )
            
            return collection
    
    async def _test_collection_access(self, collection):
        """Test if collection is accessible"""
        try:
            # Simple test query
            result = collection.find_one({})
            logger.info("✅ Collection access verified")
        except Exception as e:
            logger.info(f"🔄 Collection not accessible, will create new: {e}")
            raise
    
    async def add_chunks(self, chunks: List[Dict], request_id: str):
        """Add document chunks to vector store with batch processing"""
        if not self.initialized:
            raise Exception("Vector store not initialized")
        
        if not chunks:
            return
        
        try:
            start_time = time.time()
            
            # Prepare documents for insertion
            documents = []
            texts = [chunk['content'] for chunk in chunks]
            
            # Generate embeddings in batch (more efficient)
            embeddings = await self.embedding_service.embed_texts(texts)
            
            # Prepare documents with embeddings
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                doc = {
                    "_id": f"{request_id}_chunk_{i}",
                    "content": chunk['content'],
                    "page": chunk.get('page', 0),
                    "chunk_index": i,
                    "request_id": request_id,
                    "$vector": embedding
                }
                documents.append(doc)
            
            # Insert documents in batch
            result = self.collection.insert_many(documents)
            
            processing_time = time.time() - start_time
            logger.info(
                f"💾 [{request_id}] Stored {len(documents)} chunks in {processing_time:.2f}s"
            )
            
        except Exception as e:
            logger.error(f"❌ [{request_id}] Failed to store chunks: {e}")
            raise
    
    async def similarity_search(
        self,
        embedding: List[float],
        top_k: int = 10,
        request_id: str = None
    ) -> List[Dict]:
        """Perform similarity search with optimized retrieval"""
        if not self.initialized:
            raise Exception("Vector store not initialized")
        
        try:
            start_time = time.time()
            
            # Perform vector similarity search
            results = self.collection.vector_find(
                vector=embedding,
                limit=top_k,
                fields=["content", "page", "chunk_index", "_id"]
            )
            
            # Convert results to standard format
            chunks = []
            for doc in results:
                chunks.append({
                    "content": doc.get("content", ""),
                    "page": doc.get("page", 0),
                    "chunk_index": doc.get("chunk_index", 0),
                    "similarity_score": doc.get("$similarity", 0.0),
                    "id": doc.get("_id", "")
                })
            
            search_time = time.time() - start_time
            logger.info(
                f"🔍 [{request_id}] Retrieved {len(chunks)} chunks in {search_time:.3f}s"
            )
            
            return chunks
            
        except Exception as e:
            logger.error(f"❌ [{request_id}] Similarity search failed: {e}")
            return []
    
    async def clear_collection(self):
        """Clear all documents from collection (useful for testing)"""
        if not self.initialized:
            return
        
        try:
            # Delete all documents
            self.collection.delete_many({})
            logger.info("🗑️ Collection cleared successfully")
        except Exception as e:
            logger.error(f"❌ Failed to clear collection: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        if not self.initialized:
            return {"status": "not_initialized"}
        
        try:
            # Get document count
            count_result = self.collection.count_documents({})
            
            return {
                "status": "operational",
                "document_count": count_result,
                "collection_name": settings.ASTRA_DB_COLLECTION,
                "embedding_dimension": settings.EMBEDDING_DIMENSION
            }
        except Exception as e:
            logger.error(f"❌ Failed to get stats: {e}")
            return {"status": "error", "error": str(e)}
    
    async def close(self):
        """Clean up resources"""
        if self.db:
            # AstraDB doesn't require explicit connection closing
            self.initialized = False
            logger.info("🔄 AstraDB connection cleaned up")