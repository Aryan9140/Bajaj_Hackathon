# app/services/embedding.py - Optimized Embedding Service
import os
import logging
import time
from typing import List, Optional, Dict, Any
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Optimized service for handling embeddings with OpenAI for maximum performance"""
    
    def __init__(self, 
                 openai_api_key: str = None,
                 model: str = "text-embedding-3-small",
                 dimensions: Optional[int] = None,
                 chunk_size: int = 1000):
        """
        Initialize optimized embedding service
        
        Args:
            openai_api_key: OpenAI API key
            model: Embedding model to use (optimized for speed and accuracy)
            dimensions: Number of dimensions for embeddings
            chunk_size: Maximum chunk size for batch processing
        """
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.dimensions = dimensions
        self.chunk_size = chunk_size
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass openai_api_key parameter.")
        
        # Initialize OpenAI embeddings with performance optimizations
        self.embeddings = self._initialize_embeddings_optimized()
    
    def _initialize_embeddings_optimized(self) -> OpenAIEmbeddings:
        """Initialize OpenAI embeddings with performance optimizations"""
        try:
            init_start = time.time()
            
            embedding_params = {
                "openai_api_key": self.openai_api_key,
                "model": self.model,
                "show_progress_bar": False,  # Disable for better performance
                "chunk_size": self.chunk_size,
                "max_retries": 3,  # Retry failed requests
                "request_timeout": 30  # Timeout for requests
            }
            
            # Add dimensions parameter if specified
            if self.dimensions:
                embedding_params["dimensions"] = self.dimensions
            
            embeddings = OpenAIEmbeddings(**embedding_params)
            
            # Test embedding with performance tracking
            test_start = time.time()
            test_embedding = embeddings.embed_query("test")
            test_time = time.time() - test_start
            
            init_time = time.time() - init_start
            logger.info(f"✅ OpenAI embeddings initialized in {init_time:.3f}s")
            logger.info(f"📊 Model: {self.model}, Dimensions: {len(test_embedding)}, Test time: {test_time:.3f}s")
            
            return embeddings
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI embeddings: {str(e)}")
            raise e
    
    def get_embeddings_instance(self) -> OpenAIEmbeddings:
        """
        Get the optimized embeddings instance
        
        Returns:
            OpenAIEmbeddings: The embeddings instance
        """
        return self.embeddings
    
    def embed_query_fast(self, text: str) -> List[float]:
        """
        Create embedding for a single query with performance tracking
        
        Args:
            text: Text to embed
            
        Returns:
            List[float]: Embedding vector
        """
        try:
            embed_start = time.time()
            
            embedding = self.embeddings.embed_query(text)
            
            embed_time = time.time() - embed_start
            logger.info(f"🔍 Created embedding for query in {embed_time:.3f}s: {text[:50]}...")
            return embedding
        except Exception as e:
            logger.error(f"❌ Failed to create query embedding: {str(e)}")
            raise e
    
    def embed_documents_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for multiple documents with optimized batch processing
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        try:
            batch_start = time.time()
            
            # Process in optimized batches
            if len(texts) > self.chunk_size:
                logger.info(f"📦 Processing {len(texts)} texts in batches of {self.chunk_size}")
                
                all_embeddings = []
                for i in range(0, len(texts), self.chunk_size):
                    batch = texts[i:i + self.chunk_size]
                    batch_embeddings = self.embeddings.embed_documents(batch)
                    all_embeddings.extend(batch_embeddings)
                    logger.info(f"📊 Processed batch {i//self.chunk_size + 1}/{(len(texts)-1)//self.chunk_size + 1}")
                
                embeddings = all_embeddings
            else:
                embeddings = self.embeddings.embed_documents(texts)
            
            batch_time = time.time() - batch_start
            logger.info(f"📚 Created {len(embeddings)} embeddings in {batch_time:.3f}s ({len(texts)/batch_time:.1f} texts/sec)")
            return embeddings
        except Exception as e:
            logger.error(f"❌ Failed to create document embeddings: {str(e)}")
            raise e
    
    def embed_document_objects_optimized(self, documents: List[Document]) -> List[List[float]]:
        """
        Create embeddings for Document objects with optimization
        
        Args:
            documents: List of Document objects
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        try:
            extract_start = time.time()
            
            texts = [doc.page_content for doc in documents]
            
            extract_time = time.time() - extract_start
            logger.info(f"📄 Extracted {len(texts)} texts from documents in {extract_time:.3f}s")
            
            embeddings = self.embed_documents_batch(texts)
            return embeddings
        except Exception as e:
            logger.error(f"❌ Failed to create embeddings for Document objects: {str(e)}")
            raise e
    
    def calculate_similarity_fast(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Fast calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            float: Similarity score (-1 to 1)
        """
        try:
            calc_start = time.time()
            
            import numpy as np
            
            # Convert to numpy arrays for speed
            vec1 = np.array(embedding1, dtype=np.float32)
            vec2 = np.array(embedding2, dtype=np.float32)
            
            # Fast cosine similarity calculation
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            
            calc_time = time.time() - calc_start
            logger.info(f"📐 Calculated similarity in {calc_time:.4f}s: {similarity:.4f}")
            return float(similarity)
        except Exception as e:
            logger.error(f"❌ Failed to calculate similarity: {str(e)}")
            return 0.0
    
    def find_most_similar_optimized(self, query_embedding: List[float], 
                                  document_embeddings: List[List[float]], 
                                  top_k: int = 5) -> List[tuple]:
        """
        Find indices of most similar documents with performance optimization
        
        Args:
            query_embedding: Query embedding vector
            document_embeddings: List of document embedding vectors
            top_k: Number of top results to return
            
        Returns:
            List[tuple]: List of (similarity_score, index) tuples
        """
        try:
            search_start = time.time()
            
            import numpy as np
            
            # Convert to numpy for vectorized operations
            query_vec = np.array(query_embedding, dtype=np.float32)
            doc_vecs = np.array(document_embeddings, dtype=np.float32)
            
            # Vectorized cosine similarity calculation
            dot_products = np.dot(doc_vecs, query_vec)
            query_norm = np.linalg.norm(query_vec)
            doc_norms = np.linalg.norm(doc_vecs, axis=1)
            
            similarities = dot_products / (doc_norms * query_norm)
            
            # Get top k indices
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
            
            # Return similarity scores and indices
            results = [(float(similarities[idx]), int(idx)) for idx in top_indices]
            
            search_time = time.time() - search_start
            logger.info(f"🎯 Found {len(results)} most similar documents in {search_time:.4f}s")
            return results
        except Exception as e:
            logger.error(f"❌ Failed to find most similar documents: {str(e)}")
            return []
    
    def get_embedding_info_detailed(self) -> Dict[str, Any]:
        """
        Get detailed information about the embedding model
        
        Returns:
            Dict[str, Any]: Embedding model information with performance metrics
        """
        try:
            info_start = time.time()
            
            # Test embedding to get dimension and performance
            test_start = time.time()
            test_embedding = self.embed_query_fast("performance test")
            test_time = time.time() - test_start
            
            info = {
                "model": self.model,
                "dimensions": len(test_embedding),
                "chunk_size": self.chunk_size,
                "api_key_configured": bool(self.openai_api_key),
                "status": "ready",
                "performance": {
                    "test_embedding_time": f"{test_time:.3f}s",
                    "optimizations": {
                        "batch_processing": True,
                        "vectorized_similarity": True,
                        "progress_bar_disabled": True,
                        "request_timeout": "30s",
                        "max_retries": 3
                    }
                },
                "capabilities": {
                    "single_query_embedding": True,
                    "batch_document_embedding": True,
                    "similarity_search": True,
                    "performance_tracking": True
                }
            }
            
            info_time = time.time() - info_start
            logger.info(f"📋 Retrieved detailed embedding info in {info_time:.3f}s")
            return info
        except Exception as e:
            logger.error(f"❌ Failed to get embedding info: {str(e)}")
            return {"error": str(e)}
    
    def validate_embeddings_fast(self, texts: List[str]) -> Dict[str, Any]:
        """
        Fast validate that embeddings can be created for given texts
        
        Args:
            texts: List of texts to validate
            
        Returns:
            Dict[str, Any]: Validation results with performance info
        """
        try:
            validation_start = time.time()
            
            if not texts:
                return {
                    "valid": False,
                    "error": "No texts provided for validation",
                    "validation_time": 0
                }
            
            # Try to create embeddings for first 3 texts (sample validation)
            sample_texts = texts[:3]
            embeddings = self.embed_documents_batch(sample_texts)
            
            validation_time = time.time() - validation_start
            
            if embeddings and all(len(emb) > 0 for emb in embeddings):
                result = {
                    "valid": True,
                    "sample_size": len(sample_texts),
                    "total_texts": len(texts),
                    "embedding_dimension": len(embeddings[0]) if embeddings else 0,
                    "validation_time": f"{validation_time:.3f}s",
                    "performance": "optimized"
                }
                logger.info(f"✅ Embedding validation successful in {validation_time:.3f}s")
                return result
            else:
                return {
                    "valid": False,
                    "error": "Empty embeddings generated",
                    "validation_time": f"{validation_time:.3f}s"
                }
                
        except Exception as e:
            validation_time = time.time() - validation_start if 'validation_start' in locals() else 0
            logger.error(f"❌ Embedding validation failed: {str(e)}")
            return {
                "valid": False,
                "error": str(e),
                "validation_time": f"{validation_time:.3f}s"
            }
    
    def health_check_embeddings(self) -> bool:
        """
        Quick health check for embedding service
        
        Returns:
            bool: True if healthy
        """
        try:
            health_start = time.time()
            
            # Quick test embedding
            test_embedding = self.embeddings.embed_query("health check")
            
            health_time = time.time() - health_start
            
            if test_embedding and len(test_embedding) > 0:
                logger.info(f"✅ Embedding service health check passed in {health_time:.3f}s")
                return True
            else:
                logger.warning("⚠️ Embedding service health check failed - empty embedding")
                return False
                
        except Exception as e:
            logger.warning(f"⚠️ Embedding service health check failed: {str(e)}")
            return False