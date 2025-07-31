# # app/services/embedding_service.py
# """
# High-performance embedding service using HuggingFace models
# Optimized for batch processing and caching
# """

# import asyncio
# import time
# from typing import List, Optional
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from concurrent.futures import ThreadPoolExecutor
# import threading

# from app.core.config import settings
# from app.utils.logger import get_logger

# logger = get_logger(__name__)

# class EmbeddingService:
#     """Embedding service with connection pooling and caching"""
    
#     def __init__(self):
#         self.model = None
#         self.executor = ThreadPoolExecutor(max_workers=2)
#         self._lock = threading.Lock()
#         self.initialized = False
    
#     async def initialize(self):
#         """Initialize the embedding model"""
#         try:
#             logger.info(f"🔢 Loading embedding model: {settings.EMBEDDING_MODEL}")
            
#             # Load model in thread pool to avoid blocking
#             loop = asyncio.get_event_loop()
#             self.model = await loop.run_in_executor(
#                 self.executor,
#                 self._load_model
#             )
            
#             self.initialized = True
#             logger.info("✅ Embedding service initialized successfully")
            
#         except Exception as e:
#             logger.error(f"❌ Failed to initialize embedding service: {e}")
#             raise
    
#     def _load_model(self) -> SentenceTransformer:
#         """Load the sentence transformer model"""
#         return SentenceTransformer(
#             settings.EMBEDDING_MODEL,
#             device='cpu'  # Use CPU for consistency across deployments
#         )
    
#     async def embed_texts(self, texts: List[str]) -> List[List[float]]:
#         """Generate embeddings for a list of texts"""
#         if not self.initialized:
#             raise Exception("Embedding service not initialized")
        
#         if not texts:
#             return []
        
#         start_time = time.time()
        
#         try:
#             # Process embeddings in thread pool (CPU-bound operation)
#             loop = asyncio.get_event_loop()
#             embeddings = await loop.run_in_executor(
#                 self.executor,
#                 self._generate_embeddings,
#                 texts
#             )
            
#             processing_time = time.time() - start_time
#             logger.info(f"🔢 Generated {len(embeddings)} embeddings in {processing_time:.3f}s")
            
#             return embeddings.tolist()
            
#         except Exception as e:
#             logger.error(f"❌ Embedding generation failed: {e}")
#             raise
    
#     def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
#         """Generate embeddings using the model (runs in thread pool)"""
#         with self._lock:  # Ensure thread safety
#             return self.model.encode(
#                 texts,
#                 batch_size=32,  # Optimized batch size
#                 show_progress_bar=False,
#                 convert_to_numpy=True,
#                 normalize_embeddings=True  # Better for cosine similarity
#             )
    
#     async def embed_single_text(self, text: str) -> List[float]:
#         """Generate embedding for a single text"""
#         embeddings = await self.embed_texts([text])
#         return embeddings[0] if embeddings else []
    
#     async def close(self):
#         """Clean up resources"""
#         if self.executor:
#             self.executor.shutdown(wait=True)
#         self.initialized = False
#         logger.info("🔄 Embedding service cleanup completed")


# app/services/embedding_service.py
"""
HackRx 6.0 - Embedding Service with FAISS Vector Search
Implements semantic search and clause retrieval as required
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple
import pickle
import os
import logging

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Handles document embeddings and semantic search using FAISS
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.index = None
        self.chunks = []
        self.chunk_metadata = []
        self.is_initialized = False
        self.embedding_dim = 384  # MiniLM-L6-v2 dimension
        
    async def initialize(self):
        """Initialize the embedding model and FAISS index"""
        try:
            print("🔧 Initializing embedding service...")
            
            # Load sentence transformer model
            self.model = SentenceTransformer(self.model_name)
            print(f"✅ Loaded embedding model: {self.model_name}")
            
            # Initialize FAISS index
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product for cosine similarity
            print("✅ FAISS index initialized")
            
            self.is_initialized = True
            print("🎯 Embedding service ready for semantic search")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding service: {e}")
            raise
    
    def chunk_document(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[Dict[str, Any]]:
        """
        Split document into semantic chunks for better retrieval
        """
        # Split by sentences first
        sentences = text.split('.')
        
        chunks = []
        current_chunk = ""
        current_length = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_length = len(sentence)
            
            # If adding this sentence exceeds chunk size, save current chunk
            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append({
                    'id': chunk_id,
                    'text': current_chunk.strip(),
                    'length': current_length,
                    'start_sentence': len(chunks) * 10,  # Approximate
                    'type': self._classify_chunk_type(current_chunk)
                })
                
                # Start new chunk with overlap
                overlap_text = ' '.join(current_chunk.split()[-overlap:]) if overlap > 0 else ""
                current_chunk = overlap_text + " " + sentence
                current_length = len(current_chunk)
                chunk_id += 1
            else:
                current_chunk += " " + sentence
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                'id': chunk_id,
                'text': current_chunk.strip(),
                'length': current_length,
                'start_sentence': len(chunks) * 10,
                'type': self._classify_chunk_type(current_chunk)
            })
        
        print(f"📄 Document chunked into {len(chunks)} semantic pieces")
        return chunks
    
    def _classify_chunk_type(self, text: str) -> str:
        """Classify chunk type for better retrieval"""
        text_lower = text.lower()
        
        if any(term in text_lower for term in ['grace period', 'premium', 'payment']):
            return 'payment_terms'
        elif any(term in text_lower for term in ['waiting period', 'pre-existing', 'coverage']):
            return 'coverage_conditions'
        elif any(term in text_lower for term in ['hospital', 'institution', 'medical']):
            return 'definitions'
        elif any(term in text_lower for term in ['benefit', 'claim', 'discount']):
            return 'benefits'
        elif any(term in text_lower for term in ['exclude', 'not covered', 'limitation']):
            return 'exclusions'
        else:
            return 'general'
    
    async def create_embeddings(self, documents: List[str], document_metadata: List[Dict] = None) -> bool:
        """
        Create embeddings for document chunks and add to FAISS index
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            print(f"🔄 Creating embeddings for {len(documents)} document chunks...")
            
            # Generate embeddings
            embeddings = self.model.encode(documents, convert_to_tensor=False, normalize_embeddings=True)
            embeddings = np.array(embeddings).astype('float32')
            
            # Add to FAISS index
            self.index.add(embeddings)
            
            # Store chunks and metadata
            self.chunks.extend(documents)
            if document_metadata:
                self.chunk_metadata.extend(document_metadata)
            else:
                self.chunk_metadata.extend([{'id': i, 'type': 'unknown'} for i in range(len(documents))])
            
            print(f"✅ Added {len(documents)} embeddings to FAISS index")
            print(f"📊 Total chunks in index: {self.index.ntotal}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create embeddings: {e}")
            return False
    
    async def semantic_search(self, query: str, top_k: int = 5, min_similarity: float = 0.3) -> List[Dict[str, Any]]:
        """
        Perform semantic search to find relevant document chunks
        """
        if not self.is_initialized or self.index.ntotal == 0:
            print("⚠️ No embeddings available for search")
            return []
        
        try:
            # Encode query
            query_embedding = self.model.encode([query], convert_to_tensor=False, normalize_embeddings=True)
            query_embedding = np.array(query_embedding).astype('float32')
            
            # Search in FAISS index
            similarities, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
            
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if similarity >= min_similarity:
                    result = {
                        'rank': i + 1,
                        'chunk_id': idx,
                        'text': self.chunks[idx],
                        'similarity_score': float(similarity),
                        'metadata': self.chunk_metadata[idx] if idx < len(self.chunk_metadata) else {},
                        'chunk_type': self.chunk_metadata[idx].get('type', 'unknown') if idx < len(self.chunk_metadata) else 'unknown'
                    }
                    results.append(result)
            
            print(f"🔍 Semantic search found {len(results)} relevant chunks for: '{query[:50]}...'")
            return results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    async def clause_matching(self, query: str, document_chunks: List[str]) -> List[Dict[str, Any]]:
        """
        Advanced clause matching with semantic similarity
        """
        try:
            if not document_chunks:
                return []
            
            # Create temporary embeddings for these specific chunks
            chunk_embeddings = self.model.encode(document_chunks, convert_to_tensor=False, normalize_embeddings=True)
            query_embedding = self.model.encode([query], convert_to_tensor=False, normalize_embeddings=True)
            
            # Calculate similarities
            similarities = np.dot(chunk_embeddings, query_embedding.T).flatten()
            
            # Rank clauses by similarity
            clause_matches = []
            for i, (chunk, similarity) in enumerate(zip(document_chunks, similarities)):
                if similarity > 0.2:  # Minimum threshold
                    clause_matches.append({
                        'clause_id': i,
                        'text': chunk,
                        'similarity': float(similarity),
                        'relevance': 'high' if similarity > 0.7 else 'medium' if similarity > 0.5 else 'low'
                    })
            
            # Sort by similarity
            clause_matches.sort(key=lambda x: x['similarity'], reverse=True)
            
            print(f"📋 Clause matching found {len(clause_matches)} relevant clauses")
            return clause_matches
            
        except Exception as e:
            logger.error(f"Clause matching failed: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get embedding service statistics"""
        return {
            'initialized': self.is_initialized,
            'model': self.model_name,
            'total_chunks': len(self.chunks),
            'faiss_index_size': self.index.ntotal if self.index else 0,
            'embedding_dimension': self.embedding_dim
        }
    
    def save_index(self, filepath: str):
        """Save FAISS index and metadata to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, f"{filepath}.faiss")
            
            # Save metadata
            metadata = {
                'chunks': self.chunks,
                'chunk_metadata': self.chunk_metadata,
                'model_name': self.model_name
            }
            with open(f"{filepath}.pkl", 'wb') as f:
                pickle.dump(metadata, f)
            
            print(f"💾 FAISS index saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
    
    def load_index(self, filepath: str):
        """Load FAISS index and metadata from disk"""
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{filepath}.faiss")
            
            # Load metadata
            with open(f"{filepath}.pkl", 'rb') as f:
                metadata = pickle.load(f)
            
            self.chunks = metadata['chunks']
            self.chunk_metadata = metadata['chunk_metadata']
            
            print(f"📂 FAISS index loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False  