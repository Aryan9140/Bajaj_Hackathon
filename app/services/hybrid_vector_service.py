"""
HackRx 6.0 - Hybrid Vector Service (FAISS + AstraDB)
Implements both FAISS and AstraDB with intelligent fallback
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Tuple
import asyncio
import time
import logging
from astrapy import DataAPIClient
import os

logger = logging.getLogger(__name__)

class HybridVectorService:
    """
    Hybrid vector search using FAISS (primary) and AstraDB (fallback)
    Automatically selects fastest performing option
    """
    
    def __init__(self):
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.model = None
        self.embedding_dim = 384
        
        # FAISS components
        self.faiss_index = None
        self.faiss_chunks = []
        self.faiss_metadata = []
        self.faiss_available = False
        
        # AstraDB components
        self.astra_client = None
        self.astra_collection = None
        self.astra_available = False
        
        # Performance tracking
        self.faiss_avg_time = 0.0
        self.astra_avg_time = 0.0
        self.performance_samples = {'faiss': [], 'astra': []}
        
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize both FAISS and AstraDB systems"""
        try:
            print("🔧 Initializing Hybrid Vector Service...")
            
            # Load embedding model
            self.model = SentenceTransformer(self.model_name)
            print(f"✅ Loaded embedding model: {self.model_name}")
            
            # Initialize FAISS
            await self._init_faiss()
            
            # Initialize AstraDB
            await self._init_astradb()
            
            if not self.faiss_available and not self.astra_available:
                raise Exception("Neither FAISS nor AstraDB could be initialized")
            
            self.is_initialized = True
            print(f"🎯 Hybrid Vector Service ready - FAISS: {self.faiss_available}, AstraDB: {self.astra_available}")
            
        except Exception as e:
            logger.error(f"Failed to initialize hybrid vector service: {e}")
            raise
    
    async def _init_faiss(self):
        """Initialize FAISS index"""
        try:
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
            self.faiss_available = True
            print("✅ FAISS index initialized successfully")
        except Exception as e:
            logger.warning(f"FAISS initialization failed: {e}")
            self.faiss_available = False
    
    async def _init_astradb(self):
        """Initialize AstraDB connection"""
        try:
            astra_endpoint = os.getenv('ASTRA_DB_API_ENDPOINT')
            astra_token = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
            
            if not astra_endpoint or not astra_token:
                raise Exception("AstraDB credentials not configured")
            
            self.astra_client = DataAPIClient(astra_token)
            database = self.astra_client.get_database_by_api_endpoint(astra_endpoint)
            
            # Create or get collection
            try:
                self.astra_collection = database.create_collection(
                    "hackrx_embeddings",
                    dimension=self.embedding_dim,
                    metric="cosine"
                )
            except:
                self.astra_collection = database.get_collection("hackrx_embeddings")
            
            self.astra_available = True
            print("✅ AstraDB connection established successfully")
            
        except Exception as e:
            logger.warning(f"AstraDB initialization failed: {e}")
            self.astra_available = False
    
    def chunk_document(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[Dict[str, Any]]:
        """
        Advanced document chunking with semantic awareness
        """
        # Split by sentences for better semantic coherence
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        chunks = []
        current_chunk = ""
        chunk_id = 0
        
        for i, sentence in enumerate(sentences):
            # Check if adding sentence exceeds chunk size
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                # Create chunk with metadata
                chunk_data = {
                    'id': chunk_id,
                    'text': current_chunk.strip(),
                    'length': len(current_chunk),
                    'sentence_start': max(0, i - 10),
                    'sentence_end': i,
                    'type': self._classify_chunk_type(current_chunk),
                    'keywords': self._extract_keywords(current_chunk)
                }
                chunks.append(chunk_data)
                
                # Start new chunk with overlap
                overlap_words = current_chunk.split()[-overlap:] if overlap > 0 else []
                current_chunk = ' '.join(overlap_words) + ' ' + sentence
                chunk_id += 1
            else:
                current_chunk += ' ' + sentence
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                'id': chunk_id,
                'text': current_chunk.strip(),
                'length': len(current_chunk),
                'sentence_start': max(0, len(sentences) - 10),
                'sentence_end': len(sentences),
                'type': self._classify_chunk_type(current_chunk),
                'keywords': self._extract_keywords(current_chunk)
            })
        
        print(f"📄 Document chunked into {len(chunks)} semantic pieces")
        return chunks
    
    def _classify_chunk_type(self, text: str) -> str:
        """Classify chunk type for better organization"""
        text_lower = text.lower()
        
        # Insurance-specific classification
        if any(term in text_lower for term in ['grace period', 'premium payment', 'due date']):
            return 'payment_terms'
        elif any(term in text_lower for term in ['waiting period', 'pre-existing', 'ped']):
            return 'coverage_conditions'
        elif any(term in text_lower for term in ['maternity', 'pregnancy', 'delivery']):
            return 'maternity_benefits'
        elif any(term in text_lower for term in ['hospital', 'institution', 'medical facility']):
            return 'definitions'
        elif any(term in text_lower for term in ['claim discount', 'ncd', 'no claim']):
            return 'discounts_benefits'
        elif any(term in text_lower for term in ['exclude', 'not covered', 'limitation']):
            return 'exclusions'
        elif any(term in text_lower for term in ['ayush', 'alternative treatment']):
            return 'alternative_medicine'
        else:
            return 'general_terms'
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        # Simple keyword extraction (can be enhanced with NLP)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}
        words = text.lower().split()
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        return list(set(keywords))[:10]  # Top 10 unique keywords
    
    async def add_documents(self, chunks: List[Dict[str, Any]]) -> bool:
        """
        Add document chunks to both FAISS and AstraDB
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            texts = [chunk['text'] for chunk in chunks]
            print(f"🔄 Adding {len(texts)} chunks to vector stores...")
            
            # Generate embeddings
            embeddings = self.model.encode(texts, convert_to_tensor=False, normalize_embeddings=True)
            embeddings = np.array(embeddings).astype('float32')
            
            success_count = 0
            
            # Add to FAISS
            if self.faiss_available:
                try:
                    self.faiss_index.add(embeddings)
                    self.faiss_chunks.extend(texts)
                    self.faiss_metadata.extend(chunks)
                    success_count += 1
                    print("✅ Added to FAISS index")
                except Exception as e:
                    logger.error(f"FAISS addition failed: {e}")
            
            # Add to AstraDB
            if self.astra_available:
                try:
                    documents = []
                    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                        doc = {
                            '_id': f"chunk_{chunk['id']}_{int(time.time())}",
                            'text': chunk['text'],
                            'chunk_type': chunk['type'],
                            'keywords': chunk['keywords'],
                            'metadata': chunk,
                            '$vector': embedding.tolist()
                        }
                        documents.append(doc)
                    
                    # Batch insert
                    self.astra_collection.insert_many(documents)
                    success_count += 1
                    print("✅ Added to AstraDB collection")
                except Exception as e:
                    logger.error(f"AstraDB addition failed: {e}")
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False
    
    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Intelligent search using fastest available method
        """
        if not self.is_initialized:
            return []
        
        # Determine which method to use based on performance
        use_faiss = self._should_use_faiss()
        
        if use_faiss and self.faiss_available:
            return await self._search_faiss(query, top_k)
        elif self.astra_available:
            return await self._search_astradb(query, top_k)
        else:
            print("⚠️ No vector search methods available")
            return []
    
    def _should_use_faiss(self) -> bool:
        """Decide whether to use FAISS based on performance"""
        if not self.faiss_available:
            return False
        if not self.astra_available:
            return True
        
        # If we have performance data, use the faster one
        if self.faiss_avg_time > 0 and self.astra_avg_time > 0:
            return self.faiss_avg_time <= self.astra_avg_time
        
        # Default to FAISS for small datasets
        return self.faiss_index.ntotal < 10000 if self.faiss_index else True
    
    async def _search_faiss(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Search using FAISS"""
        start_time = time.time()
        
        try:
            if self.faiss_index.ntotal == 0:
                return []
            
            # Encode query
            query_embedding = self.model.encode([query], convert_to_tensor=False, normalize_embeddings=True)
            query_embedding = np.array(query_embedding).astype('float32')
            
            # Search
            similarities, indices = self.faiss_index.search(query_embedding, min(top_k, self.faiss_index.ntotal))
            
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if similarity > 0.3:  # Minimum similarity threshold
                    result = {
                        'rank': i + 1,
                        'text': self.faiss_chunks[idx],
                        'similarity_score': float(similarity),
                        'source': 'faiss',
                        'metadata': self.faiss_metadata[idx] if idx < len(self.faiss_metadata) else {},
                        'chunk_type': self.faiss_metadata[idx].get('type', 'unknown') if idx < len(self.faiss_metadata) else 'unknown'
                    }
                    results.append(result)
            
            # Update performance tracking
            search_time = time.time() - start_time
            self._update_performance('faiss', search_time)
            
            print(f"🔍 FAISS search found {len(results)} results in {search_time:.3f}s")
            return results
            
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []
    
    async def _search_astradb(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Search using AstraDB"""
        start_time = time.time()
        
        try:
            # Encode query
            query_embedding = self.model.encode([query], convert_to_tensor=False, normalize_embeddings=True)
            
            # Search in AstraDB
            results = self.astra_collection.find(
                sort={"$vector": query_embedding.tolist()},
                limit=top_k,
                include_similarity=True
            )
            
            search_results = []
            for i, doc in enumerate(results):
                similarity = doc.get('$similarity', 0.0)
                if similarity > 0.3:
                    result = {
                        'rank': i + 1,
                        'text': doc.get('text', ''),
                        'similarity_score': similarity,
                        'source': 'astradb',
                        'metadata': doc.get('metadata', {}),
                        'chunk_type': doc.get('chunk_type', 'unknown')
                    }
                    search_results.append(result)
            
            # Update performance tracking
            search_time = time.time() - start_time
            self._update_performance('astra', search_time)
            
            print(f"🔍 AstraDB search found {len(search_results)} results in {search_time:.3f}s")
            return search_results
            
        except Exception as e:
            logger.error(f"AstraDB search failed: {e}")
            return []
    
    def _update_performance(self, method: str, time_taken: float):
        """Update performance tracking"""
        self.performance_samples[method].append(time_taken)
        
        # Keep only last 10 samples
        if len(self.performance_samples[method]) > 10:
            self.performance_samples[method] = self.performance_samples[method][-10:]
        
        # Update average
        if method == 'faiss':
            self.faiss_avg_time = sum(self.performance_samples['faiss']) / len(self.performance_samples['faiss'])
        else:
            self.astra_avg_time = sum(self.performance_samples['astra']) / len(self.performance_samples['astra'])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        return {
            'initialized': self.is_initialized,
            'faiss_available': self.faiss_available,
            'astra_available': self.astra_available,
            'faiss_chunks': len(self.faiss_chunks),
            'faiss_avg_time': self.faiss_avg_time,
            'astra_avg_time': self.astra_avg_time,
            'preferred_method': 'faiss' if self._should_use_faiss() else 'astradb',
            'embedding_model': self.model_name,
            'embedding_dimension': self.embedding_dim
        }