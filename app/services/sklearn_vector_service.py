# app/services/sklearn_vector_service.py - High-Performance Vector Store with Scikit-learn
import os
import logging
import time
import pickle
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class SklearnVectorService:
    """High-performance local vector store using Scikit-learn + OpenAI embeddings"""
    
    def __init__(self, embedding_function, index_path: str = "sklearn_index"):
        """
        Initialize Sklearn vector service
        
        Args:
            embedding_function: OpenAI embeddings function
            index_path: Path to save/load index
        """
        self.embedding_function = embedding_function
        self.index_path = index_path
        self.documents = []
        self.embeddings_matrix = None
        self.document_texts = []
        
        # TF-IDF vectorizer for fallback similarity
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = None
        
        # Optimized text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=150,
            length_function=len,
            separators=["\n\n", "\n", ". ", ".", " ", ""]
        )
        
        logger.info("✅ Sklearn Vector Service initialized")
    
    def add_documents_fast(self, documents: List[Document]) -> List[str]:
        """
        Add documents with high-performance processing
        
        Args:
            documents: List of documents to add
            
        Returns:
            List[str]: Document IDs
        """
        try:
            add_start = time.time()
            
            if not documents:
                logger.warning("⚠️ No documents provided")
                return []
            
            # Split documents into optimized chunks
            all_chunks = []
            for doc in documents:
                chunks = self.text_splitter.split_documents([doc])
                for i, chunk in enumerate(chunks):
                    chunk.metadata.update({
                        "chunk_id": f"{doc.metadata.get('source', 'doc')}_{i}",
                        "chunk_size": len(chunk.page_content),
                        "doc_type": "insurance_policy",
                        "chunk_index": i
                    })
                all_chunks.extend(chunks)
            
            logger.info(f"📄 Split {len(documents)} documents into {len(all_chunks)} chunks")
            
            # Create embeddings in batches
            texts = [chunk.page_content for chunk in all_chunks]
            embeddings = self._create_embeddings_batch(texts)
            
            # Store documents and embeddings
            self.documents.extend(all_chunks)
            self.document_texts.extend(texts)
            
            # Update embeddings matrix
            if self.embeddings_matrix is None:
                self.embeddings_matrix = np.array(embeddings)
            else:
                self.embeddings_matrix = np.vstack([self.embeddings_matrix, embeddings])
            
            # Update TF-IDF matrix for fallback
            self._update_tfidf_matrix()
            
            # Save index
            self._save_index()
            
            add_time = time.time() - add_start
            logger.info(f"✅ Added {len(all_chunks)} chunks in {add_time:.2f}s")
            
            return [f"sklearn_{i}" for i in range(len(all_chunks))]
            
        except Exception as e:
            logger.error(f"❌ Failed to add documents: {str(e)}")
            raise e
    
    def _create_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Create embeddings in optimized batches"""
        try:
            embed_start = time.time()
            
            # Use OpenAI embeddings
            embeddings = self.embedding_function.embed_documents_batch(texts)
            embeddings_array = np.array(embeddings)
            
            embed_time = time.time() - embed_start
            logger.info(f"🧠 Created {len(embeddings)} embeddings in {embed_time:.2f}s")
            
            return embeddings_array
            
        except Exception as e:
            logger.error(f"❌ Embedding creation failed: {str(e)}")
            raise e
    
    def _update_tfidf_matrix(self):
        """Update TF-IDF matrix for fallback similarity"""
        try:
            if self.document_texts:
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.document_texts)
                logger.info(f"📊 Updated TF-IDF matrix: {self.tfidf_matrix.shape}")
        except Exception as e:
            logger.warning(f"⚠️ TF-IDF update failed: {str(e)}")
    
    def similarity_search_with_scores(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        Perform semantic similarity search with scores
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List[Tuple[Document, float]]: Documents with similarity scores
        """
        try:
            search_start = time.time()
            
            if self.embeddings_matrix is None or len(self.documents) == 0:
                logger.warning("⚠️ No documents in index")
                return []
            
            # Create query embedding
            query_embedding = self.embedding_function.embed_query_fast(query)
            query_vector = np.array([query_embedding])
            
            # Calculate cosine similarities
            similarities = cosine_similarity(query_vector, self.embeddings_matrix)[0]
            
            # Get top k indices
            top_indices = np.argsort(similarities)[::-1][:k]
            
            # Prepare results
            results = []
            for idx in top_indices:
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    score = float(similarities[idx])
                    results.append((doc, score))
            
            search_time = time.time() - search_start
            logger.info(f"🔍 Semantic search completed in {search_time:.3f}s, found {len(results)} results")
            
            # Log top results
            for i, (doc, score) in enumerate(results[:3]):
                logger.info(f"   Result {i+1}: Score {score:.3f}, Content: {doc.page_content[:100]}...")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Semantic search failed: {str(e)}")
            return self._fallback_tfidf_search(query, k)
    
    def _fallback_tfidf_search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Fallback TF-IDF search if embeddings fail"""
        try:
            if self.tfidf_matrix is None:
                return []
            
            # Transform query
            query_vector = self.tfidf_vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
            
            # Get top k
            top_indices = np.argsort(similarities)[::-1][:k]
            
            results = []
            for idx in top_indices:
                if idx < len(self.documents) and similarities[idx] > 0:
                    doc = self.documents[idx]
                    score = float(similarities[idx])
                    results.append((doc, score))
            
            logger.info(f"🔄 Fallback TF-IDF search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"❌ Fallback search failed: {str(e)}")
            return []
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Simple similarity search returning documents only"""
        docs_with_scores = self.similarity_search_with_scores(query, k)
        return [doc for doc, score in docs_with_scores]
    
    def advanced_search(self, query: str, k: int = 8) -> List[Document]:
        """
        Advanced search with multiple strategies
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List[Document]: Best matching documents
        """
        try:
            search_start = time.time()
            
            # Strategy 1: Direct semantic search
            semantic_results = self.similarity_search_with_scores(query, k)
            
            # Strategy 2: Enhanced query for insurance terms
            enhanced_query = self._enhance_insurance_query(query)
            if enhanced_query != query:
                enhanced_results = self.similarity_search_with_scores(enhanced_query, k//2)
                semantic_results.extend(enhanced_results)
            
            # Strategy 3: TF-IDF fallback for keyword matching
            tfidf_results = self._fallback_tfidf_search(query, k//2)
            semantic_results.extend(tfidf_results)
            
            # Remove duplicates and sort by score
            seen_content = set()
            unique_results = []
            for doc, score in semantic_results:
                content_hash = hash(doc.page_content[:200])
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_results.append((doc, score))
            
            # Sort by relevance score (higher is better)
            unique_results.sort(key=lambda x: x[1], reverse=True)
            
            # Return top results
            final_docs = [doc for doc, score in unique_results[:k]]
            
            search_time = time.time() - search_start
            logger.info(f"🎯 Advanced search completed in {search_time:.3f}s, returning {len(final_docs)} unique results")
            
            return final_docs
            
        except Exception as e:
            logger.error(f"❌ Advanced search failed: {str(e)}")
            return self.similarity_search(query, k)
    
    def _enhance_insurance_query(self, query: str) -> str:
        """Enhance query with insurance-specific terms"""
        enhancements = {
            "grace period": "grace period premium payment due date thirty days",
            "waiting period": "waiting period coverage exclusion months years",
            "ncd": "no claim discount bonus premium reduction cumulative",
            "maternity": "maternity childbirth pregnancy coverage delivery",
            "cataract": "cataract surgery eye treatment procedure",
            "organ donor": "organ donor transplant medical expenses harvesting",
            "health check": "health checkup preventive examination annual",
            "hospital": "hospital definition inpatient facility beds nursing",
            "ayush": "ayush ayurveda unani siddha homeopathy treatment",
            "room rent": "room rent daily charges accommodation icu"
        }
        
        query_lower = query.lower()
        for term, enhancement in enhancements.items():
            if term in query_lower:
                return f"{query} {enhancement}"
        
        return query
    
    def _save_index(self):
        """Save index to disk"""
        try:
            index_data = {
                'documents': self.documents,
                'embeddings_matrix': self.embeddings_matrix,
                'document_texts': self.document_texts,
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'tfidf_matrix': self.tfidf_matrix
            }
            
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            with open(f"{self.index_path}.pkl", 'wb') as f:
                pickle.dump(index_data, f)
            
            logger.info(f"💾 Saved index to {self.index_path}.pkl")
        except Exception as e:
            logger.warning(f"⚠️ Failed to save index: {str(e)}")
    
    def load_index(self) -> bool:
        """Load index from disk"""
        try:
            if os.path.exists(f"{self.index_path}.pkl"):
                with open(f"{self.index_path}.pkl", 'rb') as f:
                    index_data = pickle.load(f)
                
                self.documents = index_data.get('documents', [])
                self.embeddings_matrix = index_data.get('embeddings_matrix')
                self.document_texts = index_data.get('document_texts', [])
                self.tfidf_vectorizer = index_data.get('tfidf_vectorizer', TfidfVectorizer())
                self.tfidf_matrix = index_data.get('tfidf_matrix')
                
                logger.info(f"📁 Loaded index from {self.index_path}.pkl")
                logger.info(f"📊 Loaded {len(self.documents)} documents")
                return True
            else:
                logger.info("📁 No existing index found")
                return False
        except Exception as e:
            logger.warning(f"⚠️ Failed to load index: {str(e)}")
            return False
    
    def get_stats(self) -> Dict:
        """Get vector store statistics"""
        try:
            stats = {
                "index_path": self.index_path,
                "total_documents": len(self.documents),
                "embeddings_shape": self.embeddings_matrix.shape if self.embeddings_matrix is not None else "None",
                "tfidf_shape": self.tfidf_matrix.shape if self.tfidf_matrix is not None else "None",
                "embedding_model": getattr(self.embedding_function, 'model', 'text-embedding-3-small'),
                "search_methods": ["semantic_similarity", "tfidf_fallback", "advanced_multi_strategy"],
                "performance": "optimized_for_accuracy_and_speed",
                "status": "ready" if len(self.documents) > 0 else "empty"
            }
            return stats
        except Exception as e:
            logger.error(f"❌ Failed to get stats: {str(e)}")
            return {"error": str(e)}
    
    def clear_index(self):
        """Clear the index"""
        try:
            self.documents = []
            self.embeddings_matrix = None
            self.document_texts = []
            self.tfidf_matrix = None
            
            # Remove saved file
            if os.path.exists(f"{self.index_path}.pkl"):
                os.remove(f"{self.index_path}.pkl")
            
            logger.info("🗑️ Cleared index")
        except Exception as e:
            logger.warning(f"⚠️ Error clearing index: {str(e)}")
    
    def health_check(self) -> bool:
        """Health check"""
        try:
            if len(self.documents) == 0:
                return True  # Empty is OK
            
            # Test search
            test_results = self.similarity_search("test", k=1)
            return True  # If no exception, it's healthy
            
        except Exception as e:
            logger.warning(f"⚠️ Health check failed: {str(e)}")
            return False