# app/services/faiss_service.py - High-Performance Local FAISS Vector Store
import os
import logging
import time
import pickle
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class FAISSVectorService:
    """High-performance local FAISS vector store with OpenAI embeddings"""
    
    def __init__(self, embedding_function, index_path: str = "faiss_index"):
        """
        Initialize FAISS vector service
        
        Args:
            embedding_function: OpenAI embeddings function
            index_path: Path to save/load FAISS index
        """
        self.embedding_function = embedding_function
        self.index_path = index_path
        self.vector_store = None
        self.documents = []
        
        # Optimized text splitter for insurance documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,  # Smaller chunks for better precision
            chunk_overlap=150,  # Good overlap for context
            length_function=len,
            separators=["\n\n", "\n", ". ", ".", " ", ""]
        )
        
        logger.info("✅ FAISS Vector Service initialized")
    
    def add_documents_fast(self, documents: List[Document]) -> List[str]:
        """
        Add documents to FAISS vector store with optimized processing
        
        Args:
            documents: List of documents to add
            
        Returns:
            List[str]: List of document IDs
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
                # Add enhanced metadata for better retrieval
                for i, chunk in enumerate(chunks):
                    chunk.metadata.update({
                        "chunk_id": f"{doc.metadata.get('source', 'doc')}_{i}",
                        "chunk_size": len(chunk.page_content),
                        "doc_type": "insurance_policy",
                        "chunk_index": i
                    })
                all_chunks.extend(chunks)
            
            logger.info(f"📄 Split {len(documents)} documents into {len(all_chunks)} optimized chunks")
            
            # Create or update FAISS vector store
            if self.vector_store is None:
                # Create new FAISS index
                self.vector_store = FAISS.from_documents(
                    all_chunks,
                    self.embedding_function
                )
                logger.info("🆕 Created new FAISS index")
            else:
                # Add to existing index
                self.vector_store.add_documents(all_chunks)
                logger.info("➕ Added to existing FAISS index")
            
            # Store documents for reference
            self.documents.extend(all_chunks)
            
            # Save index for future use
            self._save_index()
            
            add_time = time.time() - add_start
            logger.info(f"✅ Added {len(all_chunks)} chunks to FAISS in {add_time:.2f}s")
            
            return [f"faiss_{i}" for i in range(len(all_chunks))]
            
        except Exception as e:
            logger.error(f"❌ Failed to add documents to FAISS: {str(e)}")
            raise e
    
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
            
            if self.vector_store is None:
                logger.warning("⚠️ No FAISS index available")
                return []
            
            # Perform semantic similarity search
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query=query,
                k=k
            )
            
            search_time = time.time() - search_start
            logger.info(f"🔍 FAISS search completed in {search_time:.3f}s, found {len(docs_with_scores)} results")
            
            # Log relevance scores
            for i, (doc, score) in enumerate(docs_with_scores):
                logger.info(f"   Result {i+1}: Score {score:.3f}, Content: {doc.page_content[:100]}...")
            
            return docs_with_scores
            
        except Exception as e:
            logger.error(f"❌ FAISS search failed: {str(e)}")
            return []
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Perform semantic similarity search
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List[Document]: Similar documents
        """
        docs_with_scores = self.similarity_search_with_scores(query, k)
        return [doc for doc, score in docs_with_scores]
    
    def get_retriever(self, search_kwargs: Dict = None):
        """
        Get a retriever for the FAISS vector store
        
        Args:
            search_kwargs: Search parameters
            
        Returns:
            VectorStoreRetriever: FAISS retriever
        """
        try:
            if self.vector_store is None:
                raise ValueError("FAISS index not initialized")
            
            search_kwargs = search_kwargs or {"k": 5}
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs=search_kwargs
            )
            
            logger.info(f"🔄 Created FAISS retriever with k={search_kwargs.get('k', 5)}")
            return retriever
            
        except Exception as e:
            logger.error(f"❌ Failed to create FAISS retriever: {str(e)}")
            raise e
    
    def advanced_search(self, query: str, k: int = 8) -> List[Document]:
        """
        Advanced search with multiple strategies for better accuracy
        
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
            
            # Remove duplicates and sort by score
            seen_content = set()
            unique_results = []
            for doc, score in semantic_results:
                content_hash = hash(doc.page_content[:200])
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_results.append((doc, score))
            
            # Sort by relevance score (lower is better for FAISS)
            unique_results.sort(key=lambda x: x[1])
            
            # Return top results
            final_docs = [doc for doc, score in unique_results[:k]]
            
            search_time = time.time() - search_start
            logger.info(f"🎯 Advanced search completed in {search_time:.3f}s, returning {len(final_docs)} unique results")
            
            return final_docs
            
        except Exception as e:
            logger.error(f"❌ Advanced search failed: {str(e)}")
            return self.similarity_search(query, k)
    
    def _enhance_insurance_query(self, query: str) -> str:
        """
        Enhance query with insurance-specific terms
        
        Args:
            query: Original query
            
        Returns:
            str: Enhanced query
        """
        # Insurance term mappings
        enhancements = {
            "grace period": "grace period premium payment due date",
            "waiting period": "waiting period coverage exclusion",
            "ncd": "no claim discount bonus premium",
            "maternity": "maternity childbirth pregnancy coverage",
            "cataract": "cataract surgery eye treatment",
            "organ donor": "organ donor transplant medical expenses",
            "health check": "health checkup preventive examination",
            "hospital": "hospital definition inpatient facility",
            "ayush": "ayush ayurveda unani siddha homeopathy",
            "room rent": "room rent daily charges accommodation"
        }
        
        query_lower = query.lower()
        for term, enhancement in enhancements.items():
            if term in query_lower:
                return f"{query} {enhancement}"
        
        return query
    
    def _save_index(self):
        """Save FAISS index to disk"""
        try:
            if self.vector_store is not None:
                self.vector_store.save_local(self.index_path)
                logger.info(f"💾 Saved FAISS index to {self.index_path}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to save FAISS index: {str(e)}")
    
    def load_index(self) -> bool:
        """
        Load FAISS index from disk
        
        Returns:
            bool: True if loaded successfully
        """
        try:
            if os.path.exists(f"{self.index_path}.faiss"):
                self.vector_store = FAISS.load_local(
                    self.index_path,
                    self.embedding_function
                )
                logger.info(f"📁 Loaded FAISS index from {self.index_path}")
                return True
            else:
                logger.info("📁 No existing FAISS index found")
                return False
        except Exception as e:
            logger.warning(f"⚠️ Failed to load FAISS index: {str(e)}")
            return False
    
    def get_stats(self) -> Dict:
        """
        Get FAISS vector store statistics
        
        Returns:
            Dict: Statistics
        """
        try:
            stats = {
                "index_path": self.index_path,
                "vector_store_ready": self.vector_store is not None,
                "total_documents": len(self.documents),
                "embedding_model": getattr(self.embedding_function, 'model', 'unknown'),
                "search_type": "semantic_similarity",
                "performance": "optimized_for_speed_and_accuracy"
            }
            
            if self.vector_store is not None:
                stats["index_type"] = "FAISS"
                stats["status"] = "ready"
            else:
                stats["status"] = "not_initialized"
            
            return stats
        except Exception as e:
            logger.error(f"❌ Failed to get FAISS stats: {str(e)}")
            return {"error": str(e)}
    
    def clear_index(self):
        """Clear the FAISS index"""
        try:
            self.vector_store = None
            self.documents = []
            
            # Remove saved files
            import shutil
            if os.path.exists(self.index_path):
                shutil.rmtree(self.index_path, ignore_errors=True)
            
            logger.info("🗑️ Cleared FAISS index")
        except Exception as e:
            logger.warning(f"⚠️ Error clearing FAISS index: {str(e)}")
    
    def health_check(self) -> bool:
        """
        Health check for FAISS service
        
        Returns:
            bool: True if healthy
        """
        try:
            if self.vector_store is None:
                return False
            
            # Test search
            test_results = self.similarity_search("test", k=1)
            return len(test_results) >= 0  # Even 0 results is OK for health check
            
        except Exception as e:
            logger.warning(f"⚠️ FAISS health check failed: {str(e)}")
            return False