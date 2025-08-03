# app/services/chromadb_service.py - Fixed ChromaDB Vector Store
import os
import logging
import time
import uuid
from typing import List, Optional, Dict, Any, Tuple
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class OpenAIEmbeddingFunction:
    """ChromaDB-compatible OpenAI embedding function"""
    
    def __init__(self, embedding_service):
        self.embedding_service = embedding_service
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """ChromaDB-compatible interface"""
        try:
            if len(input) == 1:
                embedding = self.embedding_service.embed_query_fast(input[0])
                return [embedding]
            else:
                return self.embedding_service.embed_documents_batch(input)
        except Exception as e:
            logger.error(f"❌ Embedding function failed: {str(e)}")
            raise e

class ChromaDBService:
    """Ultra-fast ChromaDB vector store with OpenAI embeddings"""
    
    def __init__(self, embedding_function, collection_name: str = "insurance_docs", persist_directory: str = "./chroma_db"):
        """
        Initialize ChromaDB service for maximum speed
        
        Args:
            embedding_function: OpenAI embeddings function
            collection_name: Name of the collection
            persist_directory: Directory to persist ChromaDB
        """
        self.embedding_service = embedding_function
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Create ChromaDB-compatible embedding function
        self.chromadb_embedding_fn = OpenAIEmbeddingFunction(embedding_function)
        
        # Optimized text splitter for speed
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Smaller chunks for faster processing
            chunk_overlap=100,  # Reduced overlap for speed
            length_function=len,
            separators=["\n\n", "\n", ". ", ".", " ", ""]
        )
        
        # Initialize ChromaDB with speed optimizations
        self._initialize_chromadb()
        
        logger.info("✅ ChromaDB Service initialized for ULTRA-FAST processing")
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB with performance optimizations"""
        try:
            init_start = time.time()
            
            # Create ChromaDB client with optimizations
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,  # Disable telemetry for speed
                    allow_reset=True
                )
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name
                )
                logger.info(f"📁 Loaded existing ChromaDB collection: {self.collection_name}")
            except Exception:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=self.chromadb_embedding_fn,
                    metadata={"hnsw:space": "cosine"}  # Optimized for semantic similarity
                )
                logger.info(f"🆕 Created new ChromaDB collection: {self.collection_name}")
            
            init_time = time.time() - init_start
            logger.info(f"✅ ChromaDB initialized in {init_time:.3f}s")
            
        except Exception as e:
            logger.error(f"❌ ChromaDB initialization failed: {str(e)}")
            raise e
    
    def add_documents_ultra_fast(self, documents: List[Document]) -> List[str]:
        """
        Add documents with ultra-fast processing
        
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
            
            # Fast chunking
            all_chunks = []
            for doc in documents:
                chunks = self.text_splitter.split_documents([doc])
                all_chunks.extend(chunks)
            
            logger.info(f"📄 Split {len(documents)} documents into {len(all_chunks)} chunks")
            
            # Prepare data for ChromaDB
            chunk_ids = [str(uuid.uuid4()) for _ in all_chunks]
            chunk_texts = [chunk.page_content for chunk in all_chunks]
            chunk_metadatas = []
            
            for i, chunk in enumerate(all_chunks):
                metadata = chunk.metadata.copy()
                metadata.update({
                    "chunk_id": chunk_ids[i],
                    "chunk_index": i,
                    "doc_type": "insurance_policy",
                    "chunk_size": len(chunk.page_content)
                })
                # ChromaDB requires string values in metadata
                metadata = {k: str(v) for k, v in metadata.items()}
                chunk_metadatas.append(metadata)
            
            # Add to ChromaDB in batch (ultra-fast)
            self.collection.add(
                ids=chunk_ids,
                documents=chunk_texts,
                metadatas=chunk_metadatas
            )
            
            add_time = time.time() - add_start
            logger.info(f"⚡ ULTRA-FAST: Added {len(all_chunks)} chunks to ChromaDB in {add_time:.3f}s")
            
            return chunk_ids
            
        except Exception as e:
            logger.error(f"❌ Failed to add documents to ChromaDB: {str(e)}")
            raise e
    
    def similarity_search_ultra_fast(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        Ultra-fast semantic similarity search
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List[Tuple[Document, float]]: Documents with similarity scores
        """
        try:
            search_start = time.time()
            
            # ChromaDB ultra-fast query
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Convert results to Document objects
            docs_with_scores = []
            
            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0] if results['metadatas'] else [{}] * len(documents)
                distances = results['distances'][0] if results['distances'] else [0.0] * len(documents)
                
                for doc_text, metadata, distance in zip(documents, metadatas, distances):
                    # Convert distance to similarity score (ChromaDB returns distances)
                    similarity_score = 1.0 - distance
                    
                    doc = Document(
                        page_content=doc_text,
                        metadata=metadata or {}
                    )
                    
                    docs_with_scores.append((doc, similarity_score))
            
            search_time = time.time() - search_start
            logger.info(f"⚡ ULTRA-FAST ChromaDB search: {search_time:.3f}s, found {len(docs_with_scores)} results")
            
            # Log top results for debugging
            for i, (doc, score) in enumerate(docs_with_scores[:3]):
                logger.info(f"   Result {i+1}: Score {score:.3f}, Content: {doc.page_content[:100]}...")
            
            return docs_with_scores
            
        except Exception as e:
            logger.error(f"❌ ChromaDB search failed: {str(e)}")
            return []
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Simple similarity search returning documents only"""
        docs_with_scores = self.similarity_search_ultra_fast(query, k)
        return [doc for doc, score in docs_with_scores]
    
    def advanced_search_ultra_fast(self, query: str, k: int = 8) -> List[Document]:
        """
        Advanced search with multiple strategies for maximum accuracy and speed
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List[Document]: Best matching documents
        """
        try:
            search_start = time.time()
            
            # Strategy 1: Direct semantic search
            semantic_results = self.similarity_search_ultra_fast(query, k)
            
            # Strategy 2: Enhanced query for insurance terms (only if time permits)
            enhanced_query = self._enhance_insurance_query(query)
            if enhanced_query != query:
                enhanced_results = self.similarity_search_ultra_fast(enhanced_query, k//2)
                semantic_results.extend(enhanced_results)
            
            # Remove duplicates by content hash
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
            logger.info(f"⚡ ULTRA-FAST advanced search: {search_time:.3f}s, returning {len(final_docs)} results")
            
            return final_docs
            
        except Exception as e:
            logger.error(f"❌ Advanced search failed: {str(e)}")
            return self.similarity_search(query, k)
    
    def _enhance_insurance_query(self, query: str) -> str:
        """Quick enhancement of insurance queries"""
        # Simplified enhancement for speed
        enhancements = {
            "grace period": "grace period premium payment thirty days",
            "waiting period": "waiting period coverage exclusion 36 months",
            "ncd": "no claim discount bonus premium",
            "maternity": "maternity childbirth pregnancy coverage",
            "cataract": "cataract surgery eye treatment",
            "organ donor": "organ donor transplant medical expenses",
            "health check": "health checkup preventive examination",
            "hospital": "hospital definition inpatient facility",
            "ayush": "ayush ayurveda unani siddha homeopathy",
            "room rent": "room rent daily charges icu"
        }
        
        query_lower = query.lower()
        for term, enhancement in enhancements.items():
            if term in query_lower:
                return f"{query} {enhancement}"
        
        return query
    
    def get_collection_stats(self) -> Dict:
        """Get ChromaDB collection statistics"""
        try:
            count = self.collection.count()
            
            stats = {
                "collection_name": self.collection_name,
                "total_documents": count,
                "persist_directory": self.persist_directory,
                "vector_store": "ChromaDB",
                "embedding_model": getattr(self.embedding_service, 'model', 'text-embedding-3-small'),
                "search_type": "ultra_fast_semantic_similarity",
                "performance": "optimized_for_speed",
                "status": "ready" if count > 0 else "empty"
            }
            
            return stats
        except Exception as e:
            logger.error(f"❌ Failed to get stats: {str(e)}")
            return {"error": str(e)}
    
    def clear_collection(self):
        """Clear the ChromaDB collection"""
        try:
            # Delete and recreate collection for clean slate
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.chromadb_embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("🗑️ Cleared ChromaDB collection")
        except Exception as e:
            logger.warning(f"⚠️ Error clearing collection: {str(e)}")
    
    def health_check(self) -> bool:
        """Ultra-fast health check"""
        try:
            # Quick count check
            count = self.collection.count()
            logger.info(f"✅ ChromaDB health check passed: {count} documents")
            return True
        except Exception as e:
            logger.warning(f"⚠️ ChromaDB health check failed: {str(e)}")
            return False
    
    def add_documents_ultra_fast(self, documents: List[Document]) -> List[str]:
        """
        Add documents with ultra-fast processing
        
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
            
            # Fast chunking
            all_chunks = []
            for doc in documents:
                chunks = self.text_splitter.split_documents([doc])
                all_chunks.extend(chunks)
            
            logger.info(f"📄 Split {len(documents)} documents into {len(all_chunks)} chunks")
            
            # Prepare data for ChromaDB
            chunk_ids = [str(uuid.uuid4()) for _ in all_chunks]
            chunk_texts = [chunk.page_content for chunk in all_chunks]
            chunk_metadatas = []
            
            for i, chunk in enumerate(all_chunks):
                metadata = chunk.metadata.copy()
                metadata.update({
                    "chunk_id": chunk_ids[i],
                    "chunk_index": i,
                    "doc_type": "insurance_policy",
                    "chunk_size": len(chunk.page_content)
                })
                chunk_metadatas.append(metadata)
            
            # Add to ChromaDB in batch (ultra-fast)
            self.collection.add(
                ids=chunk_ids,
                documents=chunk_texts,
                metadatas=chunk_metadatas
            )
            
            add_time = time.time() - add_start
            logger.info(f"⚡ ULTRA-FAST: Added {len(all_chunks)} chunks to ChromaDB in {add_time:.3f}s")
            
            return chunk_ids
            
        except Exception as e:
            logger.error(f"❌ Failed to add documents to ChromaDB: {str(e)}")
            raise e
    
    def similarity_search_ultra_fast(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        Ultra-fast semantic similarity search
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List[Tuple[Document, float]]: Documents with similarity scores
        """
        try:
            search_start = time.time()
            
            # ChromaDB ultra-fast query
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Convert results to Document objects
            docs_with_scores = []
            
            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0] if results['metadatas'] else [{}] * len(documents)
                distances = results['distances'][0] if results['distances'] else [0.0] * len(documents)
                
                for doc_text, metadata, distance in zip(documents, metadatas, distances):
                    # Convert distance to similarity score (ChromaDB returns distances)
                    similarity_score = 1.0 - distance
                    
                    doc = Document(
                        page_content=doc_text,
                        metadata=metadata or {}
                    )
                    
                    docs_with_scores.append((doc, similarity_score))
            
            search_time = time.time() - search_start
            logger.info(f"⚡ ULTRA-FAST ChromaDB search: {search_time:.3f}s, found {len(docs_with_scores)} results")
            
            # Log top results for debugging
            for i, (doc, score) in enumerate(docs_with_scores[:3]):
                logger.info(f"   Result {i+1}: Score {score:.3f}, Content: {doc.page_content[:100]}...")
            
            return docs_with_scores
            
        except Exception as e:
            logger.error(f"❌ ChromaDB search failed: {str(e)}")
            return []
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Simple similarity search returning documents only"""
        docs_with_scores = self.similarity_search_ultra_fast(query, k)
        return [doc for doc, score in docs_with_scores]
    
    def advanced_search_ultra_fast(self, query: str, k: int = 8) -> List[Document]:
        """
        Advanced search with multiple strategies for maximum accuracy and speed
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List[Document]: Best matching documents
        """
        try:
            search_start = time.time()
            
            # Strategy 1: Direct semantic search
            semantic_results = self.similarity_search_ultra_fast(query, k)
            
            # Strategy 2: Enhanced query for insurance terms (only if time permits)
            enhanced_query = self._enhance_insurance_query(query)
            if enhanced_query != query:
                enhanced_results = self.similarity_search_ultra_fast(enhanced_query, k//2)
                semantic_results.extend(enhanced_results)
            
            # Remove duplicates by content hash
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
            logger.info(f"⚡ ULTRA-FAST advanced search: {search_time:.3f}s, returning {len(final_docs)} results")
            
            return final_docs
            
        except Exception as e:
            logger.error(f"❌ Advanced search failed: {str(e)}")
            return self.similarity_search(query, k)
    
    def _enhance_insurance_query(self, query: str) -> str:
        """Quick enhancement of insurance queries"""
        # Simplified enhancement for speed
        enhancements = {
            "grace period": "grace period premium payment thirty days",
            "waiting period": "waiting period coverage exclusion 36 months",
            "ncd": "no claim discount bonus premium",
            "maternity": "maternity childbirth pregnancy coverage",
            "cataract": "cataract surgery eye treatment",
            "organ donor": "organ donor transplant medical expenses",
            "health check": "health checkup preventive examination",
            "hospital": "hospital definition inpatient facility",
            "ayush": "ayush ayurveda unani siddha homeopathy",
            "room rent": "room rent daily charges icu"
        }
        
        query_lower = query.lower()
        for term, enhancement in enhancements.items():
            if term in query_lower:
                return f"{query} {enhancement}"
        
        return query
    
    def get_collection_stats(self) -> Dict:
        """Get ChromaDB collection statistics"""
        try:
            count = self.collection.count()
            
            stats = {
                "collection_name": self.collection_name,
                "total_documents": count,
                "persist_directory": self.persist_directory,
                "vector_store": "ChromaDB",
                "embedding_model": getattr(self.embedding_function, 'model', 'text-embedding-3-small'),
                "search_type": "ultra_fast_semantic_similarity",
                "performance": "optimized_for_speed",
                "status": "ready" if count > 0 else "empty"
            }
            
            return stats
        except Exception as e:
            logger.error(f"❌ Failed to get stats: {str(e)}")
            return {"error": str(e)}
    
    def clear_collection(self):
        """Clear the ChromaDB collection"""
        try:
            # Delete and recreate collection for clean slate
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self._chromadb_embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("🗑️ Cleared ChromaDB collection")
        except Exception as e:
            logger.warning(f"⚠️ Error clearing collection: {str(e)}")
    
    def health_check(self) -> bool:
        """Ultra-fast health check"""
        try:
            # Quick count check
            count = self.collection.count()
            logger.info(f"✅ ChromaDB health check passed: {count} documents")
            return True
        except Exception as e:
            logger.warning(f"⚠️ ChromaDB health check failed: {str(e)}")
            return False
    
    def get_retriever_interface(self, search_kwargs: Dict = None):
        """Get a retriever-like interface for compatibility"""
        search_kwargs = search_kwargs or {"k": 5}
        
        class ChromaDBRetriever:
            def __init__(self, chromadb_service, k=5):
                self.chromadb_service = chromadb_service
                self.k = k
            
            def get_relevant_documents(self, query: str) -> List[Document]:
                return self.chromadb_service.similarity_search(query, self.k)
        
        return ChromaDBRetriever(self, search_kwargs.get("k", 5))