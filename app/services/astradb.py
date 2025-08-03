# app/services/astradb.py - Optimized AstraDB Vector Store Service
import os
import logging
from typing import List, Optional, Dict, Any
from langchain_astradb import AstraDBVectorStore
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class AstraDBService:
    """Optimized AstraDB service for high-performance vector storage and retrieval"""
    
    def __init__(self, 
                 embedding_function,
                 astra_db_application_token: str = None,
                 astra_db_id: str = None,
                 astra_db_region: str = None,
                 astra_db_keyspace: str = None,
                 collection_name: str = "insurance_documents"):
        """
        Initialize AstraDB service with optimized settings
        
        Args:
            embedding_function: OpenAI embeddings function
            astra_db_application_token: AstraDB application token
            astra_db_id: AstraDB database ID
            astra_db_region: AstraDB region
            astra_db_keyspace: AstraDB keyspace
            collection_name: Collection name for documents
        """
        # AstraDB configuration
        self.astra_db_application_token = astra_db_application_token or os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        self.astra_db_id = astra_db_id or os.getenv("ASTRA_DB_ID")
        self.astra_db_region = astra_db_region or os.getenv("ASTRA_DB_REGION")
        self.astra_db_keyspace = astra_db_keyspace or os.getenv("ASTRA_DB_KEYSPACE")
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        
        # Validate required parameters
        if not all([self.astra_db_application_token, self.astra_db_id, self.astra_db_keyspace]):
            raise ValueError("AstraDB credentials are required. Please set environment variables or pass parameters.")
        
        # Initialize optimized text splitter for speed
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Smaller chunks for faster processing
            chunk_overlap=100,  # Reduced overlap for speed
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        # Initialize vector store
        self.vector_store = None
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize AstraDB vector store with performance optimizations"""
        try:
            # Construct API endpoint
            api_endpoint = f"https://{self.astra_db_id}-{self.astra_db_region}.apps.astra.datastax.com"
            
            self.vector_store = AstraDBVectorStore(
                embedding=self.embedding_function,
                collection_name=self.collection_name,
                token=self.astra_db_application_token,
                api_endpoint=api_endpoint,
                namespace=self.astra_db_keyspace,
                # Performance optimizations
                metric="cosine",  # Optimized for semantic similarity
                batch_size=100    # Batch operations for speed
            )
            logger.info("✅ AstraDB vector store initialized with performance optimizations")
        except Exception as e:
            logger.error(f"❌ Failed to initialize AstraDB vector store: {str(e)}")
            raise e
    
    def add_documents_optimized(self, documents: List[Document]) -> List[str]:
        """
        Add documents to AstraDB with optimized chunking and batching
        
        Args:
            documents: List of documents to add
            
        Returns:
            List[str]: List of document IDs
        """
        try:
            if not documents:
                logger.warning("⚠️ No documents provided to add")
                return []
            
            # Optimized chunking
            all_chunks = []
            for doc in documents:
                chunks = self.text_splitter.split_documents([doc])
                # Add metadata for better retrieval
                for chunk in chunks:
                    chunk.metadata.update({
                        "chunk_size": len(chunk.page_content),
                        "source_doc": doc.metadata.get("source", "unknown")
                    })
                all_chunks.extend(chunks)
            
            logger.info(f"📄 Split {len(documents)} documents into {len(all_chunks)} optimized chunks")
            
            # Batch add to vector store for performance
            doc_ids = self.vector_store.add_documents(all_chunks)
            logger.info(f"✅ Added {len(doc_ids)} document chunks to AstraDB in batches")
            
            return doc_ids
        except Exception as e:
            logger.error(f"❌ Failed to add documents to AstraDB: {str(e)}")
            raise e
    
    def add_text_chunks_fast(self, texts: List[str], metadatas: List[Dict] = None) -> List[str]:
        """
        Add text chunks directly with fast batch processing
        
        Args:
            texts: List of text chunks
            metadatas: Optional list of metadata for each chunk
            
        Returns:
            List[str]: List of document IDs
        """
        try:
            if not texts:
                logger.warning("⚠️ No texts provided to add")
                return []
            
            # Prepare metadata with performance tags
            if not metadatas:
                metadatas = [{"type": "insurance_policy", "indexed_at": str(time.time())} for _ in texts]
            
            # Batch add for performance
            doc_ids = self.vector_store.add_texts(
                texts=texts,
                metadatas=metadatas
            )
            logger.info(f"✅ Added {len(doc_ids)} text chunks to AstraDB via fast batch processing")
            
            return doc_ids
        except Exception as e:
            logger.error(f"❌ Failed to add text chunks to AstraDB: {str(e)}")
            raise e
    
    def similarity_search_optimized(self, query: str, k: int = 5, filter: Dict = None) -> List[Document]:
        """
        Perform optimized similarity search with performance tracking
        
        Args:
            query: Search query
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List[Document]: List of similar documents
        """
        try:
            import time
            search_start = time.time()
            
            docs = self.vector_store.similarity_search(
                query=query,
                k=k,
                filter=filter
            )
            
            search_time = time.time() - search_start
            logger.info(f"🔍 Found {len(docs)} similar documents in {search_time:.3f}s for query: {query[:50]}...")
            return docs
        except Exception as e:
            logger.error(f"❌ Failed to perform similarity search: {str(e)}")
            return []
    
    def similarity_search_with_score_fast(self, query: str, k: int = 5, filter: Dict = None) -> List[tuple]:
        """
        Perform fast similarity search with relevance scores
        
        Args:
            query: Search query
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List[tuple]: List of (document, score) tuples
        """
        try:
            import time
            search_start = time.time()
            
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter
            )
            
            search_time = time.time() - search_start
            logger.info(f"🔍 Found {len(docs_with_scores)} documents with scores in {search_time:.3f}s")
            return docs_with_scores
        except Exception as e:
            logger.error(f"❌ Failed to perform similarity search with scores: {str(e)}")
            return []
    
    def get_retriever_optimized(self, search_type: str = "similarity", search_kwargs: Dict = None):
        """
        Get an optimized retriever for the vector store
        
        Args:
            search_type: Type of search ("similarity", "mmr", etc.)
            search_kwargs: Additional search parameters
            
        Returns:
            VectorStoreRetriever: Optimized retriever object
        """
        try:
            # Optimized search parameters for speed and accuracy balance
            search_kwargs = search_kwargs or {
                "k": 5,  # Retrieve top 5 most relevant chunks
                "score_threshold": 0.7  # Filter by relevance score
            }
            
            retriever = self.vector_store.as_retriever(
                search_type=search_type,
                search_kwargs=search_kwargs
            )
            logger.info(f"🔄 Created optimized retriever with search_type: {search_type}")
            return retriever
        except Exception as e:
            logger.error(f"❌ Failed to create retriever: {str(e)}")
            raise e
    
    def process_pdf_content_fast(self, content: str, metadata: Dict = None) -> List[str]:
        """
        Fast process PDF content and add to vector store
        
        Args:
            content: PDF text content
            metadata: Optional metadata for the document
            
        Returns:
            List[str]: List of document IDs
        """
        try:
            import time
            process_start = time.time()
            
            # Create document with enhanced metadata
            enhanced_metadata = {
                "source": "pdf",
                "type": "insurance_policy",
                "processed_at": str(time.time()),
                "content_length": len(content)
            }
            if metadata:
                enhanced_metadata.update(metadata)
            
            doc = Document(
                page_content=content,
                metadata=enhanced_metadata
            )
            
            # Fast chunking and addition
            doc_ids = self.add_documents_optimized([doc])
            
            process_time = time.time() - process_start
            logger.info(f"📋 Processed PDF content in {process_time:.2f}s and created {len(doc_ids)} chunks")
            return doc_ids
        except Exception as e:
            logger.error(f"❌ Failed to process PDF content: {str(e)}")
            raise e
    
    def delete_documents_fast(self, ids: List[str]) -> bool:
        """
        Fast delete documents by IDs
        
        Args:
            ids: List of document IDs to delete
            
        Returns:
            bool: True if successful
        """
        try:
            self.vector_store.delete(ids)
            logger.info(f"🗑️ Deleted {len(ids)} documents from AstraDB")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to delete documents: {str(e)}")
            return False
    
    def get_collection_info(self) -> Dict:
        """
        Get optimized collection information
        
        Returns:
            Dict: Collection information with performance stats
        """
        try:
            stats = {
                "collection_name": self.collection_name,
                "database_id": self.astra_db_id,
                "region": self.astra_db_region,
                "keyspace": self.astra_db_keyspace,
                "status": "active",
                "optimizations": {
                    "chunking": "optimized for speed",
                    "batch_processing": "enabled",
                    "similarity_metric": "cosine",
                    "retrieval_threshold": 0.7
                },
                "embedding_info": {
                    "model": getattr(self.embedding_function, 'model', 'text-embedding-3-small'),
                    "dimension": getattr(self.embedding_function, 'dimension', 'auto')
                }
            }
            logger.info("📊 Retrieved optimized collection info")
            return stats
        except Exception as e:
            logger.error(f"❌ Failed to get collection info: {str(e)}")
            return {"error": str(e)}
    
    def health_check(self) -> bool:
        """
        Quick health check for AstraDB connection
        
        Returns:
            bool: True if healthy
        """
        try:
            # Simple test - try to perform a minimal search
            test_docs = self.vector_store.similarity_search("test", k=1)
            logger.info("✅ AstraDB health check passed")
            return True
        except Exception as e:
            logger.warning(f"⚠️ AstraDB health check failed: {str(e)}")
            return False