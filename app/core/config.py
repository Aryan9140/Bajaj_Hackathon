"""
HackRx 6.0 - Fixed Configuration Management
Compatible with your existing .env variables
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Optional
import os
from pathlib import Path

class Settings(BaseSettings):
    """Complete system configuration - Compatible with existing .env"""
    
    # ===== CORE API SETTINGS =====
    API_KEY: str = Field(default="6d2683f80eca9847d20948e1e5508885d08fdc65d943182f85de250687859193")
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8000)
    DEBUG: bool = Field(default=False)
    LOG_LEVEL: str = Field(default="INFO")
    
    # ===== LLM API KEYS =====
    OPENAI_API_KEY: Optional[str] = Field(default=None)
    CLAUDE_API_KEY: Optional[str] = Field(default=None)
    GROQ_API_KEY: Optional[str] = Field(default=None)
    
    # ===== ASTRADB CONFIGURATION (Your existing format) =====
    # Your current variables
    ASTRA_DB_ID: Optional[str] = Field(default=None)
    ASTRA_DB_REGION: Optional[str] = Field(default=None)
    ASTRA_DB_TOKEN: Optional[str] = Field(default=None)
    ASTRA_DB_DATABASE_ID: Optional[str] = Field(default=None)
    ASTRA_DB_COLLECTION_NAME: Optional[str] = Field(default="insurance_documents")
    
    # Standard format (constructed from your variables)
    ASTRA_DB_API_ENDPOINT: Optional[str] = Field(default=None)
    ASTRA_DB_APPLICATION_TOKEN: Optional[str] = Field(default=None)
    ASTRA_DB_KEYSPACE: str = Field(default="default_keyspace")
    ASTRA_DB_COLLECTION: str = Field(default="document_embeddings")
    
    # ===== OTHER SERVICES =====
    FOURSQUARE_CLIENT_ID: Optional[str] = Field(default=None)  # Your existing variable
    
    # ===== PERFORMANCE SETTINGS =====
    MAX_CONCURRENT_REQUESTS: int = Field(default=10)
    REQUEST_TIMEOUT: int = Field(default=30)
    LLM_TIMEOUT: int = Field(default=15)
    VECTOR_SEARCH_TIMEOUT: int = Field(default=10)
    
    # ===== DOCUMENT PROCESSING =====
    MAX_DOCUMENT_SIZE: str = Field(default="50MB")
    MAX_CHUNK_SIZE: int = Field(default=1000)
    CHUNK_OVERLAP: int = Field(default=100)
    MAX_CHUNKS_PER_DOCUMENT: int = Field(default=500)
    
    # ===== VECTOR SEARCH SETTINGS =====
    EMBEDDING_MODEL: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    EMBEDDING_DIMENSION: int = Field(default=384)
    VECTOR_SEARCH_TOP_K: int = Field(default=10)
    MIN_SIMILARITY_THRESHOLD: float = Field(default=0.3)
    
    # FAISS Configuration
    FAISS_INDEX_TYPE: str = Field(default="IndexFlatIP")
    FAISS_CACHE_SIZE: int = Field(default=1000000)
    
    # ===== LLM CONFIGURATION =====
    PRIMARY_LLM: str = Field(default="openai")
    SECONDARY_LLM: str = Field(default="claude") 
    TERTIARY_LLM: str = Field(default="groq")
    
    # Model names
    OPENAI_MODEL: str = Field(default="gpt-4-1106-preview")
    CLAUDE_MODEL: str = Field(default="claude-3-5-sonnet-20241022")
    GROQ_MODEL: str = Field(default="llama-3.1-70b-versatile")
    
    # Generation parameters
    MAX_TOKENS: int = Field(default=1000)
    TEMPERATURE: float = Field(default=0.1)
    TOP_P: float = Field(default=0.9)
    
    # ===== EXPLAINABLE AI =====
    ENABLE_EXPLANATIONS: bool = Field(default=True)
    MAX_EXPLANATION_TRACES: int = Field(default=100)
    TRACE_CLEANUP_INTERVAL: int = Field(default=3600)
    
    # ===== CACHING =====
    ENABLE_CACHING: bool = Field(default=True)
    CACHE_TTL: int = Field(default=3600)
    MAX_CACHE_SIZE: int = Field(default=1000)
    
    # ===== CONNECTION SETTINGS =====
    MAX_CONNECTIONS: int = Field(default=50)
    CONNECTION_TIMEOUT: int = Field(default=30)
    
    # ===== SECURITY =====
    ALLOWED_ORIGINS: str = Field(default="*")
    ALLOWED_METHODS: str = Field(default="GET,POST,OPTIONS")
    ALLOWED_HEADERS: str = Field(default="*")
    
    # Rate limiting
    RATE_LIMIT_REQUESTS: int = Field(default=100)
    RATE_LIMIT_WINDOW: int = Field(default=3600)
    
    # ===== MONITORING =====
    LOG_FORMAT: str = Field(default="json")
    LOG_FILE: str = Field(default="logs/hackrx.log")
    MAX_LOG_SIZE: str = Field(default="100MB")
    LOG_ROTATION_COUNT: int = Field(default=5)
    
    ENABLE_METRICS: bool = Field(default=True)
    METRICS_PORT: int = Field(default=9090)
    
    # ===== DEVELOPMENT =====
    RELOAD: bool = Field(default=False)
    DEBUG_MODE: bool = Field(default=False)
    VERBOSE_LOGGING: bool = Field(default=False)
    TEST_MODE: bool = Field(default=False)
    MOCK_LLMS: bool = Field(default=False)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        # Allow extra fields to prevent validation errors
        extra = "allow"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_logging_directory()
        self._construct_astra_config()
        self._validate_required_settings()
    
    def _setup_logging_directory(self):
        """Ensure logging directory exists"""
        try:
            log_path = Path(self.LOG_FILE).parent
            log_path.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass  # Skip if can't create directory
    
    def _construct_astra_config(self):
        """Construct standard AstraDB config from your variables"""
        if self.ASTRA_DB_ID and self.ASTRA_DB_REGION:
            # Construct endpoint from your variables
            self.ASTRA_DB_API_ENDPOINT = f"https://{self.ASTRA_DB_ID}-{self.ASTRA_DB_REGION}.apps.astra.datastax.com"
        
        if self.ASTRA_DB_TOKEN:
            # Use your token
            self.ASTRA_DB_APPLICATION_TOKEN = self.ASTRA_DB_TOKEN
        
        if self.ASTRA_DB_COLLECTION_NAME:
            # Use your collection name
            self.ASTRA_DB_COLLECTION = self.ASTRA_DB_COLLECTION_NAME
    
    def _validate_required_settings(self):
        """Validate that required settings are present"""
        # Only require API_KEY for basic functionality
        if not self.API_KEY and not self.TEST_MODE:
            print("⚠️ Warning: API_KEY not configured")
    
    @property
    def llm_priority_list(self) -> List[str]:
        """Get LLM priority list"""
        return [self.PRIMARY_LLM, self.SECONDARY_LLM, self.TERTIARY_LLM]
    
    @property
    def has_openai_config(self) -> bool:
        """Check if OpenAI is configured"""
        return bool(self.OPENAI_API_KEY)
    
    @property
    def has_claude_config(self) -> bool:
        """Check if Claude is configured"""
        return bool(self.CLAUDE_API_KEY)
    
    @property
    def has_groq_config(self) -> bool:
        """Check if Groq is configured"""
        return bool(self.GROQ_API_KEY)
    
    @property
    def has_astradb_config(self) -> bool:
        """Check if AstraDB is configured"""
        return bool(self.ASTRA_DB_API_ENDPOINT and self.ASTRA_DB_APPLICATION_TOKEN)
    
    @property
    def configured_llms(self) -> List[str]:
        """Get list of configured LLMs"""
        llms = []
        if self.has_openai_config:
            llms.append("openai")
        if self.has_claude_config:
            llms.append("claude")
        if self.has_groq_config:
            llms.append("groq")
        return llms
    
    def get_llm_config(self, llm_name: str) -> dict:
        """Get configuration for specific LLM"""
        configs = {
            "openai": {
                "api_key": self.OPENAI_API_KEY,
                "model": self.OPENAI_MODEL,
                "max_tokens": self.MAX_TOKENS,
                "temperature": self.TEMPERATURE,
                "timeout": self.LLM_TIMEOUT
            },
            "claude": {
                "api_key": self.CLAUDE_API_KEY,
                "model": self.CLAUDE_MODEL,
                "max_tokens": self.MAX_TOKENS,
                "temperature": self.TEMPERATURE,
                "timeout": self.LLM_TIMEOUT
            },
            "groq": {
                "api_key": self.GROQ_API_KEY,
                "model": self.GROQ_MODEL,
                "max_tokens": self.MAX_TOKENS,
                "temperature": self.TEMPERATURE,
                "timeout": self.LLM_TIMEOUT
            }
        }
        
        return configs.get(llm_name, {})
    
    def get_vector_config(self) -> dict:
        """Get vector search configuration"""
        return {
            "embedding_model": self.EMBEDDING_MODEL,
            "dimension": self.EMBEDDING_DIMENSION,
            "top_k": self.VECTOR_SEARCH_TOP_K,
            "min_similarity": self.MIN_SIMILARITY_THRESHOLD,
            "faiss_index_type": self.FAISS_INDEX_TYPE,
            "cache_size": self.FAISS_CACHE_SIZE,
            "timeout": self.VECTOR_SEARCH_TIMEOUT
        }
    
    def get_astradb_config(self) -> dict:
        """Get AstraDB configuration"""
        return {
            "endpoint": self.ASTRA_DB_API_ENDPOINT,
            "token": self.ASTRA_DB_APPLICATION_TOKEN,
            "keyspace": self.ASTRA_DB_KEYSPACE,
            "collection": self.ASTRA_DB_COLLECTION
        }
    
    def get_performance_config(self) -> dict:
        """Get performance configuration"""
        return {
            "max_concurrent_requests": self.MAX_CONCURRENT_REQUESTS,
            "request_timeout": self.REQUEST_TIMEOUT,
            "max_connections": self.MAX_CONNECTIONS,
            "connection_timeout": self.CONNECTION_TIMEOUT,
            "enable_caching": self.ENABLE_CACHING,
            "cache_ttl": self.CACHE_TTL,
            "max_cache_size": self.MAX_CACHE_SIZE
        }
    
    def get_document_config(self) -> dict:
        """Get document processing configuration"""
        return {
            "max_document_size": self.MAX_DOCUMENT_SIZE,
            "max_chunk_size": self.MAX_CHUNK_SIZE,
            "chunk_overlap": self.CHUNK_OVERLAP,
            "max_chunks_per_document": self.MAX_CHUNKS_PER_DOCUMENT
        }
    
    def get_system_info(self) -> dict:
        """Get complete system configuration info"""
        return {
            "version": "6.0.0",
            "features_enabled": {
                "6_step_workflow": True,
                "hybrid_vector_search": True,
                "clause_matching": True,
                "explainable_ai": self.ENABLE_EXPLANATIONS,
                "multi_format_documents": True,
                "multi_llm_support": True,
                "caching": self.ENABLE_CACHING,
                "metrics": self.ENABLE_METRICS
            },
            "configured_services": {
                "openai": self.has_openai_config,
                "claude": self.has_claude_config,
                "groq": self.has_groq_config,
                "astradb": self.has_astradb_config
            },
            "performance_settings": self.get_performance_config(),
            "vector_settings": self.get_vector_config(),
            "document_settings": self.get_document_config()
        }

# Create global settings instance
try:
    settings = Settings()
    print("✅ Configuration loaded successfully")
    print(f"🔧 AstraDB Endpoint: {settings.ASTRA_DB_API_ENDPOINT}")
    print(f"🔑 Configured LLMs: {settings.configured_llms}")
except Exception as e:
    print(f"❌ Configuration error: {e}")
    # Create basic settings for testing
    settings = Settings(
        API_KEY="6d2683f80eca9847d20948e1e5508885d08fdc65d943182f85de250687859193",
        TEST_MODE=True
    )