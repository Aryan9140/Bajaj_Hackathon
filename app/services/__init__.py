# app/services/__init__.py - Services Package Initialization
"""
Services package for Enhanced RAG System

This package contains:
- atlas.py: MongoDB Atlas service for chat history
- astradb.py: AstraDB vector store service  
- embedding.py: OpenAI embedding service
"""

__version__ = "1.0.0"
__author__ = "HackRx 6.0 Team"

# Import main service classes for easy access
try:
    from .atlas import AtlasService
    from .astradb import AstraDBService
    from .embedding import EmbeddingService
    
    __all__ = ['AtlasService', 'AstraDBService', 'EmbeddingService']
    
except ImportError as e:
    # Handle missing dependencies gracefully
    print(f"Warning: Some service dependencies are missing: {e}")
    __all__ = []