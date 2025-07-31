"""
HackRx 6.0 - Basic Response Models
Pydantic models for API requests and responses
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class QueryRequest(BaseModel):
    """Request model for document queries"""
    documents: str = Field(..., description="Document URL to process")
    questions: List[str] = Field(..., description="List of questions to answer")

class QueryResponse(BaseModel):
    """Response model for query results"""
    answers: List[str] = Field(..., description="List of answers corresponding to questions")
    processing_time: float = Field(..., description="Time taken to process the request in seconds")
    request_id: str = Field(..., description="Unique identifier for the request")
    cached: bool = Field(default=False, description="Whether the response was cached")
    error: Optional[str] = Field(default=None, description="Error message if any")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional response metadata")
    accuracy_score: Optional[float] = Field(default=None, description="Confidence score for the answers")
    answer_sources: Optional[List[str]] = Field(default=None, description="Sources used for each answer")
    validation_passed: Optional[bool] = Field(default=None, description="Whether validation checks passed")

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="System health status")
    message: str = Field(..., description="Health status message")
    services_loaded: bool = Field(..., description="Whether all services are loaded")
    components: Optional[Dict[str, bool]] = Field(default=None, description="Individual component status")
    performance: Optional[Dict[str, Any]] = Field(default=None, description="Performance metrics")

class ConfigResponse(BaseModel):
    """Configuration response model"""
    version: str = Field(..., description="System version")
    services_loaded: bool = Field(..., description="Whether services are loaded")
    configuration: Dict[str, Any] = Field(..., description="System configuration details")
    service_errors: Optional[List[str]] = Field(default=None, description="Any service loading errors")