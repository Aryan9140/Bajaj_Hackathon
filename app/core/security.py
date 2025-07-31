# app/core/security.py
"""
Security and authentication for HackRx 6.0 API
"""

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

def verify_api_key(api_key: str) -> bool:
    """Verify the provided API key against the hackathon key"""
    if not api_key:
        return False
    
    # The hackathon provides a specific API key
    expected_key = settings.API_KEY
    
    if api_key == expected_key:
        return True
    
    logger.warning(f"🔐 Invalid API key attempt: {api_key[:10]}...")
    return False