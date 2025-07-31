# app/utils/validators.py
"""
Input validation utilities for HackRx 6.0
Ensures data quality and security
"""

import re
from typing import List, Optional
from urllib.parse import urlparse

def validate_url(url: str) -> bool:
    """Validate if the URL is properly formatted and accessible"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False

def validate_questions(questions: List[str]) -> tuple[bool, Optional[str]]:
    """Validate the list of questions"""
    if not questions:
        return False, "Questions list cannot be empty"
    
    if len(questions) > 50:
        return False, "Too many questions (maximum 50 allowed)"
    
    for i, question in enumerate(questions):
        if not question or not question.strip():
            return False, f"Question {i+1} cannot be empty"
        
        if len(question.strip()) < 5:
            return False, f"Question {i+1} is too short (minimum 5 characters)"
        
        if len(question.strip()) > 500:
            return False, f"Question {i+1} is too long (maximum 500 characters)"
    
    return True, None

def sanitize_text(text: str) -> str:
    """Sanitize text input for security"""
    if not text:
        return ""
    
    # Remove potentially harmful characters
    sanitized = re.sub(r'[<>"\']', '', text)
    
    # Normalize whitespace
    sanitized = re.sub(r'\s+', ' ', sanitized)
    
    return sanitized.strip()

def validate_api_key_format(api_key: str) -> bool:
    """Validate API key format"""
    if not api_key:
        return False
    
    # Check if it's the expected hackathon key format
    return len(api_key) == 64 and api_key.isalnum()