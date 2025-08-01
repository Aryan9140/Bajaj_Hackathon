"""
HackRx 6.0 - 100% Dynamic Universal Document Processing System
Works with ANY document type, NO templates, completely dynamic responses
"""

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import aiohttp
import PyPDF2
import docx
import io
import re
import asyncio
import time
from typing import List, Dict, Any, Optional
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="HackRx 6.0 - Universal Document AI",
    description="Dynamic document processing for any PDF, DOCX, or text document",
    version="6.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# API Key
HACKRX_API_KEY = "6d2683f80eca9847d20948e1e5508885d08fdc65d943182f85de250687859193"

# ADD THESE PYDANTIC MODELS FOR YOUR ORIGINAL ENDPOINT
class QueryRequest(BaseModel):
    data: list

class QueryResponse(BaseModel):
    is_success: bool
    user_id: str
    email: str
    roll_number: str
    numbers: list
    alphabets: list
    highest_lowercase_alphabet: list

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key authentication"""
    if credentials.credentials != HACKRX_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials

class UniversalDocumentProcessor:
    """
    Universal document processor that works with ANY document type
    Completely dynamic, no templates or hardcoded responses
    """
    
    def __init__(self):
        self.session_timeout = aiohttp.ClientTimeout(total=30)
    
    async def download_document(self, url: str) -> Optional[bytes]:
        """Download document from URL"""
        try:
            async with aiohttp.ClientSession(timeout=self.session_timeout) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.read()
                        logger.info(f"Downloaded document: {len(content)} bytes")
                        return content
                    else:
                        logger.error(f"Download failed: HTTP {response.status}")
                        return None
        except Exception as e:
            logger.error(f"Download error: {e}")
            return None
    
    def detect_document_type(self, data: bytes, url: str = "") -> str:
        """Detect document type from content and URL"""
        if data.startswith(b'%PDF'):
            return 'pdf'
        elif data.startswith(b'PK\x03\x04') or data.startswith(b'PK\x05\x06'):
            return 'docx'
        elif url.lower().endswith('.pdf'):
            return 'pdf'
        elif url.lower().endswith(('.docx', '.doc')):
            return 'docx'
        else:
            return 'text'
    
    async def extract_text_from_pdf(self, pdf_data: bytes) -> str:
        """Extract text from PDF"""
        try:
            pdf_file = io.BytesIO(pdf_data)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text_parts = []
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_parts.append(page_text)
                except Exception as e:
                    logger.warning(f"Page {page_num} extraction failed: {e}")
                    continue
            
            full_text = "\n".join(text_parts)
            logger.info(f"PDF text extracted: {len(full_text)} characters")
            return full_text
            
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            return ""
    
    async def extract_text_from_docx(self, docx_data: bytes) -> str:
        """Extract text from DOCX"""
        try:
            docx_file = io.BytesIO(docx_data)
            document = docx.Document(docx_file)
            
            text_parts = []
            
            # Extract paragraph text
            for paragraph in document.paragraphs:
                if paragraph.text.strip():
                    text_parts.append(paragraph.text.strip())
            
            # Extract table text
            for table in document.tables:
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        if cell.text.strip():
                            row_text.append(cell.text.strip())
                    if row_text:
                        text_parts.append(" | ".join(row_text))
            
            full_text = "\n".join(text_parts)
            logger.info(f"DOCX text extracted: {len(full_text)} characters")
            return full_text
            
        except Exception as e:
            logger.error(f"DOCX extraction failed: {e}")
            return ""
    
    async def process_document(self, document_url: str) -> str:
        """Process any document and extract text"""
        try:
            logger.info(f"Processing document: {document_url[:100]}...")
            
            # Download document
            document_data = await self.download_document(document_url)
            if not document_data:
                return ""
            
            # Detect type and extract text
            doc_type = self.detect_document_type(document_data, document_url)
            logger.info(f"Document type detected: {doc_type}")
            
            if doc_type == 'pdf':
                text = await self.extract_text_from_pdf(document_data)
            elif doc_type == 'docx':
                text = await self.extract_text_from_docx(document_data)
            else:
                # Try as text
                text = document_data.decode('utf-8', errors='ignore')
            
            # Clean and optimize text
            cleaned_text = self.clean_text(text)
            logger.info(f"Final text length: {len(cleaned_text)} characters")
            
            return cleaned_text
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean and optimize extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page breaks and form feeds
        text = re.sub(r'[\f\r]+', '\n', text)
        
        # Normalize line breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove very short lines (artifacts)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if len(line) > 2:
                cleaned_lines.append(line)
        
        cleaned_text = '\n'.join(cleaned_lines)
        return cleaned_text.strip()

class DynamicAnswerGenerator:
    """
    100% Dynamic answer generator - NO templates
    Uses actual document content to generate responses
    """
    
    def __init__(self):
        pass
    
    def extract_relevant_context(self, question: str, document_text: str, max_context: int = 3000) -> str:
        """Extract relevant context from document for the question"""
        if not document_text:
            return ""
        
        # Extract keywords from question
        keywords = self.extract_question_keywords(question)
        
        # Split document into sentences
        sentences = self.split_into_sentences(document_text)
        
        # Score sentences by relevance to question
        scored_sentences = []
        for sentence in sentences:
            score = self.calculate_relevance_score(sentence, keywords, question)
            if score > 0.1:  # Only keep relevant sentences
                scored_sentences.append((sentence, score))
        
        # Sort by relevance score
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Build context from most relevant sentences
        context = ""
        for sentence, score in scored_sentences:
            if len(context) + len(sentence) <= max_context:
                context += sentence + " "
            else:
                break
        
        return context.strip()
    
    def extract_question_keywords(self, question: str) -> List[str]:
        """Extract important keywords from question"""
        # Remove common question words
        stop_words = {
            'what', 'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 
            'at', 'to', 'for', 'of', 'with', 'by', 'how', 'when', 'where', 'why',
            'does', 'do', 'did', 'will', 'would', 'could', 'should', 'can', 'may',
            'this', 'that', 'these', 'those', 'any', 'some', 'all', 'there'
        }
        
        # Extract words and phrases
        words = re.findall(r'\b\w+\b', question.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Extract multi-word phrases (more important)
        phrases = []
        question_lower = question.lower()
        
        # Common multi-word patterns in questions
        phrase_patterns = [
            r'\b(\w+\s+period)\b',
            r'\b(\w+\s+coverage)\b', 
            r'\b(\w+\s+benefit)\b',
            r'\b(\w+\s+discount)\b',
            r'\b(\w+\s+expenses?)\b',
            r'\b(\w+\s+charges?)\b',
            r'\b(\w+\s+surgery)\b',
            r'\b(\w+\s+treatment)\b',
        ]
        
        for pattern in phrase_patterns:
            matches = re.findall(pattern, question_lower)
            phrases.extend(matches)
        
        return keywords + phrases
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        
        # Clean and filter sentences
        clean_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Only keep substantial sentences
                clean_sentences.append(sentence)
        
        return clean_sentences
    
    def calculate_relevance_score(self, sentence: str, keywords: List[str], question: str) -> float:
        """Calculate how relevant a sentence is to the question"""
        sentence_lower = sentence.lower()
        question_lower = question.lower()
        
        score = 0.0
        
        # Keyword matching
        for keyword in keywords:
            if keyword in sentence_lower:
                score += 1.0
        
        # Exact phrase matching (higher weight)
        question_words = question_lower.split()
        for i in range(len(question_words) - 1):
            phrase = f"{question_words[i]} {question_words[i+1]}"
            if phrase in sentence_lower:
                score += 2.0
        
        # Length penalty for very long sentences
        if len(sentence) > 500:
            score *= 0.8
        
        # Bonus for sentences with numbers (often contain specific info)
        if re.search(r'\d+', sentence):
            score += 0.5
        
        return score
    
    def generate_dynamic_answer(self, question: str, context: str) -> str:
        """Generate answer dynamically from context"""
        if not context:
            return "Information about this topic is not available in the provided document."
        
        # Find the most relevant sentence for the question
        sentences = self.split_into_sentences(context)
        if not sentences:
            return "Information about this topic is not available in the provided document."
        
        # Score sentences again for final selection
        keywords = self.extract_question_keywords(question)
        best_sentences = []
        
        for sentence in sentences:
            score = self.calculate_relevance_score(sentence, keywords, question)
            if score > 0.5:
                best_sentences.append((sentence, score))
        
        # Sort by score and take top sentences
        best_sentences.sort(key=lambda x: x[1], reverse=True)
        
        if not best_sentences:
            # Fallback to first substantial sentence
            for sentence in sentences:
                if len(sentence) > 30:
                    return sentence
            return "Information about this topic is not available in the provided document."
        
        # Build answer from best sentences
        answer_parts = []
        total_length = 0
        max_length = 500  # Reasonable answer length
        
        for sentence, score in best_sentences:
            if total_length + len(sentence) <= max_length:
                answer_parts.append(sentence)
                total_length += len(sentence)
            else:
                break
        
        if answer_parts:
            return ". ".join(answer_parts) + "."
        else:
            return best_sentences[0][0]

# Initialize processors
doc_processor = UniversalDocumentProcessor()
answer_generator = DynamicAnswerGenerator()

# ADD YOUR ORIGINAL ENDPOINT HERE (for the competition)
@app.post("/hackrx/run")
async def process_data(request: QueryRequest):
    """Original endpoint for the competition"""
    try:
        data = request.data
        
        numbers = [item for item in data if item.isdigit()]
        alphabets = [item for item in data if item.isalpha()]
        lowercase_alphabets = [item for item in alphabets if item.islower()]
        highest_lowercase = [max(lowercase_alphabets)] if lowercase_alphabets else []
        
        return QueryResponse(
            is_success=True,
            user_id="patel",
            email="aryanpatel77462@gmail.com",
            roll_number="1047",
            numbers=numbers,
            alphabets=alphabets,
            highest_lowercase_alphabet=highest_lowercase
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Document processing endpoint (separate endpoint)
@app.post("/hackrx/document")
async def hackrx_universal_endpoint(
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    """
    Universal HackRx endpoint - Works with ANY document type
    Returns only clean JSON format: {"answers": [...]}
    """
    try:
        # Parse request
        request_data = await request.json()
        documents = request_data.get("documents", "")
        questions = request_data.get("questions", [])
        
        logger.info(f"Processing {len(questions)} questions for document: {documents[:100]}...")
        
        # Process document (works with ANY format)
        document_text = await doc_processor.process_document(documents)
        
        if not document_text:
            logger.warning("No text extracted from document")
            return {
                "answers": ["Unable to extract information from the provided document."] * len(questions)
            }
        
        # Generate dynamic answers for each question
        answers = []
        for i, question in enumerate(questions):
            logger.info(f"Processing question {i+1}: {question[:50]}...")
            
            # Extract relevant context for this specific question
            context = answer_generator.extract_relevant_context(question, document_text)
            
            # Generate dynamic answer from context
            answer = answer_generator.generate_dynamic_answer(question, context)
            answers.append(answer)
            
            logger.info(f"Generated answer {i+1}: {answer[:100]}...")
        
        logger.info(f"Successfully processed all {len(questions)} questions")
        
        # Return ONLY the answers array (HackRx format)
        return {
            "answers": answers
        }
        
    except Exception as e:
        logger.error(f"Request processing failed: {e}")
        
        # Error fallback
        question_count = len(request_data.get("questions", [])) if 'request_data' in locals() else 1
        return {
            "answers": ["Unable to process this question at the moment. Please try again."] * question_count
        }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "HackRx 6.0 - Universal Document AI + Competition API",
        "status": "ready",
        "version": "6.0.0",
        "endpoints": {
            "/hackrx/run": "Competition endpoint for data processing",
            "/hackrx/document": "Document processing endpoint"
        },
        "features": [
            "Universal document processing (PDF, DOCX, TXT)",
            "100% dynamic responses (no templates)",
            "Intelligent context extraction",
            "Multi-format document support",
            "Clean JSON output format",
            "Competition data processing"
        ]
    }

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy", "version": "6.0.0"}

# Remove the if __name__ == "__main__" section for production deployment