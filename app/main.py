"""
HackRx 6.0 - Universal Document Processing System
Handles both simple data processing AND document processing
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

app = FastAPI(
    title="HackRx 6.0 - Universal Document AI",
    description="Handles both simple data processing and document processing",
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

# Security (optional for simple data processing)
security = HTTPBearer(auto_error=False)
HACKRX_API_KEY = "6d2683f80eca9847d20948e1e5508885d08fdc65d943182f85de250687859193"

# Pydantic Models
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
    """Optional API key verification"""
    if credentials and credentials.credentials != HACKRX_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials if credentials else None

class UniversalDocumentProcessor:
    def __init__(self):
        self.session_timeout = aiohttp.ClientTimeout(total=30)
    
    async def download_document(self, url: str) -> Optional[bytes]:
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
        try:
            docx_file = io.BytesIO(docx_data)
            document = docx.Document(docx_file)
            
            text_parts = []
            for paragraph in document.paragraphs:
                if paragraph.text.strip():
                    text_parts.append(paragraph.text.strip())
            
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
        try:
            logger.info(f"Processing document: {document_url[:100]}...")
            
            document_data = await self.download_document(document_url)
            if not document_data:
                return ""
            
            doc_type = self.detect_document_type(document_data, document_url)
            logger.info(f"Document type detected: {doc_type}")
            
            if doc_type == 'pdf':
                text = await self.extract_text_from_pdf(document_data)
            elif doc_type == 'docx':
                text = await self.extract_text_from_docx(document_data)
            else:
                text = document_data.decode('utf-8', errors='ignore')
            
            cleaned_text = self.clean_text(text)
            logger.info(f"Final text length: {len(cleaned_text)} characters")
            return cleaned_text
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[\f\r]+', '\n', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if len(line) > 2:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()

class DynamicAnswerGenerator:
    def extract_relevant_context(self, question: str, document_text: str, max_context: int = 3000) -> str:
        if not document_text:
            return ""
        
        keywords = self.extract_question_keywords(question)
        sentences = self.split_into_sentences(document_text)
        
        scored_sentences = []
        for sentence in sentences:
            score = self.calculate_relevance_score(sentence, keywords, question)
            if score > 0.1:
                scored_sentences.append((sentence, score))
        
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        context = ""
        for sentence, score in scored_sentences:
            if len(context) + len(sentence) <= max_context:
                context += sentence + " "
            else:
                break
        
        return context.strip()
    
    def extract_question_keywords(self, question: str) -> List[str]:
        stop_words = {
            'what', 'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 
            'at', 'to', 'for', 'of', 'with', 'by', 'how', 'when', 'where', 'why',
            'does', 'do', 'did', 'will', 'would', 'could', 'should', 'can', 'may',
            'this', 'that', 'these', 'those', 'any', 'some', 'all', 'there'
        }
        
        words = re.findall(r'\b\w+\b', question.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        phrases = []
        question_lower = question.lower()
        phrase_patterns = [
            r'\b(\w+\s+period)\b', r'\b(\w+\s+coverage)\b', r'\b(\w+\s+benefit)\b',
            r'\b(\w+\s+discount)\b', r'\b(\w+\s+expenses?)\b', r'\b(\w+\s+charges?)\b',
            r'\b(\w+\s+surgery)\b', r'\b(\w+\s+treatment)\b',
        ]
        
        for pattern in phrase_patterns:
            matches = re.findall(pattern, question_lower)
            phrases.extend(matches)
        
        return keywords + phrases
    
    def split_into_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'[.!?]+', text)
        clean_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:
                clean_sentences.append(sentence)
        return clean_sentences
    
    def calculate_relevance_score(self, sentence: str, keywords: List[str], question: str) -> float:
        sentence_lower = sentence.lower()
        question_lower = question.lower()
        
        score = 0.0
        
        for keyword in keywords:
            if keyword in sentence_lower:
                score += 1.0
        
        question_words = question_lower.split()
        for i in range(len(question_words) - 1):
            phrase = f"{question_words[i]} {question_words[i+1]}"
            if phrase in sentence_lower:
                score += 2.0
        
        if len(sentence) > 500:
            score *= 0.8
        
        if re.search(r'\d+', sentence):
            score += 0.5
        
        return score
    
    def generate_dynamic_answer(self, question: str, context: str) -> str:
        if not context:
            return "Information about this topic is not available in the provided document."
        
        sentences = self.split_into_sentences(context)
        if not sentences:
            return "Information about this topic is not available in the provided document."
        
        keywords = self.extract_question_keywords(question)
        best_sentences = []
        
        for sentence in sentences:
            score = self.calculate_relevance_score(sentence, keywords, question)
            if score > 0.5:
                best_sentences.append((sentence, score))
        
        best_sentences.sort(key=lambda x: x[1], reverse=True)
        
        if not best_sentences:
            for sentence in sentences:
                if len(sentence) > 30:
                    return sentence
            return "Information about this topic is not available in the provided document."
        
        answer_parts = []
        total_length = 0
        max_length = 500
        
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

@app.post("/hackrx/run")
async def unified_hackrx_endpoint(
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    """
    Unified endpoint that handles BOTH:
    1. Simple data processing: {"data": [...]}
    2. Document processing: {"documents": "...", "questions": [...]}
    """
    try:
        request_data = await request.json()
        
        # Simple data processing
        if "data" in request_data:
            data = request_data["data"]
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
        
        # Document processing
        elif "documents" in request_data and "questions" in request_data:
            documents = request_data.get("documents", "")
            questions = request_data.get("questions", [])
            
            logger.info(f"Processing {len(questions)} questions for document: {documents[:100]}...")
            
            # Process document
            document_text = await doc_processor.process_document(documents)
            
            if not document_text:
                logger.warning("No text extracted from document")
                return {
                    "answers": ["Unable to extract information from the provided document."] * len(questions)
                }
            
            # Generate answers
            answers = []
            for i, question in enumerate(questions):
                logger.info(f"Processing question {i+1}: {question[:50]}...")
                
                context = answer_generator.extract_relevant_context(question, document_text)
                answer = answer_generator.generate_dynamic_answer(question, context)
                answers.append(answer)
                
                logger.info(f"Generated answer {i+1}: {answer[:100]}...")
            
            logger.info(f"Successfully processed all {len(questions)} questions")
            
            return {"answers": answers}
        
        else:
            raise HTTPException(status_code=400, detail="Invalid request format")
            
    except Exception as e:
        logger.error(f"Request processing failed: {e}")
        
        # Error fallback
        if 'request_data' in locals():
            if "questions" in request_data:
                question_count = len(request_data.get("questions", []))
                return {"answers": ["Unable to process this question at the moment."] * question_count}
        
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "HackRx 6.0 - Universal Document AI + Competition API",
        "status": "ready",
        "version": "6.0.0",
        "endpoint": "/hackrx/run supports both data processing and document processing",
        "features": [
            "Simple data processing (data array)",
            "Universal document processing (PDF, DOCX, TXT)",
            "100% dynamic responses (no templates)",
            "Intelligent context extraction",
            "Clean JSON output format"
        ]
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "6.0.0"}