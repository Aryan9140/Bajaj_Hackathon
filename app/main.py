# app/main.py - Competition Document Processing for HackRx 6.0
import os
import re
import time
import hashlib
import asyncio
import logging
from typing import List, Optional, Tuple

# Safe imports for Render deployment
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    print("aiohttp not available")

try:
    import PyPDF2
    import io
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    print("PyPDF2 not available")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("numpy not available")

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("Groq not available")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class DocumentProcessor:
    """Document processor for HackRx competition"""
    
    def __init__(self):
        self.cache = {}
        logger.info("DocumentProcessor initialized")
    
    async def download_document(self, url: str) -> Optional[bytes]:
        """Download document with retry strategy"""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        if url_hash in self.cache:
            logger.info("Document retrieved from cache")
            return self.cache[url_hash]
        
        if not AIOHTTP_AVAILABLE:
            # Fallback to requests
            try:
                import requests
                response = requests.get(url, timeout=60)
                if response.status_code == 200:
                    content = response.content
                    self.cache[url_hash] = content
                    logger.info(f"Document downloaded: {len(content)} bytes")
                    return content
            except Exception as e:
                logger.error(f"Download failed: {e}")
                return None
        
        # Use aiohttp if available
        max_retries = 3
        for attempt in range(max_retries):
            try:
                timeout = aiohttp.ClientTimeout(total=60)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    headers = {'User-Agent': 'HackRx/6.0 Competition Bot'}
                    async with session.get(url, headers=headers) as response:
                        if response.status == 200:
                            content = await response.read()
                            self.cache[url_hash] = content
                            logger.info(f"Document downloaded: {len(content)} bytes")
                            return content
                        else:
                            logger.warning(f"Download failed with status {response.status}")
            except Exception as e:
                logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        logger.error("All download attempts failed")
        return None
    
    def extract_text_from_pdf(self, pdf_data: bytes) -> str:
        """Extract text from PDF"""
        if not PYPDF2_AVAILABLE:
            logger.error("PyPDF2 not available")
            return ""
        
        try:
            pdf_file = io.BytesIO(pdf_data)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text_parts = []
            total_pages = len(pdf_reader.pages)
            logger.info(f"Processing PDF with {total_pages} pages")
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text and len(page_text.strip()) > 10:
                        cleaned_text = self.clean_text(page_text)
                        if cleaned_text:
                            text_parts.append(cleaned_text)
                            logger.debug(f"Page {page_num + 1}: {len(cleaned_text)} chars")
                except Exception as e:
                    logger.warning(f"Error on page {page_num + 1}: {e}")
                    continue
            
            full_text = '\n\n'.join(text_parts)
            logger.info(f"PDF extraction complete: {len(full_text)} chars from {len(text_parts)} pages")
            return full_text
            
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text"""
        if not text:
            return ""
        
        # Fix hyphenated words across lines
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
        
        # Fix spacing issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'(\d+)([A-Za-z])', r'\1 \2', text)
        text = re.sub(r'([A-Za-z])(\d+)', r'\1 \2', text)
        
        # Fix punctuation
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
        text = re.sub(r'([,;:])([A-Za-z])', r'\1 \2', text)
        
        # Clean whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove page numbers and short lines
        lines = []
        for line in text.split('\n'):
            line = line.strip()
            # Skip page numbers and very short lines
            if len(line) > 3 and not re.match(r'^\d{1,3}$', line):
                lines.append(line)
        
        return '\n'.join(lines)
    
    async def process_document(self, document_url: str) -> str:
        """Main document processing method"""
        start_time = time.time()
        
        try:
            document_data = await self.download_document(document_url)
            if not document_data:
                return ""
            
            if document_data.startswith(b'%PDF'):
                logger.info("Processing as PDF")
                extracted_text = self.extract_text_from_pdf(document_data)
            else:
                logger.info("Processing as plain text")
                extracted_text = document_data.decode('utf-8', errors='ignore')
            
            final_text = self.clean_text(extracted_text)
            
            processing_time = time.time() - start_time
            logger.info(f"Document processed in {processing_time:.2f}s: {len(final_text)} chars")
            
            return final_text
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            return ""

class CompetitionAnswerEngine:
    """Answer engine using Groq API with gemma2-9b-it model"""
    
    def __init__(self):
        self.groq_client = None
        self.document_text = ""
        
        # Initialize Groq client
        if GROQ_AVAILABLE and GROQ_API_KEY:
            try:
                self.groq_client = Groq(api_key=GROQ_API_KEY)
                logger.info("Groq client initialized with gemma2-9b-it model")
            except Exception as e:
                logger.error(f"Groq initialization failed: {e}")
        else:
            logger.warning("Groq not available or API key missing")
    
    def prepare_document(self, document_text: str):
        """Prepare document for answering"""
        self.document_text = document_text
        logger.info(f"Document prepared: {len(document_text)} characters")
    
    def find_relevant_text(self, question: str) -> str:
        """Find relevant text using keyword matching"""
        if not self.document_text:
            return ""
        
        question_lower = question.lower()
        
        # Define keyword patterns for policy questions
        keyword_patterns = {
            "grace period": ["grace period", "grace", "days", "payment", "premium", "due date"],
            "waiting period": ["waiting period", "waiting", "months", "years", "continuous", "coverage"],
            "maternity": ["maternity", "pregnancy", "childbirth", "delivery", "termination"],
            "cataract": ["cataract", "surgery", "waiting", "period"],
            "organ donor": ["organ donor", "transplantation", "harvesting", "medical expenses"],
            "discount": ["discount", "ncd", "no claim", "premium", "renewal"],
            "health check": ["health check", "preventive", "reimbursement", "policy years"],
            "hospital": ["hospital", "institution", "beds", "qualified", "medical", "practitioners"],
            "ayush": ["ayush", "ayurveda", "yoga", "naturopathy", "unani", "siddha", "homeopathy"],
            "room rent": ["room rent", "icu", "charges", "sum insured", "limit", "daily"]
        }
        
        # Find matching pattern
        relevant_keywords = []
        for pattern_name, keywords in keyword_patterns.items():
            if any(keyword in question_lower for keyword in keywords):
                relevant_keywords.extend(keywords)
                break
        
        if not relevant_keywords:
            # Extract key words from question
            relevant_keywords = [word for word in re.findall(r'\b\w+\b', question_lower) 
                               if len(word) > 3 and word not in ['what', 'when', 'where', 'does', 'this']]
        
        # Find relevant sentences
        sentences = re.split(r'[.!?]+', self.document_text)
        relevant_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
            
            sentence_lower = sentence.lower()
            match_count = sum(1 for keyword in relevant_keywords if keyword in sentence_lower)
            
            if match_count >= 1:
                relevant_sentences.append((sentence, match_count))
        
        if relevant_sentences:
            # Sort by relevance and return best matches
            relevant_sentences.sort(key=lambda x: x[1], reverse=True)
            best_sentences = [sent for sent, count in relevant_sentences[:3]]
            return '. '.join(best_sentences).strip()
        
        return ""
    
    async def generate_competition_answer(self, question: str) -> str:
        """Generate answer using Groq API with gemma2-9b-it"""
        
        # Find relevant context
        context = self.find_relevant_text(question)
        
        if not context:
            return "Information about this topic is not available in the provided document."
        
        # Try Groq API with gemma2-9b-it model
        if self.groq_client:
            try:
                system_prompt = """You are a policy document analyst. Answer questions based ONLY on the provided context.

RULES:
- Use EXACT details from the context: numbers, periods, percentages
- Be comprehensive and precise
- Include all specific conditions and terms
- If information is not in context, state clearly

Answer based only on the provided context."""

                user_prompt = f"""Context from policy document:
{context}

Question: {question}

Answer:"""

                response = self.groq_client.chat.completions.create(
                    model="gemma2-9b-it",  # Using the specified model
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=500,
                    temperature=0.0
                )
                
                answer = response.choices[0].message.content.strip()
                logger.info(f"Groq generated answer: {len(answer)} chars")
                return answer
                
            except Exception as e:
                logger.warning(f"Groq generation failed: {e}")
        
        # Fallback: return context directly if Groq fails
        if len(context) > 50:
            logger.info("Using direct context as answer")
            return context
        
        return "Unable to find specific information about this topic in the document."

# Initialize global instances for run.py to import
doc_processor = DocumentProcessor()
answer_engine = CompetitionAnswerEngine()