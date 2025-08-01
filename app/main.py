"""
HackRx 6.0 - LLM-Powered Intelligent Query-Retrieval System
Optimized for MAXIMUM COMPETITION SCORING
- Handles both Known Documents (0.5 weight) and Unknown Documents (2.0 weight)
- Focuses on high accuracy for maximum score contribution
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
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
import os
import json
import time
from datetime import datetime
import hashlib
from collections import defaultdict

# Advanced libraries with fallbacks
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="HackRx 6.0 - Competition Scoring System",
    description="Maximum accuracy document processing for competition scoring",
    version="6.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)
HACKRX_API_KEY = "6d2683f80eca9847d20948e1e5508885d08fdc65d943182f85de250687859193"

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key authentication"""
    if credentials and credentials.credentials != HACKRX_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials if credentials else None

class DocumentProcessor:
    """
    Ultra-high accuracy document processor
    Focus: Extract maximum information with minimal loss
    """
    
    def __init__(self):
        self.session_timeout = aiohttp.ClientTimeout(total=120)  # Extended timeout
        self.cache = {}  # Document cache for efficiency
        logger.info("DocumentProcessor initialized for competition")
    
    async def download_document(self, url: str) -> Optional[bytes]:
        """Download with aggressive retry and error handling"""
        # Check cache first
        url_hash = hashlib.md5(url.encode()).hexdigest()
        if url_hash in self.cache:
            logger.info("Document retrieved from cache")
            return self.cache[url_hash]
        
        max_retries = 5
        backoff_delays = [1, 2, 4, 8, 16]  # Exponential backoff
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Downloading document (attempt {attempt + 1}/{max_retries})")
                
                async with aiohttp.ClientSession(
                    timeout=self.session_timeout,
                    connector=aiohttp.TCPConnector(limit=10)
                ) as session:
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    }
                    
                    async with session.get(url, headers=headers) as response:
                        if response.status == 200:
                            content = await response.read()
                            # Cache the document
                            self.cache[url_hash] = content
                            logger.info(f"Document downloaded successfully: {len(content)} bytes")
                            return content
                        else:
                            logger.warning(f"Download failed with status {response.status}")
                            
            except Exception as e:
                logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(backoff_delays[attempt])
        
        logger.error("All download attempts failed")
        return None
    
    async def extract_text_from_pdf(self, pdf_data: bytes) -> str:
        """Ultra-comprehensive PDF text extraction"""
        try:
            logger.info("Starting comprehensive PDF extraction")
            pdf_file = io.BytesIO(pdf_data)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text_parts = []
            total_pages = len(pdf_reader.pages)
            logger.info(f"Processing {total_pages} pages")
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    # Multiple extraction attempts per page
                    page_text = ""
                    
                    # Method 1: Standard extraction
                    try:
                        text1 = page.extract_text()
                        if text1 and len(text1.strip()) > 10:
                            page_text = text1
                    except:
                        pass
                    
                    # Method 2: Alternative extraction if first fails
                    if not page_text:
                        try:
                            # Try different extraction parameters
                            if hasattr(page, 'extract_text'):
                                text2 = page.extract_text(space_width=200)
                                if text2 and len(text2.strip()) > 10:
                                    page_text = text2
                        except:
                            pass
                    
                    if page_text:
                        # Advanced text cleaning and reconstruction
                        cleaned_text = self.advanced_pdf_cleaning(page_text)
                        if cleaned_text:
                            text_parts.append(cleaned_text)
                            logger.debug(f"Page {page_num + 1}: {len(cleaned_text)} chars extracted")
                    else:
                        logger.warning(f"No text extracted from page {page_num + 1}")
                        
                except Exception as e:
                    logger.warning(f"Error processing page {page_num + 1}: {e}")
                    continue
            
            # Combine all text with proper formatting
            full_text = self.combine_pdf_pages(text_parts)
            logger.info(f"PDF extraction complete: {len(full_text)} total characters from {len(text_parts)} pages")
            
            return full_text
            
        except Exception as e:
            logger.error(f"PDF extraction completely failed: {e}")
            return ""
    
    def advanced_pdf_cleaning(self, text: str) -> str:
        """Advanced PDF text cleaning for maximum accuracy"""
        if not text:
            return ""
        
        # Fix common PDF extraction issues
        # 1. Fix broken words across lines
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
        
        # 2. Fix spacing issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between camelCase
        text = re.sub(r'(\d+)([A-Za-z])', r'\1 \2', text)  # Space between numbers and letters
        text = re.sub(r'([A-Za-z])(\d+)', r'\1 \2', text)  # Space between letters and numbers
        
        # 3. Fix punctuation spacing
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
        text = re.sub(r'([,;:])([A-Za-z])', r'\1 \2', text)
        
        # 4. Clean up whitespace but preserve structure
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple newlines to double
        
        # 5. Remove likely page artifacts
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip likely page numbers
            if re.match(r'^\d{1,3}$', line):
                continue
            
            # Skip very short lines that are likely artifacts
            if len(line) < 3:
                continue
                
            # Skip lines that are just special characters
            if re.match(r'^[^\w\s]*$', line):
                continue
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def combine_pdf_pages(self, text_parts: List[str]) -> str:
        """Intelligently combine PDF pages"""
        if not text_parts:
            return ""
        
        combined = []
        for i, part in enumerate(text_parts):
            if i > 0:
                # Check if previous part ends mid-sentence
                prev_part = text_parts[i-1].strip()
                if prev_part and not prev_part[-1] in '.!?':
                    # Continue from previous page
                    combined.append(' ' + part)
                else:
                    # New section
                    combined.append('\n\n' + part)
            else:
                combined.append(part)
        
        return ''.join(combined)
    
    async def process_document(self, document_url: str) -> str:
        """Main document processing with maximum accuracy focus"""
        start_time = time.time()
        
        try:
            # Download with retries
            document_data = await self.download_document(document_url)
            if not document_data:
                logger.error("Failed to download document")
                return ""
            
            # Detect and process
            if document_data.startswith(b'%PDF'):
                logger.info("Processing as PDF document")
                extracted_text = await self.extract_text_from_pdf(document_data)
            elif document_data.startswith(b'PK'):
                logger.info("Processing as DOCX document")
                extracted_text = self.extract_text_from_docx(document_data)
            else:
                logger.info("Processing as plain text")
                extracted_text = document_data.decode('utf-8', errors='ignore')
            
            # Final cleanup and validation
            final_text = self.final_text_validation(extracted_text)
            
            processing_time = time.time() - start_time
            logger.info(f"Document processing completed in {processing_time:.2f}s: {len(final_text)} characters")
            
            return final_text
            
        except Exception as e:
            logger.error(f"Document processing failed completely: {e}")
            return ""
    
    def extract_text_from_docx(self, docx_data: bytes) -> str:
        """Enhanced DOCX extraction"""
        try:
            docx_file = io.BytesIO(docx_data)
            document = docx.Document(docx_file)
            
            text_parts = []
            
            # Extract paragraphs
            for paragraph in document.paragraphs:
                if paragraph.text.strip():
                    text_parts.append(paragraph.text.strip())
            
            # Extract tables
            for table in document.tables:
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        if cell.text.strip():
                            row_text.append(cell.text.strip())
                    if row_text:
                        text_parts.append(' | '.join(row_text))
            
            return '\n'.join(text_parts)
            
        except Exception as e:
            logger.error(f"DOCX extraction failed: {e}")
            return ""
    
    def final_text_validation(self, text: str) -> str:
        """Final validation and cleanup"""
        if not text or len(text.strip()) < 50:
            logger.warning("Extracted text is too short or empty")
            return text
        
        # Final cleanup
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Log text statistics for debugging
        word_count = len(text.split())
        char_count = len(text)
        logger.info(f"Final text stats: {char_count} characters, {word_count} words")
        
        return text

class CompetitionSemanticSearch:
    """
    Competition-optimized semantic search for maximum accuracy
    """
    
    def __init__(self):
        self.embedding_model = None
        self.faiss_index = None
        self.text_chunks = []
        self.chunk_metadata = []
        
        # Initialize the best available embedding model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # Use a high-quality model for better accuracy
                self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
                logger.info("High-quality embedding model loaded")
            except:
                try:
                    self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                    logger.info("Fallback embedding model loaded")
                except Exception as e:
                    logger.warning(f"No embedding model available: {e}")
    
    def intelligent_chunking(self, text: str) -> List[Dict[str, Any]]:
        """Advanced text chunking optimized for policy/legal documents"""
        if not text:
            return []
        
        chunks = []
        
        # Strategy 1: Split by clear sections (for policy documents)
        section_patterns = [
            r'\n\s*(?:SECTION|Section|CLAUSE|Clause|ARTICLE|Article)\s*\d+',
            r'\n\s*(?:\d+\.|\(\d+\))\s*[A-Z]',
            r'\n\s*[A-Z][A-Z\s]{10,}:',  # All caps headers
        ]
        
        sections = []
        current_pos = 0
        
        for pattern in section_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            for match in matches:
                if match.start() > current_pos:
                    section_text = text[current_pos:match.start()].strip()
                    if len(section_text) > 100:  # Substantial sections only
                        sections.append({
                            'text': section_text,
                            'start': current_pos,
                            'end': match.start(),
                            'type': 'section'
                        })
                current_pos = match.start()
        
        # Add final section
        if current_pos < len(text):
            final_text = text[current_pos:].strip()
            if len(final_text) > 100:
                sections.append({
                    'text': final_text,
                    'start': current_pos,
                    'end': len(text),
                    'type': 'section'
                })
        
        # Strategy 2: If no clear sections, use sentence-based chunking
        if not sections:
            sentences = re.split(r'[.!?]+', text)
            current_chunk = ""
            chunk_start = 0
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                if len(current_chunk) + len(sentence) > 800 and current_chunk:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'start': chunk_start,
                        'end': chunk_start + len(current_chunk),
                        'type': 'paragraph',
                        'sentence_count': len(re.split(r'[.!?]+', current_chunk))
                    })
                    chunk_start += len(current_chunk)
                    current_chunk = sentence
                else:
                    current_chunk += " " + sentence if current_chunk else sentence
            
            # Add final chunk
            if current_chunk.strip():
                chunks.append({
                    'text': current_chunk.strip(),
                    'start': chunk_start,
                    'end': chunk_start + len(current_chunk),
                    'type': 'paragraph',
                    'sentence_count': len(re.split(r'[.!?]+', current_chunk))
                })
        else:
            chunks = sections
        
        logger.info(f"Document chunked into {len(chunks)} intelligent segments")
        return chunks
    
    def build_semantic_index(self, text: str):
        """Build optimized semantic search index"""
        try:
            if not self.embedding_model:
                logger.warning("No embedding model for semantic search")
                return
            
            # Create intelligent chunks
            chunk_info = self.intelligent_chunking(text)
            if not chunk_info:
                return
            
            self.text_chunks = [chunk['text'] for chunk in chunk_info]
            self.chunk_metadata = chunk_info
            
            # Generate embeddings
            logger.info("Generating embeddings for semantic search")
            embeddings = self.embedding_model.encode(self.text_chunks, show_progress_bar=False)
            
            # Build FAISS index
            if FAISS_AVAILABLE and len(embeddings) > 0:
                dimension = embeddings.shape[1]
                # Use IndexFlatIP for exact cosine similarity
                self.faiss_index = faiss.IndexFlatIP(dimension)
                
                # Normalize for cosine similarity
                faiss.normalize_L2(embeddings)
                self.faiss_index.add(embeddings.astype('float32'))
                
                logger.info(f"FAISS index built with {len(self.text_chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Failed to build semantic index: {e}")
    
    def semantic_retrieval(self, query: str, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """High-accuracy semantic retrieval"""
        try:
            if not self.embedding_model or not self.text_chunks:
                return self.fallback_search(query, top_k)
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            
            if FAISS_AVAILABLE and self.faiss_index:
                # Normalize query
                faiss.normalize_L2(query_embedding)
                
                # Search
                scores, indices = self.faiss_index.search(
                    query_embedding.astype('float32'), 
                    min(top_k, len(self.text_chunks))
                )
                
                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx < len(self.text_chunks) and score > 0.1:  # Threshold for relevance
                        results.append((
                            self.text_chunks[idx],
                            float(score),
                            self.chunk_metadata[idx] if idx < len(self.chunk_metadata) else {}
                        ))
                
                return results
            else:
                return self.fallback_search(query, top_k)
                
        except Exception as e:
            logger.error(f"Semantic retrieval failed: {e}")
            return self.fallback_search(query, top_k)
    
    def fallback_search(self, query: str, top_k: int) -> List[Tuple[str, float, Dict]]:
        """Enhanced fallback search"""
        if not self.text_chunks:
            return []
        
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        scored_chunks = []
        
        for i, chunk in enumerate(self.text_chunks):
            chunk_terms = set(re.findall(r'\b\w+\b', chunk.lower()))
            
            # Calculate multiple similarity scores
            jaccard = len(query_terms & chunk_terms) / len(query_terms | chunk_terms) if query_terms | chunk_terms else 0
            
            # Bonus for exact phrase matches
            phrase_bonus = 0
            query_words = query.lower().split()
            for j in range(len(query_words) - 1):
                phrase = f"{query_words[j]} {query_words[j+1]}"
                if phrase in chunk.lower():
                    phrase_bonus += 0.3
            
            # Bonus for numbers (important in policy documents)
            number_bonus = 0
            query_numbers = set(re.findall(r'\b\d+\b', query))
            chunk_numbers = set(re.findall(r'\b\d+\b', chunk))
            if query_numbers & chunk_numbers:
                number_bonus = 0.2
            
            total_score = jaccard + phrase_bonus + number_bonus
            
            if total_score > 0:
                metadata = self.chunk_metadata[i] if i < len(self.chunk_metadata) else {}
                scored_chunks.append((chunk, total_score, metadata))
        
        # Sort and return top results
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_k]

class CompetitionAnswerEngine:
    """
    Competition-focused answer generation for maximum scoring
    """
    
    def __init__(self):
        self.openai_client = None
        self.groq_client = None
        self.search_engine = CompetitionSemanticSearch()
        
        # Initialize LLM clients with optimal configurations
        if OPENAI_AVAILABLE and OPENAI_API_KEY:
            try:
                self.openai_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
                logger.info("OpenAI client ready for competition")
            except Exception as e:
                logger.error(f"OpenAI setup failed: {e}")
        
        if GROQ_AVAILABLE and GROQ_API_KEY:
            try:
                self.groq_client = Groq(api_key=GROQ_API_KEY)
                logger.info("Groq client ready for competition")
            except Exception as e:
                logger.error(f"Groq setup failed: {e}")
    
    def prepare_document(self, document_text: str):
        """Prepare document for competitive answering"""
        logger.info("Preparing document for competition-level accuracy")
        self.search_engine.build_semantic_index(document_text)
        self.full_document = document_text  # Keep full document for fallbacks
    
    async def generate_competition_answer(self, question: str) -> str:
        """Generate answer optimized for competition scoring"""
        try:
            # Multi-strategy answer generation for maximum accuracy
            
            # Strategy 1: Semantic search + LLM
            semantic_results = self.search_engine.semantic_retrieval(question, top_k=3)
            
            if semantic_results:
                # Use best chunks for context
                top_chunks = [chunk for chunk, score, metadata in semantic_results if score > 0.15]
                context = "\n\n".join(top_chunks[:2])  # Limit context for token efficiency
                
                if context:
                    # Try LLM generation
                    llm_answer = await self.llm_answer_generation(question, context)
                    if self.validate_answer_quality(llm_answer, question):
                        return llm_answer
            
            # Strategy 2: Full document search as fallback
            fallback_answer = self.comprehensive_fallback(question)
            if self.validate_answer_quality(fallback_answer, question):
                return fallback_answer
            
            # Strategy 3: Last resort
            return "Information about this topic is not available in the provided document."
            
        except Exception as e:
            logger.error(f"Competition answer generation failed: {e}")
            return "Unable to process this question due to system error."
    
    async def llm_answer_generation(self, question: str, context: str) -> str:
        """LLM generation optimized for competition accuracy"""
        
        # Ultra-optimized prompt for maximum accuracy
        system_prompt = """You are an expert document analyst for a competition scoring system. Your accuracy is critical for scoring.

SCORING RULES:
- Known documents have 0.5x weight, Unknown documents have 2.0x weight
- Each question has different weights based on complexity
- Wrong answers score 0 points regardless of weight
- Your goal is MAXIMUM ACCURACY to achieve highest score

ANSWER REQUIREMENTS:
1. Answer ONLY from the provided context - no external knowledge
2. Include ALL relevant details: numbers, periods, percentages, conditions
3. Use EXACT terminology from the document
4. If information is not in context, clearly state it's not available
5. Be comprehensive but precise
6. Include specific conditions, limitations, and requirements mentioned
7. Preserve exact numbers, dates, and percentages as written"""

        competition_prompt = f"""Document Context:
{context}

Question: {question}

Provide a detailed, accurate answer based ONLY on the context above. Include all specific details, numbers, conditions, and exact terminology from the document. This is for competition scoring where accuracy is critical."""

        # Try OpenAI first (typically more accurate)
        if self.openai_client:
            try:
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",  # Latest and most accurate
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": competition_prompt}
                    ],
                    max_tokens=1200,
                    temperature=0.0,  # Zero temperature for maximum consistency
                    top_p=0.1,  # Very focused sampling
                    frequency_penalty=0.0,
                    presence_penalty=0.0
                )
                
                answer = response.choices[0].message.content.strip()
                logger.info(f"OpenAI generated answer: {len(answer)} chars")
                return answer
                
            except Exception as e:
                logger.warning(f"OpenAI failed: {e}")
        
        # Try Groq as fallback
        if self.groq_client:
            try:
                response = self.groq_client.chat.completions.create(
                    model="llama-3.1-70b-versatile",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": competition_prompt}
                    ],
                    max_tokens=1200,
                    temperature=0.0
                )
                
                answer = response.choices[0].message.content.strip()
                logger.info(f"Groq generated answer: {len(answer)} chars")
                return answer
                
            except Exception as e:
                logger.warning(f"Groq failed: {e}")
        
        # Fallback to rule-based
        return self.rule_based_extraction(question, context)
    
    def comprehensive_fallback(self, question: str) -> str:
        """Comprehensive fallback using full document"""
        if not hasattr(self, 'full_document') or not self.full_document:
            return "Document not available for processing."
        
        # Advanced pattern matching for policy documents
        question_lower = question.lower()
        doc_lower = self.full_document.lower()
        
        # Extract key terms from question
        key_terms = []
        
        # Look for specific patterns
        if "grace period" in question_lower:
            key_terms.extend(["grace period", "grace", "days", "payment"])
        elif "waiting period" in question_lower:
            key_terms.extend(["waiting period", "waiting", "months", "years"])
        elif "maternity" in question_lower:
            key_terms.extend(["maternity", "pregnancy", "childbirth"])
        elif "discount" in question_lower:
            key_terms.extend(["discount", "ncd", "no claim"])
        elif "hospital" in question_lower and "define" in question_lower:
            key_terms.extend(["hospital", "institution", "beds", "qualified"])
        
        # Find sentences containing these terms
        sentences = re.split(r'[.!?]+', self.full_document)
        relevant_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
                
            sentence_lower = sentence.lower()
            match_count = sum(1 for term in key_terms if term in sentence_lower)
            
            if match_count >= 2 or any(term in sentence_lower for term in key_terms[:1]):
                relevant_sentences.append((sentence, match_count))
        
        if relevant_sentences:
            # Sort by relevance and combine best sentences
            relevant_sentences.sort(key=lambda x: x[1], reverse=True)
            best_sentences = [sent for sent, count in relevant_sentences[:3]]
            return '. '.join(best_sentences).strip() + '.'
        
        return "Information about this topic is not available in the provided document."
    
    def rule_based_extraction(self, question: str, context: str) -> str:
        """Rule-based extraction for high accuracy"""
        if not context:
            return "Information not available in the provided context."
        
        sentences = re.split(r'[.!?]+', context)
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        
        best_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 15:
                continue
                
            sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
            overlap = len(question_words & sentence_words)
            
            # Bonus for containing numbers (important in policy docs)
            number_bonus = 1 if re.search(r'\d+', sentence) else 0
            
            total_score = overlap + number_bonus
            if total_score >= 2:
                best_sentences.append((sentence, total_score))
        
        if best_sentences:
            best_sentences.sort(key=lambda x: x[1], reverse=True)
            return best_sentences[0][0].strip()
        
        # Return first substantial sentence as last resort
        for sentence in sentences:
            if len(sentence.strip()) > 30:
                return sentence.strip()
        
        return "Information about this topic is not available in the provided document."
    
    def validate_answer_quality(self, answer: str, question: str) -> bool:
        """Validate answer quality for competition"""
        if not answer or len(answer.strip()) < 10:
            return False
        
        # Check for generic fallback responses
        generic_responses = [
            "information about this topic is not available",
            "unable to process",
            "system error",
            "not found in the document"
        ]
        
        answer_lower = answer.lower()
        if any(generic in answer_lower for generic in generic_responses):
            return len(answer) > 50  # Accept longer explanatory responses
        
        # Check for relevance to question
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        answer_words = set(re.findall(r'\b\w+\b', answer.lower()))
        
        overlap = len(question_words & answer_words)
        return overlap >= 2  # At least 2 words should match

# Initialize global processors for competition
doc_processor = DocumentProcessor()
answer_engine = CompetitionAnswerEngine()

@app.post("/hackrx/run")
async def competition_endpoint(
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    """
    COMPETITION ENDPOINT - OPTIMIZED FOR MAXIMUM SCORING
    
    Scoring Strategy:
    - Known Documents: 0.5x weight → Focus on accuracy over speed
    - Unknown Documents: 2.0x weight → Maximum accuracy critical
    - Question weights vary → Every correct answer counts
    - Wrong answers = 0 points → Better to be conservative than wrong
    """
    competition_start = time.time()
    
    try:
        # Parse request
        request_data = await request.json()
        documents = request_data.get("documents", "")
        questions = request_data.get("questions", [])
        
        logger.info(f"COMPETITION REQUEST: {len(questions)} questions")
        logger.info(f"Document URL: {documents[:100]}...")
        
        # Document processing with maximum accuracy
        logger.info("Stage 1: Document Processing (Maximum Accuracy Mode)")
        document_text = await doc_processor.process_document(documents)
        
        if not document_text or len(document_text.strip()) < 100:
            logger.error("Document processing failed or insufficient content")
            return {
                "answers": ["Unable to extract sufficient information from the provided document."] * len(questions)
            }
        
        # Document analysis
        word_count = len(document_text.split())
        char_count = len(document_text)
        logger.info(f"Document processed: {char_count} characters, {word_count} words")
        
        # Prepare for answer generation
        logger.info("Stage 2: Preparing Competition Answer Engine")
        answer_engine.prepare_document(document_text)
        
        # Generate competition-optimized answers
        logger.info("Stage 3: Generating Competition Answers")
        answers = []
        
        for i, question in enumerate(questions):
            question_start = time.time()
            logger.info(f"Processing Q{i+1}/{len(questions)}: {question[:80]}...")
            
            try:
                # Generate answer with maximum accuracy focus
                answer = await answer_engine.generate_competition_answer(question)
                answers.append(answer)
                
                question_time = time.time() - question_start
                logger.info(f"Q{i+1} completed in {question_time:.2f}s - Length: {len(answer)} chars")
                logger.debug(f"Q{i+1} Answer Preview: {answer[:120]}...")
                
            except Exception as e:
                logger.error(f"Failed to process Q{i+1}: {e}")
                answers.append("Unable to process this question due to processing error.")
        
        # Competition performance metrics
        total_time = time.time() - competition_start
        avg_time_per_question = total_time / len(questions)
        
        logger.info(f"COMPETITION COMPLETED:")
        logger.info(f"- Total Time: {total_time:.2f} seconds")
        logger.info(f"- Average per Question: {avg_time_per_question:.2f} seconds")
        logger.info(f"- Total Answers Generated: {len(answers)}")
        logger.info(f"- Average Answer Length: {sum(len(a) for a in answers) / len(answers):.1f} chars")
        
        # Quality validation
        quality_metrics = {
            "non_empty_answers": sum(1 for a in answers if len(a.strip()) > 10),
            "detailed_answers": sum(1 for a in answers if len(a.strip()) > 50),
            "specific_answers": sum(1 for a in answers if any(char.isdigit() for char in a))
        }
        
        logger.info(f"Answer Quality: {quality_metrics}")
        
        return {"answers": answers}
        
    except Exception as e:
        logger.error(f"COMPETITION ENDPOINT CRITICAL FAILURE: {e}")
        
        # Emergency fallback to ensure valid response format
        try:
            question_count = len(request_data.get("questions", [])) if 'request_data' in locals() else 10
            emergency_response = [
                "Unable to process this question due to system error. Please try again."
            ] * question_count
            
            return {"answers": emergency_response}
            
        except:
            # Absolute last resort
            return {
                "answers": [
                    "System error occurred during processing."
                ]
            }

@app.get("/")
async def competition_status():
    """Competition system status and capabilities"""
    
    # Check system capabilities
    capabilities = {
        "document_processing": "Advanced PDF/DOCX extraction",
        "semantic_search": "Available" if SENTENCE_TRANSFORMERS_AVAILABLE else "Fallback mode",
        "vector_indexing": "FAISS available" if FAISS_AVAILABLE else "In-memory search",
        "llm_integration": {
            "openai_gpt4": "Available" if (OPENAI_AVAILABLE and OPENAI_API_KEY) else "Not configured", 
            "groq_llama": "Available" if (GROQ_AVAILABLE and GROQ_API_KEY) else "Not configured"
        },
        "fallback_systems": "Multi-layer fallbacks active"
    }
    
    # Calculate readiness score
    readiness_score = 0
    if doc_processor:
        readiness_score += 25
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        readiness_score += 25
    if FAISS_AVAILABLE:
        readiness_score += 20
    if (OPENAI_AVAILABLE and OPENAI_API_KEY) or (GROQ_AVAILABLE and GROQ_API_KEY):
        readiness_score += 30
    
    return {
        "message": "HackRx 6.0 - Competition-Optimized Document Processing System",
        "status": "COMPETITION READY" if readiness_score >= 70 else "PARTIALLY READY",
        "version": "6.0.0",
        "readiness_score": f"{readiness_score}/100",
        "capabilities": capabilities,
        "optimization_focus": [
            "Maximum accuracy for both known (0.5x) and unknown (2.0x) documents",
            "Advanced semantic search with FAISS indexing",
            "Multi-strategy answer generation with fallbacks", 
            "Comprehensive PDF extraction with error recovery",
            "Competition-specific prompt optimization",
            "Real-time performance monitoring"
        ],
        "competition_ready": readiness_score >= 70,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Detailed health check for competition deployment"""
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "6.0.0",
        "system_checks": {
            "document_processor": bool(doc_processor),
            "answer_engine": bool(answer_engine),
            "openai_client": bool(answer_engine.openai_client) if answer_engine else False,
            "groq_client": bool(answer_engine.groq_client) if answer_engine else False,
            "semantic_search": SENTENCE_TRANSFORMERS_AVAILABLE,
            "vector_search": FAISS_AVAILABLE
        }
    }
    
    # Overall health score
    checks = health_status["system_checks"]
    health_score = sum(1 for check in checks.values() if check) / len(checks) * 100
    
    health_status["health_score"] = f"{health_score:.1f}%"
    health_status["competition_ready"] = health_score >= 60
    
    return health_status

@app.get("/test-processing")
async def test_document_processing():
    """Test endpoint for validating document processing capabilities"""
    
    # Test with a simple document
    test_text = """
    SAMPLE POLICY DOCUMENT
    
    1. Grace Period: A grace period of thirty (30) days is provided for premium payment.
    
    2. Waiting Period: Pre-existing diseases have a waiting period of thirty-six (36) months.
    
    3. Maternity Coverage: Maternity expenses are covered after 24 months of continuous coverage.
    """
    
    try:
        # Test the answer engine
        answer_engine.prepare_document(test_text)
        
        test_questions = [
            "What is the grace period for premium payment?",
            "What is the waiting period for pre-existing diseases?"
        ]
        
        test_results = []
        for question in test_questions:
            answer = await answer_engine.generate_competition_answer(question)
            test_results.append({
                "question": question,
                "answer": answer,
                "answer_length": len(answer)
            })
        
        return {
            "status": "processing_test_successful",
            "test_document_length": len(test_text),
            "questions_processed": len(test_questions),
            "results": test_results
        }
        
    except Exception as e:
        return {
            "status": "processing_test_failed", 
            "error": str(e)
        }

# Error handlers for competition stability
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions gracefully during competition"""
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return {"error": f"HTTP {exc.status_code}: {exc.detail}"}

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions during competition"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return {"error": "Internal server error occurred during processing"}

# Startup event for competition preparation
@app.on_event("startup")
async def startup_event():
    """Initialize system for competition"""
    logger.info("🚀 Starting HackRx 6.0 Competition System")
    logger.info("📊 System capabilities check:")
    logger.info(f"   - Document Processing: Ready")
    logger.info(f"   - Semantic Search: {'Ready' if SENTENCE_TRANSFORMERS_AVAILABLE else 'Fallback'}")
    logger.info(f"   - Vector Search: {'FAISS Ready' if FAISS_AVAILABLE else 'In-memory'}")
    logger.info(f"   - OpenAI: {'Ready' if (OPENAI_AVAILABLE and OPENAI_API_KEY) else 'Not configured'}")
    logger.info(f"   - Groq: {'Ready' if (GROQ_AVAILABLE and GROQ_API_KEY) else 'Not configured'}")
    logger.info("🏆 Competition optimization: ACTIVE")
    logger.info("💯 Maximum accuracy mode: ENABLED")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)