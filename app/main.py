"""
app/main.py - Competition LLM Features with Groq Primary
Optimized for free tier usage and maximum accuracy
"""

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

# Groq LLM (Primary - Free tier)
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# OpenAI (Secondary - if available)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Semantic search (Optional)
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

# Environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class DocumentProcessor:
    """
    Competition-grade document processor
    Optimized for maximum text extraction accuracy
    """
    
    def __init__(self):
        self.session_timeout = aiohttp.ClientTimeout(total=120)
        self.cache = {}
        logger.info("DocumentProcessor initialized for competition")
    
    async def download_document(self, url: str) -> Optional[bytes]:
        """Download with aggressive retry strategy"""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        if url_hash in self.cache:
            logger.info("Document retrieved from cache")
            return self.cache[url_hash]
        
        max_retries = 5
        backoff_delays = [1, 2, 4, 8, 16]
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Downloading document (attempt {attempt + 1}/{max_retries})")
                
                async with aiohttp.ClientSession(
                    timeout=self.session_timeout,
                    connector=aiohttp.TCPConnector(limit=10)
                ) as session:
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
                    
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
                    await asyncio.sleep(backoff_delays[attempt])
        
        logger.error("All download attempts failed")
        return None
    
    async def extract_text_from_pdf(self, pdf_data: bytes) -> str:
        """Advanced PDF text extraction with multiple strategies"""
        try:
            logger.info("Starting advanced PDF extraction")
            pdf_file = io.BytesIO(pdf_data)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text_parts = []
            total_pages = len(pdf_reader.pages)
            logger.info(f"Processing {total_pages} pages")
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = ""
                    
                    # Primary extraction method
                    try:
                        text1 = page.extract_text()
                        if text1 and len(text1.strip()) > 10:
                            page_text = text1
                    except:
                        pass
                    
                    # Alternative extraction for difficult pages
                    if not page_text:
                        try:
                            if hasattr(page, 'extract_text'):
                                text2 = page.extract_text(space_width=200)
                                if text2 and len(text2.strip()) > 10:
                                    page_text = text2
                        except:
                            pass
                    
                    if page_text:
                        cleaned_text = self.clean_pdf_text(page_text)
                        if cleaned_text:
                            text_parts.append(cleaned_text)
                            logger.debug(f"Page {page_num + 1}: {len(cleaned_text)} chars")
                    else:
                        logger.warning(f"No text from page {page_num + 1}")
                        
                except Exception as e:
                    logger.warning(f"Error on page {page_num + 1}: {e}")
                    continue
            
            full_text = self.combine_pages(text_parts)
            logger.info(f"PDF extraction complete: {len(full_text)} chars from {len(text_parts)} pages")
            
            return full_text
            
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            return ""
    
    def clean_pdf_text(self, text: str) -> str:
        """Clean PDF text for maximum readability"""
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
        
        # Remove artifacts
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip page numbers
            if re.match(r'^\d{1,3}$', line):
                continue
            
            # Skip very short lines
            if len(line) < 3:
                continue
                
            # Skip special character lines
            if re.match(r'^[^\w\s]*$', line):
                continue
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def combine_pages(self, text_parts: List[str]) -> str:
        """Intelligently combine PDF pages"""
        if not text_parts:
            return ""
        
        combined = []
        for i, part in enumerate(text_parts):
            if i > 0:
                prev_part = text_parts[i-1].strip()
                if prev_part and not prev_part[-1] in '.!?':
                    combined.append(' ' + part)
                else:
                    combined.append('\n\n' + part)
            else:
                combined.append(part)
        
        return ''.join(combined)
    
    async def process_document(self, document_url: str) -> str:
        """Main document processing method"""
        start_time = time.time()
        
        try:
            document_data = await self.download_document(document_url)
            if not document_data:
                return ""
            
            if document_data.startswith(b'%PDF'):
                logger.info("Processing as PDF")
                extracted_text = await self.extract_text_from_pdf(document_data)
            elif document_data.startswith(b'PK'):
                logger.info("Processing as DOCX")
                extracted_text = self.extract_text_from_docx(document_data)
            else:
                logger.info("Processing as plain text")
                extracted_text = document_data.decode('utf-8', errors='ignore')
            
            final_text = self.final_cleanup(extracted_text)
            
            processing_time = time.time() - start_time
            logger.info(f"Document processed in {processing_time:.2f}s: {len(final_text)} chars")
            
            return final_text
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            return ""
    
    def extract_text_from_docx(self, docx_data: bytes) -> str:
        """Extract text from DOCX documents"""
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
                        text_parts.append(' | '.join(row_text))
            
            return '\n'.join(text_parts)
            
        except Exception as e:
            logger.error(f"DOCX extraction failed: {e}")
            return ""
    
    def final_cleanup(self, text: str) -> str:
        """Final text cleanup and validation"""
        if not text or len(text.strip()) < 50:
            return text
        
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        word_count = len(text.split())
        char_count = len(text)
        logger.info(f"Final text: {char_count} chars, {word_count} words")
        
        return text

class GroqSemanticSearch:
    """
    Semantic search optimized for Groq LLM integration
    """
    
    def __init__(self):
        self.embedding_model = None
        self.text_chunks = []
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # Use lightweight model for better performance
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Semantic search model loaded")
            except Exception as e:
                logger.warning(f"Semantic search not available: {e}")
    
    def chunk_document(self, text: str) -> List[str]:
        """Chunk document for better semantic search"""
        if not text:
            return []
        
        # Strategy: Split by sentences but keep reasonable chunk sizes
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if len(current_chunk) + len(sentence) > 600 and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        logger.info(f"Document chunked into {len(chunks)} parts")
        return chunks
    
    def build_index(self, text: str):
        """Build search index from document"""
        self.text_chunks = self.chunk_document(text)
        logger.info(f"Search index built with {len(self.text_chunks)} chunks")
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Search for relevant chunks"""
        if not self.text_chunks:
            return []
        
        if self.embedding_model:
            try:
                # Semantic search
                query_embedding = self.embedding_model.encode([query])
                chunk_embeddings = self.embedding_model.encode(self.text_chunks)
                
                similarities = np.dot(query_embedding, chunk_embeddings.T)[0]
                top_indices = np.argsort(similarities)[-top_k:][::-1]
                
                results = [(self.text_chunks[idx], similarities[idx]) for idx in top_indices]
                return results
            except:
                pass
        
        # Fallback to keyword search
        return self.keyword_search(query, top_k)
    
    def keyword_search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """Keyword-based search fallback"""
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        scored_chunks = []
        
        for chunk in self.text_chunks:
            chunk_words = set(re.findall(r'\b\w+\b', chunk.lower()))
            
            # Calculate similarity
            intersection = len(query_words & chunk_words)
            union = len(query_words | chunk_words)
            jaccard = intersection / union if union > 0 else 0
            
            # Bonus for exact phrases
            phrase_bonus = 0
            query_text = query.lower()
            if len(query_text) > 10:
                query_parts = query_text.split()
                for i in range(len(query_parts) - 1):
                    phrase = f"{query_parts[i]} {query_parts[i+1]}"
                    if phrase in chunk.lower():
                        phrase_bonus += 0.2
            
            # Bonus for numbers (important in policy docs)
            number_bonus = 0.1 if re.search(r'\d+', chunk) else 0
            
            total_score = jaccard + phrase_bonus + number_bonus
            if total_score > 0:
                scored_chunks.append((chunk, total_score))
        
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_k]

class CompetitionAnswerEngine:
    """
    Competition answer engine with Groq LLM primary
    Optimized for free tier usage and maximum accuracy
    """
    
    def __init__(self):
        self.groq_client = None
        self.openai_client = None
        self.search_engine = GroqSemanticSearch()
        
        # Initialize Groq (Primary)
        if GROQ_AVAILABLE and GROQ_API_KEY:
            try:
                self.groq_client = Groq(api_key=GROQ_API_KEY)
                logger.info("Groq client initialized (Primary LLM)")
            except Exception as e:
                logger.error(f"Groq initialization failed: {e}")
        
        # Initialize OpenAI (Secondary)
        if OPENAI_AVAILABLE and OPENAI_API_KEY:
            try:
                self.openai_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
                logger.info("OpenAI client initialized (Secondary LLM)")
            except Exception as e:
                logger.error(f"OpenAI initialization failed: {e}")
    
    def prepare_document(self, document_text: str):
        """Prepare document for competition answering"""
        logger.info("Preparing document for Groq-optimized processing")
        self.search_engine.build_index(document_text)
        self.full_document = document_text
    
    async def generate_competition_answer(self, question: str) -> str:
        """Generate answer optimized for competition scoring"""
        try:
            # Strategy 1: Semantic search + Groq LLM
            relevant_chunks = self.search_engine.search(question, top_k=3)
            
            if relevant_chunks:
                best_chunks = [chunk for chunk, score in relevant_chunks if score > 0.1]
                context = "\n\n".join(best_chunks[:2])  # Limit for token efficiency
                
                if context:
                    groq_answer = await self.groq_generation(question, context)
                    if self.is_valid_answer(groq_answer, question):
                        return groq_answer
            
            # Strategy 2: Full document fallback
            fallback_answer = self.document_fallback(question)
            if self.is_valid_answer(fallback_answer, question):
                return fallback_answer
            
            return "Information about this topic is not available in the provided document."
            
        except Exception as e:
            logger.error(f"Competition answer generation failed: {e}")
            return "Unable to process this question due to system error."
    
    async def groq_generation(self, question: str, context: str) -> str:
        """Generate answer using Groq LLM (Primary)"""
        
        # Groq-optimized prompt for competition accuracy
        system_prompt = """You are a document analyst for a competition scoring system. Maximum accuracy is critical.

COMPETITION RULES:
- Answer ONLY from the provided context
- Include ALL specific details: numbers, periods, percentages, conditions
- Use EXACT terminology from the document
- Be comprehensive but precise
- If info not in context, state it clearly

ACCURACY IS CRITICAL FOR SCORING."""

        user_prompt = f"""Document Context:
{context}

Question: {question}

Provide a detailed, accurate answer based ONLY on the context above. Include all specific details and exact terminology from the document."""

        # Try Groq first (Primary LLM)
        if self.groq_client:
            try:
                response = self.groq_client.chat.completions.create(
                    model="llama-3.1-70b-versatile",  # Best free model
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=800,
                    temperature=0.0,  # Maximum consistency
                    top_p=0.1
                )
                
                answer = response.choices[0].message.content.strip()
                logger.info(f"Groq generated answer: {len(answer)} chars")
                return answer
                
            except Exception as e:
                logger.warning(f"Groq generation failed: {e}")
        
        # Try OpenAI as fallback
        if self.openai_client:
            try:
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=800,
                    temperature=0.0
                )
                
                answer = response.choices[0].message.content.strip()
                logger.info(f"OpenAI fallback answer: {len(answer)} chars")
                return answer
                
            except Exception as e:
                logger.warning(f"OpenAI fallback failed: {e}")
        
        # Rule-based fallback
        return self.rule_based_extraction(question, context)
    
    def document_fallback(self, question: str) -> str:
        """Comprehensive fallback using full document"""
        if not hasattr(self, 'full_document') or not self.full_document:
            return "Document not available for processing."
        
        question_lower = question.lower()
        
        # Extract key terms based on question type
        key_terms = []
        
        if "grace period" in question_lower:
            key_terms.extend(["grace period", "grace", "days", "payment", "premium"])
        elif "waiting period" in question_lower:
            key_terms.extend(["waiting period", "waiting", "months", "years", "continuous"])
        elif "maternity" in question_lower:
            key_terms.extend(["maternity", "pregnancy", "childbirth", "delivery"])
        elif "discount" in question_lower or "ncd" in question_lower:
            key_terms.extend(["discount", "ncd", "no claim", "premium"])
        elif "hospital" in question_lower and "define" in question_lower:
            key_terms.extend(["hospital", "institution", "beds", "qualified", "medical"])
        elif "ayush" in question_lower:
            key_terms.extend(["ayush", "ayurveda", "yoga", "naturopathy", "unani", "siddha", "homeopathy"])
        elif "room rent" in question_lower or "icu" in question_lower:
            key_terms.extend(["room rent", "icu", "charges", "sum insured", "limit"])
        elif "cataract" in question_lower:
            key_terms.extend(["cataract", "surgery", "waiting", "period"])
        elif "organ donor" in question_lower:
            key_terms.extend(["organ donor", "transplantation", "harvesting", "medical expenses"])
        elif "health check" in question_lower:
            key_terms.extend(["health check", "preventive", "reimbursement", "policy years"])
        
        # Find relevant sentences
        sentences = re.split(r'[.!?]+', self.full_document)
        relevant_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
                
            sentence_lower = sentence.lower()
            match_count = sum(1 for term in key_terms if term in sentence_lower)
            
            # Also check for any key term presence
            if match_count >= 1 or any(term in sentence_lower for term in key_terms):
                relevant_sentences.append((sentence, match_count))
        
        if relevant_sentences:
            # Sort by relevance and combine best sentences
            relevant_sentences.sort(key=lambda x: x[1], reverse=True)
            best_sentences = [sent for sent, count in relevant_sentences[:3] if count > 0]
            
            if best_sentences:
                return '. '.join(best_sentences).strip() + '.'
        
        return "Information about this topic is not available in the provided document."
    
    def rule_based_extraction(self, question: str, context: str) -> str:
        """Enhanced rule-based extraction"""
        if not context:
            return "Information not available in the provided context."
        
        sentences = re.split(r'[.!?]+', context)
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        
        # Remove common stop words from question
        stop_words = {'what', 'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        question_words = question_words - stop_words
        
        best_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 15:
                continue
                
            sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
            overlap = len(question_words & sentence_words)
            
            # Bonus scoring
            number_bonus = 2 if re.search(r'\d+', sentence) else 0
            length_bonus = 1 if len(sentence) > 50 else 0
            
            total_score = overlap + number_bonus + length_bonus
            if total_score >= 2:
                best_sentences.append((sentence, total_score))
        
        if best_sentences:
            best_sentences.sort(key=lambda x: x[1], reverse=True)
            return best_sentences[0][0].strip()
        
        # Final fallback - return first substantial sentence
        for sentence in sentences:
            if len(sentence.strip()) > 30:
                return sentence.strip()
        
        return "Information about this topic is not available in the provided document."
    
    def is_valid_answer(self, answer: str, question: str) -> bool:
        """Validate answer quality for competition"""
        if not answer or len(answer.strip()) < 10:
            return False
        
        # Check for generic responses
        generic_phrases = [
            "information about this topic is not available",
            "unable to process",  
            "system error",
            "not found in the document",
            "context above"
        ]
        
        answer_lower = answer.lower()
        
        # If it's a generic response, only accept if it's detailed
        if any(phrase in answer_lower for phrase in generic_phrases):
            return len(answer) > 60
        
        # Check relevance
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        answer_words = set(re.findall(r'\b\w+\b', answer.lower()))
        
        # Remove stop words
        stop_words = {'what', 'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        question_words = question_words - stop_words
        
        overlap = len(question_words & answer_words)
        return overlap >= 1  # At least 1 meaningful word should match

# Initialize global processors
doc_processor = DocumentProcessor()
answer_engine = CompetitionAnswerEngine()

# Export for use by run.py
__all__ = ['DocumentProcessor', 'CompetitionAnswerEngine', 'doc_processor', 'answer_engine']