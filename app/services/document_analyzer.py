# app/services/document_analyzer.py
"""
Advanced Document Analyzer Service
Handles ANY type of document with intelligent content analysis
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class DocumentType(Enum):
    INSURANCE_POLICY = "insurance_policy"
    LEGAL_CONTRACT = "legal_contract"
    MEDICAL_DOCUMENT = "medical_document"
    FINANCIAL_REPORT = "financial_report"
    TECHNICAL_MANUAL = "technical_manual"
    GENERAL_DOCUMENT = "general_document"

@dataclass
class DocumentMetadata:
    """Metadata about the analyzed document"""
    document_type: DocumentType
    total_pages: int
    total_words: int
    key_sections: List[str]
    confidence_score: float

class AdvancedDocumentAnalyzer:
    """
    Production-grade document analyzer that adapts to any document type
    """
    
    def __init__(self):
        self.document_type_indicators = {
            DocumentType.INSURANCE_POLICY: [
                'policy', 'premium', 'coverage', 'claim', 'insured', 'beneficiary',
                'deductible', 'rider', 'exclusion', 'waiting period', 'sum insured'
            ],
            DocumentType.LEGAL_CONTRACT: [
                'contract', 'agreement', 'party', 'clause', 'terms', 'conditions',
                'liability', 'breach', 'termination', 'jurisdiction'
            ],
            DocumentType.MEDICAL_DOCUMENT: [
                'patient', 'diagnosis', 'treatment', 'medication', 'symptoms',
                'medical', 'hospital', 'doctor', 'clinical', 'therapeutic'
            ],
            DocumentType.FINANCIAL_REPORT: [
                'revenue', 'profit', 'loss', 'balance sheet', 'assets', 'liabilities',
                'cash flow', 'investment', 'financial', 'quarterly'
            ],
            DocumentType.TECHNICAL_MANUAL: [
                'manual', 'instructions', 'procedure', 'specifications', 'operation',
                'maintenance', 'technical', 'system', 'configuration'
            ]
        }
        
        self.question_type_patterns = {
            'definition': [
                r'what is', r'define', r'meaning of', r'definition of'
            ],
            'yes_no': [
                r'does', r'is', r'are', r'can', r'will', r'has'
            ],
            'amount': [
                r'how much', r'what amount', r'cost', r'price', r'fee'
            ],
            'time_period': [
                r'how long', r'when', r'duration', r'period', r'time'
            ],
            'process': [
                r'how to', r'process', r'procedure', r'steps'
            ],
            'coverage': [
                r'cover', r'include', r'benefit', r'eligible'
            ]
        }
    
    def analyze_document(self, content: str) -> DocumentMetadata:
        """Analyze document and determine its type and characteristics"""
        content_lower = content.lower()
        word_count = len(content.split())
        
        # Determine document type
        doc_type = self._determine_document_type(content_lower)
        
        # Extract key sections
        key_sections = self._extract_key_sections(content, doc_type)
        
        # Calculate confidence score
        confidence = self._calculate_confidence_score(content_lower, doc_type)
        
        # Count pages
        page_count = max(1, content.count('[Page') + content.count('Page '))
        
        return DocumentMetadata(
            document_type=doc_type,
            total_pages=page_count,
            total_words=word_count,
            key_sections=key_sections,
            confidence_score=confidence
        )
    
    def _determine_document_type(self, content_lower: str) -> DocumentType:
        """Determine the type of document based on content analysis"""
        type_scores = {}
        
        for doc_type, indicators in self.document_type_indicators.items():
            score = sum(1 for indicator in indicators if indicator in content_lower)
            type_scores[doc_type] = score
        
        # Return the type with highest score, or GENERAL_DOCUMENT if no clear match
        if type_scores:
            best_type = max(type_scores, key=type_scores.get)
            if type_scores[best_type] > 2:  # Require at least 3 matching indicators
                return best_type
        
        return DocumentType.GENERAL_DOCUMENT
    
    def _extract_key_sections(self, content: str, doc_type: DocumentType) -> List[str]:
        """Extract key sections based on document type"""
        sections = []
        
        if doc_type == DocumentType.INSURANCE_POLICY:
            section_patterns = [
                r'(coverage|benefits?|exclusions?|definitions?|claims?|premiums?)',
                r'(waiting period|grace period|policy terms|conditions)',
                r'(table of benefits|schedule|riders?)'
            ]
        elif doc_type == DocumentType.LEGAL_CONTRACT:
            section_patterns = [
                r'(terms and conditions|obligations|responsibilities)',
                r'(termination|breach|liability|damages)',
                r'(jurisdiction|governing law|dispute resolution)'
            ]
        elif doc_type == DocumentType.FINANCIAL_REPORT:
            section_patterns = [
                r'(executive summary|financial highlights)',
                r'(income statement|balance sheet|cash flow)',
                r'(assets|liabilities|equity|revenue)'
            ]
        else:
            # Generic section detection
            section_patterns = [
                r'(introduction|overview|summary)',
                r'(terms|conditions|requirements)',
                r'(procedures?|processes?|instructions?)'
            ]
        
        content_lower = content.lower()
        for pattern in section_patterns:
            matches = re.findall(pattern, content_lower)
            sections.extend(matches)
        
        return list(set(sections))  # Remove duplicates
    
    def _calculate_confidence_score(self, content_lower: str, doc_type: DocumentType) -> float:
        """Calculate confidence score for document type classification"""
        if doc_type == DocumentType.GENERAL_DOCUMENT:
            return 0.5  # Low confidence for generic classification
        
        indicators = self.document_type_indicators[doc_type]
        matches = sum(1 for indicator in indicators if indicator in content_lower)
        
        # Calculate confidence based on matches and content length
        base_confidence = min(matches / len(indicators), 1.0)
        
        # Adjust based on content length (more content = higher confidence)
        length_factor = min(len(content_lower) / 10000, 1.0)  # Normalize to 10k chars
        
        return (base_confidence * 0.7) + (length_factor * 0.3)
    
    def classify_question_type(self, question: str) -> str:
        """Classify the type of question being asked"""
        question_lower = question.lower()
        
        for q_type, patterns in self.question_type_patterns.items():
            for pattern in patterns:
                if re.search(pattern, question_lower):
                    return q_type
        
        return 'general'
    
    def extract_context_for_question(self, content: str, question: str, 
                                   doc_metadata: DocumentMetadata) -> str:
        """Extract the most relevant context for a specific question"""
        question_type = self.classify_question_type(question)
        question_keywords = self._extract_question_keywords(question)
        
        # Get relevant paragraphs
        paragraphs = self._split_into_paragraphs(content)
        scored_paragraphs = []
        
        for paragraph in paragraphs:
            score = self._score_paragraph_relevance(
                paragraph, question_keywords, question_type, doc_metadata.document_type
            )
            if score > 0:
                scored_paragraphs.append((paragraph, score))
        
        # Sort by relevance and combine top paragraphs
        scored_paragraphs.sort(key=lambda x: x[1], reverse=True)
        
        # Take top 3 paragraphs, but limit total length
        relevant_context = ""
        for paragraph, score in scored_paragraphs[:3]:
            if len(relevant_context + paragraph) < 5000:  # Limit context size
                relevant_context += paragraph + "\n\n"
            else:
                break
        
        return relevant_context.strip()
    
    def _extract_question_keywords(self, question: str) -> List[str]:
        """Extract meaningful keywords from the question"""
        stop_words = {
            'what', 'is', 'the', 'are', 'does', 'this', 'that', 'these', 'those',
            'for', 'to', 'of', 'in', 'and', 'or', 'a', 'an', 'how', 'when', 
            'where', 'which', 'any', 'there', 'with', 'from', 'by', 'at', 'on'
        }
        
        words = re.findall(r'\b\w+\b', question.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Add important phrases
        important_phrases = [
            'grace period', 'waiting period', 'no claim discount', 'pre-existing',
            'sum insured', 'room rent', 'icu charges', 'organ donor'
        ]
        
        question_lower = question.lower()
        for phrase in important_phrases:
            if phrase in question_lower:
                keywords.extend(phrase.split())
        
        return list(set(keywords))
    
    def _split_into_paragraphs(self, content: str) -> List[str]:
        """Split content into meaningful paragraphs"""
        # Split by double newlines or clear paragraph breaks
        paragraphs = re.split(r'\n\s*\n|\n\s*[A-Z][A-Z\s]*\n', content)
        
        # Filter out very short paragraphs and clean
        clean_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if len(para) > 50 and not self._is_header_or_footer(para):
                clean_paragraphs.append(para)
        
        return clean_paragraphs
    
    def _is_header_or_footer(self, text: str) -> bool:
        """Check if text is likely a header or footer"""
        text_lower = text.lower()
        
        header_footer_indicators = [
            'page ', 'national insurance', 'premises no', 'uin:', 'cin -',
            'table of contents', 'index', 'appendix', 'copyright',
            'all rights reserved', 'confidential'
        ]
        
        return any(indicator in text_lower for indicator in header_footer_indicators)
    
    def _score_paragraph_relevance(self, paragraph: str, keywords: List[str], 
                                 question_type: str, doc_type: DocumentType) -> int:
        """Score how relevant a paragraph is to the question"""
        score = 0
        paragraph_lower = paragraph.lower()
        
        # Keyword matching
        for keyword in keywords:
            if keyword in paragraph_lower:
                score += 3
        
        # Question type specific scoring
        if question_type == 'definition' and any(word in paragraph_lower for word in ['means', 'defined as', 'refers to']):
            score += 10
        elif question_type == 'yes_no' and any(word in paragraph_lower for word in ['cover', 'include', 'exclude', 'benefit']):
            score += 8
        elif question_type == 'amount' and re.search(r'inr|rs\.|₹|\d+', paragraph_lower):
            score += 8
        elif question_type == 'time_period' and any(word in paragraph_lower for word in ['days', 'months', 'years', 'period']):
            score += 8
        
        # Document type specific scoring
        if doc_type == DocumentType.INSURANCE_POLICY:
            if any(word in paragraph_lower for word in ['policy', 'coverage', 'benefit', 'claim']):
                score += 5
        
        # Length penalty for very long paragraphs (prefer concise information)
        if len(paragraph) > 1000:
            score -= 2
        
        # Numerical data bonus
        if re.search(r'\d+', paragraph_lower):
            score += 2
        
        return score
    
    def get_document_specific_prompts(self, doc_type: DocumentType) -> Dict[str, str]:
        """Get document-type specific prompts for better LLM performance"""
        if doc_type == DocumentType.INSURANCE_POLICY:
            return {
                'system_prompt': "You are an expert insurance policy analyzer. Focus on extracting specific policy terms, coverage details, exclusions, and numerical values.",
                'question_prefix': "Based on this insurance policy document:",
                'answer_format': "Provide specific policy details including amounts, time periods, and conditions."
            }
        elif doc_type == DocumentType.LEGAL_CONTRACT:
            return {
                'system_prompt': "You are an expert legal document analyzer. Focus on contractual obligations, terms, conditions, and legal requirements.",
                'question_prefix': "Based on this legal contract:",
                'answer_format': "Provide specific legal terms, obligations, and conditions."
            }
        elif doc_type == DocumentType.FINANCIAL_REPORT:
            return {
                'system_prompt': "You are an expert financial analyst. Focus on financial metrics, amounts, percentages, and performance indicators.",
                'question_prefix': "Based on this financial document:",
                'answer_format': "Provide specific financial figures, percentages, and metrics."
            }
        else:
            return {
                'system_prompt': "You are an expert document analyzer. Focus on extracting factual information accurately.",
                'question_prefix': "Based on this document:",
                'answer_format': "Provide specific and accurate information from the document."
            }