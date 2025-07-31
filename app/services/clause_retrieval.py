"""
HackRx 6.0 - Advanced Clause Retrieval and Matching Service
Implements semantic clause identification and matching
"""

import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Clause:
    """Represents a document clause with metadata"""
    id: str
    text: str
    clause_type: str
    section: str
    keywords: List[str]
    importance_score: float
    semantic_embedding: np.ndarray = None
    parent_section: str = ""
    sub_clauses: List[str] = None

class ClauseRetrievalService:
    """
    Advanced clause retrieval with semantic matching and categorization
    """
    
    def __init__(self, embedding_model: SentenceTransformer = None):
        self.embedding_model = embedding_model
        self.clauses: List[Clause] = []
        self.clause_patterns = {
            'payment_terms': [
                r'grace period.*?(\d+)\s*day',
                r'premium.*?payment.*?(\d+)\s*day',
                r'due date.*?(\d+)\s*day'
            ],
            'waiting_periods': [
                r'waiting period.*?(\d+)\s*(month|year)',
                r'pre-existing.*?(\d+)\s*(month|year)',
                r'coverage.*?after.*?(\d+)\s*(month|year)'
            ],
            'coverage_conditions': [
                r'covered.*?provided',
                r'eligible.*?condition',
                r'benefit.*?subject to'
            ],
            'exclusions': [
                r'not covered',
                r'excluded.*?from',
                r'does not include'
            ],
            'definitions': [
                r'means.*?(?:shall|will|is)',
                r'defined as',
                r'definition.*?(?:of|for)'
            ],
            'benefits': [
                r'benefit.*?(?:of|include)',
                r'covered.*?(?:expense|treatment)',
                r'reimburse.*?(?:for|up to)'
            ]
        }
        self.is_initialized = False
    
    async def initialize(self, embedding_model: SentenceTransformer = None):
        """Initialize the clause retrieval service"""
        try:
            if embedding_model:
                self.embedding_model = embedding_model
            elif not self.embedding_model:
                self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
            self.is_initialized = True
            print("✅ Clause Retrieval Service initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize clause retrieval: {e}")
            raise
    
    def extract_clauses(self, document_text: str, document_metadata: Dict = None) -> List[Clause]:
        """
        Extract and classify clauses from document text
        """
        try:
            print("📋 Extracting clauses from document...")
            
            # Split document into potential clauses
            raw_clauses = self._split_into_clauses(document_text)
            
            clauses = []
            for i, clause_text in enumerate(raw_clauses):
                if len(clause_text.strip()) < 50:  # Skip very short clauses
                    continue
                
                # Classify clause
                clause_type = self._classify_clause(clause_text)
                
                # Extract keywords
                keywords = self._extract_clause_keywords(clause_text)
                
                # Calculate importance score
                importance = self._calculate_importance(clause_text, clause_type)
                
                # Identify section
                section = self._identify_section(clause_text, i)
                
                # Create clause object
                clause = Clause(
                    id=f"clause_{i:03d}",
                    text=clause_text.strip(),
                    clause_type=clause_type,
                    section=section,
                    keywords=keywords,
                    importance_score=importance,
                    sub_clauses=[]
                )
                
                # Generate semantic embedding
                if self.embedding_model:
                    clause.semantic_embedding = self.embedding_model.encode(
                        clause_text, convert_to_tensor=False, normalize_embeddings=True
                    )
                
                clauses.append(clause)
            
            self.clauses = clauses
            print(f"✅ Extracted {len(clauses)} clauses with semantic embeddings")
            
            return clauses
            
        except Exception as e:
            logger.error(f"Clause extraction failed: {e}")
            return []
    
    def _split_into_clauses(self, text: str) -> List[str]:
        """Split document into logical clauses"""
        # Strategy 1: Split by numbered items
        numbered_pattern = r'(?=\d+\.?\s+[A-Z])'
        numbered_clauses = re.split(numbered_pattern, text)
        
        # Strategy 2: Split by paragraph breaks and sentence patterns
        paragraph_clauses = []
        for clause in numbered_clauses:
            # Further split long paragraphs
            sentences = re.split(r'(?<=\.)\s+(?=[A-Z])', clause)
            current_clause = ""
            
            for sentence in sentences:
                if len(current_clause) + len(sentence) > 500:  # Max clause length
                    if current_clause:
                        paragraph_clauses.append(current_clause)
                    current_clause = sentence
                else:
                    current_clause += " " + sentence
            
            if current_clause:
                paragraph_clauses.append(current_clause)
        
        # Strategy 3: Split by specific legal/insurance markers
        final_clauses = []
        legal_markers = [
            r'(?=provided that)',
            r'(?=subject to)',
            r'(?=in case of)',
            r'(?=for the purpose of)',
            r'(?=means and includes)'
        ]
        
        for clause in paragraph_clauses:
            sub_clauses = [clause]
            for pattern in legal_markers:
                new_sub_clauses = []
                for sub_clause in sub_clauses:
                    new_sub_clauses.extend(re.split(pattern, sub_clause))
                sub_clauses = new_sub_clauses
            
            final_clauses.extend(sub_clauses)
        
        return [c.strip() for c in final_clauses if c.strip()]
    
    def _classify_clause(self, clause_text: str) -> str:
        """Classify clause type using pattern matching"""
        clause_lower = clause_text.lower()
        
        # Check each pattern category
        for clause_type, patterns in self.clause_patterns.items():
            for pattern in patterns:
                if re.search(pattern, clause_lower):
                    return clause_type
        
        # Additional semantic classification
        if any(word in clause_lower for word in ['hospital', 'institution', 'medical facility']):
            return 'definitions'
        elif any(word in clause_lower for word in ['claim', 'discount', 'benefit']):
            return 'benefits'
        elif any(word in clause_lower for word in ['ayush', 'alternative', 'homeopathy']):
            return 'alternative_medicine'
        elif any(word in clause_lower for word in ['room rent', 'icu', 'charges']):
            return 'cost_limits'
        
        return 'general'
    
    def _extract_clause_keywords(self, clause_text: str) -> List[str]:
        """Extract important keywords from clause"""
        # Insurance/legal specific keywords
        important_terms = [
            'grace period', 'waiting period', 'pre-existing', 'coverage', 'benefit',
            'premium', 'claim', 'discount', 'hospital', 'medical', 'treatment',
            'exclude', 'include', 'provided', 'subject to', 'means', 'defined',
            'maternity', 'surgery', 'organ donor', 'ayush', 'room rent', 'icu'
        ]
        
        keywords = []
        clause_lower = clause_text.lower()
        
        for term in important_terms:
            if term in clause_lower:
                keywords.append(term)
        
        # Extract numbers (important for periods, limits, etc.)
        numbers = re.findall(r'\d+', clause_text)
        for num in numbers[:3]:  # Top 3 numbers
            keywords.append(f"number_{num}")
        
        # Extract capitalized terms (likely important concepts)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', clause_text)
        keywords.extend(capitalized[:5])  # Top 5 capitalized terms
        
        return list(set(keywords))
    
    def _calculate_importance(self, clause_text: str, clause_type: str) -> float:
        """Calculate importance score for clause"""
        base_score = 0.5
        
        # Type-based scoring
        type_scores = {
            'payment_terms': 0.9,
            'waiting_periods': 0.9,
            'coverage_conditions': 0.8,
            'benefits': 0.7,
            'definitions': 0.6,
            'exclusions': 0.7,
            'general': 0.3
        }
        
        score = type_scores.get(clause_type, base_score)
        
        # Length factor (moderate length clauses are often more important)
        length = len(clause_text)
        if 100 <= length <= 500:
            score += 0.1
        elif length > 500:
            score -= 0.1
        
        # Keyword density
        important_keywords = ['grace', 'waiting', 'coverage', 'benefit', 'premium']
        keyword_count = sum(1 for keyword in important_keywords if keyword in clause_text.lower())
        score += keyword_count * 0.05
        
        # Presence of numbers (often indicates specific terms)
        if re.search(r'\d+', clause_text):
            score += 0.1
        
        return min(1.0, score)
    
    def _identify_section(self, clause_text: str, position: int) -> str:
        """Identify which document section this clause belongs to"""
        clause_lower = clause_text.lower()
        
        if any(term in clause_lower for term in ['payment', 'premium', 'grace']):
            return 'payment_section'
        elif any(term in clause_lower for term in ['waiting', 'coverage', 'benefit']):
            return 'coverage_section'
        elif any(term in clause_lower for term in ['exclude', 'not covered', 'limitation']):
            return 'exclusions_section'
        elif any(term in clause_lower for term in ['define', 'means', 'interpretation']):
            return 'definitions_section'
        else:
            return f'section_{position // 10}'  # Group by position
    
    async def find_matching_clauses(self, query: str, top_k: int = 5, min_similarity: float = 0.4) -> List[Dict[str, Any]]:
        """
        Find clauses that match the query using semantic similarity
        """
        if not self.clauses or not self.embedding_model:
            return []
        
        try:
            # Encode query
            query_embedding = self.embedding_model.encode(
                query, convert_to_tensor=False, normalize_embeddings=True
            )
            
            # Calculate similarities
            similarities = []
            for clause in self.clauses:
                if clause.semantic_embedding is not None:
                    similarity = np.dot(clause.semantic_embedding, query_embedding)
                    similarities.append((clause, float(similarity)))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Format results
            matching_clauses = []
            for i, (clause, similarity) in enumerate(similarities[:top_k]):
                if similarity >= min_similarity:
                    match_result = {
                        'rank': i + 1,
                        'clause_id': clause.id,
                        'text': clause.text,
                        'clause_type': clause.clause_type,
                        'section': clause.section,
                        'similarity_score': similarity,
                        'importance_score': clause.importance_score,
                        'keywords': clause.keywords,
                        'relevance_level': self._get_relevance_level(similarity),
                        'explanation': self._generate_match_explanation(query, clause, similarity)
                    }
                    matching_clauses.append(match_result)
            
            print(f"🔍 Found {len(matching_clauses)} matching clauses for query: '{query[:50]}...'")
            return matching_clauses
            
        except Exception as e:
            logger.error(f"Clause matching failed: {e}")
            return []
    
    def _get_relevance_level(self, similarity: float) -> str:
        """Determine relevance level based on similarity score"""
        if similarity >= 0.8:
            return 'very_high'
        elif similarity >= 0.6:
            return 'high'
        elif similarity >= 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _generate_match_explanation(self, query: str, clause: Clause, similarity: float) -> str:
        """Generate explanation for why this clause matches the query"""
        query_words = set(query.lower().split())
        clause_words = set(clause.text.lower().split())
        common_words = query_words.intersection(clause_words)
        
        explanation_parts = []
        
        if similarity >= 0.8:
            explanation_parts.append("Strong semantic match")
        elif similarity >= 0.6:
            explanation_parts.append("Good semantic similarity")
        else:
            explanation_parts.append("Moderate semantic match")
        
        if common_words:
            explanation_parts.append(f"Common keywords: {', '.join(list(common_words)[:3])}")
        
        if clause.clause_type != 'general':
            explanation_parts.append(f"Relevant clause type: {clause.clause_type}")
        
        return "; ".join(explanation_parts)
    
    def get_clause_statistics(self) -> Dict[str, Any]:
        """Get comprehensive clause statistics"""
        if not self.clauses:
            return {'total_clauses': 0}
        
        # Count by type
        type_counts = {}
        importance_scores = []
        section_counts = {}
        
        for clause in self.clauses:
            # Type distribution
            type_counts[clause.clause_type] = type_counts.get(clause.clause_type, 0) + 1
            
            # Importance scores
            importance_scores.append(clause.importance_score)
            
            # Section distribution
            section_counts[clause.section] = section_counts.get(clause.section, 0) + 1
        
        return {
            'total_clauses': len(self.clauses),
            'clause_types': type_counts,
            'sections': section_counts,
            'avg_importance': sum(importance_scores) / len(importance_scores),
            'high_importance_count': sum(1 for score in importance_scores if score >= 0.7),
            'initialized': self.is_initialized
        }