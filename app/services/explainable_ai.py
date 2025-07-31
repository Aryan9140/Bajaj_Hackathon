"""
HackRx 6.0 - Explainable AI Service
Provides decision rationale and reasoning traces for all system decisions
"""

import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)

class DecisionType(Enum):
    """Types of decisions the system makes"""
    DOCUMENT_PROCESSING = "document_processing"
    CHUNK_EXTRACTION = "chunk_extraction"
    VECTOR_SEARCH = "vector_search"
    CLAUSE_MATCHING = "clause_matching"
    LLM_INFERENCE = "llm_inference"
    ANSWER_GENERATION = "answer_generation"

class ConfidenceLevel(Enum):
    """Confidence levels for decisions"""
    VERY_HIGH = "very_high"  # 90-100%
    HIGH = "high"           # 75-89%
    MEDIUM = "medium"       # 60-74%
    LOW = "low"            # 40-59%
    VERY_LOW = "very_low"  # 0-39%

@dataclass
class DecisionStep:
    """Represents a single decision step in the reasoning chain"""
    step_id: str
    decision_type: DecisionType
    timestamp: float
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    confidence_score: float
    confidence_level: ConfidenceLevel
    reasoning: str
    evidence: List[str] = field(default_factory=list)
    alternatives_considered: List[Dict] = field(default_factory=list)
    processing_time: float = 0.0
    sources: List[str] = field(default_factory=list)

@dataclass
class ExplanationTrace:
    """Complete explanation trace for a question-answer pair"""
    trace_id: str
    question: str
    final_answer: str
    overall_confidence: float
    decision_chain: List[DecisionStep] = field(default_factory=list)
    total_processing_time: float = 0.0
    key_sources: List[str] = field(default_factory=list)
    reasoning_summary: str = ""
    quality_metrics: Dict[str, float] = field(default_factory=dict)

class ExplainableAI:
    """
    Provides comprehensive explanations for all system decisions
    """
    
    def __init__(self):
        self.current_traces: Dict[str, ExplanationTrace] = {}
        self.step_counter = 0
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize the explainable AI service"""
        self.is_initialized = True
        print("✅ Explainable AI Service initialized")
    
    def start_explanation_trace(self, question: str, request_id: str) -> str:
        """Start a new explanation trace for a question"""
        trace_id = f"trace_{request_id}_{int(time.time())}"
        
        self.current_traces[trace_id] = ExplanationTrace(
            trace_id=trace_id,
            question=question,
            final_answer="",
            overall_confidence=0.0
        )
        
        print(f"🔍 Started explanation trace: {trace_id}")
        return trace_id
    
    def log_decision_step(self, 
                         trace_id: str,
                         decision_type: DecisionType,
                         input_data: Dict[str, Any],
                         output_data: Dict[str, Any],
                         reasoning: str,
                         confidence_score: float,
                         evidence: List[str] = None,
                         sources: List[str] = None,
                         processing_time: float = 0.0) -> str:
        """Log a decision step in the reasoning chain"""
        
        if trace_id not in self.current_traces:
            logger.warning(f"Trace {trace_id} not found")
            return ""
        
        self.step_counter += 1
        step_id = f"step_{self.step_counter:03d}"
        
        step = DecisionStep(
            step_id=step_id,
            decision_type=decision_type,
            timestamp=time.time(),
            input_data=input_data or {},
            output_data=output_data or {},
            confidence_score=confidence_score,
            confidence_level=self._get_confidence_level(confidence_score),
            reasoning=reasoning,
            evidence=evidence or [],
            processing_time=processing_time,
            sources=sources or []
        )
        
        self.current_traces[trace_id].decision_chain.append(step)
        
        # Update key sources
        if sources:
            self.current_traces[trace_id].key_sources.extend(sources)
            self.current_traces[trace_id].key_sources = list(set(self.current_traces[trace_id].key_sources))
        
        print(f"📝 Logged decision step: {step_id} ({decision_type.value})")
        return step_id
    
    def _get_confidence_level(self, score: float) -> ConfidenceLevel:
        """Convert confidence score to level"""
        if score >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif score >= 0.75:
            return ConfidenceLevel.HIGH
        elif score >= 0.6:
            return ConfidenceLevel.MEDIUM
        elif score >= 0.4:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def finalize_explanation(self, trace_id: str, final_answer: str) -> ExplanationTrace:
        """Finalize the explanation trace with the final answer"""
        if trace_id not in self.current_traces:
            logger.warning(f"Trace {trace_id} not found for finalization")
            return None
        
        trace = self.current_traces[trace_id]
        trace.final_answer = final_answer
        
        # Calculate overall metrics
        trace.overall_confidence = self._calculate_overall_confidence(trace)
        trace.total_processing_time = sum(step.processing_time for step in trace.decision_chain)
        trace.reasoning_summary = self._generate_reasoning_summary(trace)
        trace.quality_metrics = self._calculate_quality_metrics(trace)
        
        print(f"✅ Finalized explanation trace: {trace_id}")
        return trace
    
    def _calculate_overall_confidence(self, trace: ExplanationTrace) -> float:
        """Calculate overall confidence from all decision steps"""
        if not trace.decision_chain:
            return 0.0
        
        # Weighted average based on decision importance
        weights = {
            DecisionType.DOCUMENT_PROCESSING: 0.1,
            DecisionType.CHUNK_EXTRACTION: 0.15,
            DecisionType.VECTOR_SEARCH: 0.25,
            DecisionType.CLAUSE_MATCHING: 0.25,
            DecisionType.LLM_INFERENCE: 0.15,
            DecisionType.ANSWER_GENERATION: 0.1
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for step in trace.decision_chain:
            weight = weights.get(step.decision_type, 0.1)
            weighted_sum += step.confidence_score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _generate_reasoning_summary(self, trace: ExplanationTrace) -> str:
        """Generate a human-readable reasoning summary"""
        summary_parts = []
        
        # Document processing summary
        doc_steps = [s for s in trace.decision_chain if s.decision_type == DecisionType.DOCUMENT_PROCESSING]
        if doc_steps:
            summary_parts.append(f"Document processed into {len(doc_steps)} sections")
        
        # Search summary
        search_steps = [s for s in trace.decision_chain if s.decision_type == DecisionType.VECTOR_SEARCH]
        if search_steps:
            avg_similarity = sum(s.confidence_score for s in search_steps) / len(search_steps)
            summary_parts.append(f"Vector search found relevant content (avg similarity: {avg_similarity:.2f})")
        
        # Clause matching summary
        clause_steps = [s for s in trace.decision_chain if s.decision_type == DecisionType.CLAUSE_MATCHING]
        if clause_steps:
            summary_parts.append(f"Identified {len(clause_steps)} relevant clauses")
        
        # LLM inference summary
        llm_steps = [s for s in trace.decision_chain if s.decision_type == DecisionType.LLM_INFERENCE]
        if llm_steps:
            avg_confidence = sum(s.confidence_score for s in llm_steps) / len(llm_steps)
            summary_parts.append(f"LLM generated answer with {avg_confidence:.2f} confidence")
        
        return "; ".join(summary_parts)
    
    def _calculate_quality_metrics(self, trace: ExplanationTrace) -> Dict[str, float]:
        """Calculate quality metrics for the decision trace"""
        if not trace.decision_chain:
            return {}
        
        # Processing efficiency
        avg_step_time = trace.total_processing_time / len(trace.decision_chain)
        
        # Confidence consistency
        confidences = [step.confidence_score for step in trace.decision_chain]
        confidence_std = sum((c - trace.overall_confidence) ** 2 for c in confidences) / len(confidences)
        confidence_consistency = 1.0 - min(confidence_std, 1.0)
        
        # Evidence coverage
        total_evidence = sum(len(step.evidence) for step in trace.decision_chain)
        evidence_coverage = min(total_evidence / 10.0, 1.0)  # Normalize to 0-1
        
        # Source diversity
        unique_sources = len(set(trace.key_sources))
        source_diversity = min(unique_sources / 5.0, 1.0)  # Normalize to 0-1
        
        return {
            'processing_efficiency': max(0.0, 1.0 - (avg_step_time / 5.0)),  # Penalty for slow steps
            'confidence_consistency': confidence_consistency,
            'evidence_coverage': evidence_coverage,
            'source_diversity': source_diversity,
            'overall_quality': (confidence_consistency + evidence_coverage + source_diversity) / 3
        }
    
    def get_explanation_for_human(self, trace_id: str) -> Dict[str, Any]:
        """Get human-readable explanation"""
        if trace_id not in self.current_traces:
            return {"error": "Trace not found"}
        
        trace = self.current_traces[trace_id]
        
        # Create step-by-step explanation
        step_explanations = []
        for step in trace.decision_chain:
            step_explanation = {
                'step': step.step_id,
                'action': self._humanize_decision_type(step.decision_type),
                'reasoning': step.reasoning,
                'confidence': f"{step.confidence_score:.1%} ({step.confidence_level.value})",
                'key_evidence': step.evidence[:3],  # Top 3 pieces of evidence
                'processing_time': f"{step.processing_time:.2f}s"
            }
            step_explanations.append(step_explanation)
        
        return {
            'question': trace.question,
            'final_answer': trace.final_answer,
            'overall_confidence': f"{trace.overall_confidence:.1%}",
            'reasoning_summary': trace.reasoning_summary,
            'total_time': f"{trace.total_processing_time:.2f}s",
            'key_sources': trace.key_sources,
            'step_by_step': step_explanations,
            'quality_metrics': {
                k: f"{v:.1%}" for k, v in trace.quality_metrics.items()
            }
        }
    
    def _humanize_decision_type(self, decision_type: DecisionType) -> str:
        """Convert decision type to human-readable action"""
        humanized = {
            DecisionType.DOCUMENT_PROCESSING: "Processed document structure",
            DecisionType.CHUNK_EXTRACTION: "Extracted relevant text chunks",
            DecisionType.VECTOR_SEARCH: "Searched for semantically similar content",
            DecisionType.CLAUSE_MATCHING: "Matched relevant policy clauses",
            DecisionType.LLM_INFERENCE: "Generated answer using language model",
            DecisionType.ANSWER_GENERATION: "Formatted final response"
        }
        return humanized.get(decision_type, decision_type.value)
    
    def get_trace_statistics(self) -> Dict[str, Any]:
        """Get statistics about all traces"""
        if not self.current_traces:
            return {'total_traces': 0}
        
        traces = list(self.current_traces.values())
        
        # Overall statistics
        total_traces = len(traces)
        avg_confidence = sum(t.overall_confidence for t in traces) / total_traces
        avg_processing_time = sum(t.total_processing_time for t in traces) / total_traces
        
        # Decision type frequency
        decision_counts = {}
        for trace in traces:
            for step in trace.decision_chain:
                decision_type = step.decision_type.value
                decision_counts[decision_type] = decision_counts.get(decision_type, 0) + 1
        
        # Confidence distribution
        confidence_levels = {'very_high': 0, 'high': 0, 'medium': 0, 'low': 0, 'very_low': 0}
        for trace in traces:
            level = self._get_confidence_level(trace.overall_confidence).value
            confidence_levels[level] += 1
        
        return {
            'total_traces': total_traces,
            'average_confidence': avg_confidence,
            'average_processing_time': avg_processing_time,
            'decision_type_frequency': decision_counts,
            'confidence_distribution': confidence_levels,
            'initialized': self.is_initialized
        }
    
    def cleanup_old_traces(self, max_traces: int = 100):
        """Clean up old traces to prevent memory issues"""
        if len(self.current_traces) > max_traces:
            # Keep only the most recent traces
            sorted_traces = sorted(
                self.current_traces.items(),
                key=lambda x: x[1].decision_chain[0].timestamp if x[1].decision_chain else 0,
                reverse=True
            )
            
            traces_to_keep = dict(sorted_traces[:max_traces])
            removed_count = len(self.current_traces) - len(traces_to_keep)
            
            self.current_traces = traces_to_keep
            print(f"🧹 Cleaned up {removed_count} old explanation traces")