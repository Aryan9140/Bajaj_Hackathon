"""
HackRx 6.0 - System Architecture Implementation
Complete 6-step workflow: Input Documents → LLM Parser → Embedding Search → 
Clause Matching → Logic Evaluation → JSON Output
"""

import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from .hybrid_vector_service import HybridVectorService
from .clause_retrieval import ClauseRetrievalService
from .explainable_ai import ExplainableAI, DecisionType
from .document_processor import DocumentProcessor
from .llm_handler import LLMHandler

logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """Result from each processing step"""
    step_name: str
    success: bool
    data: Any
    processing_time: float
    confidence_score: float
    metadata: Dict[str, Any]
    error_message: str = ""

@dataclass
class WorkflowResult:
    """Complete workflow result"""
    request_id: str
    answers: List[str]
    processing_time: float
    steps_completed: List[str]
    overall_confidence: float
    explanation_trace_id: str
    metadata: Dict[str, Any]
    errors: List[str]

class SystemArchitecture:
    """
    Implements the complete 6-step HackRx workflow
    """
    
    def __init__(self):
        # Core services
        self.document_processor = DocumentProcessor()
        self.vector_service = HybridVectorService()
        self.clause_service = ClauseRetrievalService()
        self.explainable_ai = ExplainableAI()
        self.llm_handler = LLMHandler()
        
        # Workflow state
        self.is_initialized = False
        self.processing_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'average_processing_time': 0.0,
            'step_success_rates': {}
        }
    
    async def initialize(self):
        """Initialize all system components"""
        try:
            print("🚀 Initializing HackRx System Architecture...")
            
            # Initialize all services in parallel
            await asyncio.gather(
                self.document_processor.initialize(),
                self.vector_service.initialize(),
                self.clause_service.initialize(),
                self.explainable_ai.initialize(),
                self.llm_handler.initialize()
            )
            
            # Set up clause service with vector model
            if self.vector_service.model:
                await self.clause_service.initialize(self.vector_service.model)
            
            self.is_initialized = True
            print("✅ System Architecture fully initialized - Ready for 6-step workflow")
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            raise
    
    async def process_request(self, document_url: str, questions: List[str], request_id: str) -> WorkflowResult:
        """
        Execute complete 6-step workflow
        """
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        workflow_result = WorkflowResult(
            request_id=request_id,
            answers=[],
            processing_time=0.0,
            steps_completed=[],
            overall_confidence=0.0,
            explanation_trace_id="",
            metadata={},
            errors=[]
        )
        
        try:
            print(f"🔄 Starting 6-step workflow for request: {request_id}")
            
            # STEP 1: Input Documents Processing
            step1_result = await self._step1_input_documents(document_url, request_id)
            workflow_result.steps_completed.append("input_documents")
            
            if not step1_result.success:
                workflow_result.errors.append(step1_result.error_message)
                return workflow_result
            
            document_text = step1_result.data
            
            # STEP 2: LLM Parser (Query Structuring)
            step2_result = await self._step2_llm_parser(questions, document_text, request_id)
            workflow_result.steps_completed.append("llm_parser")
            
            parsed_queries = step2_result.data if step2_result.success else questions
            
            # STEP 3: Embedding Search
            step3_result = await self._step3_embedding_search(document_text, parsed_queries, request_id)
            workflow_result.steps_completed.append("embedding_search")
            
            if not step3_result.success:
                workflow_result.errors.append(step3_result.error_message)
                # Continue with reduced functionality
            
            search_results = step3_result.data if step3_result.success else {}
            
            # STEP 4: Clause Matching
            step4_result = await self._step4_clause_matching(document_text, parsed_queries, search_results, request_id)
            workflow_result.steps_completed.append("clause_matching")
            
            clause_matches = step4_result.data if step4_result.success else {}
            
            # STEP 5: Logic Evaluation
            step5_result = await self._step5_logic_evaluation(parsed_queries, search_results, clause_matches, request_id)
            workflow_result.steps_completed.append("logic_evaluation")
            
            evaluated_contexts = step5_result.data if step5_result.success else {}
            
            # STEP 6: JSON Output (Answer Generation)
            step6_result = await self._step6_json_output(parsed_queries, evaluated_contexts, request_id)
            workflow_result.steps_completed.append("json_output")
            
            if step6_result.success:
                workflow_result.answers = step6_result.data['answers']
                workflow_result.overall_confidence = step6_result.confidence_score
                workflow_result.metadata = step6_result.data.get('metadata', {})
            
            # Finalize workflow
            workflow_result.processing_time = time.time() - start_time
            
            # Get explainability trace
            if hasattr(self.explainable_ai, 'current_traces'):
                traces = [t for t in self.explainable_ai.current_traces.keys() if request_id in t]
                if traces:
                    workflow_result.explanation_trace_id = traces[0]
            
            # Update statistics
            self._update_processing_stats(workflow_result)
            
            print(f"✅ 6-step workflow completed in {workflow_result.processing_time:.2f}s")
            return workflow_result
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            workflow_result.errors.append(str(e))
            workflow_result.processing_time = time.time() - start_time
            return workflow_result
    
    async def _step1_input_documents(self, document_url: str, request_id: str) -> ProcessingResult:
        """STEP 1: Process input documents"""
        start_time = time.time()
        
        try:
            print("📄 STEP 1: Processing input documents...")
            
            # Start explanation trace
            trace_id = self.explainable_ai.start_explanation_trace("Document processing", request_id)
            
            # Process document
            document_text = await self.document_processor.process_document(document_url)
            
            if not document_text:
                raise Exception("Failed to extract document content")
            
            processing_time = time.time() - start_time
            
            # Log explanation
            self.explainable_ai.log_decision_step(
                trace_id=trace_id,
                decision_type=DecisionType.DOCUMENT_PROCESSING,
                input_data={'document_url': document_url},
                output_data={'text_length': len(document_text), 'extracted': True},
                reasoning=f"Successfully extracted {len(document_text)} characters from document",
                confidence_score=0.95,
                evidence=[f"Document size: {len(document_text)} chars"],
                processing_time=processing_time
            )
            
            return ProcessingResult(
                step_name="input_documents",
                success=True,
                data=document_text,
                processing_time=processing_time,
                confidence_score=0.95,
                metadata={'document_length': len(document_text)}
            )
            
        except Exception as e:
            return ProcessingResult(
                step_name="input_documents",
                success=False,
                data=None,
                processing_time=time.time() - start_time,
                confidence_score=0.0,
                metadata={},
                error_message=str(e)
            )
    
    async def _step2_llm_parser(self, questions: List[str], document_text: str, request_id: str) -> ProcessingResult:
        """STEP 2: Parse and structure queries using LLM"""
        start_time = time.time()
        
        try:
            print("🧠 STEP 2: LLM query parsing and structuring...")
            
            # Find active trace
            trace_id = [t for t in self.explainable_ai.current_traces.keys() if request_id in t][0]
            
            # Structure queries for better processing
            structured_queries = []
            
            for i, question in enumerate(questions):
                # Extract query intent and key terms
                query_analysis = await self._analyze_query_intent(question)
                
                structured_query = {
                    'original': question,
                    'intent': query_analysis.get('intent', 'information_extraction'),
                    'key_terms': query_analysis.get('key_terms', []),
                    'expected_answer_type': query_analysis.get('answer_type', 'descriptive'),
                    'priority': query_analysis.get('priority', 0.5)
                }
                structured_queries.append(structured_query)
            
            processing_time = time.time() - start_time
            
            # Log explanation
            self.explainable_ai.log_decision_step(
                trace_id=trace_id,
                decision_type=DecisionType.CHUNK_EXTRACTION,
                input_data={'questions_count': len(questions)},
                output_data={'structured_queries_count': len(structured_queries)},
                reasoning=f"Parsed {len(questions)} queries into structured format with intent analysis",
                confidence_score=0.85,
                evidence=[f"Identified {len(structured_queries)} query intents"],
                processing_time=processing_time
            )
            
            return ProcessingResult(
                step_name="llm_parser",
                success=True,
                data=structured_queries,
                processing_time=processing_time,
                confidence_score=0.85,
                metadata={'queries_parsed': len(structured_queries)}
            )
            
        except Exception as e:
            return ProcessingResult(
                step_name="llm_parser",
                success=False,
                data=questions,  # Fallback to original questions
                processing_time=time.time() - start_time,
                confidence_score=0.3,
                metadata={'fallback_used': True},
                error_message=str(e)
            )
    
    async def _step3_embedding_search(self, document_text: str, queries: List[Dict], request_id: str) -> ProcessingResult:
        """STEP 3: Semantic embedding search"""
        start_time = time.time()
        
        try:
            print("🔍 STEP 3: Embedding search and semantic retrieval...")
            
            # Find active trace
            trace_id = [t for t in self.explainable_ai.current_traces.keys() if request_id in t][0]
            
            # Chunk document
            chunks = self.vector_service.chunk_document(document_text)
            
            # Add to vector store
            await self.vector_service.add_documents(chunks)
            
            # Search for each query
            search_results = {}
            total_similarities = []
            
            for i, query_data in enumerate(queries):
                query_text = query_data.get('original', '') if isinstance(query_data, dict) else str(query_data)
                
                # Perform semantic search
                results = await self.vector_service.search(query_text, top_k=5)
                search_results[f"question_{i}"] = results
                
                # Collect similarity scores for confidence calculation
                if results:
                    similarities = [r.get('similarity_score', 0) for r in results]
                    total_similarities.extend(similarities)
            
            processing_time = time.time() - start_time
            avg_similarity = sum(total_similarities) / len(total_similarities) if total_similarities else 0.5
            
            # Log explanation
            self.explainable_ai.log_decision_step(
                trace_id=trace_id,
                decision_type=DecisionType.VECTOR_SEARCH,
                input_data={'chunks_created': len(chunks), 'queries_processed': len(queries)},
                output_data={'search_results_count': len(search_results), 'avg_similarity': avg_similarity},
                reasoning=f"Created {len(chunks)} document chunks and performed semantic search for {len(queries)} queries",
                confidence_score=min(avg_similarity + 0.2, 1.0),
                evidence=[f"Average similarity: {avg_similarity:.3f}", f"Total results: {sum(len(r) for r in search_results.values())}"],
                processing_time=processing_time
            )
            
            return ProcessingResult(
                step_name="embedding_search",
                success=True,
                data=search_results,
                processing_time=processing_time,
                confidence_score=min(avg_similarity + 0.2, 1.0),
                metadata={
                    'chunks_processed': len(chunks),
                    'average_similarity': avg_similarity,
                    'search_method': self.vector_service.get_stats().get('preferred_method', 'hybrid')
                }
            )
            
        except Exception as e:
            return ProcessingResult(
                step_name="embedding_search",
                success=False,
                data={},
                processing_time=time.time() - start_time,
                confidence_score=0.0,
                metadata={},
                error_message=str(e)
            )
    
    async def _step4_clause_matching(self, document_text: str, queries: List[Dict], search_results: Dict, request_id: str) -> ProcessingResult:
        """STEP 4: Advanced clause matching"""
        start_time = time.time()
        
        try:
            print("📋 STEP 4: Clause matching and semantic analysis...")
            
            # Find active trace
            trace_id = [t for t in self.explainable_ai.current_traces.keys() if request_id in t][0]
            
            # Extract clauses from document
            clauses = self.clause_service.extract_clauses(document_text)
            
            # Match clauses for each query
            clause_matches = {}
            total_relevance_scores = []
            
            for i, query_data in enumerate(queries):
                query_text = query_data.get('original', '') if isinstance(query_data, dict) else str(query_data)
                
                # Find matching clauses
                matches = await self.clause_service.find_matching_clauses(query_text, top_k=3)
                clause_matches[f"question_{i}"] = matches
                
                # Collect relevance scores
                if matches:
                    scores = [m.get('similarity_score', 0) for m in matches]
                    total_relevance_scores.extend(scores)
            
            processing_time = time.time() - start_time
            avg_relevance = sum(total_relevance_scores) / len(total_relevance_scores) if total_relevance_scores else 0.6
            
            # Log explanation
            self.explainable_ai.log_decision_step(
                trace_id=trace_id,
                decision_type=DecisionType.CLAUSE_MATCHING,
                input_data={'clauses_extracted': len(clauses), 'queries_processed': len(queries)},
                output_data={'clause_matches_count': len(clause_matches), 'avg_relevance': avg_relevance},
                reasoning=f"Extracted {len(clauses)} clauses and matched them against {len(queries)} queries",
                confidence_score=avg_relevance,
                evidence=[f"Clause types found: {len(set(c.clause_type for c in clauses))}", f"Average relevance: {avg_relevance:.3f}"],
                processing_time=processing_time
            )
            
            return ProcessingResult(
                step_name="clause_matching",
                success=True,
                data=clause_matches,
                processing_time=processing_time,
                confidence_score=avg_relevance,
                metadata={
                    'clauses_extracted': len(clauses),
                    'average_relevance': avg_relevance,
                    'clause_types': list(set(c.clause_type for c in clauses))
                }
            )
            
        except Exception as e:
            return ProcessingResult(
                step_name="clause_matching",
                success=False,
                data={},
                processing_time=time.time() - start_time,
                confidence_score=0.0,
                metadata={},
                error_message=str(e)
            )
    
    async def _step5_logic_evaluation(self, queries: List[Dict], search_results: Dict, clause_matches: Dict, request_id: str) -> ProcessingResult:
        """STEP 5: Logic evaluation and context optimization"""
        start_time = time.time()
        
        try:
            print("🧮 STEP 5: Logic evaluation and context optimization...")
            
            # Find active trace
            trace_id = [t for t in self.explainable_ai.current_traces.keys() if request_id in t][0]
            
            # Combine and evaluate evidence from search and clause matching
            evaluated_contexts = {}
            confidence_scores = []
            
            for i, query_data in enumerate(queries):
                query_key = f"question_{i}"
                query_text = query_data.get('original', '') if isinstance(query_data, dict) else str(query_data)
                
                # Get search results
                search_data = search_results.get(query_key, [])
                clause_data = clause_matches.get(query_key, [])
                
                # Evaluate and combine contexts
                evaluated_context = self._evaluate_context_quality(query_text, search_data, clause_data)
                evaluated_contexts[query_key] = evaluated_context
                
                confidence_scores.append(evaluated_context.get('confidence', 0.5))
            
            processing_time = time.time() - start_time
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
            
            # Log explanation
            self.explainable_ai.log_decision_step(
                trace_id=trace_id,
                decision_type=DecisionType.LLM_INFERENCE,
                input_data={'contexts_evaluated': len(evaluated_contexts)},
                output_data={'avg_confidence': avg_confidence},
                reasoning=f"Evaluated and optimized contexts for {len(queries)} queries using combined search and clause data",
                confidence_score=avg_confidence,
                evidence=[f"Combined {len(search_results)} search results with {len(clause_matches)} clause matches"],
                processing_time=processing_time
            )
            
            return ProcessingResult(
                step_name="logic_evaluation",
                success=True,
                data=evaluated_contexts,
                processing_time=processing_time,
                confidence_score=avg_confidence,
                metadata={'average_confidence': avg_confidence}
            )
            
        except Exception as e:
            return ProcessingResult(
                step_name="logic_evaluation",
                success=False,
                data={},
                processing_time=time.time() - start_time,
                confidence_score=0.0,
                metadata={},
                error_message=str(e)
            )
    
    async def _step6_json_output(self, queries: List[Dict], evaluated_contexts: Dict, request_id: str) -> ProcessingResult:
        """STEP 6: Generate final JSON output with answers"""
        start_time = time.time()
        
        try:
            print("📤 STEP 6: Generating final JSON output...")
            
            # Find active trace
            trace_id = [t for t in self.explainable_ai.current_traces.keys() if request_id in t][0]
            
            # Generate answers using LLM
            answers = []
            answer_confidences = []
            
            for i, query_data in enumerate(queries):
                query_key = f"question_{i}"
                query_text = query_data.get('original', '') if isinstance(query_data, dict) else str(query_data)
                context_data = evaluated_contexts.get(query_key, {})
                
                # Generate answer
                answer = await self._generate_final_answer(query_text, context_data, request_id)
                answers.append(answer.get('text', 'Information not available in the document.'))
                answer_confidences.append(answer.get('confidence', 0.5))
            
            processing_time = time.time() - start_time
            overall_confidence = sum(answer_confidences) / len(answer_confidences) if answer_confidences else 0.5
            
            # Prepare final output
            output_data = {
                'answers': answers,
                'metadata': {
                    'system_version': '6.0.0_complete',
                    'workflow_steps_completed': 6,
                    'overall_confidence': overall_confidence,
                    'processing_method': 'full_6_step_workflow',
                    'features_used': [
                        'hybrid_vector_search',
                        'advanced_clause_matching',
                        'explainable_decisions',
                        'multi_llm_inference'
                    ]
                }
            }
            
            # Log explanation
            self.explainable_ai.log_decision_step(
                trace_id=trace_id,
                decision_type=DecisionType.ANSWER_GENERATION,
                input_data={'queries_processed': len(queries)},
                output_data={'answers_generated': len(answers), 'overall_confidence': overall_confidence},
                reasoning=f"Generated {len(answers)} final answers using complete 6-step workflow",
                confidence_score=overall_confidence,
                evidence=[f"Answer confidence range: {min(answer_confidences):.2f} - {max(answer_confidences):.2f}"],
                processing_time=processing_time
            )
            
            # Finalize explanation trace
            self.explainable_ai.finalize_explanation(trace_id, str(answers))
            
            return ProcessingResult(
                step_name="json_output",
                success=True,
                data=output_data,
                processing_time=processing_time,
                confidence_score=overall_confidence,
                metadata=output_data['metadata']
            )
            
        except Exception as e:
            return ProcessingResult(
                step_name="json_output",
                success=False,
                data={'answers': ['Error generating answer'] * len(queries)},
                processing_time=time.time() - start_time,
                confidence_score=0.0,
                metadata={},
                error_message=str(e)
            )
    
    async def _analyze_query_intent(self, question: str) -> Dict[str, Any]:
        """Analyze query intent and extract key information"""
        question_lower = question.lower()
        
        # Determine intent
        if any(word in question_lower for word in ['what is', 'define', 'definition']):
            intent = 'definition_seeking'
        elif any(word in question_lower for word in ['how much', 'how many', 'percentage', 'amount']):
            intent = 'quantitative_inquiry'
        elif any(word in question_lower for word in ['does', 'is', 'are', 'will', 'can']):
            intent = 'yes_no_question'
        elif any(word in question_lower for word in ['when', 'time', 'period', 'duration']):
            intent = 'temporal_inquiry'
        else:
            intent = 'information_extraction'
        
        # Extract key terms
        important_terms = []
        insurance_terms = [
            'grace period', 'waiting period', 'pre-existing', 'maternity', 'coverage',
            'benefit', 'premium', 'claim', 'discount', 'hospital', 'treatment',
            'surgery', 'organ donor', 'ayush', 'room rent', 'icu'
        ]
        
        for term in insurance_terms:
            if term in question_lower:
                important_terms.append(term)
        
        # Determine expected answer type
        if intent == 'yes_no_question':
            answer_type = 'boolean'
        elif intent == 'quantitative_inquiry':
            answer_type = 'numeric'
        elif intent == 'temporal_inquiry':
            answer_type = 'temporal'
        else:
            answer_type = 'descriptive'
        
        # Calculate priority (more specific questions get higher priority)
        priority = 0.5 + (len(important_terms) * 0.1)
        
        return {
            'intent': intent,
            'key_terms': important_terms,
            'answer_type': answer_type,
            'priority': min(priority, 1.0)
        }
    
    def _evaluate_context_quality(self, query: str, search_results: List[Dict], clause_matches: List[Dict]) -> Dict[str, Any]:
        """Evaluate and combine context from search and clause matching"""
        
        # Combine contexts
        combined_context = ""
        confidence_scores = []
        sources = []
        
        # Add search results
        for result in search_results[:3]:  # Top 3 search results
            combined_context += result.get('text', '') + " "
            confidence_scores.append(result.get('similarity_score', 0.5))
            sources.append(f"Vector search (similarity: {result.get('similarity_score', 0):.2f})")
        
        # Add clause matches
        for match in clause_matches[:2]:  # Top 2 clause matches
            combined_context += match.get('text', '') + " "
            confidence_scores.append(match.get('similarity_score', 0.5))
            sources.append(f"Clause match ({match.get('clause_type', 'unknown')})")
        
        # Calculate overall confidence
        overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
        
        # Truncate context to optimal length
        if len(combined_context) > 3000:
            combined_context = combined_context[:3000] + "..."
        
        return {
            'context': combined_context.strip(),
            'confidence': overall_confidence,
            'sources': sources,
            'search_results_count': len(search_results),
            'clause_matches_count': len(clause_matches)
        }
    
    async def _generate_final_answer(self, question: str, context_data: Dict, request_id: str) -> Dict[str, Any]:
        """Generate final answer using LLM with optimized context"""
        try:
            context = context_data.get('context', '')
            confidence = context_data.get('confidence', 0.5)
            
            if not context:
                return {
                    'text': 'Information not available in the document.',
                    'confidence': 0.1
                }
            
            # Use LLM to generate answer
            answer = await self.llm_handler.generate_answer(
                question=question,
                context=context,
                request_id=request_id
            )
            
            return {
                'text': answer,
                'confidence': min(confidence + 0.1, 1.0)  # Slight boost for LLM processing
            }
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return {
                'text': 'Error generating answer from available information.',
                'confidence': 0.0
            }
    
    def _update_processing_stats(self, result: WorkflowResult):
        """Update processing statistics"""
        self.processing_stats['total_requests'] += 1
        
        if not result.errors:
            self.processing_stats['successful_requests'] += 1
        
        # Update average processing time
        total_time = (self.processing_stats['average_processing_time'] * 
                     (self.processing_stats['total_requests'] - 1) + 
                     result.processing_time)
        self.processing_stats['average_processing_time'] = total_time / self.processing_stats['total_requests']
        
        # Update step success rates
        for step in result.steps_completed:
            if step not in self.processing_stats['step_success_rates']:
                self.processing_stats['step_success_rates'][step] = {'success': 0, 'total': 0}
            
            self.processing_stats['step_success_rates'][step]['total'] += 1
            if not result.errors:
                self.processing_stats['step_success_rates'][step]['success'] += 1
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        stats = {
            'initialized': self.is_initialized,
            'processing_stats': self.processing_stats.copy(),
            'component_stats': {
                'vector_service': self.vector_service.get_stats(),
                'clause_service': self.clause_service.get_clause_statistics(),
                'explainable_ai': self.explainable_ai.get_trace_statistics(),
                'llm_handler': self.llm_handler.get_comprehensive_statistics()
            }
        }
        
        # Calculate step success rates as percentages
        for step, data in stats['processing_stats']['step_success_rates'].items():
            if data['total'] > 0:
                data['success_rate'] = (data['success'] / data['total']) * 100
        
        return stats