# app/services/llm_handler.py
"""
HackRx 6.0 - Complete Multi-LLM Handler Service
Handles OpenAI, Claude, and Groq with intelligent fallback
Complete implementation with all methods
"""

import aiohttp
import asyncio
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
import time

from app.core.config import settings

logger = logging.getLogger(__name__)

class LLMHandler:
    """
    Complete multi-LLM handler with intelligent fallback and performance tracking
    """
    
    def __init__(self):
        self.active_models = []
        self.model_performance = {}
        self.is_initialized = False
        
        # Model priorities (best to worst for accuracy)
        self.model_priority = ['openai', 'claude', 'groq']
        
        # Model configurations
        self.model_configs = {
            'openai': {
                'model': 'gpt-4-turbo-preview',
                'max_tokens': 150,
                'temperature': 0.0,
                'timeout': 3.0,
                'cost_per_1k_tokens': 0.01  # Approximate
            },
            'claude': {
                'model': 'claude-3-5-sonnet-20241022',
                'max_tokens': 120,
                'temperature': 0.0,
                'timeout': 3.0,
                'cost_per_1k_tokens': 0.015  # Approximate
            },
            'groq': {
                'model': 'llama-3.1-70b-versatile',
                'max_tokens': 100,
                'temperature': 0.0,
                'timeout': 2.0,
                'cost_per_1k_tokens': 0.0005  # Very cheap
            }
        }
        
        # Performance tracking
        self.total_requests = 0
        self.successful_requests = 0
        self.total_processing_time = 0.0
        self.model_usage_stats = {}
    
    async def initialize(self):
        """Initialize and test all available LLM models"""
        try:
            print("🤖 Initializing Multi-LLM Handler...")
            
            # Test each model availability in priority order
            initialization_results = {}
            
            if settings.OPENAI_API_KEY:
                print("🔍 Testing OpenAI API...")
                if await self._test_openai():
                    self.active_models.append('openai')
                    initialization_results['openai'] = 'SUCCESS'
                    print("✅ OpenAI GPT-4 initialized and ready")
                else:
                    initialization_results['openai'] = 'FAILED'
                    print("❌ OpenAI initialization failed")
            else:
                initialization_results['openai'] = 'NO_API_KEY'
                print("⚠️ OpenAI API key not configured")
            
            if settings.CLAUDE_API_KEY:
                print("🔍 Testing Claude API...")
                if await self._test_claude():
                    self.active_models.append('claude')
                    initialization_results['claude'] = 'SUCCESS'
                    print("✅ Claude 3.5 Sonnet initialized and ready")
                else:
                    initialization_results['claude'] = 'FAILED'
                    print("❌ Claude initialization failed")
            else:
                initialization_results['claude'] = 'NO_API_KEY'
                print("⚠️ Claude API key not configured")
            
            if settings.GROQ_API_KEY:
                print("🔍 Testing Groq API...")
                if await self._test_groq():
                    self.active_models.append('groq')
                    initialization_results['groq'] = 'SUCCESS'
                    print("✅ Groq Llama initialized and ready")
                else:
                    initialization_results['groq'] = 'FAILED'
                    print("❌ Groq initialization failed")
            else:
                initialization_results['groq'] = 'NO_API_KEY'
                print("⚠️ Groq API key not configured")
            
            self.is_initialized = True
            
            # Summary
            if self.active_models:
                print(f"🎯 LLM Handler initialized with {len(self.active_models)} active models:")
                for i, model in enumerate(self.active_models, 1):
                    print(f"   {i}. {model.upper()} (Priority {self.model_priority.index(model) + 1})")
                print(f"🔄 Fallback chain: {' → '.join([m.upper() for m in self.model_priority if m in self.active_models])}")
            else:
                print("⚠️ No LLM models available - system will use pattern-based fallbacks only")
            
            return initialization_results
            
        except Exception as e:
            logger.error(f"LLM Handler initialization failed: {e}")
            raise
    
    async def _test_openai(self) -> bool:
        """Test OpenAI API connectivity with minimal request"""
        try:
            headers = {
                "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "gpt-3.5-turbo",  # Use cheaper model for testing
                "messages": [{"role": "user", "content": "Test connection. Reply with 'OK'."}],
                "max_tokens": 5,
                "temperature": 0.0
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        return 'OK' in result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    elif response.status == 401:
                        print("🔑 OpenAI API key authentication failed")
                        return False
                    elif response.status == 429:
                        print("⏰ OpenAI API rate limited - but key is valid")
                        return True  # Consider rate limit as success
                    else:
                        print(f"❌ OpenAI API error: {response.status}")
                        return False
                    
        except asyncio.TimeoutError:
            print("⏰ OpenAI API timeout during test")
            return False
        except Exception as e:
            print(f"❌ OpenAI test exception: {e}")
            return False
    
    async def _test_claude(self) -> bool:
        """Test Claude API connectivity with minimal request"""
        try:
            headers = {
                "x-api-key": settings.CLAUDE_API_KEY,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            payload = {
                "model": "claude-3-haiku-20240307",  # Use cheaper model for testing
                "max_tokens": 5,
                "messages": [{"role": "user", "content": "Test connection. Reply with 'OK'."}]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        return 'OK' in result.get("content", [{}])[0].get("text", "")
                    elif response.status == 401:
                        print("🔑 Claude API key authentication failed")
                        return False
                    elif response.status == 429:
                        print("⏰ Claude API rate limited - but key is valid")
                        return True
                    else:
                        print(f"❌ Claude API error: {response.status}")
                        return False
                    
        except asyncio.TimeoutError:
            print("⏰ Claude API timeout during test")
            return False
        except Exception as e:
            print(f"❌ Claude test exception: {e}")
            return False
    
    async def _test_groq(self) -> bool:
        """Test Groq API connectivity with minimal request"""
        try:
            headers = {
                "Authorization": f"Bearer {settings.GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "llama3-8b-8192",  # Use smaller model for testing
                "messages": [{"role": "user", "content": "Test connection. Reply with 'OK'."}],
                "max_tokens": 5,
                "temperature": 0.0
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        return 'OK' in result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    elif response.status == 401:
                        print("🔑 Groq API key authentication failed")
                        return False
                    elif response.status == 429:
                        print("⏰ Groq API rate limited - but key is valid")
                        return True
                    else:
                        print(f"❌ Groq API error: {response.status}")
                        return False
                    
        except asyncio.TimeoutError:
            print("⏰ Groq API timeout during test")
            return False
        except Exception as e:
            print(f"❌ Groq test exception: {e}")
            return False
    
    async def generate_answer(self, question: str, context: str, request_id: str) -> str:
        """
        Generate answer using the best available LLM with intelligent fallback
        """
        if not self.active_models:
            print(f"⚠️ [{request_id}] No LLM models available")
            return f"No LLM models available to process: {question[:50]}..."
        
        start_time = time.time()
        self.total_requests += 1
        
        # Try models in priority order
        for model_name in self.model_priority:
            if model_name in self.active_models:
                try:
                    print(f"🔄 [{request_id}] Trying {model_name.upper()}...")
                    
                    answer = await self._query_model(model_name, question, context, request_id)
                    
                    if answer and self._validate_answer_quality(answer, question):
                        processing_time = time.time() - start_time
                        self._update_model_performance(model_name, True, processing_time)
                        self.successful_requests += 1
                        self.total_processing_time += processing_time
                        
                        print(f"✅ [{request_id}] {model_name.upper()} succeeded in {processing_time:.2f}s")
                        return answer
                    else:
                        print(f"❌ [{request_id}] {model_name.upper()} returned invalid answer")
                        self._update_model_performance(model_name, False, time.time() - start_time)
                        
                except Exception as e:
                    processing_time = time.time() - start_time
                    logger.warning(f"Model {model_name} failed for {request_id}: {e}")
                    self._update_model_performance(model_name, False, processing_time)
                    print(f"❌ [{request_id}] {model_name.upper()} failed: {str(e)[:50]}...")
                    continue
        
        # If all models fail, return a graceful fallback response
        total_time = time.time() - start_time
        self.total_processing_time += total_time
        
        print(f"🚨 [{request_id}] All LLM models failed - using fallback")
        return f"Unable to generate answer for: {question[:50]}... (All LLM models unavailable)"
    
    async def _query_model(self, model_name: str, question: str, context: str, request_id: str) -> Optional[str]:
        """Query specific LLM model with optimized prompts"""
        config = self.model_configs[model_name]
        
        # Create model-specific optimized prompt
        prompt = self._create_optimized_prompt(model_name, question, context)
        
        # Truncate context if too long
        if len(prompt) > 8000:  # Conservative limit
            context_limit = 3000
            truncated_context = context[:context_limit] + "..." if len(context) > context_limit else context
            prompt = self._create_optimized_prompt(model_name, question, truncated_context)
        
        # Query the specific model
        if model_name == 'openai':
            return await self._query_openai(prompt, config, request_id)
        elif model_name == 'claude':
            return await self._query_claude(prompt, config, request_id)
        elif model_name == 'groq':
            return await self._query_groq(prompt, config, request_id)
        
        return None
    
    def _create_optimized_prompt(self, model_name: str, question: str, context: str) -> str:
        """Create model-specific optimized prompts"""
        
        if model_name == 'openai':
            return f"""You are an expert document analyst. Extract the precise answer from the document content provided.

DOCUMENT CONTENT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Provide a direct, factual answer based strictly on the document content
- Include specific details like numbers, percentages, time periods, and conditions
- Keep the answer concise but complete (maximum 2 sentences)
- If the information is not clearly stated in the document, respond with "Information not available in the document"
- Do not include document headers, page numbers, or contact information

PRECISE ANSWER:"""
        
        elif model_name == 'claude':
            return f"""Please analyze the following document content and answer the question precisely.

DOCUMENT:
{context}

QUESTION: {question}

Please provide a direct answer based only on the information in the document. Include specific details when available. If the information is not in the document, please state "Information not available in the document".

Answer:"""
        
        elif model_name == 'groq':
            return f"""Document: {context}

Question: {question}

Instructions: Answer based only on the document content. Be specific and concise. If not found, say "Information not available".

Answer:"""
        
        return f"Document: {context}\n\nQuestion: {question}\n\nAnswer:"
    
    async def _query_openai(self, prompt: str, config: Dict, request_id: str) -> Optional[str]:
        """Query OpenAI GPT with error handling"""
        try:
            headers = {
                "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": config['model'],
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": config['max_tokens'],
                "temperature": config['temperature']
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=config['timeout'])
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        answer = result["choices"][0]["message"]["content"].strip()
                        
                        if len(answer) > 10:
                            return answer
                    elif response.status == 429:
                        print(f"⏰ [{request_id}] OpenAI rate limited")
                    elif response.status == 401:
                        print(f"🔑 [{request_id}] OpenAI authentication failed")
                    else:
                        print(f"❌ [{request_id}] OpenAI API error: {response.status}")
                        
        except asyncio.TimeoutError:
            print(f"⏰ [{request_id}] OpenAI timeout")
        except Exception as e:
            print(f"❌ [{request_id}] OpenAI error: {e}")
        
        return None
    
    async def _query_claude(self, prompt: str, config: Dict, request_id: str) -> Optional[str]:
        """Query Claude with error handling"""
        try:
            headers = {
                "x-api-key": settings.CLAUDE_API_KEY,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            payload = {
                "model": config['model'],
                "max_tokens": config['max_tokens'],
                "temperature": config['temperature'],
                "messages": [{"role": "user", "content": prompt}]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=config['timeout'])
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        answer = result["content"][0]["text"].strip()
                        
                        if len(answer) > 10:
                            return answer
                    elif response.status == 429:
                        print(f"⏰ [{request_id}] Claude rate limited")
                    elif response.status == 401:
                        print(f"🔑 [{request_id}] Claude authentication failed")
                    else:
                        print(f"❌ [{request_id}] Claude API error: {response.status}")
                        
        except asyncio.TimeoutError:
            print(f"⏰ [{request_id}] Claude timeout")
        except Exception as e:
            print(f"❌ [{request_id}] Claude error: {e}")
        
        return None
    
    async def _query_groq(self, prompt: str, config: Dict, request_id: str) -> Optional[str]:
        """Query Groq with error handling"""
        try:
            headers = {
                "Authorization": f"Bearer {settings.GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": config['model'],
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": config['max_tokens'],
                "temperature": config['temperature']
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=config['timeout'])
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        answer = result["choices"][0]["message"]["content"].strip()
                        
                        if len(answer) > 10:
                            return answer
                    elif response.status == 429:
                        print(f"⏰ [{request_id}] Groq rate limited")
                    elif response.status == 401:
                        print(f"🔑 [{request_id}] Groq authentication failed")
                    else:
                        print(f"❌ [{request_id}] Groq API error: {response.status}")
                        
        except asyncio.TimeoutError:
            print(f"⏰ [{request_id}] Groq timeout")
        except Exception as e:
            print(f"❌ [{request_id}] Groq error: {e}")
        
        return None
    
    def _validate_answer_quality(self, answer: str, question: str) -> bool:
        """Validate if the answer meets quality criteria"""
        if not answer or len(answer.strip()) < 5:
            return False
        
        # Check for common failure patterns
        failure_patterns = [
            "information not available",
            "cannot find",
            "unable to locate",
            "not mentioned",
            "no information",
            "error occurred",
            "failed to process"
        ]
        
        answer_lower = answer.lower()
        
        # If answer contains failure patterns, it's still valid (honest response)
        # But check if it's a generic error
        if any(pattern in answer_lower for pattern in ["error", "failed", "exception"]):
            return False
        
        # Check if answer is just repeating the question
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        overlap = len(question_words & answer_words) / len(question_words) if question_words else 0
        
        if overlap > 0.8:  # Too much overlap suggests repetition
            return False
        
        return True
    
    def _update_model_performance(self, model_name: str, success: bool, processing_time: float):
        """Update model performance tracking with detailed metrics"""
        if model_name not in self.model_performance:
            self.model_performance[model_name] = {
                'total_calls': 0,
                'successful_calls': 0,
                'failed_calls': 0,
                'total_time': 0.0,
                'avg_response_time': 0.0,
                'last_success': None,
                'last_failure': None
            }
        
        perf = self.model_performance[model_name]
        perf['total_calls'] += 1
        perf['total_time'] += processing_time
        
        if success:
            perf['successful_calls'] += 1
            perf['last_success'] = datetime.now().isoformat()
        else:
            perf['failed_calls'] += 1
            perf['last_failure'] = datetime.now().isoformat()
        
        # Update average response time
        perf['avg_response_time'] = perf['total_time'] / perf['total_calls']
        
        # Update usage stats
        if model_name not in self.model_usage_stats:
            self.model_usage_stats[model_name] = 0
        self.model_usage_stats[model_name] += 1
    
    def get_active_model(self) -> str:
        """Get the currently preferred active model"""
        for model in self.model_priority:
            if model in self.active_models:
                return model
        return 'none'
    
    def get_best_performing_model(self) -> str:
        """Get the model with best success rate"""
        best_model = None
        best_rate = 0.0
        
        for model, perf in self.model_performance.items():
            if perf['total_calls'] > 0:
                success_rate = perf['successful_calls'] / perf['total_calls']
                if success_rate > best_rate:
                    best_rate = success_rate
                    best_model = model
        
        return best_model or self.get_active_model()
    
    async def health_check(self) -> bool:
        """Comprehensive health check of LLM handler"""
        if not self.is_initialized:
            return False
        
        if not self.active_models:
            return False
        
        # Quick test of primary model
        primary_model = self.get_active_model()
        if primary_model == 'none':
            return False
        
        try:
            # Quick test with minimal request
            test_answer = await self.generate_answer(
                question="Test question?",
                context="Test context for health check.",
                request_id="health_check"
            )
            
            return test_answer is not None and len(test_answer) > 0
            
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive LLM handler statistics"""
        stats = {
            'initialization': {
                'initialized': self.is_initialized,
                'active_models': self.active_models,
                'total_models_available': len(self.active_models),
                'model_priority': self.model_priority
            },
            'performance': {
                'total_requests': self.total_requests,
                'successful_requests': self.successful_requests,
                'success_rate': (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0,
                'total_processing_time': self.total_processing_time,
                'avg_processing_time': (self.total_processing_time / self.total_requests) if self.total_requests > 0 else 0
            },
            'model_statistics': {},
            'usage_distribution': self.model_usage_stats,
            'current_status': {
                'primary_model': self.get_active_model(),
                'best_performing_model': self.get_best_performing_model(),
                'models_online': len(self.active_models),
                'last_updated': datetime.now().isoformat()
            }
        }
        
        # Calculate detailed model statistics
        for model, perf in self.model_performance.items():
            if perf['total_calls'] > 0:
                success_rate = (perf['successful_calls'] / perf['total_calls']) * 100
                stats['model_statistics'][model] = {
                    'success_rate': round(success_rate, 2),
                    'total_calls': perf['total_calls'],
                    'successful_calls': perf['successful_calls'],
                    'failed_calls': perf['failed_calls'],
                    'avg_response_time': round(perf['avg_response_time'], 3),
                    'last_success': perf['last_success'],
                    'last_failure': perf['last_failure'],
                    'status': 'active' if model in self.active_models else 'inactive'
                }
        
        return stats
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get simplified statistics (backwards compatibility)"""
        return self.get_comprehensive_statistics()
    
    async def close(self):
        """Clean up resources and connections"""
        print("🔄 Closing LLM Handler...")
        
        self.active_models.clear()
        self.model_performance.clear()
        self.model_usage_stats.clear()
        self.is_initialized = False
        
        print("✅ LLM Handler closed successfully")
    
    def __str__(self) -> str:
        """String representation of LLM Handler"""
        if not self.is_initialized:
            return "LLMHandler(not initialized)"
        
        return f"LLMHandler(models={len(self.active_models)}, primary={self.get_active_model()}, requests={self.total_requests})"
    
    def __repr__(self) -> str:
        """Detailed representation"""
        return f"LLMHandler(active_models={self.active_models}, initialized={self.is_initialized}, total_requests={self.total_requests})"