# app/services/conversation_memory.py - Fast Conversation Memory Service
import logging
import time
from typing import Dict, List, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

logger = logging.getLogger(__name__)

class ConversationMemoryService:
    """Fast in-memory conversation history with MongoDB Atlas fallback"""
    
    def __init__(self, atlas_service=None):
        """
        Initialize conversation memory service
        
        Args:
            atlas_service: Optional MongoDB Atlas service for persistence
        """
        self.atlas_service = atlas_service
        self.memory_store = {}  # In-memory store for fast access
        self.session_stats = {}
        
        logger.info("✅ Conversation Memory Service initialized")
    
    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """
        Get conversation history for session (fast in-memory with Atlas backup)
        
        Args:
            session_id: Session identifier
            
        Returns:
            BaseChatMessageHistory: Chat history
        """
        try:
            # First try in-memory store for speed
            if session_id in self.memory_store:
                logger.info(f"📚 Retrieved history from memory for session: {session_id}")
                return self.memory_store[session_id]
            
            # Create new history
            if self.atlas_service:
                # Use Atlas if available
                try:
                    history = self.atlas_service.get_chat_history_fast(session_id)
                    self.memory_store[session_id] = history
                    logger.info(f"📚 Retrieved history from Atlas for session: {session_id}")
                    return history
                except Exception as e:
                    logger.warning(f"⚠️ Atlas fallback failed, using in-memory: {str(e)}")
            
            # Fallback to in-memory
            history = ChatMessageHistory()
            self.memory_store[session_id] = history
            logger.info(f"📚 Created new in-memory history for session: {session_id}")
            return history
            
        except Exception as e:
            logger.error(f"❌ Failed to get session history: {str(e)}")
            # Always return something
            return ChatMessageHistory()
    
    def add_exchange(self, session_id: str, human_message: str, ai_message: str):
        """
        Add a conversation exchange (question + answer)
        
        Args:
            session_id: Session identifier
            human_message: User question
            ai_message: AI response
        """
        try:
            start_time = time.time()
            
            history = self.get_session_history(session_id)
            
            # Add messages
            history.add_message(HumanMessage(content=human_message))
            history.add_message(AIMessage(content=ai_message))
            
            # Update stats
            if session_id not in self.session_stats:
                self.session_stats[session_id] = {
                    "exchanges": 0,
                    "total_questions": 0,
                    "total_answers": 0,
                    "created_at": time.time()
                }
            
            self.session_stats[session_id]["exchanges"] += 1
            self.session_stats[session_id]["total_questions"] += 1
            self.session_stats[session_id]["total_answers"] += 1
            self.session_stats[session_id]["last_updated"] = time.time()
            
            add_time = time.time() - start_time
            logger.info(f"💬 Added conversation exchange for session {session_id} in {add_time:.3f}s")
            
        except Exception as e:
            logger.error(f"❌ Failed to add conversation exchange: {str(e)}")
    
    def get_recent_context(self, session_id: str, max_exchanges: int = 3) -> str:
        """
        Get recent conversation context for enhanced prompting
        
        Args:
            session_id: Session identifier
            max_exchanges: Maximum number of recent exchanges
            
        Returns:
            str: Formatted conversation context
        """
        try:
            context_start = time.time()
            
            history = self.get_session_history(session_id)
            messages = history.messages
            
            if not messages:
                return ""
            
            # Get recent messages (pairs of human + AI)
            recent_messages = messages[-(max_exchanges * 2):]
            
            context_parts = []
            for i in range(0, len(recent_messages), 2):
                if i + 1 < len(recent_messages):
                    human_msg = recent_messages[i]
                    ai_msg = recent_messages[i + 1]
                    
                    if human_msg.type == "human" and ai_msg.type == "ai":
                        context_parts.append(f"Previous Q: {human_msg.content}")
                        context_parts.append(f"Previous A: {ai_msg.content}")
            
            context = "\n".join(context_parts)
            
            context_time = time.time() - context_start
            logger.info(f"🔄 Generated context for session {session_id} in {context_time:.3f}s")
            
            return context
            
        except Exception as e:
            logger.error(f"❌ Failed to get recent context: {str(e)}")
            return ""
    
    def enhance_query_with_context(self, session_id: str, current_query: str) -> str:
        """
        Enhance current query with conversation context
        
        Args:
            session_id: Session identifier
            current_query: Current user query
            
        Returns:
            str: Enhanced query with context
        """
        try:
            context = self.get_recent_context(session_id, max_exchanges=2)
            
            if not context:
                return current_query
            
            # Create enhanced query
            enhanced_query = f"""Based on our previous conversation:
{context}

Current question: {current_query}

Please answer the current question while considering the conversation history for better context."""
            
            logger.info(f"🔍 Enhanced query with conversation context for session: {session_id}")
            return enhanced_query
            
        except Exception as e:
            logger.error(f"❌ Failed to enhance query: {str(e)}")
            return current_query
    
    def get_session_summary(self, session_id: str) -> Dict:
        """
        Get conversation session summary
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dict: Session summary
        """
        try:
            history = self.get_session_history(session_id)
            messages = history.messages
            stats = self.session_stats.get(session_id, {})
            
            summary = {
                "session_id": session_id,
                "total_messages": len(messages),
                "conversation_exchanges": len(messages) // 2,
                "memory_type": "in_memory_with_atlas_backup" if self.atlas_service else "in_memory_only",
                "session_active": len(messages) > 0,
                "performance": {
                    "fast_retrieval": True,
                    "context_enhancement": True,
                    "conversation_aware": True
                }
            }
            
            # Add timing stats if available
            if stats:
                summary.update({
                    "created_at": stats.get("created_at"),
                    "last_updated": stats.get("last_updated"),
                    "total_exchanges": stats.get("exchanges", 0)
                })
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Failed to get session summary: {str(e)}")
            return {"error": str(e), "session_id": session_id}
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear conversation history for session
        
        Args:
            session_id: Session identifier
            
        Returns:
            bool: True if successful
        """
        try:
            # Clear from memory
            if session_id in self.memory_store:
                del self.memory_store[session_id]
            
            if session_id in self.session_stats:
                del self.session_stats[session_id]
            
            # Clear from Atlas if available
            if self.atlas_service:
                try:
                    self.atlas_service.clear_chat_history_fast(session_id)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to clear from Atlas: {str(e)}")
            
            logger.info(f"🗑️ Cleared conversation history for session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to clear session: {str(e)}")
            return False
    
    def get_all_sessions(self) -> List[str]:
        """
        Get list of all active session IDs
        
        Returns:
            List[str]: List of session IDs
        """
        return list(self.memory_store.keys())
    
    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """
        Clean up old sessions to free memory
        
        Args:
            max_age_hours: Maximum age of sessions in hours
            
        Returns:
            int: Number of sessions cleaned
        """
        try:
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            sessions_to_remove = []
            for session_id, stats in self.session_stats.items():
                if "created_at" in stats:
                    age = current_time - stats["created_at"]
                    if age > max_age_seconds:
                        sessions_to_remove.append(session_id)
            
            # Remove old sessions
            for session_id in sessions_to_remove:
                self.clear_session(session_id)
            
            logger.info(f"🧹 Cleaned up {len(sessions_to_remove)} old sessions")
            return len(sessions_to_remove)
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup old sessions: {str(e)}")
            return 0
    
    def health_check(self) -> bool:
        """
        Health check for conversation memory service
        
        Returns:
            bool: True if healthy
        """
        try:
            # Test creating and retrieving a session
            test_session = "health_check_session"
            history = self.get_session_history(test_session)
            
            # Clean up test session
            self.clear_session(test_session)
            
            logger.info("✅ Conversation memory health check passed")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Conversation memory health check failed: {str(e)}")
            return False