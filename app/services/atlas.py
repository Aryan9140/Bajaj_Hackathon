# app/services/atlas.py - Clean MongoDB Atlas Service
import os
import logging
import time
from typing import Dict, List, Optional
from pymongo import MongoClient
from langchain_community.chat_message_histories import MongoDBChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)

class AtlasService:
    """MongoDB Atlas service for chat history storage"""
    
    def __init__(self, mongo_uri: str = None, database_name: str = "chat_history", collection_name: str = "message_store"):
        self.mongo_uri = mongo_uri or os.getenv("MONGODB_URI")
        self.database_name = database_name
        self.collection_name = collection_name
        self.client = None
        
        if not self.mongo_uri:
            raise ValueError("MongoDB URI is required")
        
        self._connect()
    
    def _connect(self):
        """Connect to MongoDB Atlas"""
        try:
            self.client = MongoClient(
                self.mongo_uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                maxPoolSize=10,
                retryWrites=True,
                w="majority"
            )
            self.client.admin.command('ping')
            logger.info("✅ Connected to MongoDB Atlas")
        except Exception as e:
            logger.error(f"❌ Failed to connect to MongoDB Atlas: {str(e)}")
            raise e
    
    def get_chat_history_fast(self, session_id: str) -> BaseChatMessageHistory:
        """Get chat history for session"""
        try:
            chat_history = MongoDBChatMessageHistory(
                connection_string=self.mongo_uri,
                session_id=session_id,
                database_name=self.database_name,
                collection_name=self.collection_name
            )
            logger.info(f"📚 Retrieved chat history for session: {session_id}")
            return chat_history
        except Exception as e:
            logger.error(f"❌ Failed to get chat history: {str(e)}")
            raise e
    
    def clear_chat_history_fast(self, session_id: str) -> bool:
        """Clear chat history for session"""
        try:
            chat_history = self.get_chat_history_fast(session_id)
            chat_history.clear()
            logger.info(f"🗑️ Cleared chat history for session: {session_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to clear chat history: {str(e)}")
            return False
    
    def get_recent_messages_optimized(self, session_id: str, limit: int = 10) -> List[BaseMessage]:
        """Get recent messages"""
        try:
            chat_history = self.get_chat_history_fast(session_id)
            messages = chat_history.messages
            recent_messages = messages[-limit:] if len(messages) > limit else messages
            logger.info(f"📖 Retrieved {len(recent_messages)} recent messages")
            return recent_messages
        except Exception as e:
            logger.error(f"❌ Failed to get recent messages: {str(e)}")
            return []
    
    def add_message_fast(self, session_id: str, message: BaseMessage) -> bool:
        """Add message to chat history"""
        try:
            chat_history = self.get_chat_history_fast(session_id)
            chat_history.add_message(message)
            logger.info(f"💬 Added message to session: {session_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to add message: {str(e)}")
            return False
    
    def get_session_stats_fast(self, session_id: str) -> Dict:
        """Get session statistics"""
        try:
            chat_history = self.get_chat_history_fast(session_id)
            messages = chat_history.messages
            
            user_messages = sum(1 for m in messages if m.type == "human")
            assistant_messages = sum(1 for m in messages if m.type == "ai")
            
            stats = {
                "session_id": session_id,
                "total_messages": len(messages),
                "user_messages": user_messages,
                "assistant_messages": assistant_messages,
                "conversation_active": len(messages) > 0,
                "performance": {
                    "atlas_connection": "optimized",
                    "retrieval_speed": "fast",
                    "storage_type": "mongodb_atlas"
                }
            }
            
            logger.info(f"📊 Generated stats for session: {session_id}")
            return stats
        except Exception as e:
            logger.error(f"❌ Failed to get stats: {str(e)}")
            return {"error": str(e), "session_id": session_id}
    
    def get_conversation_context(self, session_id: str, max_messages: int = 5) -> str:
        """Get conversation context"""
        try:
            recent_messages = self.get_recent_messages_optimized(session_id, max_messages)
            
            if not recent_messages:
                return ""
            
            context_parts = []
            for msg in recent_messages:
                if msg.type == "human":
                    context_parts.append(f"User: {msg.content}")
                elif msg.type == "ai":
                    context_parts.append(f"Assistant: {msg.content}")
            
            context = "\n".join(context_parts)
            logger.info(f"🔄 Generated conversation context for session: {session_id}")
            return context
        except Exception as e:
            logger.error(f"❌ Failed to get conversation context: {str(e)}")
            return ""
    
    def health_check_fast(self) -> bool:
        """Health check"""
        try:
            self.client.admin.command('ping')
            logger.info("✅ Atlas health check passed")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Atlas health check failed: {str(e)}")
            return False
    
    def close_connection_gracefully(self):
        """Close connection"""
        if self.client:
            try:
                self.client.close()
                logger.info("🔌 Closed MongoDB Atlas connection")
            except Exception as e:
                logger.warning(f"⚠️ Error closing connection: {str(e)}")
    
    def __del__(self):
        """Cleanup"""
        self.close_connection_gracefully()