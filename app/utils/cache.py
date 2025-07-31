# app/utils/cache.py
"""
Cache manager for improving response latency
Uses in-memory caching with optional Redis support
"""

import asyncio
import time
from typing import Any, Optional, Dict
import json
import hashlib

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

class CacheManager:
    """High-performance cache manager"""
    
    def __init__(self):
        self.memory_cache: Dict[str, Dict] = {}
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        self.redis_client = None
        self.max_memory_items = 1000  # Prevent memory overflow
    
    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        self.stats["total_requests"] += 1
        
        # Try memory cache first
        if key in self.memory_cache:
            cache_item = self.memory_cache[key]
            
            # Check if expired
            if cache_item["expires_at"] > time.time():
                self.stats["cache_hits"] += 1
                logger.debug(f"📦 Cache hit for key: {key[:20]}...")
                return cache_item["data"]
            else:
                # Remove expired item
                del self.memory_cache[key]
        
        self.stats["cache_misses"] += 1
        return None
    
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set item in cache"""
        if ttl is None:
            ttl = settings.CACHE_TTL
        
        try:
            # Clean up memory cache if too large
            if len(self.memory_cache) >= self.max_memory_items:
                await self._cleanup_expired()
            
            # Store in memory cache
            expires_at = time.time() + ttl
            self.memory_cache[key] = {
                "data": value,
                "expires_at": expires_at,
                "created_at": time.time()
            }
            
            logger.debug(f"💾 Cached item with key: {key[:20]}...")
            return True
            
        except Exception as e:
            logger.error(f"❌ Cache set failed: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all cache"""
        try:
            self.memory_cache.clear()
            self.stats = {
                "total_requests": 0,
                "cache_hits": 0,
                "cache_misses": 0
            }
            logger.info("🗑️ Cache cleared successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Cache clear failed: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.stats["total_requests"]
        hits = self.stats["cache_hits"]
        
        hit_rate = (hits / total * 100) if total > 0 else 0.0
        
        return {
            "total_requests": total,
            "cache_hits": hits,
            "cache_misses": self.stats["cache_misses"],
            "hit_rate": round(hit_rate, 2),
            "memory_items": len(self.memory_cache)
        }
    
    async def _cleanup_expired(self):
        """Remove expired items from memory cache"""
        current_time = time.time()
        expired_keys = [
            key for key, item in self.memory_cache.items()
            if item["expires_at"] <= current_time
        ]
        
        for key in expired_keys:
            del self.memory_cache[key]
        
        # If still too many items, remove oldest
        if len(self.memory_cache) >= self.max_memory_items:
            # Sort by creation time and remove oldest 20%
            sorted_items = sorted(
                self.memory_cache.items(),
                key=lambda x: x[1]["created_at"]
            )
            
            items_to_remove = len(sorted_items) // 5  # Remove 20%
            for key, _ in sorted_items[:items_to_remove]:
                del self.memory_cache[key]
        
        logger.info(f"🧹 Cleaned up cache: {len(self.memory_cache)} items remaining")
    
    def create_cache_key(self, *args) -> str:
        """Create a consistent cache key from arguments"""
        # Create hash from arguments
        content = json.dumps(args, sort_keys=True, default=str)
        return hashlib.md5(content.encode()).hexdigest()