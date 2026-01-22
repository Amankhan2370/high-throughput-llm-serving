"""
Caching layer for inference results.
Supports in-memory and Redis backends.
"""
import hashlib
import json
import time
from typing import Optional, Dict, Any
from cachetools import TTLCache
import logging

logger = logging.getLogger(__name__)


class Cache:
    """Unified caching interface."""
    
    def __init__(
        self,
        cache_type: str = "memory",
        redis_url: Optional[str] = None,
        ttl: int = 3600,
        max_size: int = 10000
    ):
        self.cache_type = cache_type
        self.ttl = ttl
        self.max_size = max_size
        
        if cache_type == "memory":
            self._cache: TTLCache = TTLCache(maxsize=max_size, ttl=ttl)
            self._redis_client = None
        elif cache_type == "redis":
            if not redis_url:
                raise ValueError("Redis URL required for Redis cache")
            self._init_redis(redis_url)
        else:
            raise ValueError(f"Unsupported cache type: {cache_type}")
    
    def _init_redis(self, redis_url: str):
        """Initialize Redis client."""
        try:
            import redis.asyncio as aioredis
            self._redis_client = aioredis.from_url(
                redis_url,
                decode_responses=True
            )
            logger.info("Redis cache initialized")
        except ImportError:
            logger.warning("aioredis not available, falling back to memory cache")
            self.cache_type = "memory"
            self._cache = TTLCache(maxsize=self.max_size, ttl=self.ttl)
            self._redis_client = None
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {str(e)}")
            logger.warning("Falling back to memory cache")
            self.cache_type = "memory"
            self._cache = TTLCache(maxsize=self.max_size, ttl=self.ttl)
            self._redis_client = None
    
    def _generate_key(self, text: str, config: Dict[str, Any]) -> str:
        """Generate cache key from input and config."""
        key_data = {
            "text": text,
            "config": config
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    async def get(self, text: str, config: Dict[str, Any]) -> Optional[str]:
        """
        Get cached result.
        
        Returns:
            Cached result or None if not found
        """
        key = self._generate_key(text, config)
        
        if self.cache_type == "memory":
            return self._cache.get(key)
        elif self.cache_type == "redis" and self._redis_client:
            try:
                result = await self._redis_client.get(key)
                return result
            except Exception as e:
                logger.error(f"Redis get error: {str(e)}")
                return None
        
        return None
    
    async def set(self, text: str, config: Dict[str, Any], result: str):
        """Set cache entry."""
        key = self._generate_key(text, config)
        
        if self.cache_type == "memory":
            self._cache[key] = result
        elif self.cache_type == "redis" and self._redis_client:
            try:
                await self._redis_client.setex(key, self.ttl, result)
            except Exception as e:
                logger.error(f"Redis set error: {str(e)}")
    
    async def clear(self):
        """Clear all cache entries."""
        if self.cache_type == "memory":
            self._cache.clear()
        elif self.cache_type == "redis" and self._redis_client:
            try:
                await self._redis_client.flushdb()
            except Exception as e:
                logger.error(f"Redis clear error: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self.cache_type == "memory":
            return {
                "type": "memory",
                "size": len(self._cache),
                "max_size": self.max_size,
                "ttl": self.ttl
            }
        elif self.cache_type == "redis":
            return {
                "type": "redis",
                "ttl": self.ttl
            }
        return {}
