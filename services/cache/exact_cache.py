import json
import logging
from typing import Optional
import redis.asyncio as aioredis

logger = logging.getLogger("cache.exact")

EXACT_CACHE_TTL = 3600  # 1 hour


class ExactCache:
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client

    def _key(self, query: str) -> str:
        return f"exact:{query.strip().lower()}"

    async def get(self, query: str) -> Optional[dict]:
        try:
            val = await self.redis.get(self._key(query))
            if val:
                logger.info(f"Exact cache HIT for query: {query[:60]}")
                return json.loads(val)
        except Exception as e:
            logger.warning(f"Exact cache GET failed: {e}")
        return None

    async def set(self, query: str, answer: str, model: str) -> None:
        try:
            payload = json.dumps({"answer": answer, "model": model})
            await self.redis.setex(self._key(query), EXACT_CACHE_TTL, payload)
            logger.info(f"Exact cache SET for query: {query[:60]}")
        except Exception as e:
            logger.warning(f"Exact cache SET failed: {e}")
