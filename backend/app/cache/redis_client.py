import redis
from app.config import get_settings
from functools import lru_cache

@lru_cache
def get_redis():
    settings = get_settings()
    r = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
    try:
        r.ping()
    except Exception as e:
        raise RuntimeError(f"Redis connect failed: {e}")
    return r
