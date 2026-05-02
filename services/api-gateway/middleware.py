import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from fastapi.responses import JSONResponse

RATE_LIMIT = 10          # requests
RATE_WINDOW = 60         # per 60 seconds

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request.state.request_id = str(uuid.uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, redis_client_getter):
        super().__init__(app)
        self.get_redis = redis_client_getter

    async def dispatch(self, request: Request, call_next):
        if request.url.path in ("/health", "/favicon.ico"):
            return await call_next(request)

        redis = self.get_redis()

        if redis is None:
            # Redis unavailable — allow request through, don't crash
            return await call_next(request)

        try:
            client_ip = request.client.host
            key = f"rate:{client_ip}"
            count = await redis.incr(key)
            if count == 1:
                await redis.expire(key, RATE_WINDOW)

            if count > RATE_LIMIT:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded. Try again later."},
                    headers={"X-Request-ID": getattr(request.state, "request_id", "")},
                )
        except Exception:
            # Redis connection failed mid-request — allow through gracefully
            pass

        return await call_next(request)
