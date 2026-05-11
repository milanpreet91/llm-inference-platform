import asyncio
import json
import logging
import os
import time
import sys

import httpx
import redis.asyncio as aioredis

from cost_tracker import CostTracker
from complexity import score_complexity

sys.path.append("/app/rag-pipeline")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ollama-worker")

REDIS_URL = f"redis://{os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', 6379)}"
OLLAMA_URL = f"http://{os.getenv('OLLAMA_HOST', 'localhost')}:{os.getenv('OLLAMA_PORT', 11434)}"
SMALL_MODEL = os.getenv("SMALL_MODEL", "llama3.2:1b")
LARGE_MODEL = os.getenv("LARGE_MODEL", "llama3.2:3b")
COMPLEXITY_THRESHOLD = float(os.getenv("COMPLEXITY_THRESHOLD", 0.5))
QUEUE_KEY = "inference_queue"


async def call_ollama(model: str, prompt: str) -> dict:
    """Call Ollama and return response + token count."""
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "answer": data["response"],
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0),
        }


async def process_job(redis: aioredis.Redis, raw: str) -> None:
    job = json.loads(raw)
    job_id = job["job_id"]
    query = job["query"]
    request_id = job.get("request_id", "unknown")

    start = time.monotonic()

    # Route: score complexity → pick model
    complexity = score_complexity(query)
    model = LARGE_MODEL if complexity >= COMPLEXITY_THRESHOLD else SMALL_MODEL

    logger.info(json.dumps({
        "event": "routing",
        "request_id": request_id,
        "job_id": job_id,
        "complexity": complexity,
        "model": model,
    }))

    # Build RAG prompt
    try:
        from retriever import RAGRetriever
        retriever = RAGRetriever()
        prompt = retriever.build_prompt(query)
    except Exception as e:
        logger.warning(f"RAG unavailable, using raw query: {e}")
        prompt = query

    # Inference with fallback
    try:
        result = await call_ollama(model, prompt)

