# LLM Inference Platform
A production-style LLM inference platform serving Llama 3 via Ollama, exposing a unified API with intelligent model routing, two-tier caching, 
RAG over EDA documentation, async request queuing, and a full observability stack.
Built to demonstrate production ML infrastructure patterns: cost-aware routing, backpressure control, cache-first architecture, and three-pillar observability.

<img width="575" height="701" alt="image" src="https://github.com/user-attachments/assets/c8238f69-b0e1-42fa-b33e-20cac2459a3a" />

#### Project Structure
```
│   .env
│   .env.example
│   docker-compose.yml
│
├───docs
│       eda_basics.txt
│
└───services
    ├───api-gateway
    │   │   1`
    │   │   Dockerfile
    │   │   main.py
    │   │   middleware.py
    │   │   requirements.txt
    │   │
    │   └───__pycache__
    │           main.cpython-311.pyc
    │           middleware.cpython-311.pyc
    │
    ├───cache
    │       exact_cache.py
    │       requirements.txt
    │       semantic_cache.py
    │
    ├───model-router
    ├───observability
    │       prometheus.yml
    │
    ├───ollama-worker
    │   │   complexity.py
    │   │   cost_tracker.py
    │   │   Dockerfile
    │   │   requirements.txt
    │   │   test.py
    │   │   worker.py
    │   │
    │   └───__pycache__
    │           complexity.cpython-311.pyc
    │           cost_tracker.cpython-311.pyc
    │
    └───rag-pipeline
            Dockerfile
            ingest.py
            retriever.py
```
#### Running Locally
##### Prerequisites
  - Docker Desktop running
  - 4GB+ free RAM recommended

----------------------------------------------------------------------
##### Steps to run:
```
cd llm-inference-platform
cp .env.example .env
mkdir docs
(Add your .txt documentation files to docs/)
docker compose up --build
```
--------------------------------------------------------------------------
#### API Reference
```
Health Check: GET /health
Query: POST /query
```
#### Observability
Every request generates three types of observability data:
1. **Structured logs**: JSON logs for events where every event is tagged with request_id
2. **Prometheus metrics** tracked per request
    - request_latency_ms -> histogram
    - cache_hit_total -> counter by type (exact/semantic/none)
    - token_cost_usd -> cumulative spend
    - model_requests_total -> counter by model
3. **Distributed tracing**: request_id threads through every component log, enabling full request timeline reconstruction in Jaeger.

#### Cost Model
Cost is simulated per request using token counts returned by Ollama:
*cost = (prompt_tokens + completion_tokens) / 1000 × rate_per_1k*
Cache hits cost $0 -> the primary motivation for the two-tier caching architecture.

