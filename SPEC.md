# Mini LLM Inference Server

A lightweight LLM serving project focused on **backend systems, concurrent inference, dynamic batching, streaming responses, caching, observability, and performance benchmarking**.

The goal is **not** to implement a language model from scratch. Instead, the project uses an existing small Hugging Face causal language model and focuses on the systems engineering required to serve that model efficiently.

---

## 1. Project Goal

Build a production-style inference service that accepts multiple concurrent text-generation requests and serves them through an HTTP API.

The project should demonstrate:

- API design
- asynchronous request handling
- request queueing
- dynamic batching
- token streaming
- inference scheduling
- caching
- timeout and error handling
- containerization
- testing
- performance measurement
- latency/throughput trade-offs

The main engineering question is:

> How can multiple concurrent LLM inference requests be served efficiently while balancing throughput and latency?

---

## 2. High-Level Architecture

```text
Clients
   |
   v
FastAPI / HTTP API
   |
   v
Request Validation
   |
   v
Async Request Queue
   |
   v
Dynamic Batch Scheduler
   |
   v
Model Worker
   |
   v
Hugging Face / PyTorch Model
   |
   +----> Streaming Responses
   |
   +----> Metrics / Benchmarking
```

A request should not immediately invoke the model.

Instead, incoming requests enter a queue. A scheduler collects compatible requests over a short batching window and executes them together as a batch.

---

## 3. Suggested Technology Stack

- **Python**
- **FastAPI**
- **PyTorch**
- **Hugging Face Transformers**
- **asyncio**
- **Pydantic**
- **Docker**
- **pytest**
- **httpx**
- optional: **Prometheus client**
- optional: **Redis**

Use a small model that can run locally on available hardware, for example:

- Qwen2.5-0.5B-Instruct
- TinyLlama
- another small Hugging Face causal language model

The exact model is not important. The focus is the inference system around it.

---

## 4. Core API

### `POST /generate`

Example request:

```json
{
  "prompt": "Explain virtual memory in simple terms.",
  "max_new_tokens": 128,
  "temperature": 0.7,
  "top_p": 0.9
}
```

Example response:

```json
{
  "text": "...",
  "input_tokens": 9,
  "output_tokens": 103,
  "latency_ms": 842
}
```

### `POST /v1/chat/completions`

Optional OpenAI-style endpoint.

Example:

```json
{
  "model": "local-model",
  "messages": [
    {
      "role": "user",
      "content": "What is a page fault?"
    }
  ],
  "max_tokens": 128,
  "temperature": 0.7,
  "stream": false
}
```

Supporting a familiar API format makes the project closer to real inference infrastructure.

### `GET /health`

Returns whether the service and model worker are healthy.

Example:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### `GET /metrics`

Optional endpoint for runtime metrics such as:

- total requests
- failed requests
- current queue size
- active requests
- batches executed
- average batch size
- generated tokens
- request latency

---

## 5. Async Request Queue

Incoming requests should be converted into internal request objects and placed into an asynchronous queue.

Example internal request structure:

```python
InferenceRequest(
    request_id=...,
    prompt=...,
    max_new_tokens=...,
    temperature=...,
    top_p=...,
    created_at=...,
    future=...
)
```

Each request waits on a future or another synchronization primitive until the model worker completes its inference.

This separates:

- HTTP request handling
- scheduling
- model execution

That separation is one of the key architectural ideas in the project.

---

## 6. Dynamic Batching

Implement a scheduler that collects several requests into a batch.

Example policy:

```text
MAX_BATCH_SIZE = 8
BATCH_WAIT_MS = 20
```

The scheduler executes a batch when either:

1. the queue reaches `MAX_BATCH_SIZE`, or
2. the oldest request has waited `BATCH_WAIT_MS`.

Example:

```text
t = 0 ms    request A arrives
t = 4 ms    request B arrives
t = 9 ms    request C arrives
t = 20 ms   batching window expires

A + B + C -> one model batch
```

This creates a measurable trade-off:

- longer wait window -> potentially larger batches and higher throughput
- shorter wait window -> lower latency but smaller batches

This trade-off should be one of the main results discussed in the README.

---

## 7. Batch Compatibility

Initially, requests can be batched even if their prompt lengths differ by using tokenizer padding.

A later improvement can group requests using simple bucketing, for example:

```text
short prompts
medium prompts
long prompts
```

This reduces wasted computation caused by excessive padding.

Do not over-engineer this in the first version.

---

## 8. Model Worker

The model worker owns the model and performs inference.

Responsibilities:

- load tokenizer
- load model
- tokenize prompts
- create padded batch
- run generation
- decode outputs
- return results to waiting requests
- record timing information

Only the model worker should directly interact with the model.

The API layer should not call `model.generate()` directly.

---

## 9. Concurrency

The service should support multiple simultaneous clients.

Test scenarios such as:

```text
1 concurrent request
4 concurrent requests
8 concurrent requests
16 concurrent requests
32 concurrent requests
```

The important distinction is:

> HTTP concurrency is not the same as parallel model execution.

Many clients can submit requests concurrently, while the scheduler decides how those requests are grouped into model batches.

---

## 10. Streaming Responses

Add token streaming as a second phase of the project.

Possible implementation:

- Server-Sent Events (SSE), or
- FastAPI `StreamingResponse`

Example:

```text
data: Virtual
data: memory
data: allows
data: ...
```

Streaming improves perceived latency because the user receives generated tokens before the full response is complete.

The benchmark should distinguish:

- total request latency
- time to first token

---

## 11. Cache

Implement a small cache for identical deterministic requests.

Cache key can include:

```text
prompt
max_new_tokens
temperature
top_p
model version
```

For deterministic generation:

```text
temperature = 0
```

an identical request can safely reuse a cached result.

Start with an in-memory LRU cache.

Optional extension:

- Redis-backed cache

Measure:

- cache-hit latency
- cache-miss latency
- cache hit ratio

---

## 12. Timeouts and Cancellation

Requests should not wait forever.

Support:

- request timeout
- queue timeout
- client cancellation
- graceful error response

Example:

```json
{
  "error": "inference_timeout"
}
```

This is important because production inference systems must handle overloaded or slow workers safely.

---

## 13. Backpressure

Do not allow the request queue to grow without limit.

Example:

```text
MAX_QUEUE_SIZE = 100
```

If the queue is full:

```http
HTTP 503 Service Unavailable
```

or:

```http
HTTP 429 Too Many Requests
```

This protects the server during overload.

---

## 14. Graceful Shutdown

On shutdown:

1. stop accepting new requests
2. allow the current batch to finish
3. resolve or cancel remaining queued requests
4. release model resources
5. exit cleanly

This is a useful production-style systems feature and a good interview discussion point.

---

## 15. Docker

Provide a `Dockerfile`.

Example usage:

```bash
docker build -t mini-llm-server .
docker run -p 8000:8000 mini-llm-server
```

The container should expose the API server and load the configured model at startup.

Optional environment configuration:

```text
MODEL_NAME
MAX_BATCH_SIZE
BATCH_WAIT_MS
MAX_QUEUE_SIZE
DEVICE
```

---

## 16. Testing

### Unit tests

Test:

- request validation
- batching logic
- queue behavior
- cache behavior
- timeout handling
- error handling

### Integration tests

Test:

```text
client -> API -> queue -> scheduler -> model worker -> response
```

### Concurrency tests

Launch many requests simultaneously and verify:

- no lost requests
- no duplicate responses
- correct request/result mapping
- server remains stable

---

## 17. Benchmarking

Benchmarking is a central part of the project.

Create a script such as:

```text
benchmarks/load_test.py
```

Parameters:

```text
concurrency
number of requests
batch size
batch wait time
prompt length
max output tokens
```

---

## 18. Metrics to Measure

### Throughput

```text
requests / second
```

and:

```text
generated tokens / second
```

### End-to-End Latency

Measure:

```text
p50
p95
p99
```

### Time To First Token

For streaming:

```text
TTFT = first_token_time - request_arrival_time
```

### Time Per Output Token

Optional:

```text
TPOT
```

### Queue Time

```text
queue_time = batch_start_time - request_arrival_time
```

This helps distinguish scheduler waiting from model compute time.

### Average Batch Size

Measure how large batches become under different concurrency levels.

---

## 19. Experiments

### Experiment 1 — No batching vs dynamic batching

Compare:

```text
one request -> one model invocation
```

against:

```text
multiple requests -> dynamically formed batch
```

Measure:

- requests/sec
- tokens/sec
- p50 latency
- p95 latency
- p99 latency

### Experiment 2 — Batching window

Try:

```text
0 ms
5 ms
10 ms
20 ms
50 ms
```

Show how waiting slightly longer can improve throughput but worsen latency.

### Experiment 3 — Maximum batch size

Try:

```text
1
2
4
8
16
```

Find the point where larger batches stop helping.

### Experiment 4 — Concurrency

Try:

```text
1
4
8
16
32 clients
```

Observe:

- queue growth
- batch size
- throughput
- tail latency

### Experiment 5 — Cache

Compare repeated deterministic requests:

```text
cache disabled
cache enabled
```

---

## 20. Example Benchmark Table

```text
| Mode             | Concurrency | Avg Batch | Req/s | Tokens/s | p50 | p95 | p99 |
|------------------|-------------|-----------|-------|----------|-----|-----|-----|
| No batching      | 8           | 1.0       | ...   | ...      | ... | ... | ... |
| Dynamic batching | 8           | 3.7       | ...   | ...      | ... | ... | ... |
| Dynamic batching | 32          | 7.4       | ...   | ...      | ... | ... | ... |
```

For streaming:

```text
| Configuration | TTFT p50 | TTFT p95 | Tokens/s |
|---------------|----------|----------|
| ...           | ...      | ...      |
```

---

## 21. Suggested Repository Structure

```text
mini-llm-inference-server/
│
├── app/
│   ├── main.py
│   ├── api/
│   │   ├── routes.py
│   │   └── schemas.py
│   │
│   ├── inference/
│   │   ├── worker.py
│   │   ├── scheduler.py
│   │   ├── request.py
│   │   └── batching.py
│   │
│   ├── cache/
│   │   └── lru.py
│   │
│   ├── metrics/
│   │   └── metrics.py
│   │
│   └── config.py
│
├── benchmarks/
│   ├── load_test.py
│   └── analyze_results.py
│
├── tests/
│   ├── test_api.py
│   ├── test_scheduler.py
│   ├── test_batching.py
│   └── test_cache.py
│
├── Dockerfile
├── requirements.txt
├── README.md
└── results/
    └── benchmark_results.csv
```

---

## 22. Implementation Order

### Phase 1 — Minimal server

Implement:

- FastAPI
- model loading
- `/generate`
- single-request inference

Goal:

```text
HTTP request -> model -> response
```

### Phase 2 — Queue and worker

Separate API handling from model execution.

Implement:

- async queue
- inference request object
- background model worker

### Phase 3 — Dynamic batching

Implement:

- batching window
- maximum batch size
- padded batched inference

This is the core systems feature.

### Phase 4 — Benchmarking

Before adding more features, benchmark:

- no batching
- batching
- several concurrency levels

### Phase 5 — Streaming

Add:

- SSE or streaming HTTP response
- TTFT measurement

### Phase 6 — Reliability

Add:

- timeouts
- bounded queue
- overload response
- graceful shutdown
- error handling

### Phase 7 — Cache

Add:

- in-memory LRU cache
- cache metrics

### Phase 8 — Docker and documentation

Add:

- Dockerfile
- configuration
- architecture diagram
- benchmark tables
- explanation of observed trade-offs

---

## 23. What Not to Implement

The project should remain focused.

Do **not** spend time implementing:

- a Transformer model from scratch
- custom CUDA kernels
- custom tokenizer
- distributed training
- full Kubernetes deployment
- full vLLM clone
- complicated frontend
- authentication system
- database unless it serves a clear purpose

The value of the project is in the serving architecture and performance analysis.

---

## 24. What the Project Demonstrates

### Backend Engineering

- FastAPI
- REST API design
- async programming
- request validation
- error handling
- streaming responses

### Systems Engineering

- concurrency
- queues
- scheduling
- batching
- backpressure
- timeouts
- graceful shutdown
- latency/throughput trade-offs

### ML Systems

- model loading
- inference
- batched generation
- tokenization
- model serving

### Performance Engineering

- load testing
- benchmarking
- throughput
- TTFT
- p50/p95/p99 latency
- queue time
- batch efficiency

### Software Engineering

- modular architecture
- testing
- Docker
- configuration
- reproducible benchmarks

---

## 25. Possible CV Entry

**Mini LLM Inference Server**
*Python | FastAPI | PyTorch | Hugging Face | Docker*

- Built a concurrent LLM serving system with asynchronous request queueing, dynamic batching and streaming text generation.
- Implemented bounded queues, request timeouts, caching and graceful shutdown for reliable inference under concurrent load.
- Benchmarked throughput, TTFT and p50/p95/p99 latency across different concurrency levels, batch sizes and batching windows, quantifying the latency-throughput trade-off.

---

## 26. Strong README Result

The final README should not only describe what was implemented. It should explain what was learned.

A strong result section could say something like:

> Dynamic batching increased throughput by X% under 16 concurrent clients compared with single-request execution. A 20 ms batching window produced larger average batches and higher token throughput, but increased p95 TTFT by Y%. Increasing the batching window beyond 20 ms provided little additional throughput while continuing to increase latency.

That type of result shows understanding of inference systems rather than simply demonstrating that the code runs.

---

## 27. Definition of Done

The project is complete when:

- the server accepts concurrent requests
- requests pass through an async queue
- the scheduler dynamically creates batches
- batches are executed by a model worker
- normal and streaming inference work
- the queue is bounded
- timeout/error paths work
- the server shuts down cleanly
- tests cover core scheduling logic
- the project runs through Docker
- benchmark results compare multiple batching/concurrency configurations
- the README explains the measured latency/throughput trade-offs

At that point, the project is already strong enough to discuss in an inference/backend/systems interview.
