# Production and Deployment

Every AI prototype works in a demo. The model responds, the output looks good, and someone says "ship it." Then production happens. Users send inputs you never imagined. Costs scale linearly with adoption. The model fails silently at 3 AM, and nobody notices until a customer complains on Monday. The distance between a working prototype and a production system is the entire discipline of AI engineering.

This chapter covers everything between "it works on my laptop" and "it runs reliably at scale": architecture, observability, reliability, deployment, scaling, cost management, security, and the caching strategies that make production economics viable.

> **Practitioner's note:** Production-ready AI isn't a quality bar for the model — it's a quality bar for the system. It handles inputs the training data didn't include, degrades predictably, and someone who didn't build it can operate it.

## The Production Gap

The gap between prototype and production is not a smooth continuum. It is a phase transition. Nearly every dimension of the system changes simultaneously.

| Dimension | Prototype | Production |
|---|---|---|
| **Users** | Single developer testing | Thousands of concurrent users |
| **Inputs** | Happy path, curated examples | Edge cases, adversarial inputs, empty strings, 50K-token documents |
| **Cost sensitivity** | Doesn't matter | Every token counts; cost per request is a KPI |
| **Latency** | Flexible, seconds are fine | Under 2 seconds or users leave |
| **Failure tolerance** | Failures are learning | 99.9% uptime expected, failures page an engineer |
| **Observability** | Print statements | Structured logs, distributed traces, alerts, dashboards |
| **Rollback** | Delete the notebook | Blue-green deploys, feature flags, instant rollback |
| **Security** | API key in .env | Secrets management, input validation, rate limiting, output filtering |

Most teams underestimate this gap because the prototype was easy. The ease of the prototype is precisely why the gap is so dangerous — it creates false confidence about what remains.

## Architecture Patterns

Three architectural patterns cover the vast majority of production AI deployments. The right choice depends on latency requirements, payload size, and user experience expectations.

> **Practitioner's note:** The algorithm was never the hard part. AI fails at the handoff — between the team that built it and the team that operates it. Each handoff loses critical context about how the model works and what constitutes acceptable behavior.

### Synchronous API

The simplest pattern. The client sends a request, waits for the model response, and gets it back in one shot.

```
Client --> API Gateway --> AI Service --> LLM Provider
                                    <--
       <--              <--
```

Best for: short completions under 2 seconds, classification, extraction, structured outputs. Worst for: long-running generations, document summarization, anything where the model needs 10+ seconds.

### Async / Queue-Based

The client submits a request and gets a job ID. A worker pulls from the queue, processes the request, and stores the result. The client polls or gets a webhook callback.

```
Client --> API --> Message Queue --> Worker --> LLM Provider
       <-- (job_id)                        --> Result Store
Client --> API --> Result Store
       <-- (result)
```

Best for: batch processing, long-running tasks, workloads with unpredictable latency. This pattern also gives you natural backpressure — if the queue grows, you add workers instead of dropping requests.

### Streaming

The client opens a connection and receives tokens as they are generated. This is the pattern behind every chatbot interface.

```
Client <--SSE/WebSocket--> API --> LLM Provider (stream=True)
         token by token
```

Best for: conversational interfaces, any response over 3 seconds where users need feedback that something is happening. Streaming does not reduce total latency — it reduces perceived latency, which matters more for user experience.

In practice, most production systems combine these patterns. A chatbot uses streaming for the conversation and async queues for background tasks like document indexing or batch evaluation.

## Observability

You cannot improve what you do not measure, and you cannot debug what you did not log.

> **Practitioner's note:** If your model started returning random outputs at 2 AM Saturday, how long before a specific human knows? If the answer involves checking a dashboard Monday morning, you're watching, not monitoring. Monitoring means thresholds, alerts to specific people, runbooks, documented actions, and closed loops.

### What to Log

Every LLM call should produce a structured log entry containing:

```python
import time
import uuid
import json
from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class LLMCallLog:
    request_id: str
    timestamp: float
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    status: str  # "success", "error", "timeout"
    user_id: Optional[str] = None
    feature: Optional[str] = None
    tool_calls: Optional[list] = None
    error_message: Optional[str] = None
    cost_usd: Optional[float] = None

def log_llm_call(model, input_tokens, output_tokens, latency_ms,
                 status, user_id=None, feature=None, **kwargs):
    entry = LLMCallLog(
        request_id=str(uuid.uuid4()),
        timestamp=time.time(),
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        status=status,
        user_id=user_id,
        feature=feature,
        **kwargs
    )
    # Send to your logging backend (Datadog, CloudWatch, etc.)
    print(json.dumps(asdict(entry)))
    return entry
```

### Metrics to Track

| Metric | Why It Matters | Alert Threshold (example) |
|---|---|---|
| **p50/p95/p99 latency** | User experience degrades at the tail | p95 > 3s |
| **Error rate** | Model or provider failures | > 1% over 5 min |
| **Token usage per request** | Cost and prompt bloat | Avg > 2x baseline |
| **Cost per request** | Budget management | Daily spend > 120% of forecast |
| **Output quality score** | Drift detection (if you have evals) | Score drops > 10% from baseline |
| **Cache hit rate** | Optimization effectiveness | < 30% when expected > 60% |

### Tracing with Request IDs

Every request entering your system gets a unique ID that propagates through every service call, LLM invocation, and database query. When something goes wrong, you pull the request ID and see the entire chain: what the user sent, how the prompt was constructed, what the model returned, and what the user received.

Without request-level tracing, debugging production AI issues is guesswork.

## Reliability

> **Practitioner's note:** Most AI teams build. Very few AI teams maintain. Until organizations create roles, incentives, and accountability specifically for AI maintenance, every deployed model is an orphan slowly degrading without anyone watching.

LLM providers have outages. Rate limits get hit. Models return malformed outputs. Production systems need defensive patterns at every layer.

### Retries with Exponential Backoff and Jitter

```python
import random
import time
import httpx

def call_with_retry(fn, max_retries=3, base_delay=1.0, max_delay=30.0):
    """Retry with exponential backoff and jitter."""
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except (httpx.HTTPStatusError, httpx.TimeoutException) as e:
            if attempt == max_retries:
                raise
            # Don't retry client errors (4xx) except 429
            if hasattr(e, 'response') and 400 <= e.response.status_code < 500:
                if e.response.status_code != 429:
                    raise
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = random.uniform(0, delay * 0.5)
            time.sleep(delay + jitter)
```

The jitter is critical. Without it, all clients retry at the same time after an outage, creating a thundering herd that causes the next outage.

### Circuit Breaker

When a downstream service is failing, stop hammering it. A circuit breaker tracks failures and trips open after a threshold, returning errors immediately without making the call. After a cooldown period, it lets a single request through to test if the service recovered.

```python
import time
from enum import Enum
from threading import Lock

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery

class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=30.0,
                 success_threshold=2):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self._lock = Lock()

    def call(self, fn, fallback=None):
        with self._lock:
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                else:
                    if fallback:
                        return fallback()
                    raise RuntimeError("Circuit breaker is open")

        try:
            result = fn()
            with self._lock:
                if self.state == CircuitState.HALF_OPEN:
                    self.success_count += 1
                    if self.success_count >= self.success_threshold:
                        self.state = CircuitState.CLOSED
                        self.failure_count = 0
                else:
                    self.failure_count = 0
            return result
        except Exception as e:
            with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitState.OPEN
                if self.state == CircuitState.HALF_OPEN:
                    self.state = CircuitState.OPEN
            if fallback:
                return fallback()
            raise
```

### Model and Provider Fallback

When your primary model is unavailable or degraded, fall back to an alternative. This requires testing your application against multiple models ahead of time — a fallback you have never tested is not a fallback.

```python
FALLBACK_CHAIN = [
    {"provider": "anthropic", "model": "claude-sonnet-4-20250514"},
    {"provider": "openai", "model": "gpt-4o"},
    {"provider": "anthropic", "model": "claude-haiku-4-20250514"},
]

def call_with_fallback(messages, fallback_chain=FALLBACK_CHAIN):
    for config in fallback_chain:
        try:
            return call_provider(config["provider"], config["model"], messages)
        except Exception as e:
            log_llm_call(model=config["model"], input_tokens=0,
                        output_tokens=0, latency_ms=0, status="error",
                        error_message=str(e))
            continue
    raise RuntimeError("All providers in fallback chain failed")
```

## Deployment Strategies

AI features need the same deployment discipline as any critical service, with additional considerations for model behavior changes.

**Blue-green deployment.** Run two identical environments. Route all traffic to blue (current). Deploy the new version to green. Run smoke tests against green. Switch the router to green. If anything goes wrong, switch back to blue in seconds. For AI features, smoke tests must include evaluation checks — not just "does it respond" but "does it respond correctly."

**Canary deployment.** Route 5% of traffic to the new version. Monitor error rates, latency, and output quality metrics. If metrics hold, gradually increase to 25%, 50%, 100%. For AI features, "output quality" often requires sampling and human review, not just automated metrics.

**Feature flags.** Wrap the AI feature behind a flag. Roll out to internal users first, then beta users, then everyone. Feature flags also let you instantly kill an AI feature that starts misbehaving without deploying anything.

Rolling back an AI feature is harder than rolling back a traditional feature because AI outputs may have been stored, sent to users, or used in downstream decisions. Keep an audit trail of which model version generated which outputs.

## Scaling

Horizontal scaling for AI services follows standard patterns with one caveat: LLM calls are slow (hundreds of milliseconds to seconds), which means each worker spends most of its time waiting. This makes AI services I/O-bound, not CPU-bound.

**Async processing.** Use `asyncio` or equivalent to handle many concurrent LLM calls per worker. A single async worker can manage 50+ concurrent LLM calls because it is just waiting on network I/O.

**Batching.** When processing many items, batch them. Some providers offer batch APIs at 50% discount (OpenAI's Batch API, Anthropic's Message Batches). Even without provider batching, grouping items reduces per-request overhead.

**Queue-based architecture.** For workloads that spike (a user uploads 100 documents for processing), put work items on a queue and scale workers independently. This decouples ingestion speed from processing speed and prevents cascading failures.

## Cost Management

At prototype scale, cost is invisible. At production scale, it is the line item that kills projects.

### Model Tiering

Route requests to the cheapest model that can handle them. Simple tasks do not need frontier models.

```python
from enum import Enum

class Tier(Enum):
    SIMPLE = "simple"       # Classification, extraction, short answers
    STANDARD = "standard"   # General Q&A, summarization
    COMPLEX = "complex"     # Multi-step reasoning, code generation, analysis

MODEL_TIERS = {
    Tier.SIMPLE: {"model": "claude-haiku-4-20250514", "cost_per_1k_input": 0.0008},
    Tier.STANDARD: {"model": "claude-sonnet-4-20250514", "cost_per_1k_input": 0.003},
    Tier.COMPLEX: {"model": "claude-opus-4-20250514", "cost_per_1k_input": 0.015},
}

def classify_request_tier(message: str, tool_calls: list = None) -> Tier:
    """Route to cheapest capable model based on request characteristics."""
    if tool_calls and len(tool_calls) > 2:
        return Tier.COMPLEX
    if len(message.split()) < 20 and not any(
        kw in message.lower() for kw in ["analyze", "compare", "explain why"]
    ):
        return Tier.SIMPLE
    return Tier.STANDARD

def call_tiered(message: str, **kwargs):
    tier = classify_request_tier(message, kwargs.get("tool_calls"))
    config = MODEL_TIERS[tier]
    return call_provider("anthropic", config["model"], message, **kwargs)
```

### Cost Attribution

Tag every LLM call with the user ID and feature name. Aggregate daily. This answers "which feature costs the most?" and "which users are outliers?" — questions you will be asked when the bill arrives.

### Budget Alerts

Set hard limits per feature, per user tier, and globally. When spend hits 80% of the daily budget, alert. When it hits 100%, degrade gracefully (switch to cheaper models, disable non-critical features) rather than going dark.

## Security in Production

### Secrets Management

Never hardcode API keys. Use environment variables at minimum, a secrets manager (AWS Secrets Manager, HashiCorp Vault, GCP Secret Manager) in production. Rotate keys on a schedule and after any suspected leak.

### Input Validation

Validate every user input before it reaches the model. Enforce maximum length. Strip or reject known prompt injection patterns. Never trust that users will send what the UI allows.

```python
import re

MAX_INPUT_LENGTH = 10000

def validate_input(text: str) -> str:
    if not text or not text.strip():
        raise ValueError("Input cannot be empty")
    if len(text) > MAX_INPUT_LENGTH:
        raise ValueError(f"Input exceeds maximum length of {MAX_INPUT_LENGTH}")
    # Basic injection pattern detection
    suspicious = re.findall(
        r"(ignore previous|disregard|forget your instructions|system prompt)",
        text.lower()
    )
    if suspicious:
        # Log for review, don't necessarily block
        log_security_event("potential_injection", patterns=suspicious)
    return text.strip()
```

### Rate Limiting

Rate limit per user, not just per IP. AI calls are expensive. A single abusive user can run up significant costs in minutes.

### Output Filtering

Check model outputs before returning them to users. Filter for PII leakage, harmful content, and responses that violate your application's policies. This is your last line of defense.

## Prompt Caching and Cost Optimization

Caching is the single highest-leverage cost optimization for production AI systems. Most applications send the same system prompt, the same few-shot examples, and the same context documents on every call. Without caching, you pay full price to process identical content thousands of times per day.

### The Cost Problem

Consider a customer support bot with a 3,000-token system prompt. At 1,000 requests per hour, you process 3 million redundant input tokens per hour. At $3 per million input tokens, that is $9/hour or $216/day — just for the system prompt. The actual user messages are a fraction of the total cost.

### Provider-Level Prompt Caching

Providers now offer built-in caching that dramatically reduces costs for repeated prompt prefixes.

**Anthropic: Explicit Cache Control**

Anthropic gives you direct control over caching with `cache_control` breakpoints. Cached input tokens cost 90% less than standard input tokens. There is a small write cost on the first request, then reads are deeply discounted.

```python
import anthropic

client = anthropic.Anthropic()

SYSTEM_PROMPT = """You are a customer support agent for Acme Corp.
You have access to our complete product catalog, return policies,
and troubleshooting guides. Always be helpful and precise..."""  # ~3000 tokens

def cached_support_call(user_message: str):
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"}
            }
        ],
        messages=[{"role": "user", "content": user_message}]
    )
    # Check cache performance in response headers
    usage = response.usage
    print(f"Input tokens: {usage.input_tokens}")
    print(f"Cache read tokens: {getattr(usage, 'cache_read_input_tokens', 0)}")
    print(f"Cache creation tokens: {getattr(usage, 'cache_creation_input_tokens', 0)}")
    return response
```

After the first call writes the cache, subsequent calls with the same prefix read from cache. For a 3,000-token system prompt, the savings are immediate and substantial.

**OpenAI: Automatic Prefix Caching**

OpenAI automatically caches prompt prefixes longer than 1,024 tokens. Cached tokens receive a 50% discount. No code changes are required — if your prompt starts with the same prefix, caching happens transparently. Check `usage.prompt_tokens_details.cached_tokens` in the response to verify.

### Semantic Caching

Provider caching handles identical prefixes. Semantic caching handles similar queries — different phrasings of the same question. You compute an embedding of the query, check if a similar embedding exists in your cache, and return the cached response if the similarity is above a threshold.

```python
import hashlib
import json
import numpy as np
import redis
import openai

r = redis.Redis(host="localhost", port=6379, db=0)
embed_client = openai.OpenAI()

SIMILARITY_THRESHOLD = 0.95
CACHE_TTL = 3600  # 1 hour

def get_embedding(text: str) -> list[float]:
    response = embed_client.embeddings.create(
        model="text-embedding-3-small", input=text
    )
    return response.data[0].embedding

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def semantic_cache_get(query: str):
    query_embedding = get_embedding(query)
    # Scan cached embeddings (use vector DB in production for scale)
    for key in r.scan_iter("sem_cache:*"):
        cached = json.loads(r.get(key))
        sim = cosine_similarity(query_embedding, cached["embedding"])
        if sim >= SIMILARITY_THRESHOLD:
            return cached["response"]
    return None

def semantic_cache_set(query: str, response: str):
    embedding = get_embedding(query)
    cache_key = f"sem_cache:{hashlib.sha256(query.encode()).hexdigest()[:16]}"
    r.setex(cache_key, CACHE_TTL, json.dumps({
        "embedding": embedding,
        "query": query,
        "response": response
    }))
```

Semantic caching is powerful for FAQ-style workloads where users ask the same questions in different ways. Set the similarity threshold high (0.95+) to avoid returning irrelevant cached responses.

### Response Caching for Deterministic Queries

For queries that have a single correct answer — data lookups, status checks, factual retrieval — use exact-match caching with `temperature=0`.

```python
import hashlib
import json
import redis

r = redis.Redis(host="localhost", port=6379, db=0)

def exact_cache_key(model: str, messages: list, tools: list = None) -> str:
    """Deterministic cache key from request parameters."""
    key_data = json.dumps({"model": model, "messages": messages,
                           "tools": tools or []}, sort_keys=True)
    return f"exact:{hashlib.sha256(key_data.encode()).hexdigest()}"

def cached_llm_call(model, messages, tools=None, ttl=1800):
    key = exact_cache_key(model, messages, tools)
    cached = r.get(key)
    if cached:
        return json.loads(cached)
    response = call_provider("anthropic", model, messages, tools=tools)
    r.setex(key, ttl, json.dumps(response))
    return response
```

### Caching by Use Case

**RAG systems** benefit from multi-level caching. Cache the retrieval results (same query returns the same documents), cache the final response for exact query matches, and use provider-level caching for the system prompt and few-shot examples that stay constant.

**Chatbots** generate massive savings from FAQ caching. Analyze your logs to find the top 100 questions users ask. Pre-compute and cache responses. For a support bot, this alone can handle 40-60% of traffic without making a single LLM call.

**Batch processing** pipelines should deduplicate before sending to the model. If you are processing 10,000 documents and 3,000 are duplicates, cache after the first processing of each unique document.

### Cost Savings: A Worked Example

Consider an application making 10,000 LLM calls per day with a 3,000-token system prompt and an average 500-token user message at $3/million input tokens.

| Component | Without Caching | With Caching |
|---|---|---|
| System prompt tokens | 30M tokens/day | 30M tokens cached at 90% discount |
| System prompt cost | $90.00/day | $9.00/day + $0.30 cache writes |
| User message tokens | 5M tokens/day | 5M tokens (no caching) |
| User message cost | $15.00/day | $15.00/day |
| Semantic cache hits (40%) | — | 4,000 calls avoided |
| Avoided call savings | — | $42.00/day |
| **Daily total** | **$105.00** | **$24.30** |

Actual numbers depend on your cache hit rates and token volumes. But the pattern is consistent: combining provider-level prompt caching with semantic and exact-match response caching routinely reduces costs by 60-90%.

### Cache Invalidation

Cache invalidation is famously hard, and AI caching adds its own complications. Invalidate when:

- **The model changes.** A new model version may produce different (better) outputs. Include the model identifier in every cache key.
- **The system prompt changes.** Any prompt edit should bust the cache. Version your prompts and include the version in cache keys.
- **Knowledge is updated.** If your RAG system indexes new documents, cached responses based on old retrieval results are stale.

Use versioned cache keys to make invalidation explicit:

```python
PROMPT_VERSION = "v3"
MODEL_VERSION = "claude-sonnet-4-20250514"

def versioned_cache_key(query: str) -> str:
    raw = f"{PROMPT_VERSION}:{MODEL_VERSION}:{query}"
    return f"v_cache:{hashlib.sha256(raw.encode()).hexdigest()}"
```

When you update the prompt, bump `PROMPT_VERSION`. All old cache entries expire naturally via TTL while new requests populate fresh cache entries.

### Structuring Prompts for Caching

Provider-level caching works on prefixes. Content at the beginning of your prompt that stays the same across requests benefits from caching. Content that changes per request should go at the end.

Structure your prompts as:

1. **System prompt** (static) — instructions, persona, rules
2. **Few-shot examples** (static) — reference examples
3. **Retrieved context** (semi-static) — documents, knowledge base chunks
4. **User message** (dynamic) — the actual request

This ordering maximizes the cacheable prefix length.

### TTL Recommendations by Content Type

| Content Type | Recommended TTL | Rationale |
|---|---|---|
| System prompt responses | 24 hours | Changes only on deployment |
| FAQ / common questions | 4-8 hours | Answers are stable, refresh for freshness |
| RAG-based responses | 30-60 minutes | Source documents may be updated |
| User-specific responses | 5-15 minutes | Personalization context changes frequently |
| Real-time data queries | No caching | Stale data is worse than the cost of a fresh call |

### Monitoring Cache Performance

Track three numbers: **hit rate** (percentage of requests served from cache), **latency delta** (cached response time vs. uncached), and **cost savings** (estimated spend avoided).

```python
import time

class CacheMetrics:
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.cached_latencies = []
        self.uncached_latencies = []

    def record_hit(self, latency_ms: float):
        self.hits += 1
        self.cached_latencies.append(latency_ms)

    def record_miss(self, latency_ms: float):
        self.misses += 1
        self.uncached_latencies.append(latency_ms)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def avg_latency_savings_ms(self) -> float:
        if not self.cached_latencies or not self.uncached_latencies:
            return 0.0
        return (sum(self.uncached_latencies) / len(self.uncached_latencies) -
                sum(self.cached_latencies) / len(self.cached_latencies))

    def summary(self) -> dict:
        return {
            "hit_rate": f"{self.hit_rate:.1%}",
            "total_requests": self.hits + self.misses,
            "avg_latency_savings_ms": f"{self.avg_latency_savings_ms:.0f}",
        }
```

A healthy cache shows a hit rate above 40% for general workloads and above 70% for FAQ-heavy applications. If your hit rate is below 20%, either your traffic has high cardinality (every query is unique) or your cache keys are too specific. Review your TTLs and similarity thresholds.

## Bringing It All Together

Production AI is not one concern — it is all of these concerns operating simultaneously. Your system needs retry logic that respects circuit breakers, cost management that interacts with model tiering, caching that invalidates correctly when prompts change, and observability that covers all of it.

The teams that succeed in production are the ones that treat these as engineering problems with engineering solutions: automated, tested, monitored, and maintained. The model is the easy part. Everything around it is the work.
