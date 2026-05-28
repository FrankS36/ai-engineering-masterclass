# Chapter 4: Building with LLM APIs

Every production AI feature starts the same way: an HTTP request carrying a prompt, and a response carrying generated text. The gap between that first successful curl and a system that serves thousands of users reliably is where most engineering effort lives. This chapter covers the full surface area -- from authentication through structured outputs and streaming -- so you can build features that are correct, fast, and economical.

---

## API Fundamentals

### The Messages Array

All major providers have converged on the same core abstraction: a list of messages, each tagged with a role.

```python
from openai import OpenAI

client = OpenAI()  # reads OPENAI_API_KEY from env

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a senior code reviewer. Be concise."},
        {"role": "user", "content": "Review this function: def add(a, b): return a + b"},
    ],
    temperature=0.2,
)

print(response.choices[0].message.content)
```

Three roles matter:

- **system** -- sets behavior, persona, and constraints. Processed once at the start. Some providers call this a "system prompt" or "preamble."
- **user** -- the end-user's input.
- **assistant** -- the model's prior responses. You include these when building multi-turn conversations.

Anthropic's SDK uses a slightly different shape -- the system prompt is a top-level parameter, not a message:

```python
import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system="You are a senior code reviewer. Be concise.",
    messages=[
        {"role": "user", "content": "Review this function: def add(a, b): return a + b"},
    ],
)

print(response.content[0].text)
```

Google's Gemini API follows a similar pattern but uses `contents` instead of `messages` and calls the roles `user` and `model`.

### Authentication and Rate Limiting

Every provider uses bearer-token authentication via API keys. In production, store keys in a secrets manager (AWS Secrets Manager, GCP Secret Manager, Vault), not in environment variables baked into container images.

Rate limits come in two flavors: **requests per minute (RPM)** and **tokens per minute (TPM)**. Hitting either returns a 429. The response headers tell you your current usage and limits -- read them.

---

## Provider Landscape

| Dimension | OpenAI | Anthropic | Google (Gemini) |
|---|---|---|---|
| SDK style | `openai` Python/TS | `anthropic` Python/TS | `google-genai` Python/TS |
| System prompt | message with `role: system` | top-level `system` param | `system_instruction` param |
| Streaming | `stream=True` returns iterator | `stream()` context manager | `stream=True` on generate |
| Structured outputs | native JSON schema enforcement | tool use with schema | JSON mode, function calling |
| Pricing model | per-token (input/output split) | per-token (input/output split) | per-token, free tier available |
| Notable quirk | strict schema mode rejects invalid JSON | prefill (start assistant response) | very large context windows (1M+) |

All three providers charge differently for input versus output tokens, and output tokens are typically 3-5x more expensive. Prompt caching (available from OpenAI and Anthropic) can cut input costs by 50-90% for repeated prefixes.

---

## Multi-Turn Conversations

The model is stateless. Every request must include the full conversation history. This means you are responsible for managing state.

### Basic Conversation Loop

```python
conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
]

def chat(user_message: str) -> str:
    conversation.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=conversation,
    )

    assistant_message = response.choices[0].message.content
    conversation.append({"role": "assistant", "content": assistant_message})
    return assistant_message
```

### Context Window Management

Every model has a finite context window. When your conversation exceeds it, the API returns an error. Strategies for staying within bounds:

1. **Sliding window** -- drop the oldest messages, keeping the system prompt and the last N turns. Simple but loses early context.
2. **Summarization** -- periodically ask the model to summarize the conversation so far, replace the history with that summary, and continue. Preserves key information at the cost of an extra API call.
3. **Hybrid** -- keep the system prompt, a running summary, and the last 5-10 messages. Best balance for most applications.

```python
def summarize_and_trim(messages: list[dict], model: str = "gpt-4o-mini") -> list[dict]:
    """Replace old messages with a summary when context grows too large."""
    system_msg = messages[0]
    history = messages[1:]

    summary_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Summarize this conversation in 2-3 sentences."},
            *history,
        ],
        max_tokens=200,
    )

    summary = summary_response.choices[0].message.content
    return [
        system_msg,
        {"role": "assistant", "content": f"[Conversation summary: {summary}]"},
        *history[-6:],  # keep last 3 turns (6 messages)
    ]
```

Count tokens before sending. Use `tiktoken` for OpenAI models or the provider's token counting endpoint.

---

## Error Handling and Retries

APIs fail. Networks drop. Rate limits trigger. Production code must handle all of this gracefully.

### Exponential Backoff with Jitter

```python
import time
import random
from openai import (
    OpenAI,
    APIConnectionError,
    RateLimitError,
    APIStatusError,
)

def call_with_retries(
    client: OpenAI,
    messages: list[dict],
    model: str = "gpt-4o",
    max_retries: int = 5,
    base_delay: float = 1.0,
) -> str:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            return response.choices[0].message.content

        except RateLimitError:
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            time.sleep(delay)

        except APIConnectionError:
            if attempt == max_retries - 1:
                raise
            time.sleep(base_delay)

        except APIStatusError as e:
            if e.status_code >= 500:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
            else:
                raise  # 4xx errors (except 429) are not retryable

    raise RuntimeError(f"Failed after {max_retries} retries")
```

### Model Fallback Chains

When your primary model is down or overloaded, fall through to alternatives:

```python
FALLBACK_CHAIN = [
    ("gpt-4o", OpenAI()),
    ("claude-sonnet-4-20250514", anthropic.Anthropic()),
    ("gpt-4o-mini", OpenAI()),  # cheaper, always available
]

def resilient_completion(messages: list[dict]) -> str:
    for model, client in FALLBACK_CHAIN:
        try:
            if isinstance(client, anthropic.Anthropic):
                resp = client.messages.create(
                    model=model,
                    max_tokens=1024,
                    messages=[m for m in messages if m["role"] != "system"],
                    system=next(
                        (m["content"] for m in messages if m["role"] == "system"), ""
                    ),
                )
                return resp.content[0].text
            else:
                resp = client.chat.completions.create(model=model, messages=messages)
                return resp.choices[0].message.content
        except Exception as e:
            logging.warning(f"{model} failed: {e}")
            continue

    raise RuntimeError("All models in fallback chain failed")
```

---

## Cost Tracking and Budgeting

LLM costs sneak up on you. A single unoptimized endpoint can burn through hundreds of dollars a day.

### Per-Request Cost Formula

```
cost = (input_tokens * input_price_per_token) + (output_tokens * output_price_per_token)
```

For GPT-4o at $2.50 / 1M input and $10.00 / 1M output: a request with 2,000 input tokens and 500 output tokens costs $0.01. That is 1,000 requests for $10. Sounds cheap until your feature gets 100,000 requests a day.

### Tracking Implementation

```python
import logging
from dataclasses import dataclass

@dataclass
class UsageRecord:
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    user_id: str
    feature: str

PRICING = {
    "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
    "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
    "claude-sonnet-4-20250514": {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
}

def track_usage(response, model: str, user_id: str, feature: str) -> UsageRecord:
    usage = response.usage
    prices = PRICING[model]
    cost = (usage.prompt_tokens * prices["input"]) + (
        usage.completion_tokens * prices["output"]
    )

    record = UsageRecord(
        model=model,
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        cost_usd=cost,
        user_id=user_id,
        feature=feature,
    )

    # Ship to your monitoring system
    logging.info(f"LLM cost: ${cost:.6f} | {model} | {feature} | user={user_id}")
    return record
```

Set alerts at the user level (e.g., $5/day per user), the feature level (e.g., $200/day for the summarization endpoint), and the organization level. Kill switches that disable non-critical AI features when budgets are exceeded are not over-engineering -- they are basic operational hygiene.

---

## Structured Outputs

### The Problem

LLMs produce text. Applications consume typed data. Bridging that gap is one of the most common challenges in AI engineering. Ask a model to "extract the product name and price" and you might get:

```
The product is "Widget Pro" and it costs $29.99.
```

Useful to a human. Useless to `json.loads()`.

### JSON Mode

The simplest approach. OpenAI and others offer a `response_format` parameter:

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "Extract product info. Return JSON with keys: name, price_cents, in_stock."},
        {"role": "user", "content": "The Widget Pro costs $29.99 and is currently available."},
    ],
    response_format={"type": "json_object"},
)

data = json.loads(response.choices[0].message.content)
```

JSON mode guarantees valid JSON but does not enforce a schema. The model might return `{"product": "Widget Pro"}` instead of `{"name": "Widget Pro"}`. You still need validation.

### Function Calling / Tool Use

Define a schema as a "function" the model can call. The model returns structured arguments matching your schema:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "save_product",
            "description": "Save extracted product information",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Product name"},
                    "price_cents": {"type": "integer", "description": "Price in cents"},
                    "in_stock": {"type": "boolean", "description": "Whether the product is in stock"},
                },
                "required": ["name", "price_cents", "in_stock"],
            },
        },
    }
]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    tool_choice={"type": "function", "function": {"name": "save_product"}},
)

args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
```

This works across all three major providers. Anthropic calls it "tool use" and the response shape differs slightly, but the concept is identical.

### OpenAI Structured Outputs with Strict Schema

OpenAI offers a `strict` mode that guarantees the output matches your JSON schema exactly -- not just valid JSON, but valid according to your schema:

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "product_extraction",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "price_cents": {"type": "integer"},
                    "in_stock": {"type": "boolean"},
                },
                "required": ["name", "price_cents", "in_stock"],
                "additionalProperties": False,
            },
        },
    },
)
```

This uses constrained decoding -- the model literally cannot produce tokens that would violate the schema. The first request with a new schema has higher latency as the provider compiles the grammar.

### Instructor: The Practical Choice

The Instructor library wraps any provider's client and lets you define schemas as Pydantic models. It handles retries, validation, and provider differences:

```python
import instructor
from pydantic import BaseModel, Field, field_validator

class Product(BaseModel):
    name: str = Field(description="Product name as it appears in the listing")
    price_cents: int = Field(description="Price in US cents", ge=0)
    in_stock: bool
    category: str | None = Field(default=None, description="Product category if mentioned")

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Product name cannot be empty")
        return v.strip()

client = instructor.from_openai(OpenAI())

product = client.chat.completions.create(
    model="gpt-4o",
    response_model=Product,
    max_retries=3,  # re-prompts model on validation failure
    messages=[
        {"role": "user", "content": "The Widget Pro costs $29.99 and is available now."},
    ],
)

print(product.name)         # "Widget Pro"
print(product.price_cents)  # 2999
print(product.in_stock)     # True
```

Instructor works with OpenAI, Anthropic, Google, Mistral, and local models. The `max_retries` parameter is key: if the model returns data that fails Pydantic validation, Instructor sends the validation error back to the model and asks it to fix the output.

### Complex Schema Patterns

Real applications need more than flat objects:

```python
from enum import Enum
from pydantic import BaseModel, Field

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class SubTask(BaseModel):
    title: str
    estimated_hours: float = Field(ge=0, le=100)

class TaskExtraction(BaseModel):
    title: str
    description: str
    priority: Priority
    assignee: str | None = None
    subtasks: list[SubTask] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list, max_length=10)
```

Nested objects, enums, optional fields, constrained lists -- Pydantic handles all of it, and Instructor translates the schema to whatever format the provider expects.

### Constrained Generation for Local Models

If you run models locally (via vLLM, llama.cpp, or similar), the Outlines library gives you schema-enforced generation at the token level:

```python
import outlines

model = outlines.models.transformers("mistralai/Mistral-7B-v0.1")
generator = outlines.generate.json(model, Product)
result = generator("Extract: The Widget Pro costs $29.99 and is in stock.")
```

This compiles your Pydantic schema into a finite-state machine that masks invalid tokens during generation. Zero post-hoc retries needed.

### Best Practices for Schema Design

- **Use descriptive field names and descriptions.** `price_cents` is better than `price` -- it eliminates ambiguity about the unit.
- **Represent money as integers (cents), not floats.** Floating-point arithmetic and currency do not mix.
- **Use enums for categorical fields.** The model will pick from your options, not invent new ones.
- **Keep schemas as flat as possible.** Deep nesting increases error rates.
- **Set reasonable constraints** (`ge`, `le`, `max_length`) so validation catches nonsense early.

### Provider Comparison for Structured Outputs

| Capability | OpenAI | Anthropic | Google |
|---|---|---|---|
| JSON mode | Yes | Via tool use | Yes |
| Strict schema enforcement | Yes (native) | No (use Instructor) | Partial |
| Function calling | Yes | Yes (tool use) | Yes |
| Instructor support | Yes | Yes | Yes |
| Constrained decoding | Yes (strict mode) | No | No |

**Practitioner's note:** Structured outputs solve a real problem, but don't confuse schema compliance with correctness. A perfectly formatted JSON response that contains hallucinated data is worse than a messy response you'd have double-checked. Validate the content, not just the shape. If the model returns `{"price_cents": 0}` for a product that costs $29.99, your Pydantic model will happily accept it. Build domain-level validation -- not just type-level validation.

---

## Streaming

### Why Streaming Matters

Without streaming, the user stares at a blank screen for 2-10 seconds while the model generates its full response. With streaming, the first token appears in 200-500ms. The total generation time is identical, but the perceived experience is dramatically better.

The key metric is **Time to First Token (TTFT)** -- the delay between sending the request and receiving the first token of the response. Streaming does not reduce TTFT, but it lets you display content as soon as it arrives instead of waiting for the complete response.

### SSE Protocol Basics

Most providers use Server-Sent Events (SSE). The server sends a stream of `data:` lines, each containing a JSON chunk, terminated by `data: [DONE]`. You don't need to implement this yourself -- the SDKs handle it -- but understanding the protocol helps when debugging.

### Python Backend

```python
from openai import OpenAI

client = OpenAI()

def stream_response(messages: list[dict]):
    """Generator that yields text chunks as they arrive."""
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content
```

Expose this as an SSE endpoint in your web framework:

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.post("/chat")
async def chat(request: ChatRequest):
    def event_generator():
        for token in stream_response(request.messages):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

With Anthropic, streaming uses a context manager:

```python
with client.messages.stream(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=messages,
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

### React/TypeScript Frontend with Vercel AI SDK

The Vercel AI SDK handles the SSE parsing, state management, and rendering:

```typescript
// app/api/chat/route.ts
import { openai } from "@ai-sdk/openai";
import { streamText } from "ai";

export async function POST(req: Request) {
  const { messages } = await req.json();

  const result = streamText({
    model: openai("gpt-4o"),
    messages,
  });

  return result.toDataStreamResponse();
}
```

```typescript
// app/page.tsx
"use client";
import { useChat } from "@ai-sdk/react";

export default function Chat() {
  const { messages, input, handleInputChange, handleSubmit, isLoading, stop } =
    useChat();

  return (
    <div>
      {messages.map((m) => (
        <div key={m.id}>
          <strong>{m.role}:</strong> {m.content}
        </div>
      ))}
      <form onSubmit={handleSubmit}>
        <input value={input} onChange={handleInputChange} />
        <button type="submit" disabled={isLoading}>Send</button>
        {isLoading && <button onClick={stop}>Cancel</button>}
      </form>
    </div>
  );
}
```

### Vanilla JavaScript Client

If you are not using React:

```javascript
async function streamChat(messages) {
  const response = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages }),
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop(); // keep incomplete line in buffer

    for (const line of lines) {
      if (line.startsWith("data: ") && line !== "data: [DONE]") {
        const data = JSON.parse(line.slice(6));
        document.getElementById("output").textContent += data.token;
      }
    }
  }
}
```

### Display Patterns

How you render streamed tokens affects perceived quality:

- **Token-by-token** -- append each token immediately. Fastest display but can look jittery, especially with subword tokens.
- **Buffered (50ms interval)** -- accumulate tokens in a buffer and flush every 50ms. Smoother visual flow.
- **Word-by-word** -- buffer until a whitespace boundary, then flush. Natural reading pace.
- **Markdown rendering** -- accumulate the full response and re-render markdown on each flush. Libraries like `react-markdown` handle this well, but re-rendering on every token is expensive. Throttle to every 100ms or use a streaming-aware markdown renderer.

### Streaming with Tool Calls

When the model invokes tools during streaming, you receive the function name and arguments as fragments that must be accumulated:

```python
tool_calls = {}

for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.tool_calls:
        for tc in delta.tool_calls:
            idx = tc.index
            if idx not in tool_calls:
                tool_calls[idx] = {"name": "", "arguments": ""}
            if tc.function.name:
                tool_calls[idx]["name"] = tc.function.name
            if tc.function.arguments:
                tool_calls[idx]["arguments"] += tc.function.arguments

# After stream completes, parse and execute
for tc in tool_calls.values():
    args = json.loads(tc["arguments"])
    result = execute_tool(tc["name"], args)
```

### Streaming Structured Outputs

Instructor supports streaming partial objects, letting you display structured data as it forms:

```python
import instructor
from openai import OpenAI

client = instructor.from_openai(OpenAI())

for partial in client.chat.completions.create_partial(
    model="gpt-4o",
    response_model=Product,
    messages=[
        {"role": "user", "content": "Extract: Widget Pro, $29.99, in stock"},
    ],
    stream=True,
):
    # partial is a Product with fields populated as they arrive
    # partial.name might be "Wid" then "Widget" then "Widget Pro"
    print(partial.model_dump())
```

This is useful for progressive UI updates -- show the product name as soon as it is available, then fill in the price, then the stock status.

### Error Handling for Streams

Streams can fail mid-response. The connection might drop, the server might error after sending partial data, or the client might time out.

```python
import asyncio

async def stream_with_timeout(messages, timeout_seconds=30):
    try:
        stream = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            stream=True,
        )

        full_response = ""
        last_chunk_time = time.time()

        for chunk in stream:
            if time.time() - last_chunk_time > timeout_seconds:
                raise TimeoutError("No data received for {timeout_seconds}s")

            delta = chunk.choices[0].delta
            if delta.content:
                full_response += delta.content
                yield delta.content
            last_chunk_time = time.time()

    except Exception as e:
        # You have full_response up to the failure point
        # Decide: return partial, retry from scratch, or retry with context
        logging.error(f"Stream failed after {len(full_response)} chars: {e}")
        raise
```

### UX Considerations

- **Typing indicator** -- show a pulsing cursor or "thinking..." state during the TTFT delay, before any tokens arrive.
- **Cancellation** -- always give users a way to stop generation. On the backend, close the stream; the provider will stop generating and you stop paying for tokens.
- **Smart auto-scroll** -- scroll to follow new content, but stop auto-scrolling if the user scrolls up to read earlier content. Resume auto-scroll when they scroll back to the bottom.
- **Skeleton states** -- for structured output streaming, show placeholder UI (gray boxes for fields) that fills in as data arrives.

---

## Putting It All Together

A production LLM integration combines all of these pieces: authenticated API calls with retry logic, conversation management that respects context windows, structured output extraction with validation, streaming for responsive UX, and cost tracking on every request. None of these are optional for systems serving real users.

Start with the simplest approach that works -- a single API call with JSON mode and basic retries -- and add complexity as your requirements demand it. Instructor for structured outputs, streaming for user-facing features, fallback chains for reliability. Layer them incrementally, measure the impact of each, and resist the urge to over-engineer before you have traffic.
