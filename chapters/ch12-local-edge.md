# Local and Edge AI

## Why Run Models Locally

There are five reasons to move inference off the cloud and onto hardware you control:

**Privacy.** Some data cannot leave your infrastructure. Medical records, legal documents, classified material, and financial data may have regulatory requirements that prohibit sending content to third-party APIs. Local inference keeps everything on-premises.

**Cost at scale.** API pricing is per-token. At low volume, this is cheaper than owning hardware. At high volume, the math flips. If you're processing millions of tokens daily, a single GPU can pay for itself in weeks.

**Latency.** Network round-trips add 50-200ms minimum. For real-time applications — autocomplete, inline suggestions, voice assistants — local inference eliminates that overhead entirely.

**Offline capability.** Field workers, aircraft, submarines, remote locations. If there's no internet, there's no API. Local models work anywhere.

**Control.** No rate limits, no provider outages, no surprise model deprecations, no terms-of-service changes. You own the stack.

## Hardware Requirements

GPU memory is the primary constraint. A model must fit in VRAM to run efficiently, and quantization is how you make that happen.

| Model Size | FP16 (full) | Q8 (8-bit) | Q4 (4-bit) | Minimum GPU |
|-----------|-------------|------------|------------|-------------|
| 7B | 14 GB | 7 GB | 4 GB | RTX 4060 (8GB) |
| 13B | 26 GB | 13 GB | 7 GB | RTX 4070 Ti (12GB) |
| 34B | 68 GB | 34 GB | 17 GB | A100 40GB or 2x RTX 4090 |
| 70B | 140 GB | 70 GB | 35 GB | A100 80GB or 2x A100 40GB |

**CPU inference** is possible but slow — roughly 10-50x slower than GPU for generation. Acceptable for batch processing or low-throughput use cases. Apple Silicon (M1-M4) blurs the line with unified memory, making 7-13B models practical on laptops.

**RAM matters too.** The model loads into VRAM, but the KV cache (which stores context for generation) needs additional memory. Long conversations with 70B models can consume 20-40GB of additional VRAM depending on context length.

## Quantization

Quantization reduces the numerical precision of model weights, shrinking the model and speeding up inference at the cost of some quality.

### How It Works

A neural network's weights are typically stored as 16-bit floating point numbers (FP16). Quantization converts these to lower precision:

- **Q8 (INT8):** 8 bits per weight. ~50% size reduction. Quality loss is negligible for most tasks. This is the safe default.
- **Q6:** 6 bits. ~62% reduction. Minimal quality loss. Good middle ground.
- **Q4 (INT4):** 4 bits. ~75% reduction. Noticeable quality loss on complex reasoning and nuanced tasks. Fine for simple generation, classification, and extraction.
- **Q2:** 2 bits. ~87% reduction. Significant quality degradation. Only viable for the simplest tasks.

The common claim is "Q4 retains 95% of quality." This is roughly true for straightforward tasks like summarization and classification, but breaks down on tasks requiring precise reasoning, math, or nuanced instruction following. Always evaluate quantized models on YOUR specific use case before deploying.

### Quantization Formats

- **GGUF:** The standard for llama.cpp and Ollama. CPU + GPU hybrid inference. Most compatible.
- **GPTQ:** GPU-only, fast inference. Good for serving with vLLM.
- **AWQ:** Activation-aware quantization. Better quality than GPTQ at the same bit width. Newer, gaining adoption.
- **EXL2:** Variable bit-width within a single model. Fine-grained control over quality vs size.

## Tools for Local Inference

### Ollama — Simplest Start

One-line model download and serving. Runs llama.cpp under the hood.

```bash
# Install
curl -fsSL https://ollama.com/install.sh | sh

# Pull and run a model
ollama pull llama3.1:8b
ollama run llama3.1:8b "Explain recursion in one paragraph"

# Serve as API (OpenAI-compatible)
ollama serve
# Then: curl http://localhost:11434/v1/chat/completions ...
```

**Best for:** Development, prototyping, personal use, simple deployments. OpenAI-compatible API means your existing code works with one URL change.

### llama.cpp — Most Flexible

The C/C++ inference engine most tools build on. Direct control over quantization, batching, context length, and hardware.

```bash
# Build from source
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make -j

# Run inference
./main -m models/llama-3.1-8b-q4.gguf \
  -p "Explain recursion:" \
  -n 256 --temp 0.7
```

**Best for:** Maximum performance tuning, custom builds, embedding in C/C++ applications, edge devices.

### vLLM — Production Serving

High-throughput inference server with continuous batching, PagedAttention, and OpenAI-compatible API.

```bash
pip install vllm

# Serve a model
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --quantization awq \
  --max-model-len 8192
```

**Best for:** Production deployments with multiple concurrent users. Handles batching, scheduling, and GPU memory management automatically.

### LM Studio — GUI

Desktop app for running local models with a visual interface. Download models from HuggingFace, adjust parameters, chat. No command line required.

**Best for:** Non-technical users, experimentation, model evaluation.

## Choosing a Local Model

The open-source model landscape moves fast. As of 2025, these are the strongest options by size:

| Size | Model | Strengths | Limitations |
|------|-------|-----------|-------------|
| 1-3B | Phi-3.5 Mini, Qwen2.5-3B | Fast, runs on phones/edge | Limited reasoning, short context |
| 7-8B | Llama 3.1 8B, Mistral 7B, Qwen2.5-7B | Best quality-per-FLOP, runs on consumer GPUs | Can't match 70B+ on complex tasks |
| 13-14B | Qwen2.5-14B | Sweet spot for many tasks | Needs 12GB+ VRAM at Q4 |
| 32-34B | Qwen2.5-32B, CodeLlama 34B | Near-API quality on many tasks | Needs 24GB+ VRAM at Q4 |
| 70B+ | Llama 3.1 70B, Qwen2.5-72B | Competitive with commercial APIs | Needs datacenter GPU (A100/H100) |

**The general rule:** Start with the smallest model that passes your eval suite. A 7B model running at Q4 handles 80% of production use cases people throw at 70B models.

## Hybrid Architecture

The most practical pattern: route simple tasks to a local model, complex tasks to a cloud API.

```python
from openai import OpenAI

local_client = OpenAI(base_url="http://localhost:11434/v1", api_key="unused")
cloud_client = OpenAI()  # Uses OPENAI_API_KEY

def classify_complexity(message: str) -> str:
    """Simple heuristic -- replace with a classifier for production."""
    complex_signals = ["analyze", "compare", "explain why", "multi-step", "reason"]
    if any(signal in message.lower() for signal in complex_signals):
        return "complex"
    return "simple"

def route_request(message: str) -> str:
    complexity = classify_complexity(message)

    if complexity == "simple":
        client = local_client
        model = "llama3.1:8b"
    else:
        client = cloud_client
        model = "gpt-4.1-mini"

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": message}]
    )
    return response.choices[0].message.content
```

This pattern cuts API costs dramatically while maintaining quality where it matters.

## Local Embeddings

For privacy-sensitive RAG, you can run embedding models locally:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")  # 130MB, runs on CPU

texts = ["What is machine learning?", "How do neural networks work?"]
embeddings = model.encode(texts)

similarity = embeddings[0] @ embeddings[1]  # Dot product
```

**Popular local embedding models:**

| Model | Dimensions | Size | Quality (MTEB) |
|-------|-----------|------|-----------------|
| bge-small-en-v1.5 | 384 | 130MB | Good |
| bge-large-en-v1.5 | 1024 | 1.3GB | Very good |
| nomic-embed-text | 768 | 550MB | Very good |
| e5-mistral-7b | 4096 | 14GB | Excellent |

## Edge Deployment

### Browser (WebLLM)

Run models directly in the browser using WebGPU:

```javascript
import { CreateMLCEngine } from "@mlc-ai/web-llm";

const engine = await CreateMLCEngine("Llama-3.1-8B-Instruct-q4f16_1-MLC");

const reply = await engine.chat.completions.create({
  messages: [{ role: "user", content: "Hello!" }]
});
```

**Reality check:** Browser inference is slow (5-20 tokens/sec on good hardware), requires model download (2-4GB), and only works on devices with WebGPU support. Useful for demos and privacy-sensitive tools, not production chat interfaces.

### Mobile

Apple Core ML and Android NNAPI support small models (1-3B parameters). Practical for on-device autocomplete, classification, and simple generation.

### Docker for Deployment

```yaml
# docker-compose.yml
services:
  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

volumes:
  ollama_data:
```

## Cost Analysis: Local vs API

### Break-Even Calculation

**Scenario:** 10M tokens/day, GPT-4.1 Mini pricing ($0.40/M input, $1.60/M output, ~50/50 split)

**API cost:**
- Daily: 5M input x $0.40/M + 5M output x $1.60/M = **$10/day**
- Monthly: **$300/month**

**Local cost (Llama 3.1 8B on RTX 4090):**
- Hardware: $1,600 one-time (amortized over 2 years = $67/month)
- Power: ~350W x 24h x 30 days x $0.12/kWh = **$30/month**
- Total: **~$97/month**

**Break-even: ~3.3M tokens/day.** Below that, API is cheaper. Above that, local wins.

**Caveats:**
- Assumes the local model's quality is acceptable for your use case
- Doesn't include maintenance, monitoring, or ops time
- API prices drop regularly — recalculate quarterly

## Summary

Local and edge AI is not about replacing cloud APIs — it's about having the right tool for each situation. Use local models for privacy, cost optimization at scale, and latency-sensitive applications. Use cloud APIs for frontier capability, simplicity, and low-volume use cases. The hybrid architecture gives you both.

Start with Ollama and a 7-8B model. If quality is sufficient, you've saved yourself significant ongoing costs. If not, you know exactly which queries need the cloud and which don't.
