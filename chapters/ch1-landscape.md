# The AI Engineering Landscape

> **From the field — Frank Sellhausen, Sellhausen AI Systems.** AI doesn't introduce new failures — it speeds up and surfaces organizational failures that already exist. The failure patterns are inventory gaps, controls you can't demonstrate, and documentation that doesn't trace from business intent to monitoring. Tools don't fix culture. They reproduce it faster.

## The Paradigm Shift

AI engineering exists because of a simple change: we stopped training models and started using them. Before 2020, building an AI feature meant collecting a dataset, training a model from scratch, and deploying it — a process that required ML PhDs, GPU clusters, and months of work. Now you make an API call.

This isn't a small change. It created an entirely new engineering discipline.

**Traditional ML Engineering** was about data pipelines, feature engineering, model training, and hyperparameter tuning. You built the intelligence. **AI Engineering** is about prompting, retrieval, orchestration, and evaluation. The intelligence is pre-built — your job is to make it useful for a specific problem.

| | ML Engineering | AI Engineering |
|---|---|---|
| **Core work** | Train models from data | Compose applications from pre-trained models |
| **Key skills** | Statistics, PyTorch, feature engineering | Prompt design, API integration, RAG, evaluation |
| **Time to prototype** | Weeks to months | Hours to days |
| **Background** | PhD/MS in ML, $150-300K+ | Software engineering, $130-250K+ |
| **Primary challenge** | Getting the model to work | Getting the model to work *reliably, at scale, for real users* |

The barrier to entry dropped. The challenge shifted from "can we build this?" to "can we deploy this responsibly and make it actually useful?"

## Why Now: Six Forces That Converged

This field didn't emerge gradually. Several forces hit at the same time:

**1. Model capability crossed the usefulness threshold.** GPT-3 (2020) proved that scale creates emergent abilities — models suddenly became good enough for real applications without task-specific training. By 2023, models could reason through multi-step problems, write production code, and follow complex instructions.

**2. APIs democratized access.** OpenAI, Google, and Anthropic made frontier models available through simple REST APIs. You no longer need infrastructure expertise — just an API key and a credit card.

**3. Costs dropped dramatically.** Inference pricing fell roughly 10-30x between 2022 and 2025 across major providers. Tasks that were economically impossible at $0.06 per 1K tokens became viable at $0.002. This is real but not the "100x" number you see in blog posts — it depends heavily on which models and tasks you compare.

**4. Context windows expanded.** From 4K tokens (GPT-3) to 200K+ (Claude) and 1M+ (Gemini). This means models can now process entire codebases, books, or long conversation histories in a single call — fundamentally changing what's possible with retrieval and document analysis.

**5. The tooling ecosystem matured.** LangChain, LlamaIndex, vector databases (Pinecone, Weaviate, Chroma), observability platforms (LangSmith, Helicone), and structured output libraries (Instructor, Outlines) created an infrastructure layer that abstracts real complexity.

**6. Enterprise demand exploded.** Every company wants AI features. The talent gap between "ML researchers who can train models" and "developers who can use APIs" created massive demand for AI engineers who can bridge that gap.

## The Evolution: Language Models to Foundation Models

**Language Models (2018-2020)** — BERT, RoBERTa, DistilBERT. Task-specific: you needed a different fine-tuned model for sentiment analysis, NER, and classification. Good at understanding text, couldn't generate it well.

**Large Language Models (2020-2023)** — GPT-3, PaLM, LLaMA. Multi-task via prompting: one model could write, summarize, translate, and code. The breakthrough was that scale created abilities nobody explicitly programmed.

**Foundation Models (2023+)** — GPT-4, Claude, Gemini. Native multimodality (text, images, audio, video in one model), million-token context windows, sophisticated reasoning through chain-of-thought, and tool use. The term "foundation model" (coined by Stanford's CRFM in 2021) captures the idea: these are base layers everything else builds on.

> The shift: we stopped building task-specific models and started steering general-purpose intelligence through prompts, retrieval, and tools.

## The AI Engineering Stack

Modern AI applications have three layers:

### Layer 1: Application
Your user-facing code — the UI, API gateway, authentication, session management, and orchestration logic. This is standard software engineering. The AI-specific part is how you route requests, manage conversation state, and handle the inherent unreliability of model outputs.

### Layer 2: AI
The intelligence layer — prompt templates, RAG pipelines, agent loops, fine-tuned models, guardrails, and output parsing. This is where AI engineering lives. Your decisions here determine quality, cost, latency, and safety.

### Layer 3: Infrastructure
Model provider APIs (OpenAI, Anthropic, Google), vector databases for embeddings, caching layers (semantic and response), observability and logging, and evaluation pipelines. You're stitching together managed services more than managing servers.

## Use Cases and Their Constraints

AI features aren't magic. Each category has specific limitations that matter in production:

**Code generation** — Models write functional code, but it needs expert review and security scanning. Compilation rate should exceed 95% for production use. The real risk isn't wrong syntax; it's subtle logic bugs and security vulnerabilities.

**Content and writing** — Drafts, summaries, translations. Hallucination risk means fact-checking is non-negotiable. Quality ratings above 4.5/5 from human evaluators is a reasonable target. The harder problem is maintaining consistent voice and accuracy across thousands of outputs.

**Data extraction and analysis** — Parsing documents, extracting structured data, classification. Field accuracy above 95% is achievable but requires careful prompt engineering and validation. Edge cases will surprise you.

**Customer-facing assistants** — Chatbots, support agents, search. Response accuracy above 85% is a starting target, but the 15% failure rate means you need graceful degradation, escalation paths, and monitoring. A wrong answer confidently stated is worse than saying "I don't know."

**Education and research** — Tutoring, summarization, synthesis. Academic integrity concerns and the tendency to fabricate citations make this domain high-stakes despite seeming low-risk.

> **Practitioner's note:** The mistake teams make most often is picking the use case based on what's impressive to demo rather than what solves a real workflow problem. The impressive demo and the useful product are rarely the same thing.

## Planning an AI Project: The Five Phases

Shipping an AI feature follows a predictable arc. Teams that skip phases pay for it later.

**Phase 1: Proof of Concept (2-4 weeks)**
Demonstrate basic feasibility. Prompt engineering, rough accuracy assessment, initial cost estimates. Target: >70% accuracy on core task. Common mistake: spending too long here polishing instead of validating the core assumption.

**Phase 2: Prototype (4-8 weeks)**
Build evaluation infrastructure. This is the phase most teams skip, and it's the one that kills projects later. Create eval sets, establish baseline metrics, test edge cases. Deliverable: a system you can measure, not just one that works on demo inputs.

**Phase 3: Alpha (8-12 weeks)**
Production-ready code. Error handling, rate limiting, cost controls, monitoring, security review. This is where the engineering happens. The model was the easy part.

**Phase 4: Beta (4-8 weeks)**
Real user validation with limited rollout. A/B testing, user feedback loops, performance monitoring under real load. Common mistake: declaring victory based on internal testing without exposing the system to real-world messiness.

**Phase 5: Production (Ongoing)**
Monitoring, cost optimization, model updates, drift detection, incident response. This phase never ends. Most AI features degrade silently without active maintenance.

## Model Selection

Choosing a model is one of the highest-leverage decisions you'll make. It's not about picking "the best" — it's about the right tradeoff between capability, cost, latency, and context for your specific use case.

### Current Landscape (2025)

**OpenAI**
- *GPT-4.1*: Strong general-purpose, excellent function calling. ~$2/M input, ~$8/M output. 1M context.
- *GPT-4.1 Mini*: 80% of GPT-4.1 quality at ~15% of the cost. Good default for most tasks.
- *o3/o4-mini*: Reasoning-optimized models. Higher latency, higher cost, better on complex multi-step problems.

**Anthropic**
- *Claude Sonnet 4*: Best balance of quality, speed, and cost. Strong on long documents and instruction following. ~$3/M input, ~$15/M output. 200K context.
- *Claude Opus 4*: Highest capability, best for complex analysis. ~$15/M input, ~$75/M output.
- *Claude Haiku 4*: Fast and cheap for simple tasks. ~$0.25/M input.

**Google**
- *Gemini 2.5 Pro*: Competitive quality, massive 1M+ context window, strong multimodal. Good value.
- *Gemini 2.5 Flash*: Fast, cheap, good for high-volume simple tasks.

### How to Choose

**Start with the cheapest model that might work**, then move up only when evaluation proves you need to. Most teams over-provision — they reach for GPT-4 when GPT-4.1 Mini or Claude Haiku would handle 80% of their traffic.

**Model tiering** is the production pattern: route simple queries to a cheap/fast model, complex queries to an expensive/capable one. This alone can cut costs 40-70%.

## Cost Optimization

A single GPT-4 call might cost $0.03. At 1M requests/month, that's $30,000. Smart optimization cuts this dramatically.

**1. Model tiering (40-70% savings):** Classify incoming requests by complexity. Send "What are your hours?" to Haiku. Send "Analyze this contract for liability risks" to Opus.

**2. Prompt caching (30-60% savings):** Anthropic offers 90% discounts on cached prompt prefixes. OpenAI offers 50% on repeated prefixes >1024 tokens. Structure your prompts with static content first.

**3. Prompt compression (20-40% savings):** Shorter prompts cost less. Remove examples that don't improve quality. Use concise system prompts. Every token you cut is money saved at scale.

**4. Response caching (varies):** Cache responses to identical or semantically similar queries. A customer support bot answering the same FAQ 1,000 times should hit cache 999 times.

**5. Batching (15-30% savings):** Some providers offer batch APIs at discounted rates for non-time-sensitive work. Classification, extraction, and analysis tasks often don't need real-time responses.

## The Evaluation Gap

Here's the thing that makes AI engineering fundamentally different from traditional software: **you can't write a unit test for it.**

In traditional software, `assertEqual(add(2, 2), 4)` either passes or fails. In AI engineering, the same prompt can produce different outputs every time. "Correct" is often subjective. Edge cases are infinite. And the model's behavior changes when the provider updates it.

**The gap between "it works in my demo" and "it works reliably in production" is almost entirely an evaluation problem.** Teams that invest in evaluation infrastructure early ship better products. Teams that skip it ship demos that break in production.

How to evaluate AI systems is covered in depth in Chapter 7.

## Summary

AI engineering is a new discipline born from a simple shift: pre-trained models became good enough to use as building blocks. The work moved from training intelligence to deploying it — and that turns out to be a different set of skills entirely.

The field is young, the tools are changing fast, and the gap between what's possible and what's reliable is where the real engineering happens. The rest of this course is about closing that gap.
