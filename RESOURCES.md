# AI Engineering Resources

A curated collection of tools, frameworks, and platforms for AI engineers.

---

## LLM Providers & APIs

| Provider | Key Models | Best For | Pricing |
|----------|-----------|----------|---------|
| **OpenAI** | GPT-4o, GPT-4 Turbo, o1 | General purpose, function calling, vision | Pay-per-token |
| **Anthropic** | Claude 3.5 Sonnet, Claude 3 Opus | Long context, instruction following, safety | Pay-per-token |
| **Google** | Gemini 1.5 Pro, Gemini Ultra | Multimodal, long context (1M tokens) | Pay-per-token |
| **Mistral** | Mistral Large, Mixtral 8x22B | Open weights, European hosting | Pay-per-token |
| **Cohere** | Command R+, Embed, Rerank | Enterprise RAG, multilingual | Pay-per-token |
| **Groq** | Llama 3, Mixtral (hosted) | Ultra-fast inference | Pay-per-token |
| **Together AI** | Open models (70+ options) | Open model hosting, fine-tuning | Pay-per-token |
| **Fireworks AI** | Open models | Low-latency inference | Pay-per-token |
| **Perplexity** | pplx-api | Search-augmented generation | Pay-per-token |

**Links:**
- [OpenAI Platform](https://platform.openai.com)
- [Anthropic Console](https://console.anthropic.com)
- [Google AI Studio](https://aistudio.google.com)
- [Mistral Platform](https://console.mistral.ai)
- [Cohere Dashboard](https://dashboard.cohere.com)
- [Groq Console](https://console.groq.com)
- [Together AI](https://together.ai)
- [Fireworks AI](https://fireworks.ai)

---

## Orchestration Frameworks

### LangChain
**What:** Most popular LLM orchestration framework
**Best for:** Chains, agents, RAG pipelines, tool use
**Language:** Python, JavaScript/TypeScript
**Link:** [langchain.com](https://langchain.com)

### LlamaIndex
**What:** Data framework for LLM applications
**Best for:** RAG, document processing, knowledge graphs
**Language:** Python, TypeScript
**Link:** [llamaindex.ai](https://llamaindex.ai)

### Semantic Kernel
**What:** Microsoft's LLM orchestration SDK
**Best for:** Enterprise .NET/Python apps, Azure integration
**Language:** C#, Python, Java
**Link:** [github.com/microsoft/semantic-kernel](https://github.com/microsoft/semantic-kernel)

### Haystack
**What:** End-to-end NLP/LLM framework
**Best for:** Production RAG pipelines, search
**Language:** Python
**Link:** [haystack.deepset.ai](https://haystack.deepset.ai)

### DSPy
**What:** Programmatic prompt optimization
**Best for:** Automated prompt engineering, complex pipelines
**Language:** Python
**Link:** [github.com/stanfordnlp/dspy](https://github.com/stanfordnlp/dspy)

---

## Structured Output Libraries

### Instructor
**What:** Structured outputs using Pydantic
**Best for:** Type-safe LLM responses, validation
**Language:** Python, TypeScript
**Link:** [github.com/jxnl/instructor](https://github.com/jxnl/instructor)

### Outlines
**What:** Constrained text generation
**Best for:** Guaranteed JSON, regex-constrained output
**Language:** Python
**Link:** [github.com/outlines-dev/outlines](https://github.com/outlines-dev/outlines)

### Guidance
**What:** Microsoft's structured generation library
**Best for:** Templates, constrained decoding
**Language:** Python
**Link:** [github.com/guidance-ai/guidance](https://github.com/guidance-ai/guidance)

### Marvin
**What:** AI functions as Python functions
**Best for:** Quick prototyping, type extraction
**Language:** Python
**Link:** [github.com/prefecthq/marvin](https://github.com/prefecthq/marvin)

---

## Vector Databases

| Database | Type | Best For | Key Features |
|----------|------|----------|--------------|
| **Pinecone** | Managed | Production RAG | Serverless, hybrid search, metadata filtering |
| **Weaviate** | Open source / Managed | Semantic search | GraphQL API, modules, hybrid search |
| **Chroma** | Open source | Local development | Lightweight, embedded, easy setup |
| **Qdrant** | Open source / Managed | High performance | Rust-based, filtering, payloads |
| **Milvus** | Open source | Enterprise scale | Distributed, GPU acceleration |
| **pgvector** | Postgres extension | Existing Postgres users | SQL integration, familiar tooling |
| **LanceDB** | Open source | Embedded, serverless | Zero-copy, versioned |

**Links:**
- [Pinecone](https://pinecone.io)
- [Weaviate](https://weaviate.io)
- [Chroma](https://trychroma.com)
- [Qdrant](https://qdrant.tech)
- [Milvus](https://milvus.io)
- [pgvector](https://github.com/pgvector/pgvector)
- [LanceDB](https://lancedb.com)

---

## Evaluation & Observability

### Evaluation Platforms

| Tool | Type | Best For |
|------|------|----------|
| **Braintrust** | Platform | LLM evals, prompt playground, logging |
| **Promptfoo** | Open source CLI | Prompt testing, CI/CD integration |
| **Langfuse** | Open source | Tracing, evals, prompt management |
| **Ragas** | Open source | RAG evaluation metrics |

### Observability & Monitoring

| Tool | Type | Best For |
|------|------|----------|
| **LangSmith** | Platform | LangChain tracing, debugging, datasets |
| **Helicone** | Platform | Logging, caching, rate limiting |
| **Portkey** | Platform | Gateway, observability, fallbacks |
| **Arize Phoenix** | Open source | ML observability, embeddings analysis |
| **Weights & Biases** | Platform | Experiment tracking, prompts |

**Links:**
- [Braintrust](https://braintrust.dev)
- [Promptfoo](https://promptfoo.dev)
- [Langfuse](https://langfuse.com)
- [LangSmith](https://smith.langchain.com)
- [Helicone](https://helicone.ai)
- [Portkey](https://portkey.ai)
- [Arize](https://arize.com)

---

## Agent Frameworks

### AutoGen
**What:** Microsoft's multi-agent conversation framework
**Best for:** Multi-agent systems, human-in-the-loop
**Link:** [github.com/microsoft/autogen](https://github.com/microsoft/autogen)

### CrewAI
**What:** Role-based multi-agent orchestration
**Best for:** Team-of-agents patterns, workflows
**Link:** [crewai.com](https://crewai.com)

### LangGraph
**What:** LangChain's graph-based agent framework
**Best for:** Stateful agents, complex workflows
**Link:** [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph)

### Smolagents
**What:** Hugging Face's lightweight agent library
**Best for:** Simple agents, code execution
**Link:** [github.com/huggingface/smolagents](https://github.com/huggingface/smolagents)

---

## Fine-Tuning & Training

### Platforms

| Platform | Best For | Key Features |
|----------|----------|--------------|
| **Hugging Face** | Everything | Models, datasets, training, inference |
| **OpenAI Fine-tuning** | GPT models | Simple API, no infra needed |
| **Anyscale** | Ray-based training | Distributed, scalable |
| **Modal** | Serverless GPU | Easy deployment, pay-per-use |
| **Lambda Labs** | GPU cloud | On-demand A100s, H100s |
| **RunPod** | GPU marketplace | Affordable, spot instances |

### Libraries

| Library | Best For |
|---------|----------|
| **Axolotl** | Easy fine-tuning with configs |
| **Unsloth** | 2x faster LoRA training |
| **PEFT** | Parameter-efficient fine-tuning |
| **TRL** | RLHF, DPO, reward modeling |
| **LitGPT** | Lightning-based training |

**Links:**
- [Hugging Face](https://huggingface.co)
- [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)
- [Unsloth](https://github.com/unslothai/unsloth)
- [Modal](https://modal.com)
- [Lambda Labs](https://lambdalabs.com)

---

## Open Source Models

### Text Models

| Model Family | Provider | Sizes | License |
|--------------|----------|-------|---------|
| **Llama 3.1** | Meta | 8B, 70B, 405B | Llama 3.1 License |
| **Mistral/Mixtral** | Mistral AI | 7B, 8x7B, 8x22B | Apache 2.0 |
| **Qwen 2.5** | Alibaba | 0.5B - 72B | Apache 2.0 |
| **Gemma 2** | Google | 2B, 9B, 27B | Gemma License |
| **Phi-3** | Microsoft | 3.8B, 14B | MIT |
| **Command R** | Cohere | 35B, 104B | CC-BY-NC |

### Code Models

| Model | Provider | Best For |
|-------|----------|----------|
| **CodeLlama** | Meta | Code completion, infilling |
| **DeepSeek Coder** | DeepSeek | Code generation |
| **StarCoder 2** | BigCode | Code, 600+ languages |
| **Qwen2.5-Coder** | Alibaba | Code, long context |

### Embedding Models

| Model | Provider | Dimensions |
|-------|----------|------------|
| **text-embedding-3-large** | OpenAI | 3072 |
| **voyage-3** | Voyage AI | 1024 |
| **embed-v3** | Cohere | 1024 |
| **bge-large-en-v1.5** | BAAI | 1024 |
| **e5-mistral-7b** | Microsoft | 4096 |
| **nomic-embed-text** | Nomic | 768 |

---

## Prompt Engineering Tools

| Tool | Type | Best For |
|------|------|----------|
| **PromptLayer** | Platform | Prompt versioning, analytics |
| **Humanloop** | Platform | Prompt management, evals, deployment |
| **Promptimize** | Open source | Prompt optimization |
| **Agenta** | Open source | Prompt playground, versioning |

---

## Deployment & Infrastructure

### Inference Servers

| Server | Best For |
|--------|----------|
| **vLLM** | High-throughput serving, PagedAttention |
| **TGI (Text Generation Inference)** | Hugging Face models, production |
| **Ollama** | Local development, easy setup |
| **llama.cpp** | CPU inference, quantization |
| **TensorRT-LLM** | NVIDIA optimized inference |

### Deployment Platforms

| Platform | Best For |
|----------|----------|
| **Replicate** | One-click model deployment |
| **Baseten** | Custom model serving |
| **Banana** | Serverless GPU inference |
| **Beam** | Serverless Python |
| **Modal** | Serverless GPU, easy scaling |

**Links:**
- [vLLM](https://vllm.ai)
- [TGI](https://github.com/huggingface/text-generation-inference)
- [Ollama](https://ollama.ai)
- [Replicate](https://replicate.com)
- [Baseten](https://baseten.co)

---

## AI Product Management Frameworks

### Core AI PM Frameworks

| Framework | Description | Link |
|-----------|-------------|------|
| **AI Product Management Lifecycle** | End-to-end framework from problem definition to continuous improvement | [thinkaipm.com](https://thinkaipm.com/frameworks) |
| **AI Product Canvas** | Visual template mapping user needs, data requirements, and business outcomes | [thinkaipm.com](https://thinkaipm.com/frameworks) |
| **AI Decision Matrix** | Evaluates technical complexity, business value, and data availability | [thinkaipm.com](https://thinkaipm.com/frameworks) |
| **AI Implementation Roadmap** | Execution strategy, milestone planning, risk management | [thinkaipm.com](https://thinkaipm.com/frameworks) |
| **FOBW (Fear of Being Wrong)** | Optimizes AI adoption by enhancing user trust and confidence | [thinkaipm.com](https://thinkaipm.com/frameworks) |
| **ProductAI Framework™** | AI PM training and certification program | [productaiframework.com](https://productaiframework.com) |
| **AI Product Playbook** | Actionable frameworks and career roadmaps | [aiproduct.com](https://aiproduct.com/ai-product-playbook) |

### ML/AI Operations

| Framework | Description | Link |
|-----------|-------------|------|
| **MLOps** | Deploy and maintain ML models in production reliably | [Wikipedia](https://en.wikipedia.org/wiki/MLOps) |
| **ModelOps** | Governance and lifecycle management of operationalized AI models | [Wikipedia](https://en.wikipedia.org/wiki/ModelOps) |
| **CRISP-DM** | Cross-Industry Standard Process for Data Mining | Industry standard |
| **CRISP-ML(Q)** | Extension for machine learning with quality assurance | [arxiv.org](https://arxiv.org/abs/2003.05155) |
| **Kafka-ML** | ML pipeline management through data streams | [arxiv.org](https://arxiv.org/abs/2006.04105) |

### Governance & Risk Management

| Framework | Description | Link |
|-----------|-------------|------|
| **NIST AI RMF** | Structured approach to managing AI risks (Govern, Map, Measure, Manage) | [NIST](https://www.nist.gov/itl/ai-risk-management-framework) |
| **Responsible AI Pattern Catalogue** | Best practices for AI governance and ethics | [arxiv.org](https://arxiv.org/abs/2209.04963) |
| **TAIBOM** | Trusted AI Bill of Materials - dependency model for AI components | [arxiv.org](https://arxiv.org/abs/2510.02169) |
| **Unified Control Framework (UCF)** | Integrates risk management and regulatory compliance | [arxiv.org](https://arxiv.org/abs/2503.05937) |
| **Frontier AI Risk Management** | Guidelines for robust AI risk management | [arxiv.org](https://arxiv.org/abs/2502.06656) |
| **EU AI Act Compliance** | European regulatory framework for AI systems | [EU AI Act](https://artificialintelligenceact.eu) |

### Enterprise & Scaling

| Framework | Description | Link |
|-----------|-------------|------|
| **SAFe for AI** | Scaled Agile Framework adapted for AI teams | [scaledagileframework.com](https://scaledagileframework.com) |
| **ISPMA Framework** | Software product management practices for AI | [ispma.org](https://ispma.org) |
| **FAIGMOE** | GenAI adoption for midsize organizations | [arxiv.org](https://arxiv.org/abs/2510.19997) |
| **Product Intelligence Framework** | Embedding intelligence across product operations | Industry practice |

### General PM Frameworks (Adapted for AI)

| Framework | Description | Best For |
|-----------|-------------|----------|
| **Design Sprint** | 5-day process for testing AI ideas | Rapid prototyping |
| **Opportunity Solution Tree** | Connecting outcomes to AI solutions | Discovery |
| **RICE Scoring** | Reach, Impact, Confidence, Effort prioritization | Feature prioritization |
| **HEART Metrics** | Happiness, Engagement, Adoption, Retention, Task success | AI UX measurement |
| **Jobs-to-be-Done (JTBD)** | Understanding user needs for AI features | User research |
| **Kano Model** | Categorizing AI features by user satisfaction | Feature analysis |

**Key Resources:**
- [ThinkAIPM Frameworks](https://thinkaipm.com/frameworks)
- [ProductAI Framework](https://productaiframework.com)
- [AI Product Playbook](https://aiproduct.com/ai-product-playbook)
- [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework)

---

## Learning Resources

### Documentation
- [OpenAI Cookbook](https://cookbook.openai.com)
- [Anthropic Docs](https://docs.anthropic.com)
- [LangChain Docs](https://python.langchain.com)
- [LlamaIndex Docs](https://docs.llamaindex.ai)

### Courses
- [DeepLearning.AI Short Courses](https://deeplearning.ai/short-courses)
- [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course)
- [Full Stack LLM Bootcamp](https://fullstackdeeplearning.com)

### Newsletters & Blogs
- [The Batch (DeepLearning.AI)](https://deeplearning.ai/the-batch)
- [Simon Willison's Blog](https://simonwillison.net)
- [Chip Huyen's Blog](https://huyenchip.com/blog)
- [Eugene Yan's Blog](https://eugeneyan.com)

### Communities
- [Hugging Face Discord](https://huggingface.co/join/discord)
- [LangChain Discord](https://discord.gg/langchain)
- [r/LocalLLaMA](https://reddit.com/r/LocalLLaMA)
- [r/MachineLearning](https://reddit.com/r/MachineLearning)

---

## Recommended Stacks

### Beginner RAG Stack
- **LLM:** OpenAI GPT-4o-mini or Claude 3.5 Sonnet
- **Embeddings:** OpenAI text-embedding-3-small
- **Vector DB:** Chroma (local) or Pinecone (managed)
- **Framework:** LangChain or LlamaIndex
- **Eval:** Promptfoo

### Production RAG Stack
- **LLM:** GPT-4o or Claude 3.5 Sonnet
- **Embeddings:** Cohere embed-v3 or Voyage
- **Vector DB:** Pinecone or Qdrant Cloud
- **Framework:** LlamaIndex or custom
- **Reranking:** Cohere Rerank
- **Observability:** Langfuse or LangSmith
- **Eval:** Braintrust or Ragas

### Agent Stack
- **LLM:** GPT-4o or Claude 3.5 Sonnet (function calling)
- **Framework:** LangGraph or AutoGen
- **Memory:** Redis or Postgres
- **Observability:** LangSmith
- **Guardrails:** Guardrails AI or NeMo Guardrails

### Cost-Optimized Stack
- **LLM:** Mixtral 8x7B via Groq or Together
- **Embeddings:** nomic-embed-text or bge
- **Vector DB:** pgvector or Qdrant (self-hosted)
- **Framework:** LlamaIndex
- **Caching:** Redis semantic cache

### Self-Hosted Stack
- **LLM:** Llama 3.1 70B via vLLM
- **Embeddings:** bge-large or e5-mistral
- **Vector DB:** Qdrant or Milvus
- **Framework:** LlamaIndex
- **Infra:** Modal, RunPod, or Lambda Labs

---

*Last updated: November 2024*

