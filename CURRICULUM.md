# AI Engineering Masterclass - Curriculum

A comprehensive learning path for software engineers and product managers transitioning to AI engineering.

## Course Structure

| # | Chapter | Status | Description |
|---|---------|--------|-------------|
| 1 | **The AI Engineering Landscape** | ✅ Complete | Why AI engineering exists, the paradigm shift, roles, development lifecycle |
| 2 | **How Foundation Models Work** | ✅ Complete | Architecture, training, tokenization, inference optimization |
| 3 | **Prompt Engineering & Techniques** | ✅ Complete | Zero-shot, few-shot, CoT, structured outputs, system prompts |
| 4 | **Building with LLM APIs** | ✅ Complete | OpenAI/Anthropic/Google APIs, streaming, function calling, error handling |
| 5 | **RAG & Knowledge Systems** | ✅ Complete | Embeddings, vector databases, chunking, retrieval strategies |
| 6 | **Agents & Tool Use** | ✅ Complete | ReAct, function calling, multi-step reasoning, orchestration |
| 7 | **Evaluation & Testing** | ✅ Complete | Benchmarks, human eval, automated testing, regression detection |
| 8 | **Production & Operations** | ✅ Complete | Monitoring, cost optimization, latency, guardrails, safety |
| 9 | **Fine-Tuning & Customization** | ✅ Complete | When to fine-tune, LoRA, data preparation, evaluation |
| 10 | **AI Product Strategy** | ✅ Complete | Use case selection, build vs buy, pricing, user research |

---

## Chapter Details

### 1. The AI Engineering Landscape
**Target Question:** "What is this field and where do I fit?"

- The paradigm shift from traditional ML to foundation models
- Why AI engineering emerged as a discipline
- AI Engineer vs ML Engineer vs Full-Stack Engineer
- The development lifecycle (PoC → Production)
- Model selection and cost considerations
- Probabilistic vs deterministic systems

---

### 2. How Foundation Models Work
**Target Question:** "How does the technology actually work?"

- What makes a model a "foundation model"
- The three pillars: Scale, Generality, Adaptability
- Scaling laws and emergent capabilities
- Transformer architecture deep dive
- Training pipeline: Pre-training → SFT → RLHF → Reasoning
- Training data sources and limitations
- Tokenization and its implications
- Decoding strategies (temperature, top-p, etc.)
- Context windows and their evolution
- Inference optimization techniques

---

### 3. Prompt Engineering & Techniques
**Target Question:** "How do I get good results from models?"

- The anatomy of a prompt
- System prompts and persona design
- Zero-shot vs few-shot prompting
- Chain of Thought (CoT) and reasoning techniques
- Structured outputs (JSON mode, function schemas)
- Prompt templates and management
- Common failure modes and fixes
- Prompt injection and security
- Testing and iterating on prompts
- When prompting isn't enough

---

### 4. Building with LLM APIs
**Target Question:** "How do I integrate this into my code?"

- API landscape: OpenAI, Anthropic, Google, open-source
- Authentication and rate limiting
- Streaming responses for UX
- Function calling / tool use
- Multi-turn conversations
- Error handling and retries
- Cost tracking and budgeting
- SDK comparison and selection
- Building a simple chatbot
- Building a document Q&A system

---

### 5. RAG & Knowledge Systems
**Target Question:** "How do I give models my own data?"

- Why RAG? Limitations of context stuffing
- Embeddings: What they are and how they work
- Vector databases (Pinecone, Weaviate, Chroma, pgvector)
- Chunking strategies and their trade-offs
- Retrieval techniques (semantic, keyword, hybrid)
- Reranking and relevance scoring
- Handling updates and deletions
- Multi-modal RAG (images, tables)
- Evaluation: Retrieval quality metrics
- Common RAG failure modes

---

### 6. Agents & Tool Use
**Target Question:** "How do I build autonomous systems?"

- What is an agent? Definitions and mental models
- ReAct pattern: Reasoning + Acting
- Tool/function calling in depth
- Planning and decomposition
- Memory systems (short-term, long-term)
- Multi-agent architectures
- Orchestration frameworks (LangChain, LlamaIndex, etc.)
- Human-in-the-loop patterns
- Safety and guardrails for agents
- When NOT to use agents

---

### 7. Evaluation & Testing
**Target Question:** "How do I know if it's working?"

- Why LLM evaluation is hard
- Benchmark landscape (MMLU, HumanEval, etc.)
- Building custom evaluation sets
- LLM-as-judge patterns
- Human evaluation design
- A/B testing for AI features
- Regression detection
- Red teaming and adversarial testing
- Continuous evaluation in production
- Evaluation frameworks and tools

---

### 8. Production & Operations
**Target Question:** "How do I ship and maintain this?"

- Production architecture patterns
- Latency optimization techniques
- Cost optimization strategies
- Caching (semantic caching, prompt caching)
- Monitoring and observability
- Logging and debugging
- Guardrails and content filtering
- Rate limiting and quota management
- Fallback strategies
- Incident response for AI systems

---

### 9. Fine-Tuning & Customization
**Target Question:** "When and how do I customize models?"

- When to fine-tune vs prompt vs RAG
- Data preparation and quality
- LoRA and parameter-efficient fine-tuning
- Full fine-tuning considerations
- Evaluation during training
- Avoiding catastrophic forgetting
- Fine-tuning for specific tasks
- Cost and infrastructure
- Open-source fine-tuning workflows
- Deploying fine-tuned models

---

### 10. AI Product Strategy
**Target Question:** "How do I make good product decisions?"

- Identifying good AI use cases
- Build vs buy vs API decisions
- Pricing AI features
- User research for AI products
- Managing user expectations
- Handling failures gracefully
- Legal and compliance considerations
- AI ethics in product design
- Competitive landscape analysis
- Future-proofing your AI strategy

---

## Target Audience

This curriculum is designed for:

1. **Software Engineers** transitioning to AI/ML roles
2. **Product Managers** building AI-powered products
3. **Technical Leaders** making AI strategy decisions
4. **Full-Stack Developers** adding AI capabilities to applications

## Prerequisites

- Basic programming knowledge (Python preferred)
- Familiarity with REST APIs
- Understanding of basic ML concepts (helpful but not required)

