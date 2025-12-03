import React, { useState } from 'react';
import { ExternalLink, Server, Database, Wrench, FlaskConical, Cpu, BookOpen, Layers, Zap, Code, Search, Rocket, Binary, GraduationCap, Briefcase, ChevronDown, ChevronUp, Target, Users, TrendingUp, Shield, Workflow, BarChart3, Lightbulb, CheckCircle2 } from 'lucide-react';

type Category = 'providers' | 'frameworks' | 'vectordb' | 'eval' | 'agents' | 'finetuning' | 'models' | 'inference' | 'embeddings' | 'learning' | 'stacks' | 'aipm';

interface Resource {
    name: string;
    description: string;
    url: string;
    tags?: string[];
}

const resources: Record<Category, { title: string; icon: React.ReactNode; items: Resource[] }> = {
    providers: {
        title: 'LLM Providers',
        icon: <Server className="w-5 h-5" />,
        items: [
            { name: 'OpenAI', description: 'GPT models - General purpose, function calling, vision, reasoning', url: 'https://platform.openai.com', tags: ['API', 'Vision', 'Tools', 'Reasoning'] },
            { name: 'Anthropic', description: 'Claude models - Long context, instruction following, safety-focused', url: 'https://console.anthropic.com', tags: ['API', 'Long Context', 'Safety'] },
            { name: 'Google AI', description: 'Gemini models - Multimodal, very long context windows', url: 'https://aistudio.google.com', tags: ['API', 'Multimodal', 'Long Context'] },
            { name: 'Mistral', description: 'Mistral models - Open weights available, European hosting', url: 'https://console.mistral.ai', tags: ['API', 'Open Weights', 'Code'] },
            { name: 'Cohere', description: 'Command, Embed, Rerank models - Enterprise RAG, multilingual', url: 'https://dashboard.cohere.com', tags: ['API', 'RAG', 'Embeddings', 'Reranking'] },
            { name: 'Groq', description: 'Ultra-fast inference for open models - sub-100ms latency', url: 'https://console.groq.com', tags: ['Fast Inference', 'Low Latency'] },
            { name: 'Together AI', description: '100+ open models, fine-tuning, inference optimization', url: 'https://together.ai', tags: ['Open Models', 'Fine-tuning', 'Inference'] },
            { name: 'Fireworks AI', description: 'Low-latency inference, function calling, JSON mode', url: 'https://fireworks.ai', tags: ['Fast Inference', 'Function Calling'] },
            { name: 'Perplexity', description: 'Search-augmented generation, real-time web access', url: 'https://docs.perplexity.ai', tags: ['Search', 'RAG', 'Real-time'] },
            { name: 'Amazon Bedrock', description: 'AWS managed - Multiple model providers, enterprise features', url: 'https://aws.amazon.com/bedrock', tags: ['AWS', 'Enterprise', 'Multi-model'] },
            { name: 'Azure OpenAI', description: 'Enterprise OpenAI - Compliance, private endpoints, reserved capacity', url: 'https://azure.microsoft.com/en-us/products/ai-services/openai-service', tags: ['Enterprise', 'Azure', 'Compliance'] },
            { name: 'Vertex AI', description: 'Google Cloud - Gemini models, fine-tuning, MLOps', url: 'https://cloud.google.com/vertex-ai', tags: ['GCP', 'Enterprise', 'MLOps'] },
            { name: 'Replicate', description: 'One-click model deployment, pay-per-second', url: 'https://replicate.com', tags: ['Deployment', 'Open Models'] },
            { name: 'Anyscale', description: 'Ray-based inference, fine-tuning at scale', url: 'https://anyscale.com', tags: ['Scale', 'Ray', 'Fine-tuning'] },
            { name: 'DeepInfra', description: 'Cost-effective inference for open models', url: 'https://deepinfra.com', tags: ['Affordable', 'Open Models'] },
            { name: 'OctoAI', description: 'Optimized inference, image generation, customization', url: 'https://octo.ai', tags: ['Inference', 'Images'] },
        ]
    },
    frameworks: {
        title: 'Orchestration Frameworks',
        icon: <Layers className="w-5 h-5" />,
        items: [
            { name: 'LangChain', description: 'Most popular LLM orchestration - chains, agents, RAG pipelines, tools', url: 'https://langchain.com', tags: ['Python', 'TypeScript', 'Agents', 'RAG'] },
            { name: 'LlamaIndex', description: 'Data framework for LLM apps - RAG, document processing, knowledge graphs', url: 'https://llamaindex.ai', tags: ['Python', 'RAG', 'Knowledge Graphs'] },
            { name: 'Semantic Kernel', description: 'Microsoft SDK - Enterprise .NET/Python/Java, Azure integration, planners', url: 'https://github.com/microsoft/semantic-kernel', tags: ['C#', 'Python', 'Java', 'Enterprise'] },
            { name: 'Haystack', description: 'End-to-end NLP/LLM framework - production RAG, pipelines', url: 'https://haystack.deepset.ai', tags: ['Python', 'RAG', 'Production'] },
            { name: 'DSPy', description: 'Programmatic prompt optimization from Stanford - compile prompts', url: 'https://github.com/stanfordnlp/dspy', tags: ['Python', 'Optimization', 'Research'] },
            { name: 'Instructor', description: 'Structured outputs using Pydantic - type-safe, validated responses', url: 'https://github.com/jxnl/instructor', tags: ['Python', 'TypeScript', 'Structured Output'] },
            { name: 'Outlines', description: 'Constrained text generation - guaranteed JSON, regex, grammar', url: 'https://github.com/outlines-dev/outlines', tags: ['Python', 'Structured Output', 'Constrained'] },
            { name: 'Guidance', description: 'Microsoft structured generation - templates, constrained decoding', url: 'https://github.com/guidance-ai/guidance', tags: ['Python', 'Templates', 'Microsoft'] },
            { name: 'Marvin', description: 'AI functions as Python functions - quick prototyping, extraction', url: 'https://github.com/prefecthq/marvin', tags: ['Python', 'Simple', 'Extraction'] },
            { name: 'LMQL', description: 'Query language for LLMs - constraints, scripted prompting', url: 'https://lmql.ai', tags: ['Query Language', 'Constraints'] },
            { name: 'Guardrails AI', description: 'Input/output validation, guardrails, structured outputs', url: 'https://guardrailsai.com', tags: ['Validation', 'Safety', 'Structured'] },
            { name: 'NeMo Guardrails', description: 'NVIDIA safety rails - topical, moderation, fact-checking', url: 'https://github.com/NVIDIA/NeMo-Guardrails', tags: ['Safety', 'NVIDIA', 'Moderation'] },
            { name: 'Mirascope', description: 'Pythonic LLM toolkit - simple, type-safe, provider-agnostic', url: 'https://mirascope.io', tags: ['Python', 'Simple', 'Type-safe'] },
            { name: 'LiteLLM', description: 'Unified API for 100+ LLMs - OpenAI format, load balancing', url: 'https://github.com/BerriAI/litellm', tags: ['Unified API', 'Load Balancing'] },
        ]
    },
    vectordb: {
        title: 'Vector Databases',
        icon: <Database className="w-5 h-5" />,
        items: [
            { name: 'Pinecone', description: 'Managed vector DB - serverless, hybrid search, metadata filtering', url: 'https://pinecone.io', tags: ['Managed', 'Production', 'Serverless'] },
            { name: 'Weaviate', description: 'Open source - GraphQL API, hybrid search, modules, multi-tenancy', url: 'https://weaviate.io', tags: ['Open Source', 'Hybrid Search', 'GraphQL'] },
            { name: 'Chroma', description: 'Lightweight embedded DB - perfect for local dev and prototyping', url: 'https://trychroma.com', tags: ['Open Source', 'Local', 'Embedded'] },
            { name: 'Qdrant', description: 'Rust-based, high performance - filtering, payloads, quantization', url: 'https://qdrant.tech', tags: ['Open Source', 'Fast', 'Rust'] },
            { name: 'Milvus', description: 'Enterprise scale - distributed, GPU acceleration, billion-scale', url: 'https://milvus.io', tags: ['Open Source', 'Enterprise', 'Scale'] },
            { name: 'pgvector', description: 'Postgres extension - SQL integration, familiar tooling', url: 'https://github.com/pgvector/pgvector', tags: ['Postgres', 'SQL', 'Extension'] },
            { name: 'LanceDB', description: 'Embedded, serverless - zero-copy, versioned, multi-modal', url: 'https://lancedb.com', tags: ['Embedded', 'Serverless', 'Multi-modal'] },
            { name: 'Vespa', description: 'Yahoo search engine - hybrid search, ML ranking, real-time', url: 'https://vespa.ai', tags: ['Search', 'Ranking', 'Enterprise'] },
            { name: 'Elasticsearch', description: 'Classic search + vector - kNN, hybrid, mature ecosystem', url: 'https://elastic.co/elasticsearch', tags: ['Search', 'Hybrid', 'Enterprise'] },
            { name: 'MongoDB Atlas Vector', description: 'MongoDB with vector search - familiar, integrated', url: 'https://mongodb.com/atlas/vector-search', tags: ['MongoDB', 'Integrated'] },
            { name: 'Supabase Vector', description: 'Postgres + pgvector - open source, real-time, auth', url: 'https://supabase.com/vector', tags: ['Postgres', 'Open Source', 'BaaS'] },
            { name: 'Turbopuffer', description: 'Serverless vector DB - fast, cost-effective, simple', url: 'https://turbopuffer.com', tags: ['Serverless', 'Fast', 'Simple'] },
        ]
    },
    eval: {
        title: 'Evaluation & Observability',
        icon: <FlaskConical className="w-5 h-5" />,
        items: [
            { name: 'Braintrust', description: 'LLM evals, prompt playground, logging, datasets', url: 'https://braintrust.dev', tags: ['Evals', 'Platform', 'Datasets'] },
            { name: 'Promptfoo', description: 'Open source prompt testing, CI/CD integration, red teaming', url: 'https://promptfoo.dev', tags: ['Open Source', 'CLI', 'Testing'] },
            { name: 'Langfuse', description: 'Open source tracing, evals, prompt management, analytics', url: 'https://langfuse.com', tags: ['Open Source', 'Tracing', 'Analytics'] },
            { name: 'LangSmith', description: 'LangChain tracing, debugging, datasets, hub', url: 'https://smith.langchain.com', tags: ['Tracing', 'LangChain', 'Datasets'] },
            { name: 'Helicone', description: 'Logging, caching, rate limiting, cost tracking gateway', url: 'https://helicone.ai', tags: ['Gateway', 'Caching', 'Cost'] },
            { name: 'Portkey', description: 'AI gateway - observability, fallbacks, caching, load balancing', url: 'https://portkey.ai', tags: ['Gateway', 'Reliability', 'Load Balancing'] },
            { name: 'Ragas', description: 'Open source RAG evaluation - faithfulness, relevancy, context', url: 'https://github.com/explodinggradients/ragas', tags: ['Open Source', 'RAG Eval', 'Metrics'] },
            { name: 'Arize Phoenix', description: 'Open source ML observability - embeddings, traces, evals', url: 'https://phoenix.arize.com', tags: ['Open Source', 'Observability', 'Embeddings'] },
            { name: 'Weights & Biases', description: 'Experiment tracking, prompts, evals, artifacts', url: 'https://wandb.ai', tags: ['Experiments', 'MLOps', 'Tracking'] },
            { name: 'Humanloop', description: 'Prompt management, evals, deployment, collaboration', url: 'https://humanloop.com', tags: ['Prompts', 'Evals', 'Collaboration'] },
            { name: 'Parea AI', description: 'Prompt engineering platform - testing, evals, versioning', url: 'https://parea.ai', tags: ['Prompts', 'Testing', 'Versioning'] },
            { name: 'Galileo', description: 'LLM studio - debugging, evals, guardrails, fine-tuning', url: 'https://rungalileo.io', tags: ['Debugging', 'Evals', 'Guardrails'] },
            { name: 'TruLens', description: 'Open source eval framework - feedback functions, dashboards', url: 'https://trulens.org', tags: ['Open Source', 'Feedback', 'Dashboards'] },
            { name: 'DeepEval', description: 'Open source LLM eval framework - unit testing for LLMs', url: 'https://github.com/confident-ai/deepeval', tags: ['Open Source', 'Unit Testing'] },
            { name: 'OpenLLMetry', description: 'Open source observability - OpenTelemetry for LLMs', url: 'https://github.com/traceloop/openllmetry', tags: ['Open Source', 'OpenTelemetry'] },
        ]
    },
    agents: {
        title: 'Agent Frameworks',
        icon: <Zap className="w-5 h-5" />,
        items: [
            { name: 'LangGraph', description: 'Graph-based agents from LangChain - stateful workflows, cycles', url: 'https://langchain-ai.github.io/langgraph', tags: ['LangChain', 'Stateful', 'Graphs'] },
            { name: 'AutoGen', description: 'Microsoft multi-agent conversations, human-in-the-loop, code exec', url: 'https://github.com/microsoft/autogen', tags: ['Multi-Agent', 'Microsoft', 'Conversations'] },
            { name: 'CrewAI', description: 'Role-based multi-agent orchestration, tasks, processes', url: 'https://crewai.com', tags: ['Multi-Agent', 'Roles', 'Tasks'] },
            { name: 'Smolagents', description: 'Hugging Face lightweight agents, code execution, tools', url: 'https://github.com/huggingface/smolagents', tags: ['Lightweight', 'HuggingFace', 'Code'] },
            { name: 'OpenAI Assistants', description: 'OpenAI hosted agents - code interpreter, retrieval, tools', url: 'https://platform.openai.com/docs/assistants', tags: ['OpenAI', 'Hosted', 'Tools'] },
            { name: 'Anthropic Tool Use', description: 'Claude function calling and tool use', url: 'https://docs.anthropic.com/en/docs/build-with-claude/tool-use', tags: ['Anthropic', 'Tools', 'Functions'] },
            { name: 'Letta (MemGPT)', description: 'Stateful agents with long-term memory, persistence', url: 'https://github.com/letta-ai/letta', tags: ['Memory', 'Stateful', 'Persistence'] },
            { name: 'Agency Swarm', description: 'Agent orchestration framework, custom tools, communication', url: 'https://github.com/VRSEN/agency-swarm', tags: ['Multi-Agent', 'Tools', 'Orchestration'] },
            { name: 'Phidata', description: 'Build AI assistants with memory, knowledge, tools', url: 'https://phidata.com', tags: ['Assistants', 'Memory', 'Knowledge'] },
            { name: 'Composio', description: 'Tool integration platform - 150+ integrations for agents', url: 'https://composio.dev', tags: ['Tools', 'Integrations', 'Platform'] },
            { name: 'E2B', description: 'Sandboxed code execution for AI agents', url: 'https://e2b.dev', tags: ['Code Execution', 'Sandbox', 'Security'] },
            { name: 'Browserbase', description: 'Headless browser infrastructure for AI agents', url: 'https://browserbase.com', tags: ['Browser', 'Web', 'Automation'] },
        ]
    },
    finetuning: {
        title: 'Fine-Tuning & Training',
        icon: <Cpu className="w-5 h-5" />,
        items: [
            { name: 'Hugging Face', description: 'Models, datasets, training, inference - the ML hub', url: 'https://huggingface.co', tags: ['Models', 'Datasets', 'Training', 'Hub'] },
            { name: 'Axolotl', description: 'Easy fine-tuning with YAML configs - LoRA, QLoRA, full', url: 'https://github.com/OpenAccess-AI-Collective/axolotl', tags: ['Open Source', 'Easy', 'LoRA'] },
            { name: 'Unsloth', description: '2x faster LoRA training, 80% less memory', url: 'https://github.com/unslothai/unsloth', tags: ['Fast', 'LoRA', 'Efficient'] },
            { name: 'LLaMA-Factory', description: 'Easy fine-tuning for 100+ LLMs - WebUI, CLI', url: 'https://github.com/hiyouga/LLaMA-Factory', tags: ['Easy', 'WebUI', 'Multi-model'] },
            { name: 'OpenAI Fine-tuning', description: 'Fine-tune OpenAI models via API', url: 'https://platform.openai.com/docs/guides/fine-tuning', tags: ['OpenAI', 'API', 'Managed'] },
            { name: 'Modal', description: 'Serverless GPU compute - easy deployment, pay-per-use', url: 'https://modal.com', tags: ['GPU', 'Serverless', 'Easy'] },
            { name: 'Lambda Labs', description: 'On-demand A100s, H100s - cloud GPU instances', url: 'https://lambdalabs.com', tags: ['GPU Cloud', 'H100'] },
            { name: 'RunPod', description: 'GPU marketplace - affordable spot instances, templates', url: 'https://runpod.io', tags: ['GPU Cloud', 'Affordable', 'Spot'] },
            { name: 'Vast.ai', description: 'GPU marketplace - cheapest GPUs, spot pricing', url: 'https://vast.ai', tags: ['GPU Cloud', 'Cheap', 'Marketplace'] },
            { name: 'TRL', description: 'Transformer Reinforcement Learning - RLHF, DPO, PPO', url: 'https://github.com/huggingface/trl', tags: ['RLHF', 'DPO', 'HuggingFace'] },
            { name: 'PEFT', description: 'Parameter-Efficient Fine-Tuning - LoRA, adapters', url: 'https://github.com/huggingface/peft', tags: ['LoRA', 'Efficient', 'HuggingFace'] },
            { name: 'LitGPT', description: 'Lightning-based training and fine-tuning', url: 'https://github.com/Lightning-AI/litgpt', tags: ['Lightning', 'Training'] },
            { name: 'Predibase', description: 'Fine-tuning platform - LoRA serving, deployment', url: 'https://predibase.com', tags: ['Platform', 'LoRA', 'Serving'] },
            { name: 'Weights & Biases', description: 'Experiment tracking, hyperparameter tuning', url: 'https://wandb.ai', tags: ['Tracking', 'Experiments', 'MLOps'] },
        ]
    },
    models: {
        title: 'Open Source Models',
        icon: <Code className="w-5 h-5" />,
        items: [
            { name: 'Llama', description: 'Meta - Open weights, multiple sizes, text and vision', url: 'https://llama.meta.com', tags: ['Meta', 'Text', 'Vision'] },
            { name: 'Mistral / Mixtral', description: 'Mistral AI - 7B, 8x7B, 8x22B MoE, Codestral', url: 'https://mistral.ai', tags: ['MoE', 'Open Weights', 'Code'] },
            { name: 'Qwen 2.5', description: 'Alibaba - 0.5B to 72B, strong multilingual, code, math', url: 'https://github.com/QwenLM/Qwen2.5', tags: ['Multilingual', 'Code', 'Math'] },
            { name: 'Gemma 2', description: 'Google - 2B, 9B, 27B efficient instruction-tuned models', url: 'https://ai.google.dev/gemma', tags: ['Google', 'Efficient', 'Instruction'] },
            { name: 'Phi-3 / Phi-4', description: 'Microsoft - Small but capable 3.8B, 14B models', url: 'https://azure.microsoft.com/en-us/products/phi', tags: ['Microsoft', 'Small', 'Efficient'] },
            { name: 'DeepSeek V3', description: 'DeepSeek - 671B MoE, strong reasoning and code', url: 'https://github.com/deepseek-ai/DeepSeek-V3', tags: ['MoE', 'Reasoning', 'Code'] },
            { name: 'DeepSeek Coder', description: 'Strong code generation, 1.3B to 33B', url: 'https://github.com/deepseek-ai/DeepSeek-Coder', tags: ['Code', 'Programming'] },
            { name: 'StarCoder 2', description: 'BigCode - 3B, 7B, 15B, 600+ languages', url: 'https://github.com/bigcode-project/starcoder2', tags: ['Code', 'Multi-language'] },
            { name: 'CodeLlama', description: 'Meta - Code completion, infilling, instruction', url: 'https://github.com/meta-llama/codellama', tags: ['Meta', 'Code', 'Infilling'] },
            { name: 'Yi', description: '01.AI - 6B, 9B, 34B - strong Chinese/English', url: 'https://github.com/01-ai/Yi', tags: ['Bilingual', 'Chinese'] },
            { name: 'Command R', description: 'Cohere - 35B, 104B - RAG optimized, tool use', url: 'https://cohere.com/command', tags: ['RAG', 'Tool Use', 'Enterprise'] },
            { name: 'Falcon', description: 'TII - 7B, 40B, 180B - permissive license', url: 'https://falconllm.tii.ae', tags: ['Permissive', 'Large'] },
            { name: 'DBRX', description: 'Databricks - 132B MoE, strong general performance', url: 'https://github.com/databricks/dbrx', tags: ['Databricks', 'MoE'] },
            { name: 'Nous Research', description: 'Fine-tuned models - Hermes, Capybara series', url: 'https://nousresearch.com', tags: ['Fine-tuned', 'Community'] },
        ]
    },
    inference: {
        title: 'Inference & Deployment',
        icon: <Rocket className="w-5 h-5" />,
        items: [
            { name: 'vLLM', description: 'High-throughput serving - PagedAttention, continuous batching', url: 'https://vllm.ai', tags: ['Open Source', 'Fast', 'Production'] },
            { name: 'TGI (Text Generation Inference)', description: 'Hugging Face production server - tensor parallelism, quantization', url: 'https://github.com/huggingface/text-generation-inference', tags: ['HuggingFace', 'Production'] },
            { name: 'Ollama', description: 'Local LLM runner - easy setup, model library, API', url: 'https://ollama.ai', tags: ['Local', 'Easy', 'Desktop'] },
            { name: 'llama.cpp', description: 'CPU/GPU inference in C++ - quantization, GGUF format', url: 'https://github.com/ggerganov/llama.cpp', tags: ['CPU', 'Quantization', 'C++'] },
            { name: 'TensorRT-LLM', description: 'NVIDIA optimized inference - maximum GPU performance', url: 'https://github.com/NVIDIA/TensorRT-LLM', tags: ['NVIDIA', 'Optimized', 'GPU'] },
            { name: 'SGLang', description: 'Fast serving with RadixAttention - structured generation', url: 'https://github.com/sgl-project/sglang', tags: ['Fast', 'Structured', 'Research'] },
            { name: 'LMDeploy', description: 'Efficient LLM deployment - quantization, KV cache', url: 'https://github.com/InternLM/lmdeploy', tags: ['Efficient', 'Quantization'] },
            { name: 'ExLlamaV2', description: 'Fast inference for quantized models - EXL2 format', url: 'https://github.com/turboderp/exllamav2', tags: ['Quantized', 'Fast'] },
            { name: 'LocalAI', description: 'OpenAI-compatible local API - multiple backends', url: 'https://localai.io', tags: ['Local', 'OpenAI Compatible'] },
            { name: 'LM Studio', description: 'Desktop app for local LLMs - GUI, chat interface', url: 'https://lmstudio.ai', tags: ['Desktop', 'GUI', 'Local'] },
            { name: 'Jan', description: 'Open source ChatGPT alternative - local, private', url: 'https://jan.ai', tags: ['Desktop', 'Open Source', 'Private'] },
            { name: 'GPT4All', description: 'Local LLM ecosystem - models, chat, embeddings', url: 'https://gpt4all.io', tags: ['Local', 'Ecosystem', 'Easy'] },
            { name: 'BentoML', description: 'ML model serving framework - packaging, deployment', url: 'https://bentoml.com', tags: ['Serving', 'MLOps', 'Packaging'] },
            { name: 'Ray Serve', description: 'Scalable model serving - batching, autoscaling', url: 'https://docs.ray.io/en/latest/serve', tags: ['Scalable', 'Ray', 'Production'] },
        ]
    },
    embeddings: {
        title: 'Embeddings & Reranking',
        icon: <Binary className="w-5 h-5" />,
        items: [
            { name: 'OpenAI Embeddings', description: 'High quality embedding models with multiple dimension options', url: 'https://platform.openai.com/docs/guides/embeddings', tags: ['API', 'High Quality'] },
            { name: 'Cohere Embed v3', description: 'Multilingual embeddings - 1024 dims, compression', url: 'https://cohere.com/embed', tags: ['API', 'Multilingual', 'Compression'] },
            { name: 'Voyage AI', description: 'Specialized embeddings - code, law, finance, multilingual', url: 'https://voyageai.com', tags: ['API', 'Specialized', 'Domain'] },
            { name: 'Jina Embeddings', description: 'Open source embeddings - 8K context, multilingual', url: 'https://jina.ai/embeddings', tags: ['Open Source', 'Long Context'] },
            { name: 'BGE (BAAI)', description: 'Open source - bge-large-en-v1.5, strong MTEB scores', url: 'https://github.com/FlagOpen/FlagEmbedding', tags: ['Open Source', 'MTEB'] },
            { name: 'E5 (Microsoft)', description: 'e5-mistral-7b-instruct - 4096 dims, instruction-tuned', url: 'https://github.com/microsoft/unilm/tree/master/e5', tags: ['Microsoft', 'Instruction'] },
            { name: 'Nomic Embed', description: 'Open source - nomic-embed-text-v1.5, 768 dims', url: 'https://nomic.ai', tags: ['Open Source', 'Efficient'] },
            { name: 'GTE (Alibaba)', description: 'General Text Embeddings - strong multilingual', url: 'https://github.com/alibaba-damo-academy/FlagEmbedding', tags: ['Open Source', 'Multilingual'] },
            { name: 'Sentence Transformers', description: 'Python library for embeddings - 100+ models', url: 'https://sbert.net', tags: ['Library', 'Python', 'Models'] },
            { name: 'Cohere Rerank', description: 'Reranking API - rerank-v3, multilingual', url: 'https://cohere.com/rerank', tags: ['API', 'Reranking'] },
            { name: 'Jina Reranker', description: 'Open source reranking - jina-reranker-v2', url: 'https://jina.ai/reranker', tags: ['Open Source', 'Reranking'] },
            { name: 'BGE Reranker', description: 'Open source - bge-reranker-v2-m3, cross-encoder', url: 'https://github.com/FlagOpen/FlagEmbedding', tags: ['Open Source', 'Reranking'] },
            { name: 'Mixedbread', description: 'Embeddings and reranking - mxbai-embed-large', url: 'https://mixedbread.ai', tags: ['API', 'Embeddings', 'Reranking'] },
            { name: 'MTEB Leaderboard', description: 'Massive Text Embedding Benchmark - compare models', url: 'https://huggingface.co/spaces/mteb/leaderboard', tags: ['Benchmark', 'Comparison'] },
        ]
    },
    learning: {
        title: 'Learning Resources',
        icon: <GraduationCap className="w-5 h-5" />,
        items: [
            { name: 'OpenAI Cookbook', description: 'Official examples and guides for OpenAI APIs', url: 'https://cookbook.openai.com', tags: ['OpenAI', 'Examples', 'Official'] },
            { name: 'Anthropic Docs', description: 'Claude documentation, prompt engineering guide', url: 'https://docs.anthropic.com', tags: ['Anthropic', 'Prompting', 'Official'] },
            { name: 'DeepLearning.AI Short Courses', description: 'Free courses on LLMs, RAG, agents, fine-tuning', url: 'https://deeplearning.ai/short-courses', tags: ['Courses', 'Free', 'Beginner'] },
            { name: 'Hugging Face NLP Course', description: 'Comprehensive NLP and transformers course', url: 'https://huggingface.co/learn/nlp-course', tags: ['Course', 'Free', 'Transformers'] },
            { name: 'Full Stack LLM Bootcamp', description: 'Production LLM applications course', url: 'https://fullstackdeeplearning.com', tags: ['Course', 'Production', 'Full Stack'] },
            { name: 'LangChain Academy', description: 'Official LangChain courses and tutorials', url: 'https://academy.langchain.com', tags: ['LangChain', 'Official', 'Free'] },
            { name: 'Prompt Engineering Guide', description: 'Comprehensive prompting techniques guide', url: 'https://promptingguide.ai', tags: ['Prompting', 'Guide', 'Free'] },
            { name: 'LLM University (Cohere)', description: 'NLP and LLM fundamentals course', url: 'https://cohere.com/llmu', tags: ['Course', 'Fundamentals', 'Free'] },
            { name: 'Simon Willison Blog', description: 'Insights on LLMs, tools, and AI engineering', url: 'https://simonwillison.net', tags: ['Blog', 'Insights', 'Tools'] },
            { name: 'Chip Huyen Blog', description: 'ML systems, LLMOps, production ML', url: 'https://huyenchip.com/blog', tags: ['Blog', 'MLOps', 'Production'] },
            { name: 'Eugene Yan Blog', description: 'Applied ML, RecSys, LLM applications', url: 'https://eugeneyan.com', tags: ['Blog', 'Applied ML', 'Production'] },
            { name: 'Lilian Weng Blog', description: 'Deep dives on transformers, agents, RLHF', url: 'https://lilianweng.github.io', tags: ['Blog', 'Research', 'Deep Dives'] },
            { name: 'The Batch (DeepLearning.AI)', description: 'Weekly AI newsletter by Andrew Ng', url: 'https://deeplearning.ai/the-batch', tags: ['Newsletter', 'Weekly', 'News'] },
            { name: 'AI Engineering Newsletter', description: 'Curated AI engineering news and resources', url: 'https://aiengineering.substack.com', tags: ['Newsletter', 'Engineering'] },
            { name: 'Latent Space Podcast', description: 'AI engineering interviews and discussions', url: 'https://latent.space', tags: ['Podcast', 'Interviews'] },
            { name: 'Practical AI Podcast', description: 'Making AI practical and accessible', url: 'https://changelog.com/practicalai', tags: ['Podcast', 'Practical'] },
            { name: 'r/LocalLLaMA', description: 'Reddit community for local LLM enthusiasts', url: 'https://reddit.com/r/LocalLLaMA', tags: ['Community', 'Reddit', 'Local'] },
            { name: 'Hugging Face Discord', description: 'Active ML community, help, discussions', url: 'https://huggingface.co/join/discord', tags: ['Community', 'Discord', 'Help'] },
            { name: 'LangChain Discord', description: 'LangChain community and support', url: 'https://discord.gg/langchain', tags: ['Community', 'Discord', 'LangChain'] },
            { name: 'MLOps Community', description: 'Slack community for ML operations', url: 'https://mlops.community', tags: ['Community', 'Slack', 'MLOps'] },
        ]
    },
    stacks: {
        title: 'Recommended Stacks',
        icon: <BookOpen className="w-5 h-5" />,
        items: []
    },
    aipm: {
        title: 'AI Product Management',
        icon: <Briefcase className="w-5 h-5" />,
        items: []
    }
};

// AI PM Frameworks - Our own detailed content
interface AIPMFramework {
    id: string;
    name: string;
    tagline: string;
    icon: React.ReactNode;
    category: 'lifecycle' | 'prioritization' | 'discovery' | 'operations' | 'governance' | 'metrics';
    whenToUse: string[];
    steps: { title: string; description: string }[];
    aiSpecificTips: string[];
    example?: { scenario: string; application: string };
}

const aipmFrameworks: AIPMFramework[] = [
    {
        id: 'ai-product-lifecycle',
        name: 'AI Product Lifecycle',
        tagline: 'End-to-end framework for AI product development from ideation to iteration',
        icon: <Workflow className="w-6 h-6" />,
        category: 'lifecycle',
        whenToUse: [
            'Starting a new AI product or feature from scratch',
            'Need a structured approach to AI development',
            'Want to ensure all stakeholders are aligned on process'
        ],
        steps: [
            { title: '1. Problem Discovery', description: 'Identify problems where AI adds unique value. Validate that ML/AI is the right solution vs. rules-based approaches.' },
            { title: '2. Data Assessment', description: 'Evaluate data availability, quality, and labeling needs. Determine if you have enough data to train or if you need synthetic/external data.' },
            { title: '3. Feasibility Analysis', description: 'Assess technical feasibility with ML team. Run quick experiments to validate approach before committing resources.' },
            { title: '4. MVP Definition', description: 'Define minimum viable AI product. Start with narrow scope, single use case, and clear success metrics.' },
            { title: '5. Development & Training', description: 'Build data pipelines, train models, establish evaluation benchmarks. Iterate on model performance.' },
            { title: '6. Integration & Testing', description: 'Integrate AI into product, A/B test with users, measure real-world performance vs. offline metrics.' },
            { title: '7. Deployment & Monitoring', description: 'Deploy with proper monitoring, set up drift detection, establish feedback loops for continuous improvement.' },
            { title: '8. Iteration & Scaling', description: 'Analyze production performance, gather user feedback, iterate on model and expand to new use cases.' }
        ],
        aiSpecificTips: [
            'AI products require longer experimentation phases than traditional software',
            'Build feedback loops early - user corrections improve model over time',
            'Plan for model degradation and retraining from day one',
            'Consider the cold start problem for personalization features'
        ],
        example: {
            scenario: 'Building a document classification system for legal documents',
            application: 'Start with 3 document types, validate 85%+ accuracy on test set before expanding to 20+ types'
        }
    },
    {
        id: 'ai-rice-scoring',
        name: 'AI-RICE Scoring',
        tagline: 'Adapted RICE framework for prioritizing AI features with data and model considerations',
        icon: <BarChart3 className="w-6 h-6" />,
        category: 'prioritization',
        whenToUse: [
            'Prioritizing AI features in your backlog',
            'Comparing AI vs non-AI solutions for a problem',
            'Justifying AI investment to stakeholders'
        ],
        steps: [
            { title: 'Reach', description: 'How many users/transactions will this AI feature impact? Consider both direct users and downstream effects.' },
            { title: 'Impact', description: 'What\'s the magnitude of improvement? For AI: accuracy gains, time savings, cost reduction, new capabilities enabled.' },
            { title: 'Confidence', description: 'How sure are we this will work? Factor in: data availability (40%), technical feasibility (30%), team capability (30%).' },
            { title: 'Effort', description: 'Total effort including: data collection/labeling, model development, integration, monitoring setup, and ongoing maintenance.' },
            { title: 'Calculate Score', description: 'Score = (Reach × Impact × Confidence) / Effort. Compare AI solutions against simpler alternatives.' }
        ],
        aiSpecificTips: [
            'Confidence for AI should include data quality assessment',
            'Effort must include ongoing costs: compute, retraining, monitoring',
            'Consider "time to first value" - AI often has longer lead times',
            'Factor in regulatory/compliance requirements for AI systems'
        ],
        example: {
            scenario: 'Prioritizing between AI-powered search vs. manual tagging system',
            application: 'AI search: Reach=10K, Impact=3, Confidence=0.6 (limited training data), Effort=8 → Score: 2,250. Manual tags: Reach=10K, Impact=2, Confidence=0.9, Effort=3 → Score: 6,000. Start with tags, add AI later.'
        }
    },
    {
        id: 'ai-decision-matrix',
        name: 'AI Decision Matrix',
        tagline: 'Framework to determine when AI is the right solution for a problem',
        icon: <Target className="w-6 h-6" />,
        category: 'discovery',
        whenToUse: [
            'Evaluating if a problem needs AI or simpler solutions',
            'Choosing between different AI approaches',
            'Communicating AI decisions to non-technical stakeholders'
        ],
        steps: [
            { title: 'Problem Complexity', description: 'Rate 1-5: Is the problem too complex for rules? Does it require pattern recognition, prediction, or generation?' },
            { title: 'Data Availability', description: 'Rate 1-5: Do you have sufficient labeled data? Is data accessible and of good quality?' },
            { title: 'Value of Automation', description: 'Rate 1-5: How much value does automation create? Consider volume, speed requirements, and cost savings.' },
            { title: 'Tolerance for Errors', description: 'Rate 1-5: Can the system tolerate AI mistakes? Are there human fallbacks? What\'s the cost of errors?' },
            { title: 'Decision Threshold', description: 'Sum scores: 16-20 = Strong AI fit, 12-15 = Consider AI with caveats, 8-11 = Hybrid approach, <8 = Use rules/heuristics' }
        ],
        aiSpecificTips: [
            'High complexity + low error tolerance = needs human-in-the-loop',
            'Low data + high value = consider few-shot learning or LLMs',
            'If rules can solve 80% of cases, start there and add AI for edge cases',
            'Consider maintenance burden - AI systems need ongoing care'
        ],
        example: {
            scenario: 'Should we use AI for customer support ticket routing?',
            application: 'Complexity: 4 (many categories, context-dependent), Data: 5 (years of labeled tickets), Value: 4 (high volume), Error tolerance: 4 (human agents catch mistakes) → Score: 17, strong AI fit'
        }
    },
    {
        id: 'jobs-to-be-done-ai',
        name: 'JTBD for AI Features',
        tagline: 'Discover what jobs users hire AI to do, not just what features they want',
        icon: <Users className="w-6 h-6" />,
        category: 'discovery',
        whenToUse: [
            'Defining AI product requirements',
            'Understanding user expectations for AI',
            'Avoiding feature bloat in AI products'
        ],
        steps: [
            { title: 'Identify the Job', description: 'What progress is the user trying to make? Frame as: "When [situation], I want to [motivation], so I can [outcome]."' },
            { title: 'Map Current Solutions', description: 'How do users accomplish this job today? What\'s painful about current approaches?' },
            { title: 'Define AI\'s Role', description: 'How can AI help? Categories: Automate (do it for me), Augment (help me do it better), Accelerate (help me do it faster).' },
            { title: 'Identify Success Criteria', description: 'How will users judge if AI did the job well? Speed? Accuracy? Creativity? Cost?' },
            { title: 'Design for Failure', description: 'What happens when AI fails? How does user recover? Design graceful degradation.' }
        ],
        aiSpecificTips: [
            'Users hire AI for confidence, not just output - show your work',
            'The job often includes "without me having to learn new skills"',
            'AI anxiety is real - address "will this replace me?" concerns',
            'Users want control - let them override AI decisions'
        ],
        example: {
            scenario: 'AI writing assistant for marketing team',
            application: 'Job: "When I need to write campaign copy, I want to generate multiple variations quickly, so I can test more ideas without spending hours writing." AI role: Accelerate + Augment. Success: 5x more variations in same time.'
        }
    },
    {
        id: 'kano-ai',
        name: 'Kano Model for AI',
        tagline: 'Categorize AI features by their impact on user satisfaction',
        icon: <TrendingUp className="w-6 h-6" />,
        category: 'prioritization',
        whenToUse: [
            'Deciding which AI capabilities to build first',
            'Balancing "wow" features vs. reliability',
            'Setting user expectations for AI products'
        ],
        steps: [
            { title: 'Must-Be (Basic)', description: 'AI features users expect to work. Absence causes dissatisfaction, presence doesn\'t delight. Example: AI doesn\'t crash, responds in reasonable time.' },
            { title: 'Performance (Linear)', description: 'More is better. Satisfaction scales with capability. Example: Accuracy - 90% good, 95% better, 99% best.' },
            { title: 'Attractive (Delighters)', description: 'Unexpected capabilities that wow users. Absence is fine, presence delights. Example: AI proactively suggests improvements.' },
            { title: 'Indifferent', description: 'Features users don\'t care about. Don\'t waste resources here. Example: Showing model confidence scores to non-technical users.' },
            { title: 'Reverse', description: 'Features some users actively dislike. Example: Overly aggressive AI suggestions, "creepy" personalization.' }
        ],
        aiSpecificTips: [
            'AI accuracy often shifts from Attractive → Performance → Must-Be as users adapt',
            'Explainability is becoming a Must-Be for many AI products',
            'Speed is usually Performance (faster = better) but has diminishing returns',
            'Personalization can be Attractive or Reverse depending on execution'
        ],
        example: {
            scenario: 'AI email assistant',
            application: 'Must-Be: Grammar check works, suggestions are relevant. Performance: Better suggestions, faster response. Attractive: Learns your writing style over time. Reverse: Auto-sending emails without confirmation.'
        }
    },
    {
        id: 'mlops-lifecycle',
        name: 'MLOps Maturity Model',
        tagline: 'Framework for operationalizing ML from ad-hoc to fully automated',
        icon: <Workflow className="w-6 h-6" />,
        category: 'operations',
        whenToUse: [
            'Assessing your team\'s ML operational maturity',
            'Planning MLOps infrastructure investments',
            'Scaling from prototype to production AI'
        ],
        steps: [
            { title: 'Level 0: Manual', description: 'Data scientists work in notebooks, manual deployments, no monitoring. Good for: POCs and experiments.' },
            { title: 'Level 1: ML Pipeline', description: 'Automated training pipeline, feature store basics, model versioning. Good for: First production models.' },
            { title: 'Level 2: CI/CD for ML', description: 'Automated testing, continuous training, A/B testing infrastructure. Good for: Multiple models in production.' },
            { title: 'Level 3: Full Automation', description: 'Auto-retraining on drift, automated rollback, self-healing systems. Good for: Mission-critical AI at scale.' }
        ],
        aiSpecificTips: [
            'Most teams should aim for Level 1-2, not Level 3',
            'Monitoring is more important than automation early on',
            'Feature stores provide huge ROI for teams with multiple models',
            'Don\'t automate retraining until you understand model behavior'
        ],
        example: {
            scenario: 'E-commerce recommendation system',
            application: 'Start at Level 1 with weekly batch retraining. Move to Level 2 when you have 3+ recommendation models. Level 3 only if recommendations are core to business and you have dedicated ML platform team.'
        }
    },
    {
        id: 'responsible-ai-checklist',
        name: 'Responsible AI Checklist',
        tagline: 'Ensure your AI product is fair, transparent, and accountable',
        icon: <Shield className="w-6 h-6" />,
        category: 'governance',
        whenToUse: [
            'Before launching any AI feature',
            'During AI product reviews',
            'When AI makes decisions affecting people'
        ],
        steps: [
            { title: 'Fairness Audit', description: 'Test for bias across protected groups. Check training data representation. Measure outcome disparities.' },
            { title: 'Transparency Design', description: 'Can users understand why AI made a decision? Provide explanations appropriate to audience. Document model limitations.' },
            { title: 'Human Oversight', description: 'Define when humans must review AI decisions. Create escalation paths. Enable user appeals.' },
            { title: 'Privacy Review', description: 'What data does the model use? Is consent obtained? Can users opt out? Is data minimization practiced?' },
            { title: 'Impact Assessment', description: 'What happens if AI fails? Who is harmed? What are second-order effects? Document and mitigate risks.' }
        ],
        aiSpecificTips: [
            'Bias can emerge even with "neutral" features - test thoroughly',
            'Explainability needs vary by stakeholder - users vs. regulators vs. developers',
            'Build audit trails from day one - retrofitting is expensive',
            'Consider the "newspaper test" - would you be comfortable if this was reported?'
        ],
        example: {
            scenario: 'AI-powered hiring screening tool',
            application: 'Fairness: Test pass rates across gender, race, age. Transparency: Show candidates which factors influenced decision. Oversight: Human reviews all rejections. Privacy: Only use job-relevant data. Impact: Document risk of qualified candidates being filtered out.'
        }
    },
    {
        id: 'heart-ai',
        name: 'HEART Framework for AI',
        tagline: 'Measure AI product success with user-centered metrics',
        icon: <BarChart3 className="w-6 h-6" />,
        category: 'metrics',
        whenToUse: [
            'Defining success metrics for AI features',
            'Measuring AI product health',
            'Reporting AI impact to stakeholders'
        ],
        steps: [
            { title: 'Happiness', description: 'User satisfaction with AI. Metrics: NPS for AI features, satisfaction surveys, sentiment in feedback. AI-specific: Trust scores, comfort with AI decisions.' },
            { title: 'Engagement', description: 'How much users interact with AI. Metrics: Feature usage, session depth, return usage. AI-specific: Override rate (lower = more trust), suggestion acceptance rate.' },
            { title: 'Adoption', description: 'New users trying AI features. Metrics: Feature discovery, first-use completion, activation rate. AI-specific: Time to first successful AI interaction.' },
            { title: 'Retention', description: 'Users continuing to use AI. Metrics: Weekly/monthly active users of AI features, churn from AI features. AI-specific: Do users come back after AI errors?' },
            { title: 'Task Success', description: 'AI helping users achieve goals. Metrics: Task completion rate, time to complete, error rate. AI-specific: AI accuracy, false positive/negative rates, human correction rate.' }
        ],
        aiSpecificTips: [
            'Track "AI trust recovery" - do users return after AI makes mistakes?',
            'Measure time saved, not just accuracy - that\'s the user value',
            'Override rate is a key signal - too high means low trust, too low might mean users aren\'t checking',
            'Segment metrics by user expertise - novices and experts use AI differently'
        ],
        example: {
            scenario: 'AI code completion tool',
            application: 'Happiness: Developer satisfaction score. Engagement: Suggestions accepted per session. Adoption: % of eligible users who enabled feature. Retention: Weekly active users. Task Success: Time to complete coding tasks, bugs introduced.'
        }
    },
    {
        id: 'ai-canvas',
        name: 'AI Product Canvas',
        tagline: 'One-page template to align stakeholders on AI product vision',
        icon: <Lightbulb className="w-6 h-6" />,
        category: 'discovery',
        whenToUse: [
            'Kicking off a new AI product or feature',
            'Aligning engineering and business stakeholders',
            'Documenting AI product decisions'
        ],
        steps: [
            { title: 'Problem & Users', description: 'What problem are we solving? For whom? Why is AI the right approach?' },
            { title: 'AI Value Proposition', description: 'What can AI do that alternatives cannot? What\'s the unique advantage?' },
            { title: 'Data Requirements', description: 'What data do we need? Do we have it? How will we get more? Quality requirements?' },
            { title: 'Model Approach', description: 'What type of AI/ML? Build vs. buy vs. API? Key technical risks?' },
            { title: 'Success Metrics', description: 'How do we measure success? Offline metrics (accuracy) vs. online metrics (business impact)?' },
            { title: 'Risks & Mitigations', description: 'What could go wrong? Bias risks? Failure modes? How do we mitigate?' },
            { title: 'Resources & Timeline', description: 'What team do we need? What\'s the timeline? Key milestones?' },
            { title: 'Iteration Plan', description: 'How will we improve over time? Feedback loops? Retraining cadence?' }
        ],
        aiSpecificTips: [
            'Fill this out collaboratively with ML engineers - avoid PM assumptions',
            'Be explicit about what happens when AI is wrong',
            'Include data labeling and collection in timeline',
            'Revisit quarterly - AI products evolve as you learn'
        ],
        example: {
            scenario: 'AI-powered fraud detection',
            application: 'Problem: Manual review can\'t scale, missing 30% of fraud. Data: 2 years of labeled transactions. Approach: Gradient boosting for v1, deep learning for v2. Success: Catch 90%+ fraud, <1% false positives. Risk: Bias against new customers. Timeline: 3 months to MVP.'
        }
    },
    {
        id: 'ai-user-trust',
        name: 'AI Trust Ladder',
        tagline: 'Framework for building user trust in AI systems progressively',
        icon: <CheckCircle2 className="w-6 h-6" />,
        category: 'discovery',
        whenToUse: [
            'Launching AI to skeptical users',
            'Designing AI onboarding experiences',
            'Recovering from AI failures'
        ],
        steps: [
            { title: 'Level 1: Transparency', description: 'Users know AI is involved. Label AI-generated content. Explain what AI does and doesn\'t do.' },
            { title: 'Level 2: Control', description: 'Users can override AI. Provide manual alternatives. Let users adjust AI behavior.' },
            { title: 'Level 3: Understanding', description: 'Users understand why AI decided something. Show reasoning. Highlight key factors.' },
            { title: 'Level 4: Verification', description: 'Users can verify AI is working correctly. Show confidence levels. Provide audit trails.' },
            { title: 'Level 5: Delegation', description: 'Users trust AI to act autonomously. Requires track record of success. Still provide oversight options.' }
        ],
        aiSpecificTips: [
            'Start at Level 1-2 for new AI features, earn your way up',
            'Different users will be at different levels - let them choose',
            'After AI errors, users often drop back levels - design for this',
            'B2B products often need Level 3-4 for compliance reasons'
        ],
        example: {
            scenario: 'AI calendar scheduling assistant',
            application: 'Level 1: "AI suggested this time". Level 2: User can pick different time. Level 3: "Suggested because you\'re usually free and it\'s their timezone". Level 4: Show past scheduling success rate. Level 5: Auto-schedule with one-click undo.'
        }
    }
];

// Recommended Stacks - detailed breakdowns
interface StackComponent {
    category: string;
    name: string;
    why: string;
    link: string;
}

interface RecommendedStack {
    id: string;
    name: string;
    tagline: string;
    bestFor: string[];
    monthlyCost: string;
    complexity: 'Low' | 'Medium' | 'High';
    components: StackComponent[];
    tradeoffs: { pros: string[]; cons: string[] };
    gettingStarted: string[];
}

const recommendedStacks: RecommendedStack[] = [
    {
        id: 'beginner-rag',
        name: 'Beginner RAG Stack',
        tagline: 'Start building RAG applications with minimal setup and cost',
        bestFor: ['Learning RAG concepts', 'Hackathons and prototypes', 'Small document collections (<10K docs)'],
        monthlyCost: '$10-50',
        complexity: 'Low',
        components: [
            { category: 'LLM', name: 'OpenAI GPT-4o-mini', why: 'Cheap, fast, good quality for most use cases', link: 'https://platform.openai.com' },
            { category: 'Embeddings', name: 'OpenAI text-embedding-3-small', why: 'Simple API, good performance, same billing', link: 'https://platform.openai.com/docs/guides/embeddings' },
            { category: 'Vector DB', name: 'Chroma', why: 'Runs locally, no setup, free, Python-native', link: 'https://trychroma.com' },
            { category: 'Framework', name: 'LangChain', why: 'Most tutorials, largest community, easy to start', link: 'https://langchain.com' },
            { category: 'Evaluation', name: 'Promptfoo', why: 'Free, CLI-based, easy to add to any project', link: 'https://promptfoo.dev' },
        ],
        tradeoffs: {
            pros: ['Up and running in hours', 'Minimal cost to experiment', 'Tons of tutorials available', 'No infrastructure to manage'],
            cons: ['Chroma not suitable for production scale', 'OpenAI dependency (no offline)', 'Limited observability', 'Will need to migrate components later']
        },
        gettingStarted: [
            'pip install langchain langchain-openai chromadb promptfoo',
            'Set OPENAI_API_KEY environment variable',
            'Follow LangChain RAG quickstart tutorial',
            'Add promptfoo tests before shipping anything'
        ]
    },
    {
        id: 'production-rag',
        name: 'Production RAG Stack',
        tagline: 'Battle-tested stack for production RAG with enterprise features',
        bestFor: ['Production applications with real users', 'Large document collections (100K+ docs)', 'Teams needing observability and reliability'],
        monthlyCost: '$200-2000',
        complexity: 'Medium',
        components: [
            { category: 'LLM', name: 'Claude 3.5 Sonnet or GPT-4o', why: 'Best quality, reliable, good tool use', link: 'https://console.anthropic.com' },
            { category: 'Embeddings', name: 'Cohere embed-v3', why: 'Excellent quality, compression options, multilingual', link: 'https://cohere.com/embed' },
            { category: 'Vector DB', name: 'Pinecone', why: 'Managed, scales automatically, hybrid search', link: 'https://pinecone.io' },
            { category: 'Reranking', name: 'Cohere Rerank', why: 'Dramatically improves retrieval quality', link: 'https://cohere.com/rerank' },
            { category: 'Framework', name: 'LlamaIndex', why: 'Better for complex RAG, more retrieval options', link: 'https://llamaindex.ai' },
            { category: 'Observability', name: 'Langfuse', why: 'Open source, self-hostable, great tracing', link: 'https://langfuse.com' },
            { category: 'Evaluation', name: 'Ragas + Braintrust', why: 'RAG-specific metrics + production evals', link: 'https://github.com/explodinggradients/ragas' },
        ],
        tradeoffs: {
            pros: ['Production-ready out of the box', 'Excellent retrieval quality with reranking', 'Full observability and debugging', 'Scales to millions of documents'],
            cons: ['Higher cost than beginner stack', 'More services to manage', 'Vendor lock-in concerns', 'Requires more ML knowledge to optimize']
        },
        gettingStarted: [
            'Set up Pinecone index with appropriate dimensions for Cohere',
            'Implement chunking strategy (start with 512 tokens, 50 overlap)',
            'Add Cohere reranking after initial retrieval (top 20 → rerank → top 5)',
            'Set up Langfuse tracing from day one',
            'Create eval dataset with 50+ question-answer pairs'
        ]
    },
    {
        id: 'agent-stack',
        name: 'Agent Stack',
        tagline: 'Build reliable AI agents with tools, memory, and complex workflows',
        bestFor: ['Multi-step task automation', 'Agents that use external tools', 'Complex workflows with branching logic'],
        monthlyCost: '$100-1000',
        complexity: 'High',
        components: [
            { category: 'LLM', name: 'Claude 3.5 Sonnet or GPT-4o', why: 'Best tool use, instruction following, reasoning', link: 'https://console.anthropic.com' },
            { category: 'Framework', name: 'LangGraph', why: 'Graph-based workflows, state management, cycles', link: 'https://langchain-ai.github.io/langgraph' },
            { category: 'Memory', name: 'Redis', why: 'Fast, persistent, good for conversation history', link: 'https://redis.io' },
            { category: 'Tools', name: 'Composio', why: '150+ pre-built integrations (Gmail, Slack, etc)', link: 'https://composio.dev' },
            { category: 'Code Execution', name: 'E2B', why: 'Secure sandboxed code execution', link: 'https://e2b.dev' },
            { category: 'Observability', name: 'LangSmith', why: 'Best for LangGraph debugging, trace visualization', link: 'https://smith.langchain.com' },
        ],
        tradeoffs: {
            pros: ['Handle complex multi-step tasks', 'Rich tool ecosystem', 'Stateful conversations', 'Visual debugging with LangSmith'],
            cons: ['Agents can be unpredictable', 'Higher latency (multiple LLM calls)', 'More expensive per task', 'Harder to test and evaluate']
        },
        gettingStarted: [
            'Start with a single-tool agent before adding complexity',
            'Use LangGraph\'s checkpointing for state persistence',
            'Implement human-in-the-loop for high-stakes actions',
            'Set up comprehensive logging - agents fail in surprising ways',
            'Create guardrails to prevent infinite loops and runaway costs'
        ]
    },
    {
        id: 'cost-optimized',
        name: 'Cost-Optimized Stack',
        tagline: 'Maximum capability per dollar with open models and smart routing',
        bestFor: ['High-volume applications', 'Cost-sensitive startups', 'When you need fast inference'],
        monthlyCost: '$20-200',
        complexity: 'Medium',
        components: [
            { category: 'LLM', name: 'Llama 3.1 70B via Groq', why: 'Near-GPT-4 quality, 10x cheaper, sub-100ms latency', link: 'https://console.groq.com' },
            { category: 'Fallback LLM', name: 'GPT-4o-mini via LiteLLM', why: 'Fallback for complex tasks, unified API', link: 'https://github.com/BerriAI/litellm' },
            { category: 'Embeddings', name: 'BGE-large or Nomic', why: 'Free to run, excellent quality, no API costs', link: 'https://huggingface.co/BAAI/bge-large-en-v1.5' },
            { category: 'Vector DB', name: 'pgvector', why: 'Free with existing Postgres, good enough for most', link: 'https://github.com/pgvector/pgvector' },
            { category: 'Caching', name: 'Redis semantic cache', why: 'Cache similar queries, huge cost savings', link: 'https://redis.io' },
            { category: 'Gateway', name: 'LiteLLM Proxy', why: 'Route between providers, automatic fallbacks', link: 'https://docs.litellm.ai/docs/proxy' },
        ],
        tradeoffs: {
            pros: ['5-10x cost reduction vs premium APIs', 'Groq is incredibly fast', 'No vendor lock-in', 'Semantic caching saves repeat queries'],
            cons: ['More moving parts to manage', 'Open models slightly lower quality', 'Need to handle model routing logic', 'Self-hosted embeddings need compute']
        },
        gettingStarted: [
            'Start with Groq for speed, add fallback to GPT-4o-mini for failures',
            'Use LiteLLM proxy to unify all providers under OpenAI API format',
            'Implement semantic caching with 0.95 similarity threshold',
            'Run embeddings locally or use Hugging Face Inference Endpoints',
            'Monitor cost per query and optimize routing rules'
        ]
    },
    {
        id: 'self-hosted',
        name: 'Self-Hosted Stack',
        tagline: 'Full control and privacy with self-hosted models and infrastructure',
        bestFor: ['Data privacy requirements', 'Air-gapped environments', 'Teams wanting full control'],
        monthlyCost: '$500-5000 (compute)',
        complexity: 'High',
        components: [
            { category: 'LLM', name: 'Llama 3.1 70B', why: 'Best open model, Apache-like license', link: 'https://llama.meta.com' },
            { category: 'Inference', name: 'vLLM', why: 'Highest throughput, PagedAttention, production-ready', link: 'https://vllm.ai' },
            { category: 'Embeddings', name: 'BGE-large or E5-mistral', why: 'Run locally, no data leaves your infra', link: 'https://huggingface.co/BAAI/bge-large-en-v1.5' },
            { category: 'Vector DB', name: 'Qdrant', why: 'Self-hostable, fast, good filtering', link: 'https://qdrant.tech' },
            { category: 'Compute', name: 'RunPod or Lambda Labs', why: 'On-demand GPUs, A100/H100 available', link: 'https://runpod.io' },
            { category: 'Observability', name: 'Langfuse (self-hosted)', why: 'Keep all traces in your infrastructure', link: 'https://langfuse.com/docs/deployment/self-host' },
        ],
        tradeoffs: {
            pros: ['Complete data privacy', 'No per-token costs after infra', 'Full control over models and updates', 'Can fine-tune for your domain'],
            cons: ['High upfront infrastructure cost', 'Need ML ops expertise', 'Responsible for uptime and scaling', 'Model updates require manual work']
        },
        gettingStarted: [
            'Start with RunPod for experimentation before committing to infra',
            'Use vLLM with tensor parallelism for 70B models (needs 2x A100 or 4x A10)',
            'Benchmark throughput and latency for your use case',
            'Set up model versioning and rollback procedures',
            'Implement health checks and auto-restart for inference servers'
        ]
    },
    {
        id: 'enterprise',
        name: 'Enterprise Stack',
        tagline: 'Compliance-ready stack with security, governance, and audit trails',
        bestFor: ['Regulated industries (finance, healthcare)', 'Large organizations with compliance needs', 'When you need SOC2, HIPAA, or similar'],
        monthlyCost: '$1000-10000',
        complexity: 'High',
        components: [
            { category: 'LLM', name: 'Azure OpenAI', why: 'Enterprise SLAs, data residency, compliance certs', link: 'https://azure.microsoft.com/en-us/products/ai-services/openai-service' },
            { category: 'Vector DB', name: 'Weaviate Cloud', why: 'SOC2, enterprise support, RBAC', link: 'https://weaviate.io/developers/wcs' },
            { category: 'Framework', name: 'LangChain + Semantic Kernel', why: 'LangChain for dev speed, SK for .NET shops', link: 'https://github.com/microsoft/semantic-kernel' },
            { category: 'Gateway', name: 'Portkey', why: 'Audit logs, rate limiting, access control', link: 'https://portkey.ai' },
            { category: 'Guardrails', name: 'Guardrails AI + NeMo', why: 'Input/output validation, content filtering', link: 'https://guardrailsai.com' },
            { category: 'Observability', name: 'Datadog or Splunk', why: 'Enterprise logging, existing integrations', link: 'https://datadoghq.com' },
        ],
        tradeoffs: {
            pros: ['Compliance certifications included', 'Enterprise support and SLAs', 'Audit trails for all AI interactions', 'Integrates with existing enterprise tools'],
            cons: ['Significantly higher cost', 'Vendor lock-in (especially Azure)', 'Slower to adopt new models', 'More bureaucracy in setup']
        },
        gettingStarted: [
            'Start with Azure OpenAI provisioning (can take weeks for approval)',
            'Set up Portkey for centralized API management and audit logging',
            'Implement guardrails for PII detection and content filtering',
            'Create data classification policies for what can be sent to LLMs',
            'Document AI usage policies and get legal/compliance sign-off'
        ]
    },
    {
        id: 'multimodal',
        name: 'Multimodal Stack',
        tagline: 'Process images, audio, and video alongside text',
        bestFor: ['Image understanding applications', 'Document processing with visuals', 'Audio/video analysis'],
        monthlyCost: '$100-1000',
        complexity: 'Medium',
        components: [
            { category: 'Vision LLM', name: 'GPT-4o or Claude 3.5 Sonnet', why: 'Best vision capabilities, native multimodal', link: 'https://platform.openai.com' },
            { category: 'Document Processing', name: 'Unstructured.io', why: 'Extract text from PDFs, images, tables', link: 'https://unstructured.io' },
            { category: 'Image Embeddings', name: 'CLIP or SigLIP', why: 'Embed images for similarity search', link: 'https://huggingface.co/openai/clip-vit-large-patch14' },
            { category: 'Audio', name: 'Whisper', why: 'Best speech-to-text, runs locally or via API', link: 'https://openai.com/research/whisper' },
            { category: 'Vector DB', name: 'LanceDB or Pinecone', why: 'Native multimodal support, mixed embeddings', link: 'https://lancedb.com' },
            { category: 'Framework', name: 'LlamaIndex', why: 'Best multimodal RAG support', link: 'https://llamaindex.ai' },
        ],
        tradeoffs: {
            pros: ['Unlock non-text data sources', 'Better document understanding', 'Can process charts, diagrams, screenshots', 'Audio transcription enables voice apps'],
            cons: ['Higher latency for image processing', 'More expensive per query', 'Image quality affects results significantly', 'Harder to evaluate multimodal outputs']
        },
        gettingStarted: [
            'Start with GPT-4o for vision - best quality and simplest API',
            'Use Unstructured.io to preprocess documents before LLM',
            'For image search, generate CLIP embeddings and store in vector DB',
            'Run Whisper locally for audio (whisper.cpp is fast)',
            'Test with diverse image qualities - models struggle with low-res'
        ]
    },
    {
        id: 'code-assistant',
        name: 'Code Assistant Stack',
        tagline: 'Build AI coding assistants with code understanding and execution',
        bestFor: ['Developer tools', 'Code generation applications', 'Automated coding workflows'],
        monthlyCost: '$50-500',
        complexity: 'Medium',
        components: [
            { category: 'LLM', name: 'Claude 3.5 Sonnet', why: 'Best coding model, excellent at complex code', link: 'https://console.anthropic.com' },
            { category: 'Code LLM', name: 'DeepSeek Coder or Codestral', why: 'Specialized for code, good for autocomplete', link: 'https://github.com/deepseek-ai/DeepSeek-Coder' },
            { category: 'Code Execution', name: 'E2B', why: 'Secure sandboxed execution, multiple languages', link: 'https://e2b.dev' },
            { category: 'Code Search', name: 'Sourcegraph or Greptile', why: 'Semantic code search across repos', link: 'https://sourcegraph.com' },
            { category: 'Framework', name: 'LangGraph', why: 'Multi-step code generation workflows', link: 'https://langchain-ai.github.io/langgraph' },
            { category: 'Embeddings', name: 'Voyage Code or CodeBERT', why: 'Code-specific embeddings for retrieval', link: 'https://voyageai.com' },
        ],
        tradeoffs: {
            pros: ['Claude is remarkably good at code', 'Sandboxed execution is safe', 'Can iterate on code with test feedback', 'Semantic code search improves context'],
            cons: ['Code execution adds latency and cost', 'Security requires careful sandboxing', 'Generated code needs human review', 'Context limits matter for large codebases']
        },
        gettingStarted: [
            'Start with Claude 3.5 Sonnet - it\'s the best at code by far',
            'Use E2B for safe code execution with automatic cleanup',
            'Implement a code → test → fix loop for better results',
            'Add codebase context via embeddings for repo-aware suggestions',
            'Always show generated code to user before execution'
        ]
    },
    {
        id: 'research',
        name: 'Research & Experimentation Stack',
        tagline: 'Optimize prompts and models with rigorous experimentation',
        bestFor: ['Prompt optimization research', 'Model comparison studies', 'Teams focused on AI quality improvement'],
        monthlyCost: '$50-500',
        complexity: 'Medium',
        components: [
            { category: 'Optimization', name: 'DSPy', why: 'Programmatic prompt optimization, compiles prompts', link: 'https://github.com/stanfordnlp/dspy' },
            { category: 'LLM Access', name: 'Together AI or Fireworks', why: 'Access to many open models for comparison', link: 'https://together.ai' },
            { category: 'Evaluation', name: 'Ragas + Promptfoo', why: 'Comprehensive eval metrics, CI integration', link: 'https://promptfoo.dev' },
            { category: 'Experiment Tracking', name: 'Weights & Biases', why: 'Track prompt versions, compare results', link: 'https://wandb.ai' },
            { category: 'Dataset Management', name: 'Argilla', why: 'Label data, manage eval datasets', link: 'https://argilla.io' },
            { category: 'Notebooks', name: 'Jupyter + nbdev', why: 'Reproducible experiments, literate programming', link: 'https://nbdev.fast.ai' },
        ],
        tradeoffs: {
            pros: ['Systematic improvement over ad-hoc prompting', 'Reproducible experiments', 'Can find 10-20% quality gains', 'Great for understanding model behavior'],
            cons: ['Slower iteration than just prompting', 'Requires eval dataset investment', 'DSPy has learning curve', 'May over-optimize for benchmarks']
        },
        gettingStarted: [
            'Create a golden eval dataset (50+ examples minimum)',
            'Set up Promptfoo for baseline prompt testing',
            'Use DSPy to optimize prompts programmatically',
            'Track all experiments in W&B with prompt versions',
            'Compare at least 3 models before committing to one'
        ]
    },
    {
        id: 'startup-mvp',
        name: 'Startup MVP Stack',
        tagline: 'Ship an AI product fast with minimal infrastructure',
        bestFor: ['MVPs and prototypes', 'Solo developers or small teams', 'Validating AI product ideas quickly'],
        monthlyCost: '$20-100',
        complexity: 'Low',
        components: [
            { category: 'LLM', name: 'OpenAI GPT-4o-mini', why: 'Best price/performance, reliable, fast', link: 'https://platform.openai.com' },
            { category: 'Vector DB', name: 'Supabase Vector', why: 'Postgres + pgvector, auth, realtime included', link: 'https://supabase.com/vector' },
            { category: 'Framework', name: 'Vercel AI SDK', why: 'Streaming, React hooks, edge-ready', link: 'https://sdk.vercel.ai' },
            { category: 'Hosting', name: 'Vercel', why: 'Deploy in seconds, generous free tier', link: 'https://vercel.com' },
            { category: 'Observability', name: 'Helicone', why: 'Free tier, drop-in proxy, cost tracking', link: 'https://helicone.ai' },
            { category: 'Auth', name: 'Clerk or Supabase Auth', why: 'Auth in minutes, not days', link: 'https://clerk.com' },
        ],
        tradeoffs: {
            pros: ['Ship in days, not weeks', 'Minimal ops burden', 'Generous free tiers', 'Modern DX with streaming'],
            cons: ['Will need to re-architect to scale', 'Limited customization', 'Vercel costs grow quickly', 'pgvector has scale limits']
        },
        gettingStarted: [
            'npx create-next-app with Vercel AI SDK template',
            'Set up Supabase project with pgvector extension',
            'Add Helicone proxy for observability (one line change)',
            'Deploy to Vercel, get feedback, iterate',
            'Don\'t over-engineer - validate the idea first'
        ]
    }
];

const StacksView = () => {
    const [expandedStack, setExpandedStack] = useState<string | null>(null);

    const complexityColor = (c: string) => {
        switch (c) {
            case 'Low': return 'bg-green-100 text-green-700';
            case 'Medium': return 'bg-amber-100 text-amber-700';
            case 'High': return 'bg-red-100 text-red-700';
            default: return 'bg-stone-100 text-stone-700';
        }
    };

    return (
        <div className="space-y-4">
            {recommendedStacks.map((stack) => (
                <div
                    key={stack.id}
                    className="bg-white border border-stone-200 rounded-2xl overflow-hidden hover:shadow-lg transition-shadow"
                >
                    {/* Header */}
                    <button
                        onClick={() => setExpandedStack(expandedStack === stack.id ? null : stack.id)}
                        className="w-full p-5 flex items-start gap-4 text-left"
                    >
                        <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-1 flex-wrap">
                                <h3 className="font-bold text-stone-900">{stack.name}</h3>
                                <span className={`text-xs px-2 py-0.5 rounded ${complexityColor(stack.complexity)}`}>
                                    {stack.complexity} Complexity
                                </span>
                                <span className="text-xs px-2 py-0.5 bg-brand-100 text-brand-700 rounded">
                                    {stack.monthlyCost}/mo
                                </span>
                            </div>
                            <p className="text-stone-600 text-sm">{stack.tagline}</p>
                        </div>
                        <div className="text-stone-400 shrink-0">
                            {expandedStack === stack.id ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
                        </div>
                    </button>

                    {/* Expanded content */}
                    {expandedStack === stack.id && (
                        <div className="px-5 pb-5 border-t border-stone-100">
                            {/* Best For */}
                            <div className="pt-5 mb-5">
                                <h4 className="font-semibold text-stone-900 mb-2">Best For</h4>
                                <div className="flex flex-wrap gap-2">
                                    {stack.bestFor.map((item, i) => (
                                        <span key={i} className="text-sm px-3 py-1 bg-stone-100 text-stone-700 rounded-full">
                                            {item}
                                        </span>
                                    ))}
                                </div>
                            </div>

                            {/* Components */}
                            <div className="mb-5">
                                <h4 className="font-semibold text-stone-900 mb-3">Stack Components</h4>
                                <div className="grid gap-2">
                                    {stack.components.map((comp, i) => (
                                        <div key={i} className="bg-stone-50 rounded-xl p-3 flex items-start gap-3">
                                            <span className="text-xs font-medium px-2 py-1 bg-stone-200 text-stone-600 rounded shrink-0">
                                                {comp.category}
                                            </span>
                                            <div className="flex-1 min-w-0">
                                                <a 
                                                    href={comp.link}
                                                    target="_blank"
                                                    rel="noopener noreferrer"
                                                    className="font-medium text-brand-600 hover:text-brand-700 hover:underline"
                                                >
                                                    {comp.name}
                                                </a>
                                                <p className="text-sm text-stone-600">{comp.why}</p>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>

                            {/* Tradeoffs */}
                            <div className="grid md:grid-cols-2 gap-4 mb-5">
                                <div className="bg-green-50 rounded-xl p-4 border border-green-100">
                                    <h4 className="font-semibold text-green-900 mb-2">✓ Pros</h4>
                                    <ul className="space-y-1">
                                        {stack.tradeoffs.pros.map((pro, i) => (
                                            <li key={i} className="text-sm text-green-800">{pro}</li>
                                        ))}
                                    </ul>
                                </div>
                                <div className="bg-red-50 rounded-xl p-4 border border-red-100">
                                    <h4 className="font-semibold text-red-900 mb-2">✗ Cons</h4>
                                    <ul className="space-y-1">
                                        {stack.tradeoffs.cons.map((con, i) => (
                                            <li key={i} className="text-sm text-red-800">{con}</li>
                                        ))}
                                    </ul>
                                </div>
                            </div>

                            {/* Getting Started */}
                            <div className="bg-brand-50 rounded-xl p-4 border border-brand-100">
                                <h4 className="font-semibold text-brand-900 mb-3">🚀 Getting Started</h4>
                                <ol className="space-y-2">
                                    {stack.gettingStarted.map((step, i) => (
                                        <li key={i} className="text-sm text-brand-800 flex gap-2">
                                            <span className="font-mono text-brand-600">{i + 1}.</span>
                                            <span className="font-mono">{step}</span>
                                        </li>
                                    ))}
                                </ol>
                            </div>
                        </div>
                    )}
                </div>
            ))}
        </div>
    );
};

const AIPMFrameworksView = () => {
    const [expandedFramework, setExpandedFramework] = useState<string | null>(null);
    const [filterCategory, setFilterCategory] = useState<string>('all');

    const categories = [
        { id: 'all', label: 'All Frameworks' },
        { id: 'lifecycle', label: 'Lifecycle' },
        { id: 'prioritization', label: 'Prioritization' },
        { id: 'discovery', label: 'Discovery' },
        { id: 'operations', label: 'Operations' },
        { id: 'governance', label: 'Governance' },
        { id: 'metrics', label: 'Metrics' },
    ];

    const filteredFrameworks = filterCategory === 'all' 
        ? aipmFrameworks 
        : aipmFrameworks.filter(f => f.category === filterCategory);

    return (
        <div className="space-y-6">
            {/* Category filter */}
            <div className="flex flex-wrap gap-2">
                {categories.map(cat => (
                    <button
                        key={cat.id}
                        onClick={() => setFilterCategory(cat.id)}
                        className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                            filterCategory === cat.id
                                ? 'bg-brand-600 text-white'
                                : 'bg-stone-100 text-stone-600 hover:bg-stone-200'
                        }`}
                    >
                        {cat.label}
                    </button>
                ))}
            </div>

            {/* Frameworks */}
            <div className="space-y-4">
                {filteredFrameworks.map((framework) => (
                    <div
                        key={framework.id}
                        className="bg-white border border-stone-200 rounded-2xl overflow-hidden hover:shadow-lg transition-shadow"
                    >
                        {/* Header - always visible */}
                        <button
                            onClick={() => setExpandedFramework(expandedFramework === framework.id ? null : framework.id)}
                            className="w-full p-5 flex items-start gap-4 text-left"
                        >
                            <div className="w-12 h-12 rounded-xl bg-brand-100 text-brand-600 flex items-center justify-center shrink-0">
                                {framework.icon}
                            </div>
                            <div className="flex-1 min-w-0">
                                <div className="flex items-center gap-2 mb-1">
                                    <h3 className="font-bold text-stone-900">{framework.name}</h3>
                                    <span className="text-xs px-2 py-0.5 bg-stone-100 text-stone-500 rounded capitalize">
                                        {framework.category}
                                    </span>
                                </div>
                                <p className="text-stone-600 text-sm">{framework.tagline}</p>
                            </div>
                            <div className="text-stone-400 shrink-0">
                                {expandedFramework === framework.id ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
                            </div>
                        </button>

                        {/* Expanded content */}
                        {expandedFramework === framework.id && (
                            <div className="px-5 pb-5 border-t border-stone-100">
                                <div className="grid md:grid-cols-2 gap-6 pt-5">
                                    {/* When to Use */}
                                    <div>
                                        <h4 className="font-semibold text-stone-900 mb-3 flex items-center gap-2">
                                            <Target size={16} className="text-brand-500" />
                                            When to Use
                                        </h4>
                                        <ul className="space-y-2">
                                            {framework.whenToUse.map((item, i) => (
                                                <li key={i} className="text-sm text-stone-600 flex items-start gap-2">
                                                    <span className="text-brand-500 mt-1">•</span>
                                                    {item}
                                                </li>
                                            ))}
                                        </ul>
                                    </div>

                                    {/* AI-Specific Tips */}
                                    <div>
                                        <h4 className="font-semibold text-stone-900 mb-3 flex items-center gap-2">
                                            <Lightbulb size={16} className="text-amber-500" />
                                            AI-Specific Tips
                                        </h4>
                                        <ul className="space-y-2">
                                            {framework.aiSpecificTips.map((tip, i) => (
                                                <li key={i} className="text-sm text-stone-600 flex items-start gap-2">
                                                    <span className="text-amber-500 mt-1">→</span>
                                                    {tip}
                                                </li>
                                            ))}
                                        </ul>
                                    </div>
                                </div>

                                {/* Steps */}
                                <div className="mt-6">
                                    <h4 className="font-semibold text-stone-900 mb-4 flex items-center gap-2">
                                        <Workflow size={16} className="text-brand-500" />
                                        Framework Steps
                                    </h4>
                                    <div className="grid gap-3">
                                        {framework.steps.map((step, i) => (
                                            <div key={i} className="bg-stone-50 rounded-xl p-4">
                                                <h5 className="font-medium text-stone-900 mb-1">{step.title}</h5>
                                                <p className="text-sm text-stone-600">{step.description}</p>
                                            </div>
                                        ))}
                                    </div>
                                </div>

                                {/* Example */}
                                {framework.example && (
                                    <div className="mt-6 bg-brand-50 rounded-xl p-4 border border-brand-100">
                                        <h4 className="font-semibold text-brand-900 mb-2 flex items-center gap-2">
                                            <CheckCircle2 size={16} className="text-brand-600" />
                                            Example
                                        </h4>
                                        <p className="text-sm text-brand-800 mb-2">
                                            <strong>Scenario:</strong> {framework.example.scenario}
                                        </p>
                                        <p className="text-sm text-brand-700">
                                            <strong>Application:</strong> {framework.example.application}
                                        </p>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                ))}
            </div>
        </div>
    );
};

export const ResourcesView = () => {
    const [activeCategory, setActiveCategory] = useState<Category>('providers');
    const [searchQuery, setSearchQuery] = useState('');

    const categories: { id: Category; label: string }[] = [
        { id: 'providers', label: 'LLM Providers' },
        { id: 'frameworks', label: 'Frameworks' },
        { id: 'vectordb', label: 'Vector DBs' },
        { id: 'embeddings', label: 'Embeddings' },
        { id: 'eval', label: 'Eval & Observability' },
        { id: 'agents', label: 'Agents' },
        { id: 'inference', label: 'Inference' },
        { id: 'finetuning', label: 'Fine-Tuning' },
        { id: 'models', label: 'Open Models' },
        { id: 'aipm', label: 'AI PM Frameworks' },
        { id: 'learning', label: 'Learning' },
        { id: 'stacks', label: 'Stacks' },
    ];

    const currentResources = resources[activeCategory];
    
    const filteredItems = searchQuery 
        ? currentResources.items.filter(item => 
            item.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
            item.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
            item.tags?.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()))
          )
        : currentResources.items;

    return (
        <div className="min-h-screen bg-stone-50">
            {/* Header */}
            <div className="bg-stone-900 text-white py-12 px-6">
                <div className="max-w-5xl mx-auto">
                    <h1 className="text-3xl font-bold mb-2">AI Engineering Resources</h1>
                    <p className="text-stone-400">Curated tools, frameworks, and platforms for building AI applications</p>
                    
                    {/* Search */}
                    <div className="mt-6 relative">
                        <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-stone-500" />
                        <input
                            type="text"
                            placeholder="Search resources..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            className="w-full max-w-md pl-12 pr-4 py-3 bg-stone-800 border border-stone-700 rounded-xl text-white placeholder-stone-500 focus:outline-none focus:border-brand-500"
                        />
                    </div>
                </div>
            </div>

            <div className="max-w-5xl mx-auto px-6 py-8">
                {/* Category tabs */}
                <div className="flex flex-wrap gap-2 mb-8">
                    {categories.map(cat => (
                        <button
                            key={cat.id}
                            onClick={() => setActiveCategory(cat.id)}
                            className={`px-4 py-2 rounded-xl font-medium transition-all ${
                                activeCategory === cat.id 
                                    ? 'bg-stone-900 text-white' 
                                    : 'bg-white text-stone-600 border border-stone-200 hover:bg-stone-100'
                            }`}
                        >
                            {cat.label}
                        </button>
                    ))}
                </div>

                {/* Category header */}
                <div className="flex items-center gap-3 mb-6">
                    <div className="w-10 h-10 rounded-xl bg-brand-100 text-brand-600 flex items-center justify-center">
                        {currentResources.icon}
                    </div>
                    <h2 className="text-2xl font-bold text-stone-900">{currentResources.title}</h2>
                </div>

                {/* Resource cards or custom views */}
                {activeCategory === 'aipm' ? (
                    <AIPMFrameworksView />
                ) : activeCategory === 'stacks' ? (
                    <StacksView />
                ) : (
                    <>
                <div className="grid md:grid-cols-2 gap-4">
                    {filteredItems.map((item) => (
                        <a
                            key={item.name}
                            href={item.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="bg-white border border-stone-200 rounded-2xl p-5 hover:shadow-lg hover:border-brand-200 transition-all group"
                        >
                            <div className="flex items-start justify-between mb-2">
                                <h3 className="font-bold text-stone-900 group-hover:text-brand-600 transition-colors">
                                    {item.name}
                                </h3>
                                <ExternalLink className="w-4 h-4 text-stone-400 group-hover:text-brand-500 transition-colors" />
                            </div>
                            <p className="text-stone-600 text-sm mb-3">{item.description}</p>
                            {item.tags && (
                                <div className="flex flex-wrap gap-1">
                                    {item.tags.map(tag => (
                                        <span key={tag} className="text-xs px-2 py-1 bg-stone-100 text-stone-600 rounded">
                                            {tag}
                                        </span>
                                    ))}
                                </div>
                            )}
                        </a>
                    ))}
                </div>

                {filteredItems.length === 0 && (
                    <div className="text-center py-12 text-stone-500">
                        No resources found matching "{searchQuery}"
                    </div>
                        )}
                    </>
                )}
            </div>
        </div>
    );
};

