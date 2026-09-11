import React, { useState } from 'react';
import { Code, FileText, GitCompare, GitBranch, Calculator, Box, Copy, Check, ChevronDown, ChevronUp, ExternalLink, Download } from 'lucide-react';
import { LookUpPrompt } from './LookUpPrompt';

type TabId = 'code' | 'templates' | 'comparisons' | 'decisions' | 'calculator' | 'architecture';

// ============ CODE EXAMPLES ============
interface CodeExample {
    id: string;
    title: string;
    description: string;
    language: string;
    code: string;
    category: string;
}

const codeExamples: CodeExample[] = [
    {
        id: 'basic-chat',
        title: 'Basic Chat Completion',
        description: 'Simple chat completion with OpenAI',
        category: 'Getting Started',
        language: 'python',
        code: `import os
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model=os.environ["MODEL_ID"],
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)`
    },
    {
        id: 'streaming',
        title: 'Streaming Response',
        description: 'Stream tokens as they are generated',
        category: 'Getting Started',
        language: 'python',
        code: `import os
from openai import OpenAI

client = OpenAI()

stream = client.chat.completions.create(
    model=os.environ["MODEL_ID"],
    messages=[{"role": "user", "content": "Write a haiku"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)`
    },
    {
        id: 'function-calling',
        title: 'Function Calling',
        description: 'Let the model call your functions',
        category: 'Tools',
        language: 'python',
        code: `from openai import OpenAI
import json

client = OpenAI()

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
    }
}]

response = client.chat.completions.create(
    model=os.environ["MODEL_ID"],
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools
)

tool_call = response.choices[0].message.tool_calls[0]
args = json.loads(tool_call.function.arguments)
print(f"Function: {tool_call.function.name}")
print(f"Arguments: {args}")`
    },
    {
        id: 'structured-output',
        title: 'Structured Output with Instructor',
        description: 'Get typed responses using Pydantic',
        category: 'Tools',
        language: 'python',
        code: `import instructor
from openai import OpenAI
from pydantic import BaseModel

client = instructor.from_openai(OpenAI())

class Person(BaseModel):
    name: str
    age: int
    occupation: str

person = client.chat.completions.create(
    model=os.environ["MODEL_ID"],
    messages=[{
        "role": "user", 
        "content": "John Smith is a 32 year old software engineer."
    }],
    response_model=Person
)

print(f"Name: {person.name}")
print(f"Age: {person.age}")
print(f"Occupation: {person.occupation}")`
    },
    {
        id: 'simple-rag',
        title: 'Simple RAG with Chroma',
        description: 'Basic retrieval-augmented generation',
        category: 'RAG',
        language: 'python',
        code: `from openai import OpenAI
import chromadb

# Initialize
client = OpenAI()
chroma = chromadb.Client()
collection = chroma.create_collection("docs")

# Add documents
docs = [
    "Python is a programming language.",
    "JavaScript runs in browsers.",
    "Rust is known for memory safety."
]
collection.add(
    documents=docs,
    ids=[f"doc_{i}" for i in range(len(docs))]
)

# Query
query = "What language is good for web browsers?"
results = collection.query(query_texts=[query], n_results=2)

# Generate with context
context = "\\n".join(results['documents'][0])
response = client.chat.completions.create(
    model=os.environ["MODEL_ID"],
    messages=[
        {"role": "system", "content": f"Answer based on: {context}"},
        {"role": "user", "content": query}
    ]
)
print(response.choices[0].message.content)`
    },
    {
        id: 'embeddings',
        title: 'Generate Embeddings',
        description: 'Create vector embeddings for text',
        category: 'RAG',
        language: 'python',
        code: `from openai import OpenAI
import numpy as np

client = OpenAI()

def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# Generate embeddings
texts = ["Hello world", "Goodbye world", "Hello there"]
embeddings = [get_embedding(t) for t in texts]

# Calculate similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(f"'Hello world' vs 'Goodbye world': {cosine_similarity(embeddings[0], embeddings[1]):.3f}")
print(f"'Hello world' vs 'Hello there': {cosine_similarity(embeddings[0], embeddings[2]):.3f}")`
    },
    {
        id: 'react-agent',
        title: 'Simple ReAct Agent',
        description: 'Agent that reasons and acts',
        category: 'Agents',
        language: 'python',
        code: `from openai import OpenAI
import json

client = OpenAI()

tools = [
    {"type": "function", "function": {
        "name": "search", 
        "description": "Search the web",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
    }},
    {"type": "function", "function": {
        "name": "calculate",
        "description": "Do math",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}
    }}
]

def run_agent(query: str, max_steps: int = 5):
    messages = [{"role": "user", "content": query}]
    
    for step in range(max_steps):
        response = client.chat.completions.create(
            model=os.environ["MODEL_ID"],
            messages=messages,
            tools=tools
        )
        
        msg = response.choices[0].message
        messages.append(msg)
        
        if not msg.tool_calls:
            return msg.content  # Final answer
        
        # Execute tools
        for tc in msg.tool_calls:
            result = f"[Mock result for {tc.function.name}]"
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result
            })
    
    return "Max steps reached"

print(run_agent("What is 25 * 4?"))`
    },
    {
        id: 'prompt-caching',
        title: 'Anthropic Prompt Caching',
        description: 'Cache system prompts for cost savings',
        category: 'Optimization',
        language: 'python',
        code: `from anthropic import Anthropic

client = Anthropic()

# Long system prompt that will be cached
system_prompt = """You are an expert legal assistant...
[Imagine 5000 tokens of detailed instructions here]
"""

# First call - creates cache
response = client.messages.create(
    model=os.environ["MODEL_ID"],
    max_tokens=1024,
    system=[{
        "type": "text",
        "text": system_prompt,
        "cache_control": {"type": "ephemeral"}  # Enable caching
    }],
    messages=[{"role": "user", "content": "What is a tort?"}]
)

# Subsequent calls get 90% discount on cached tokens!
response2 = client.messages.create(
    model=os.environ["MODEL_ID"],
    max_tokens=1024,
    system=[{
        "type": "text",
        "text": system_prompt,  # Same prompt = cache hit
        "cache_control": {"type": "ephemeral"}
    }],
    messages=[{"role": "user", "content": "Explain negligence"}]
)`
    },
    {
        id: 'local-ollama',
        title: 'Local LLM with Ollama',
        description: 'Run models locally with OpenAI-compatible API',
        category: 'Local AI',
        language: 'python',
        code: `import os
from openai import OpenAI

# Point to local Ollama server
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="not-needed"  # Ollama doesn't require auth
)

response = client.chat.completions.create(
    model="llama3.1",  # or any model you've pulled
    messages=[
        {"role": "user", "content": "Explain quantum computing simply"}
    ]
)

print(response.choices[0].message.content)

# To use: 
# 1. Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh
# 2. Pull a model: ollama pull llama3.1
# 3. Run this script`
    },
    {
        id: 'vision',
        title: 'Vision - Analyze Images',
        description: 'Send images to multimodal models',
        category: 'Multimodal',
        language: 'python',
        code: `from openai import OpenAI
import base64

client = OpenAI()

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

response = client.chat.completions.create(
    model=os.environ["MODEL_ID"],
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{encode_image('photo.jpg')}",
                "detail": "high"  # high, low, or auto
            }}
        ]
    }]
)

print(response.choices[0].message.content)`
    }
];

// ============ TEMPLATES ============
interface Template {
    id: string;
    title: string;
    description: string;
    category: string;
    template: string;
    variables: string[];
}

const templates: Template[] = [
    {
        id: 'system-assistant',
        title: 'General Assistant',
        description: 'Balanced system prompt for general-purpose assistants',
        category: 'System Prompts',
        variables: ['ASSISTANT_NAME', 'COMPANY_NAME', 'DOMAIN'],
        template: `You are {{ASSISTANT_NAME}}, a helpful AI assistant for {{COMPANY_NAME}}.

## Your Role
You help users with {{DOMAIN}}-related questions and tasks.

## Guidelines
- Be concise but thorough
- Ask clarifying questions when needed
- Admit when you don't know something
- Never make up information

## Tone
- Professional but friendly
- Patient and understanding
- Clear and jargon-free unless the user is technical

## Constraints
- Do not share internal company information
- Do not make promises about timelines or guarantees
- Escalate to human support for sensitive issues`
    },
    {
        id: 'system-rag',
        title: 'RAG System Prompt',
        description: 'System prompt for retrieval-augmented generation',
        category: 'System Prompts',
        variables: ['CONTEXT_TYPE'],
        template: `You are a helpful assistant that answers questions based on the provided context.

## Instructions
1. Answer ONLY based on the provided {{CONTEXT_TYPE}}
2. If the context doesn't contain the answer, say "I don't have information about that in my knowledge base"
3. Quote relevant passages when helpful
4. Be concise but complete

## Context Format
The context will be provided in <context> tags before the user's question.

## Response Format
- Start with a direct answer
- Provide supporting details from the context
- Cite sources when available`
    },
    {
        id: 'system-code',
        title: 'Code Assistant',
        description: 'System prompt for coding assistants',
        category: 'System Prompts',
        variables: ['LANGUAGES', 'FRAMEWORKS'],
        template: `You are an expert programming assistant specializing in {{LANGUAGES}} and {{FRAMEWORKS}}.

## Guidelines
- Write clean, readable, well-documented code
- Follow best practices and design patterns
- Consider edge cases and error handling
- Explain your reasoning when helpful

## Code Style
- Use meaningful variable/function names
- Add comments for complex logic
- Keep functions small and focused
- Follow the project's existing style when visible

## When Reviewing Code
- Point out bugs and security issues
- Suggest performance improvements
- Recommend better patterns when applicable
- Be constructive, not critical`
    },
    {
        id: 'extraction',
        title: 'Data Extraction',
        description: 'Extract structured data from unstructured text',
        category: 'Task Prompts',
        variables: ['ENTITY_TYPE', 'FIELDS'],
        template: `Extract {{ENTITY_TYPE}} information from the following text.

Return a JSON object with these fields:
{{FIELDS}}

Rules:
- Use null for fields that cannot be determined
- Use exact values from the text when possible
- Normalize formats (dates as YYYY-MM-DD, phones as +1-XXX-XXX-XXXX)
- If multiple entities exist, return an array

Text to extract from:
"""
{input_text}
"""

JSON output:`
    },
    {
        id: 'summarization',
        title: 'Summarization',
        description: 'Summarize content at different lengths',
        category: 'Task Prompts',
        variables: ['LENGTH', 'FOCUS'],
        template: `Summarize the following content in {{LENGTH}}.

Focus on: {{FOCUS}}

Guidelines:
- Capture the main points and key takeaways
- Maintain the original meaning and tone
- Use clear, concise language
- Preserve important names, numbers, and dates

Content to summarize:
"""
{input_text}
"""

Summary:`
    },
    {
        id: 'classification',
        title: 'Classification',
        description: 'Classify text into categories',
        category: 'Task Prompts',
        variables: ['CATEGORIES', 'CONTEXT'],
        template: `Classify the following text into one of these categories:
{{CATEGORIES}}

Context: {{CONTEXT}}

Return your response as JSON:
{
  "category": "selected_category",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}

Text to classify:
"""
{input_text}
"""

Classification:`
    },
    {
        id: 'chain-of-thought',
        title: 'Chain of Thought',
        description: 'Encourage step-by-step reasoning',
        category: 'Reasoning',
        variables: ['PROBLEM_TYPE'],
        template: `Solve this {{PROBLEM_TYPE}} problem step by step.

Before giving your final answer:
1. Identify what information is given
2. Determine what needs to be found
3. Break down the problem into smaller steps
4. Work through each step carefully
5. Verify your answer makes sense

Problem:
{input_text}

Let's solve this step by step:`
    },
    {
        id: 'few-shot',
        title: 'Few-Shot Template',
        description: 'Template with examples for consistent output',
        category: 'Reasoning',
        variables: ['TASK_DESCRIPTION'],
        template: `{{TASK_DESCRIPTION}}

Here are some examples:

Input: "The product arrived broken and customer service was unhelpful"
Output: {"sentiment": "negative", "topics": ["product_quality", "customer_service"]}

Input: "Fast shipping, exactly what I ordered!"
Output: {"sentiment": "positive", "topics": ["shipping", "accuracy"]}

Input: "It's okay, nothing special but does the job"
Output: {"sentiment": "neutral", "topics": ["product_quality"]}

Now analyze this:
Input: "{input_text}"
Output:`
    }
];

// ============ COMPARISONS ============
interface ComparisonTable {
    id: string;
    title: string;
    description: string;
    headers: string[];
    rows: string[][];
}

const comparisons: ComparisonTable[] = [
    {
        id: 'llm-providers',
        title: 'What to compare across providers',
        description: 'Axes stay stable. SKUs and list prices do not — look them up.',
        headers: ['Provider family', 'Ask about', 'Typical strength', 'Typical watch-out', 'Where to look'],
        rows: [
            ['OpenAI', 'Cheap/fast vs flagship vs reasoning IDs', 'Tools, ecosystem, generalists', 'Price and ID churn', 'platform.openai.com'],
            ['Anthropic', 'Budget vs mid vs flagship + cache', 'Long context, coding, safety posture', 'Availability / rate limits', 'docs.anthropic.com'],
            ['Google', 'Flash vs Pro + context class', 'Long context, multimodal', 'Quality varies by task', 'ai.google.dev'],
            ['Open-weight hosts', 'Current Llama / Qwen / Mixtral SKU', 'Cost, residency, speed', 'You own evals and ops', 'Groq, Together, Fireworks, Bedrock'],
            ['Enterprise wrappers', 'Region, VPC, logging, DPA', 'Compliance path', 'Slower to get new models', 'Azure / Bedrock / Vertex'],
        ]
    },
    {
        id: 'vector-dbs',
        title: 'Vector Database Comparison',
        description: 'Compare vector databases for RAG',
        headers: ['Database', 'Type', 'Best For', 'Hybrid Search', 'Pricing'],
        rows: [
            ['Pinecone', 'Managed', 'Production, serverless', '✅', 'Pay per use'],
            ['Weaviate', 'Open/Managed', 'GraphQL, modules', '✅', 'Free / Managed'],
            ['Chroma', 'Open source', 'Local dev, prototypes', '❌', 'Free'],
            ['Qdrant', 'Open/Managed', 'Performance, filtering', '✅', 'Free / Managed'],
            ['pgvector', 'Extension', 'Existing Postgres', '✅', 'Free'],
        ]
    },
    {
        id: 'orchestration',
        title: 'Orchestration Framework Comparison',
        description: 'Compare LLM orchestration frameworks',
        headers: ['Framework', 'Language', 'Best For', 'Learning Curve', 'Community'],
        rows: [
            ['LangChain', 'Python/TS', 'General purpose, agents', 'Medium', 'Largest'],
            ['LlamaIndex', 'Python/TS', 'RAG, data pipelines', 'Medium', 'Large'],
            ['Semantic Kernel', 'C#/Python', 'Enterprise, .NET', 'Medium', 'Growing'],
            ['Haystack', 'Python', 'Production RAG', 'Low', 'Medium'],
            ['DSPy', 'Python', 'Prompt optimization', 'High', 'Growing'],
        ]
    },
    {
        id: 'embedding-models',
        title: 'Embedding checklist',
        description: 'Pick a family, then look up the current ID and dims — do not hardcode last year\'s name.',
        headers: ['Family', 'When to reach for it', 'What to look up', 'Watch-out'],
        rows: [
            ['Hosted general', 'You want one API and decent English', 'Current ID, dims, $/M', 'Vendor lock-in on dim size'],
            ['Hosted multilingual / domain', 'Legal, code, or non-English', 'Domain SKUs + rerank pair', 'Dims change across versions'],
            ['Open / local', 'Privacy or zero API spend', 'Current MTEB-ish rank + VRAM', 'You own serving quality'],
        ]
    },
    {
        id: 'open-models',
        title: 'Open-weight families',
        description: 'Families persist. Version numbers and "best open model" claims do not.',
        headers: ['Family', 'Who', 'Why people use it', 'Look up'],
        rows: [
            ['Llama', 'Meta', 'Default open generalist', 'Current sizes + license'],
            ['Qwen', 'Alibaba', 'Multilingual, code, math', 'Current sizes + license'],
            ['Mistral / Mixtral', 'Mistral', 'Efficient MoE, EU option', 'Current MoE vs dense'],
            ['Gemma', 'Google', 'Small / efficient instruct', 'Current sizes'],
            ['DeepSeek', 'DeepSeek', 'Reasoning and code', 'Current distill vs full'],
        ]
    }
];

// ============ DECISION TREES ============
interface DecisionNode {
    question: string;
    options: { label: string; next: string | null; recommendation?: string }[];
}

interface DecisionTree {
    id: string;
    title: string;
    description: string;
    nodes: Record<string, DecisionNode>;
}

const decisionTrees: DecisionTree[] = [
    {
        id: 'model-selection',
        title: 'Which LLM Should I Use?',
        description: 'Find the right model for your use case',
        nodes: {
            start: {
                question: 'What\'s your primary constraint?',
                options: [
                    { label: 'Best quality, cost is secondary', next: 'quality' },
                    { label: 'Cost-sensitive, need good quality', next: 'cost' },
                    { label: 'Data privacy / on-premise required', next: 'privacy' },
                    { label: 'Fastest possible response', next: 'speed' }
                ]
            },
            quality: {
                question: 'What\'s the main task?',
                options: [
                    { label: 'Code generation', next: null, recommendation: 'Start with a mid-tier workhorse. Escalate to flagship only when your coding evals fail. Look up the current "best at code" ID — it rotates.' },
                    { label: 'Long document processing', next: null, recommendation: 'Prefer the current longest-context API, or chunk + RAG if the corpus is large or changing. Confirm the live window on the provider docs.' },
                    { label: 'Complex reasoning', next: null, recommendation: 'Use a reasoning / slow-think tier, or a mid-tier plus a verifier loop. Measure both on your task — thinking tokens are expensive.' },
                    { label: 'General purpose', next: null, recommendation: 'Mid-tier from two providers, behind one client. Pick with YOUR eval set, not a leaderboard screenshot.' }
                ]
            },
            cost: {
                question: 'How much volume?',
                options: [
                    { label: 'Low volume (<1M tokens/day)', next: null, recommendation: 'Cheap/fast hosted model. Look up current $/M. Abstraction (LiteLLM or your own wrapper) still pays off later.' },
                    { label: 'High volume (>10M tokens/day)', next: null, recommendation: 'Cheap/fast + cache + routing. Consider a fast host for current open weights. Run the cost formula with live prices.' },
                    { label: 'Variable/unpredictable', next: null, recommendation: 'Cheap/fast default with a flagship fallback through a unified API. Set per-user and per-feature budgets.' }
                ]
            },
            privacy: {
                question: 'What infrastructure do you have?',
                options: [
                    { label: 'Have datacenter GPUs', next: null, recommendation: 'Serve the current capable open-weight family with vLLM (or equivalent). Benchmark on your evals before committing.' },
                    { label: 'Consumer GPUs', next: null, recommendation: 'A small/medium open-weight instruct model via Ollama or llama.cpp. Look up what fits your VRAM today.' },
                    { label: 'CPU only', next: null, recommendation: 'Quantized small open-weight model via llama.cpp. Expect slower tokens; keep prompts short.' },
                    { label: 'No infrastructure', next: null, recommendation: 'Enterprise-wrapped API (Azure / Bedrock / Vertex) in your tenant. Confirm region, DPA, and logging.' }
                ]
            },
            speed: {
                question: 'What latency is acceptable?',
                options: [
                    { label: 'Sub-100ms TTFT', next: null, recommendation: 'A specialized fast-inference host for open weights, or the cheapest/fastest hosted tier. Confirm live TTFT on your prompt, not a blog number.' },
                    { label: 'Sub-500ms TTFT', next: null, recommendation: 'Cheap/fast hosted models from major providers. Avoid reasoning tiers.' },
                    { label: 'Sub-1s is fine', next: null, recommendation: 'Optimize for quality and cost first. Any major provider works if your evals pass.' }
                ]
            }
        }
    },
    {
        id: 'rag-vs-finetune',
        title: 'RAG vs Fine-tuning vs Prompting',
        description: 'Choose the right approach for your knowledge needs',
        nodes: {
            start: {
                question: 'What kind of knowledge do you need to add?',
                options: [
                    { label: 'Company/domain documents', next: 'documents' },
                    { label: 'Specific output style/format', next: 'style' },
                    { label: 'Task-specific behavior', next: 'behavior' },
                    { label: 'Real-time/frequently changing data', next: null, recommendation: 'RAG - Only option for dynamic data' }
                ]
            },
            documents: {
                question: 'How much data?',
                options: [
                    { label: '<50 pages', next: null, recommendation: 'Prompting - Just include in context window' },
                    { label: '50-10,000 pages', next: null, recommendation: 'RAG - Retrieve relevant chunks as needed' },
                    { label: '>10,000 pages', next: null, recommendation: 'RAG with hybrid search and reranking' }
                ]
            },
            style: {
                question: 'How specific is the style?',
                options: [
                    { label: 'General tone (formal, casual)', next: null, recommendation: 'Prompting - Describe in system prompt with examples' },
                    { label: 'Very specific voice/brand', next: null, recommendation: 'Fine-tuning - Train on examples of desired output' },
                    { label: 'Consistent format/structure', next: null, recommendation: 'Structured outputs (Instructor) - Define schema' }
                ]
            },
            behavior: {
                question: 'How many examples do you have?',
                options: [
                    { label: '<20 examples', next: null, recommendation: 'Few-shot prompting - Include examples in prompt' },
                    { label: '20-1000 examples', next: null, recommendation: 'Fine-tuning - Enough data to learn patterns' },
                    { label: '>1000 examples', next: null, recommendation: 'Fine-tuning with eval set - Can measure improvement' }
                ]
            }
        }
    },
    {
        id: 'vector-db-selection',
        title: 'Which Vector Database?',
        description: 'Choose the right vector database for your needs',
        nodes: {
            start: {
                question: 'What\'s your deployment preference?',
                options: [
                    { label: 'Fully managed (no ops)', next: 'managed' },
                    { label: 'Self-hosted (control)', next: 'selfhosted' },
                    { label: 'Local development only', next: null, recommendation: 'Chroma - Simplest setup, runs in-memory or SQLite' }
                ]
            },
            managed: {
                question: 'What scale do you need?',
                options: [
                    { label: '<1M vectors', next: null, recommendation: 'Pinecone Serverless - Pay per use, no minimum' },
                    { label: '1M-100M vectors', next: null, recommendation: 'Pinecone or Weaviate Cloud - Both excellent' },
                    { label: '>100M vectors', next: null, recommendation: 'Pinecone or Qdrant Cloud - Enterprise tier' }
                ]
            },
            selfhosted: {
                question: 'Do you already use Postgres?',
                options: [
                    { label: 'Yes, heavily', next: null, recommendation: 'pgvector - Add to existing Postgres, familiar tooling' },
                    { label: 'No / want dedicated', next: 'dedicated' }
                ]
            },
            dedicated: {
                question: 'What\'s more important?',
                options: [
                    { label: 'Raw performance', next: null, recommendation: 'Qdrant - Rust-based, very fast, great filtering' },
                    { label: 'Ecosystem/modules', next: null, recommendation: 'Weaviate - GraphQL, built-in vectorizers' },
                    { label: 'Scale to billions', next: null, recommendation: 'Milvus - Distributed, GPU acceleration' }
                ]
            }
        }
    }
];

// ============ COST CALCULATOR ============
interface CostCalculatorState {
    inputPricePerMillion: number;
    outputPricePerMillion: number;
    inputTokensPerRequest: number;
    outputTokensPerRequest: number;
    requestsPerDay: number;
}

// ============ ARCHITECTURE DIAGRAMS ============
interface ArchitectureDiagram {
    id: string;
    title: string;
    description: string;
    diagram: string;
}

const architectureDiagrams: ArchitectureDiagram[] = [
    {
        id: 'simple-chat',
        title: 'Simple Chatbot',
        description: 'Basic chat application architecture',
        diagram: `
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Frontend  │────▶│   Backend   │────▶│   LLM API   │
│  (React)    │◀────│  (FastAPI)  │◀────│  (OpenAI)   │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   Redis     │
                    │  (Sessions) │
                    └─────────────┘
`
    },
    {
        id: 'rag-pipeline',
        title: 'RAG Pipeline',
        description: 'Retrieval-Augmented Generation architecture',
        diagram: `
┌──────────────────────────────────────────────────────────────┐
│                      INGESTION PIPELINE                       │
├──────────────────────────────────────────────────────────────┤
│  Documents ──▶ Chunker ──▶ Embeddings ──▶ Vector DB          │
│     │            │            │              │                │
│   PDFs        512 tok      OpenAI        Pinecone            │
│   Web         overlap      embed-3                            │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                       QUERY PIPELINE                          │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  User Query                                                   │
│      │                                                        │
│      ▼                                                        │
│  ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐        │
│  │ Embed  │───▶│ Search │───▶│Rerank  │───▶│  LLM   │        │
│  │ Query  │    │Top 20  │    │Top 5   │    │Generate│        │
│  └────────┘    └────────┘    └────────┘    └────────┘        │
│                                                  │            │
│                                                  ▼            │
│                                             Response          │
└──────────────────────────────────────────────────────────────┘
`
    },
    {
        id: 'agent-loop',
        title: 'Agent Architecture',
        description: 'ReAct agent with tools',
        diagram: `
                    ┌─────────────────┐
                    │   User Query    │
                    └────────┬────────┘
                             │
                             ▼
              ┌──────────────────────────┐
              │         LLM              │
              │   (Reasoning Engine)     │◀──────────────┐
              └──────────────┬───────────┘               │
                             │                           │
              ┌──────────────┴───────────┐               │
              │      Decision Point      │               │
              └──────────────┬───────────┘               │
                             │                           │
           ┌─────────────────┼─────────────────┐         │
           │                 │                 │         │
           ▼                 ▼                 ▼         │
    ┌────────────┐   ┌────────────┐   ┌────────────┐    │
    │   Tool 1   │   │   Tool 2   │   │   Tool 3   │    │
    │  (Search)  │   │   (Code)   │   │   (API)    │    │
    └─────┬──────┘   └─────┬──────┘   └─────┬──────┘    │
          │                │                │           │
          └────────────────┴────────────────┘           │
                           │                            │
                           ▼                            │
                    ┌────────────┐                      │
                    │   Result   │──────────────────────┘
                    └────────────┘
                           │
                           ▼ (when done)
                    ┌────────────┐
                    │   Answer   │
                    └────────────┘
`
    },
    {
        id: 'production-stack',
        title: 'Production LLM Stack',
        description: 'Full production architecture with observability',
        diagram: `
┌────────────────────────────────────────────────────────────────────┐
│                           CLIENTS                                   │
│         Web App              Mobile App           API Consumers     │
└───────────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│                        API GATEWAY                                  │
│              (Auth, Rate Limiting, Routing)                         │
└───────────────────────────────┬────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│   Chat API   │       │   RAG API    │       │  Agent API   │
└──────┬───────┘       └──────┬───────┘       └──────┬───────┘
       │                      │                      │
       └──────────────────────┼──────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│                      LLM GATEWAY (LiteLLM/Portkey)                  │
│         Load Balancing │ Fallbacks │ Caching │ Cost Tracking        │
└───────────────────────────────┬────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│    OpenAI    │       │  Anthropic   │       │    Groq      │
│   (Primary)  │       │  (Fallback)  │       │   (Fast)     │
└──────────────┘       └──────────────┘       └──────────────┘

┌────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                   │
├──────────────┬──────────────┬──────────────┬──────────────────────┤
│  Vector DB   │    Redis     │  Postgres    │    S3/Blob           │
│  (Pinecone)  │   (Cache)    │  (Metadata)  │   (Documents)        │
└──────────────┴──────────────┴──────────────┴──────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│                      OBSERVABILITY                                  │
├──────────────┬──────────────┬──────────────┬──────────────────────┤
│   Langfuse   │   Datadog    │   Sentry     │    PagerDuty         │
│  (Tracing)   │  (Metrics)   │   (Errors)   │    (Alerts)          │
└──────────────┴──────────────┴──────────────┴──────────────────────┘
`
    },
    {
        id: 'multimodal-rag',
        title: 'Multimodal RAG',
        description: 'RAG with images, tables, and text',
        diagram: `
┌────────────────────────────────────────────────────────────────────┐
│                      DOCUMENT PROCESSING                            │
└────────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│     Text     │       │    Images    │       │    Tables    │
│   Chunks     │       │   (Charts)   │       │   (Data)     │
└──────┬───────┘       └──────┬───────┘       └──────┬───────┘
       │                      │                      │
       ▼                      ▼                      ▼
┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│    Text      │       │    CLIP      │       │    Text      │
│  Embeddings  │       │  Embeddings  │       │  Embeddings  │
└──────┬───────┘       └──────┬───────┘       └──────┬───────┘
       │                      │                      │
       └──────────────────────┼──────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   Vector Store   │
                    │  (Multi-index)   │
                    └────────┬─────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────────┐
│                         QUERY TIME                                  │
└────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Retrieve Mixed  │
                    │  (Text + Images) │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   Vision LLM     │
                    │   (current ID)   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    Response      │
                    │  (with sources)  │
                    └──────────────────┘
`
    }
];

// ============ COMPONENT ============
const CopyButton = ({ text }: { text: string }) => {
    const [copied, setCopied] = useState(false);
    
    const handleCopy = () => {
        navigator.clipboard.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };
    
    return (
        <button
            onClick={handleCopy}
            className="p-2 text-stone-400 hover:text-stone-600 transition-colors"
            title="Copy to clipboard"
        >
            {copied ? <Check size={16} className="text-green-500" /> : <Copy size={16} />}
        </button>
    );
};

const CodeExamplesTab = () => {
    const [selectedCategory, setSelectedCategory] = useState<string>('all');
    const [expandedExample, setExpandedExample] = useState<string | null>(null);
    
    const categories = ['all', ...new Set(codeExamples.map(e => e.category))];
    const filtered = selectedCategory === 'all' 
        ? codeExamples 
        : codeExamples.filter(e => e.category === selectedCategory);
    
    return (
        <div className="space-y-6">
            <div className="flex flex-wrap gap-2">
                {categories.map(cat => (
                    <button
                        key={cat}
                        onClick={() => setSelectedCategory(cat)}
                        className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                            selectedCategory === cat
                                ? 'bg-brand-600 text-white'
                                : 'bg-stone-100 text-stone-600 hover:bg-stone-200'
                        }`}
                    >
                        {cat === 'all' ? 'All' : cat}
                    </button>
                ))}
            </div>
            
            <div className="space-y-4">
                {filtered.map(example => (
                    <div key={example.id} className="bg-white border border-stone-200 rounded-xl overflow-hidden">
                        <button
                            onClick={() => setExpandedExample(expandedExample === example.id ? null : example.id)}
                            className="w-full p-4 flex items-center justify-between text-left hover:bg-stone-50"
                        >
                            <div>
                                <span className="text-xs font-medium text-brand-600 bg-brand-50 px-2 py-0.5 rounded">
                                    {example.category}
                                </span>
                                <h3 className="font-bold text-stone-900 mt-1">{example.title}</h3>
                                <p className="text-sm text-stone-600">{example.description}</p>
                            </div>
                            {expandedExample === example.id ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
                        </button>
                        
                        {expandedExample === example.id && (
                            <div className="border-t border-stone-200">
                                <div className="flex items-center justify-between px-4 py-2 bg-stone-800">
                                    <span className="text-xs text-stone-400 font-mono">{example.language}</span>
                                    <CopyButton text={example.code} />
                                </div>
                                <pre className="p-4 bg-stone-900 text-stone-300 text-sm overflow-x-auto">
                                    <code>{example.code}</code>
                                </pre>
                            </div>
                        )}
                    </div>
                ))}
            </div>
        </div>
    );
};

const TemplatesTab = () => {
    const [selectedCategory, setSelectedCategory] = useState<string>('all');
    const [expandedTemplate, setExpandedTemplate] = useState<string | null>(null);
    
    const categories = ['all', ...new Set(templates.map(t => t.category))];
    const filtered = selectedCategory === 'all' 
        ? templates 
        : templates.filter(t => t.category === selectedCategory);
    
    return (
        <div className="space-y-6">
            <div className="flex flex-wrap gap-2">
                {categories.map(cat => (
                    <button
                        key={cat}
                        onClick={() => setSelectedCategory(cat)}
                        className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                            selectedCategory === cat
                                ? 'bg-brand-600 text-white'
                                : 'bg-stone-100 text-stone-600 hover:bg-stone-200'
                        }`}
                    >
                        {cat === 'all' ? 'All' : cat}
                    </button>
                ))}
            </div>
            
            <div className="space-y-4">
                {filtered.map(template => (
                    <div key={template.id} className="bg-white border border-stone-200 rounded-xl overflow-hidden">
                        <button
                            onClick={() => setExpandedTemplate(expandedTemplate === template.id ? null : template.id)}
                            className="w-full p-4 flex items-center justify-between text-left hover:bg-stone-50"
                        >
                            <div>
                                <span className="text-xs font-medium text-amber-600 bg-amber-50 px-2 py-0.5 rounded">
                                    {template.category}
                                </span>
                                <h3 className="font-bold text-stone-900 mt-1">{template.title}</h3>
                                <p className="text-sm text-stone-600">{template.description}</p>
                            </div>
                            {expandedTemplate === template.id ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
                        </button>
                        
                        {expandedTemplate === template.id && (
                            <div className="border-t border-stone-200 p-4">
                                <div className="mb-3">
                                    <span className="text-xs font-medium text-stone-500">Variables: </span>
                                    {template.variables.map(v => (
                                        <code key={v} className="text-xs bg-stone-100 px-1.5 py-0.5 rounded mx-1">
                                            {`{{${v}}}`}
                                        </code>
                                    ))}
                                </div>
                                <div className="relative">
                                    <div className="absolute top-2 right-2">
                                        <CopyButton text={template.template} />
                                    </div>
                                    <pre className="p-4 bg-stone-50 border border-stone-200 rounded-lg text-sm text-stone-700 whitespace-pre-wrap overflow-x-auto">
                                        {template.template}
                                    </pre>
                                </div>
                            </div>
                        )}
                    </div>
                ))}
            </div>
        </div>
    );
};

const ComparisonsTab = () => {
    return (
        <div className="space-y-8">
            <LookUpPrompt
                why="These tables are checklists. Paste this prompt to fill in today's SKUs and prices."
                prompt="For OpenAI, Anthropic, and Google: what are the current cheap/fast, mid-tier, and flagship model IDs, list prices per 1M input/output tokens, and context windows? Cite official docs only."
                sources={[
                    { label: 'OpenAI pricing', href: 'https://openai.com/pricing' },
                    { label: 'Anthropic models', href: 'https://docs.anthropic.com/en/docs/about-claude/models' },
                    { label: 'Google AI pricing', href: 'https://ai.google.dev/pricing' },
                ]}
            />
            {comparisons.map(table => (
                <div key={table.id} className="bg-white border border-stone-200 rounded-xl overflow-hidden">
                    <div className="p-4 border-b border-stone-200">
                        <h3 className="font-bold text-stone-900">{table.title}</h3>
                        <p className="text-sm text-stone-600">{table.description}</p>
                    </div>
                    <div className="overflow-x-auto">
                        <table className="w-full">
                            <thead>
                                <tr className="bg-stone-50">
                                    {table.headers.map((h, i) => (
                                        <th key={i} className="px-4 py-3 text-left text-sm font-semibold text-stone-900">
                                            {h}
                                        </th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-stone-100">
                                {table.rows.map((row, i) => (
                                    <tr key={i} className="hover:bg-stone-50">
                                        {row.map((cell, j) => (
                                            <td key={j} className={`px-4 py-3 text-sm ${j === 0 ? 'font-medium text-stone-900' : 'text-stone-600'}`}>
                                                {cell}
                                            </td>
                                        ))}
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            ))}
        </div>
    );
};

const DecisionTreesTab = () => {
    const [activeTree, setActiveTree] = useState<string>(decisionTrees[0].id);
    const [currentNode, setCurrentNode] = useState<string>('start');
    const [history, setHistory] = useState<string[]>([]);
    const [recommendation, setRecommendation] = useState<string | null>(null);
    
    const tree = decisionTrees.find(t => t.id === activeTree)!;
    const node = tree.nodes[currentNode];
    
    const handleSelect = (option: { label: string; next: string | null; recommendation?: string }) => {
        if (option.recommendation) {
            setRecommendation(option.recommendation);
        } else if (option.next) {
            setHistory([...history, currentNode]);
            setCurrentNode(option.next);
        }
    };
    
    const handleBack = () => {
        if (history.length > 0) {
            const prev = history[history.length - 1];
            setHistory(history.slice(0, -1));
            setCurrentNode(prev);
            setRecommendation(null);
        }
    };
    
    const handleReset = () => {
        setCurrentNode('start');
        setHistory([]);
        setRecommendation(null);
    };
    
    const handleTreeChange = (treeId: string) => {
        setActiveTree(treeId);
        setCurrentNode('start');
        setHistory([]);
        setRecommendation(null);
    };
    
    return (
        <div className="space-y-6">
            <div className="flex flex-wrap gap-2">
                {decisionTrees.map(t => (
                    <button
                        key={t.id}
                        onClick={() => handleTreeChange(t.id)}
                        className={`px-4 py-2 rounded-xl font-medium transition-all ${
                            activeTree === t.id
                                ? 'bg-stone-900 text-white'
                                : 'bg-white text-stone-600 border border-stone-200 hover:bg-stone-100'
                        }`}
                    >
                        {t.title}
                    </button>
                ))}
            </div>
            
            <div className="bg-white border border-stone-200 rounded-xl p-6">
                <p className="text-sm text-stone-500 mb-4">{tree.description}</p>
                
                {recommendation ? (
                    <div className="space-y-4">
                        <div className="bg-green-50 border border-green-200 rounded-xl p-6">
                            <h4 className="font-bold text-green-900 mb-2">✓ Recommendation</h4>
                            <p className="text-green-800">{recommendation}</p>
                        </div>
                        <button
                            onClick={handleReset}
                            className="px-4 py-2 bg-stone-900 text-white rounded-lg hover:bg-stone-800"
                        >
                            Start Over
                        </button>
                    </div>
                ) : (
                    <div className="space-y-4">
                        <h3 className="text-xl font-bold text-stone-900">{node.question}</h3>
                        
                        <div className="grid gap-3">
                            {node.options.map((option, i) => (
                                <button
                                    key={i}
                                    onClick={() => handleSelect(option)}
                                    className="p-4 text-left bg-stone-50 hover:bg-brand-50 border border-stone-200 hover:border-brand-300 rounded-xl transition-all"
                                >
                                    <span className="text-stone-900">{option.label}</span>
                                </button>
                            ))}
                        </div>
                        
                        {history.length > 0 && (
                            <button
                                onClick={handleBack}
                                className="text-brand-600 hover:text-brand-700 font-medium"
                            >
                                ← Back
                            </button>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
};

const CostCalculatorTab = () => {
    const [state, setState] = useState<CostCalculatorState>({
        inputPricePerMillion: 3,
        outputPricePerMillion: 15,
        inputTokensPerRequest: 1000,
        outputTokensPerRequest: 500,
        requestsPerDay: 1000
    });

    const dailyInputTokens = state.inputTokensPerRequest * state.requestsPerDay;
    const dailyOutputTokens = state.outputTokensPerRequest * state.requestsPerDay;

    const dailyCost =
        (dailyInputTokens / 1_000_000 * state.inputPricePerMillion) +
        (dailyOutputTokens / 1_000_000 * state.outputPricePerMillion);
    const monthlyCost = dailyCost * 30;
    const yearlyCost = dailyCost * 365;
    
    return (
        <div className="space-y-6">
            <LookUpPrompt
                why="Prices are inputs, not a catalog. Paste this, then type the $/M you find into the calculator."
                prompt="What is the current list price per 1M input tokens and 1M output tokens for one cheap/fast model and one flagship model from my provider? Cite the official pricing page."
                sources={[
                    { label: 'OpenAI pricing', href: 'https://openai.com/pricing' },
                    { label: 'Anthropic pricing', href: 'https://docs.anthropic.com/en/docs/about-claude/pricing' },
                    { label: 'Google AI pricing', href: 'https://ai.google.dev/pricing' },
                ]}
            />
            <p className="text-sm text-stone-600">
                Formula: <code className="font-mono text-xs bg-stone-100 px-1.5 py-0.5 rounded">cost = input_tokens × in_$/M + output_tokens × out_$/M</code>
            </p>
            <div className="bg-white border border-stone-200 rounded-xl p-6">
                <h3 className="font-bold text-stone-900 mb-4">Enter current prices and usage</h3>
                
                <div className="grid md:grid-cols-2 gap-6">
                    <div>
                        <label className="block text-sm font-medium text-stone-700 mb-2">Input price ($ / 1M tokens)</label>
                        <input
                            type="number"
                            step="0.01"
                            min="0"
                            value={state.inputPricePerMillion}
                            onChange={e => setState({ ...state, inputPricePerMillion: parseFloat(e.target.value) || 0 })}
                            className="w-full p-3 border border-stone-200 rounded-lg"
                        />
                    </div>
                    
                    <div>
                        <label className="block text-sm font-medium text-stone-700 mb-2">Output price ($ / 1M tokens)</label>
                        <input
                            type="number"
                            step="0.01"
                            min="0"
                            value={state.outputPricePerMillion}
                            onChange={e => setState({ ...state, outputPricePerMillion: parseFloat(e.target.value) || 0 })}
                            className="w-full p-3 border border-stone-200 rounded-lg"
                        />
                    </div>
                    
                    <div>
                        <label className="block text-sm font-medium text-stone-700 mb-2">
                            Input Tokens per Request
                        </label>
                        <input
                            type="number"
                            value={state.inputTokensPerRequest}
                            onChange={e => setState({ ...state, inputTokensPerRequest: parseInt(e.target.value) || 0 })}
                            className="w-full p-3 border border-stone-200 rounded-lg"
                        />
                    </div>
                    
                    <div>
                        <label className="block text-sm font-medium text-stone-700 mb-2">
                            Output Tokens per Request
                        </label>
                        <input
                            type="number"
                            value={state.outputTokensPerRequest}
                            onChange={e => setState({ ...state, outputTokensPerRequest: parseInt(e.target.value) || 0 })}
                            className="w-full p-3 border border-stone-200 rounded-lg"
                        />
                    </div>
                    
                    <div className="md:col-span-2">
                        <label className="block text-sm font-medium text-stone-700 mb-2">
                            Requests per Day: {state.requestsPerDay.toLocaleString()}
                        </label>
                        <input
                            type="range"
                            min="10"
                            max="100000"
                            step="10"
                            value={state.requestsPerDay}
                            onChange={e => setState({ ...state, requestsPerDay: parseInt(e.target.value) })}
                            className="w-full"
                        />
                        <div className="flex justify-between text-xs text-stone-500 mt-1">
                            <span>10</span>
                            <span>100,000</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <div className="bg-stone-900 text-white rounded-xl p-6">
                <h3 className="font-bold mb-4">Estimated Costs</h3>
                
                <div className="grid md:grid-cols-3 gap-6">
                    <div className="text-center">
                        <div className="text-3xl font-bold text-brand-400">${dailyCost.toFixed(2)}</div>
                        <div className="text-stone-400">Per Day</div>
                    </div>
                    <div className="text-center">
                        <div className="text-3xl font-bold text-brand-400">${monthlyCost.toFixed(2)}</div>
                        <div className="text-stone-400">Per Month</div>
                    </div>
                    <div className="text-center">
                        <div className="text-3xl font-bold text-brand-400">${yearlyCost.toFixed(2)}</div>
                        <div className="text-stone-400">Per Year</div>
                    </div>
                </div>
                
                <div className="mt-6 pt-6 border-t border-stone-700 text-sm text-stone-400">
                    <div className="flex justify-between">
                        <span>Daily Input Tokens:</span>
                        <span>{dailyInputTokens.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between">
                        <span>Daily Output Tokens:</span>
                        <span>{dailyOutputTokens.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between mt-2">
                        <span>Input Price:</span>
                        <span>${state.inputPricePerMillion}/M tokens</span>
                    </div>
                    <div className="flex justify-between">
                        <span>Output Price:</span>
                        <span>${state.outputPricePerMillion}/M tokens</span>
                    </div>
                </div>
            </div>
        </div>
    );
};

const ArchitectureTab = () => {
    const [selected, setSelected] = useState<string>(architectureDiagrams[0].id);
    const diagram = architectureDiagrams.find(d => d.id === selected)!;
    
    return (
        <div className="space-y-6">
            <div className="flex flex-wrap gap-2">
                {architectureDiagrams.map(d => (
                    <button
                        key={d.id}
                        onClick={() => setSelected(d.id)}
                        className={`px-4 py-2 rounded-xl font-medium transition-all ${
                            selected === d.id
                                ? 'bg-stone-900 text-white'
                                : 'bg-white text-stone-600 border border-stone-200 hover:bg-stone-100'
                        }`}
                    >
                        {d.title}
                    </button>
                ))}
            </div>
            
            <div className="bg-white border border-stone-200 rounded-xl overflow-hidden">
                <div className="p-4 border-b border-stone-200 flex items-center justify-between">
                    <div>
                        <h3 className="font-bold text-stone-900">{diagram.title}</h3>
                        <p className="text-sm text-stone-600">{diagram.description}</p>
                    </div>
                    <CopyButton text={diagram.diagram} />
                </div>
                <pre className="p-6 bg-stone-50 text-stone-700 text-sm overflow-x-auto font-mono">
                    {diagram.diagram}
                </pre>
            </div>
        </div>
    );
};

export const ToolkitView = () => {
    const [activeTab, setActiveTab] = useState<TabId>('code');
    
    const tabs: { id: TabId; label: string; icon: React.ReactNode }[] = [
        { id: 'code', label: 'Code Examples', icon: <Code size={18} /> },
        { id: 'templates', label: 'Templates', icon: <FileText size={18} /> },
        { id: 'comparisons', label: 'Comparisons', icon: <GitCompare size={18} /> },
        { id: 'decisions', label: 'Decision Trees', icon: <GitBranch size={18} /> },
        { id: 'calculator', label: 'Cost Calculator', icon: <Calculator size={18} /> },
        { id: 'architecture', label: 'Architecture', icon: <Box size={18} /> },
    ];
    
    return (
        <div className="min-h-screen bg-stone-50">
            <div className="bg-stone-900 text-white py-12 px-6">
                <div className="max-w-5xl mx-auto">
                    <h1 className="text-3xl font-bold mb-2">Developer Toolkit</h1>
                    <p className="text-stone-400">Code examples, templates, comparisons, and tools for AI engineering</p>
                </div>
            </div>
            
            <div className="max-w-5xl mx-auto px-6 py-8">
                <div className="flex flex-wrap gap-2 mb-8">
                    {tabs.map(tab => (
                        <button
                            key={tab.id}
                            onClick={() => setActiveTab(tab.id)}
                            className={`flex items-center gap-2 px-4 py-2 rounded-xl font-medium transition-all ${
                                activeTab === tab.id
                                    ? 'bg-stone-900 text-white'
                                    : 'bg-white text-stone-600 border border-stone-200 hover:bg-stone-100'
                            }`}
                        >
                            {tab.icon}
                            {tab.label}
                        </button>
                    ))}
                </div>
                
                {activeTab === 'code' && <CodeExamplesTab />}
                {activeTab === 'templates' && <TemplatesTab />}
                {activeTab === 'comparisons' && <ComparisonsTab />}
                {activeTab === 'decisions' && <DecisionTreesTab />}
                {activeTab === 'calculator' && <CostCalculatorTab />}
                {activeTab === 'architecture' && <ArchitectureTab />}
            </div>
        </div>
    );
};


