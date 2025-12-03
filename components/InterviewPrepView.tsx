import React, { useState } from 'react';
import { Briefcase, ChevronDown, ChevronUp, Lightbulb } from 'lucide-react';

interface Question {
    question: string;
    answer: string;
    tips?: string[];
    category: string;
}

const questions: Question[] = [
    // Fundamentals
    {
        category: 'Fundamentals',
        question: 'What is the difference between a language model and a large language model?',
        answer: 'A language model predicts the probability of a sequence of words. Traditional LMs were smaller and task-specific (e.g., n-gram models, small RNNs). Large Language Models (LLMs) are transformer-based models with billions of parameters, trained on massive datasets. The scale enables emergent capabilities like in-context learning, reasoning, and following complex instructions that smaller models lack.',
        tips: ['Mention the transformer architecture as the key enabler', 'Discuss emergent capabilities that come with scale', 'Give examples: GPT-4 (1.7T params) vs BERT (340M)']
    },
    {
        category: 'Fundamentals',
        question: 'Explain the transformer architecture and why it revolutionized NLP.',
        answer: 'Transformers use self-attention to process all tokens in parallel, unlike RNNs which process sequentially. Key components: (1) Self-attention allows each token to attend to all others, capturing long-range dependencies. (2) Multi-head attention learns different relationship types. (3) Positional encoding adds sequence order information. (4) Feed-forward layers add non-linearity. This enables: parallelization (faster training), better long-range dependencies, and scalability to billions of parameters.',
        tips: ['Draw the architecture if whiteboarding', 'Explain attention as "which words should I focus on to understand this word"', 'Mention the 2017 "Attention Is All You Need" paper']
    },
    {
        category: 'Fundamentals',
        question: 'What is tokenization and why does it matter?',
        answer: 'Tokenization converts text into numerical tokens the model can process. Modern LLMs use subword tokenization (BPE, SentencePiece) which balances vocabulary size with coverage. It matters because: (1) Token count affects cost and context limits. (2) Different tokenizers handle languages/code differently. (3) Tokenization artifacts can affect output quality. Example: "tokenization" might be ["token", "ization"] = 2 tokens.',
        tips: ['Mention that different models have different tokenizers', 'Discuss the ~4 chars per token rule of thumb for English', 'Note that non-English languages often use more tokens']
    },
    {
        category: 'Fundamentals',
        question: 'What is the context window and how do you handle documents that exceed it?',
        answer: 'The context window is the maximum tokens an LLM can process in one request. Strategies for long documents: (1) Chunking + RAG: Split into chunks, retrieve relevant ones. (2) Summarization: Hierarchical summarization of sections. (3) Map-reduce: Process chunks separately, combine results. (4) Sliding window: Process overlapping windows. (5) Use longer-context models. Trade-offs: RAG is flexible but loses global context; longer windows are simpler but more expensive.',
        tips: ['Know that context limits vary widely by model (32K to 1M+)', 'Discuss the "lost in the middle" problem', 'Mention that longer context ≠ better retrieval']
    },
    
    // Prompting
    {
        category: 'Prompting',
        question: 'What are the key components of an effective system prompt?',
        answer: 'A good system prompt includes: (1) Identity/Role: Who the AI is. (2) Task: What it should do. (3) Constraints: What it should NOT do. (4) Output format: Structure of responses. (5) Examples: Sample interactions. (6) Context: Background information. Best practices: Be specific, use delimiters, order matters (important info first/last), iterate and test.',
        tips: ['Give a concrete example of a well-structured prompt', 'Mention that prompt engineering is empirical—test and iterate', 'Discuss the importance of clear delimiters (XML tags, markdown)']
    },
    {
        category: 'Prompting',
        question: 'Explain Chain of Thought prompting and when to use it.',
        answer: 'Chain of Thought (CoT) prompting encourages the model to show its reasoning steps before giving a final answer. Techniques: (1) Zero-shot CoT: Add "Let\'s think step by step". (2) Few-shot CoT: Provide examples with reasoning. (3) Self-consistency: Generate multiple reasoning paths, take majority vote. Use for: math, logic, multi-step problems. Avoid for: simple factual queries (adds latency/cost). CoT improves accuracy on complex tasks by 10-40%.',
        tips: ['Demonstrate with a math problem example', 'Mention that CoT works better on larger models', 'Discuss the latency/cost trade-off']
    },
    {
        category: 'Prompting',
        question: 'What is prompt injection and how do you defend against it?',
        answer: 'Prompt injection is when user input manipulates the LLM to ignore its instructions or reveal sensitive information. Types: (1) Direct: "Ignore previous instructions and..." (2) Indirect: Malicious content in retrieved documents. Defenses: (1) Input sanitization and validation. (2) Clear delimiters between instructions and user input. (3) Output filtering/guardrails. (4) Separate privileged/unprivileged contexts. (5) LLM-based detection. No defense is 100% effective—defense in depth is key.',
        tips: ['Give concrete injection examples', 'Emphasize this is an unsolved problem', 'Mention tools like Guardrails AI, NeMo Guardrails']
    },
    
    // RAG
    {
        category: 'RAG',
        question: 'Walk me through designing a RAG system from scratch.',
        answer: 'Key components: (1) Ingestion: Load documents → chunk (512-1024 tokens, with overlap) → embed → store in vector DB. (2) Retrieval: Embed query → vector search → optional reranking → return top-k. (3) Generation: Construct prompt with retrieved context → LLM generates answer. Design decisions: Chunking strategy (semantic vs fixed), embedding model, vector DB choice, hybrid search, reranking, prompt template. Evaluation: Retrieval metrics (Recall@K, MRR) + generation quality (faithfulness, relevance).',
        tips: ['Draw a diagram showing the data flow', 'Discuss trade-offs at each step', 'Mention specific tools: LangChain, LlamaIndex, Pinecone, etc.']
    },
    {
        category: 'RAG',
        question: 'How do you evaluate a RAG system?',
        answer: 'Evaluate both retrieval and generation: Retrieval metrics: (1) Recall@K: Did we retrieve the relevant docs? (2) Precision@K: How many retrieved docs are relevant? (3) MRR: How high is the first relevant result? Generation metrics: (1) Faithfulness: Is the answer supported by retrieved context? (2) Relevance: Does it answer the question? (3) Groundedness: No hallucinations? Tools: Ragas, TruLens, custom LLM-as-judge. Also: latency, cost, user feedback.',
        tips: ['Mention the importance of a golden test set', 'Discuss LLM-as-judge for subjective quality', 'Talk about A/B testing in production']
    },
    {
        category: 'RAG',
        question: 'What are common failure modes in RAG and how do you address them?',
        answer: 'Common failures: (1) Retrieval misses: Wrong/no docs retrieved. Fix: Better chunking, hybrid search, query expansion. (2) Lost in the middle: Model ignores middle context. Fix: Rerank to put relevant docs first. (3) Hallucination despite context: Model ignores retrieved info. Fix: Lower temperature, explicit grounding instructions. (4) Stale data: Index not updated. Fix: Incremental updates, freshness metadata. (5) Wrong granularity: Chunks too big/small. Fix: Experiment with chunk sizes, parent-child retrieval.',
        tips: ['Give specific examples of each failure', 'Discuss debugging strategies', 'Mention observability tools for diagnosing issues']
    },
    
    // Agents
    {
        category: 'Agents',
        question: 'What is an AI agent and how does it differ from a simple LLM call?',
        answer: 'An agent is an LLM that can reason, plan, and take actions using tools to accomplish goals. Unlike simple LLM calls: (1) Autonomy: Decides what actions to take. (2) Tool use: Can call APIs, search, execute code. (3) Planning: Breaks down complex tasks. (4) Memory: Maintains state across steps. (5) Iteration: Loops until goal achieved. Common patterns: ReAct (Reason + Act), Plan-and-Execute, Tree of Thoughts. Challenges: Reliability, cost, latency, error handling.',
        tips: ['Give a concrete agent example (research agent, coding agent)', 'Discuss the ReAct pattern in detail', 'Mention frameworks: LangGraph, AutoGen, CrewAI']
    },
    {
        category: 'Agents',
        question: 'How do you implement function calling / tool use?',
        answer: 'Function calling lets LLMs output structured requests for external tools. Implementation: (1) Define tools with name, description, parameters (JSON schema). (2) Send tools in API request. (3) Model returns tool call with arguments. (4) Execute tool, return result to model. (5) Model generates final response. Best practices: Clear tool descriptions, handle errors gracefully, validate arguments, limit available tools, implement timeouts. Providers: OpenAI, Anthropic, Gemini all support native function calling.',
        tips: ['Show a code example of tool definition', 'Discuss parallel vs sequential tool calls', 'Mention structured output libraries like Instructor']
    },
    
    // Production
    {
        category: 'Production',
        question: 'How do you handle LLM failures and ensure reliability in production?',
        answer: 'Reliability strategies: (1) Retries with exponential backoff for transient errors. (2) Fallbacks: Secondary model/provider if primary fails. (3) Timeouts: Don\'t wait forever. (4) Circuit breakers: Stop calling failing services. (5) Load balancing across providers. (6) Graceful degradation: Simpler response if LLM unavailable. (7) Input validation: Reject malformed requests early. (8) Output validation: Check response format/content. Monitoring: Track error rates, latency p50/p95/p99, cost per request.',
        tips: ['Mention specific tools: Portkey, LiteLLM for multi-provider', 'Discuss SLAs and error budgets', 'Talk about alerting thresholds']
    },
    {
        category: 'Production',
        question: 'How do you optimize LLM costs in production?',
        answer: 'Cost optimization strategies: (1) Model tiering: Route simple queries to cheaper models. (2) Prompt caching: Reuse system prompts (Anthropic/OpenAI offer this). (3) Semantic caching: Cache responses for similar queries. (4) Prompt compression: Shorter prompts = fewer tokens. (5) Batch processing: Use batch APIs for non-real-time (50% discount). (6) Output limits: Set appropriate max_tokens. (7) Fine-tuning: Smaller fine-tuned model can match larger base model. Monitoring: Track cost per request, per user, per feature.',
        tips: ['Give concrete cost examples', 'Discuss the model tiering decision logic', 'Mention that 10x cost difference between models is common']
    },
    {
        category: 'Production',
        question: 'How do you evaluate and monitor LLMs in production?',
        answer: 'Evaluation: (1) Offline: Benchmark on test sets before deployment. (2) Online: A/B testing, shadow mode. (3) Metrics: Task-specific (accuracy, BLEU) + general (latency, cost). (4) LLM-as-judge for subjective quality. Monitoring: (1) Log all requests/responses. (2) Track latency, error rates, token usage. (3) Detect drift in output quality. (4) User feedback collection. (5) Alerting on anomalies. Tools: Langfuse, LangSmith, Braintrust, Arize for observability.',
        tips: ['Emphasize the importance of logging everything', 'Discuss regression detection', 'Mention the challenge of evaluating open-ended generation']
    },
    
    // Fine-tuning
    {
        category: 'Fine-tuning',
        question: 'When should you fine-tune vs use prompting?',
        answer: 'Use prompting when: (1) Quick iteration needed. (2) Task is well-defined with examples. (3) Base model performs reasonably. (4) Data is limited (<100 examples). Fine-tune when: (1) Consistent style/format needed. (2) Domain-specific knowledge required. (3) Cost optimization (smaller fine-tuned > larger base). (4) Latency critical (shorter prompts). (5) Have 100-10K+ quality examples. Often: Start with prompting, fine-tune if needed. Fine-tuning is not magic—garbage in, garbage out.',
        tips: ['Give specific use cases for each approach', 'Discuss the data quality requirements', 'Mention that fine-tuning can\'t add new knowledge effectively']
    },
    {
        category: 'Fine-tuning',
        question: 'Explain LoRA and why it\'s popular for fine-tuning.',
        answer: 'LoRA (Low-Rank Adaptation) trains small adapter matrices instead of full model weights. How it works: Freeze base weights, add low-rank decomposition matrices (A×B) to attention layers. Benefits: (1) 10-100x fewer trainable parameters. (2) Fits on consumer GPUs. (3) Can merge multiple LoRAs. (4) Easy to swap adapters. (5) Preserves base model capabilities. Trade-offs: Slightly lower quality than full fine-tuning, rank hyperparameter tuning needed. QLoRA adds 4-bit quantization for even more efficiency.',
        tips: ['Explain the math simply: W\' = W + AB where A and B are small', 'Mention typical rank values (8-64)', 'Discuss tools: Axolotl, Unsloth, PEFT']
    },
    
    // System Design
    {
        category: 'System Design',
        question: 'Design a customer support chatbot for an e-commerce company.',
        answer: 'Requirements: Handle FAQs, order status, returns, escalation. Architecture: (1) Intent classification (fine-tuned classifier or LLM). (2) RAG over knowledge base (FAQs, policies, product info). (3) Tool integration (order API, CRM). (4) Conversation memory (session state). (5) Guardrails (no competitor mentions, escalation triggers). (6) Human handoff for complex issues. Technical: Budget LLM for cost, vector DB, observability, caching for common queries. Evaluation: Resolution rate, CSAT, escalation rate, cost per conversation.',
        tips: ['Draw the architecture diagram', 'Discuss the build vs buy decision', 'Mention specific edge cases and how to handle them']
    },
    {
        category: 'System Design',
        question: 'Design a code review assistant that integrates with GitHub.',
        answer: 'Requirements: Review PRs, suggest improvements, explain changes. Architecture: (1) GitHub webhook triggers on PR. (2) Fetch diff and relevant context (related files, docs). (3) RAG over codebase + style guides. (4) LLM generates review (code-capable model). (5) Post comments via GitHub API. (6) Learn from accepted/rejected suggestions. Considerations: Context limits (large PRs), incremental reviews, confidence thresholds (don\'t spam), security (no secrets in prompts), cost management (limit reviews per PR). Similar to: Copilot, CodeRabbit, Sourcery.',
        tips: ['Discuss how to handle large diffs', 'Talk about the feedback loop for improvement', 'Mention security considerations']
    },
];

const categories = [...new Set(questions.map(q => q.category))];

export const InterviewPrepView = () => {
    const [activeCategory, setActiveCategory] = useState<string | null>(null);
    const [expandedQuestions, setExpandedQuestions] = useState<Set<string>>(new Set());

    const filteredQuestions = activeCategory 
        ? questions.filter(q => q.category === activeCategory)
        : questions;

    const toggleQuestion = (question: string) => {
        const newExpanded = new Set(expandedQuestions);
        if (newExpanded.has(question)) {
            newExpanded.delete(question);
        } else {
            newExpanded.add(question);
        }
        setExpandedQuestions(newExpanded);
    };

    return (
        <div className="min-h-screen bg-stone-50">
            <div className="bg-stone-900 text-white py-12 px-6">
                <div className="max-w-4xl mx-auto">
                    <div className="flex items-center gap-3 mb-2">
                        <Briefcase className="w-8 h-8 text-brand-400" />
                        <h1 className="text-3xl font-bold">Interview Prep</h1>
                    </div>
                    <p className="text-stone-400">Common AI engineering interview questions with detailed answers</p>
                </div>
            </div>

            <div className="max-w-4xl mx-auto px-6 py-8">
                {/* Category filters */}
                <div className="flex flex-wrap gap-2 mb-8">
                    <button
                        onClick={() => setActiveCategory(null)}
                        className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                            activeCategory === null 
                                ? 'bg-stone-900 text-white' 
                                : 'bg-white text-stone-600 border border-stone-200 hover:bg-stone-100'
                        }`}
                    >
                        All ({questions.length})
                    </button>
                    {categories.map(cat => (
                        <button
                            key={cat}
                            onClick={() => setActiveCategory(cat)}
                            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                                activeCategory === cat 
                                    ? 'bg-stone-900 text-white' 
                                    : 'bg-white text-stone-600 border border-stone-200 hover:bg-stone-100'
                            }`}
                        >
                            {cat} ({questions.filter(q => q.category === cat).length})
                        </button>
                    ))}
                </div>

                {/* Questions */}
                <div className="space-y-4">
                    {filteredQuestions.map((q, idx) => {
                        const isExpanded = expandedQuestions.has(q.question);
                        return (
                            <div key={idx} className="bg-white border border-stone-200 rounded-xl overflow-hidden">
                                <button
                                    onClick={() => toggleQuestion(q.question)}
                                    className="w-full p-5 text-left flex items-start justify-between gap-4 hover:bg-stone-50 transition-colors"
                                >
                                    <div>
                                        <span className="text-xs font-medium text-brand-600 bg-brand-50 px-2 py-0.5 rounded mb-2 inline-block">
                                            {q.category}
                                        </span>
                                        <h3 className="font-bold text-stone-900">{q.question}</h3>
                                    </div>
                                    {isExpanded ? (
                                        <ChevronUp className="w-5 h-5 text-stone-400 shrink-0 mt-1" />
                                    ) : (
                                        <ChevronDown className="w-5 h-5 text-stone-400 shrink-0 mt-1" />
                                    )}
                                </button>
                                
                                {isExpanded && (
                                    <div className="px-5 pb-5 border-t border-stone-100">
                                        <div className="pt-4">
                                            <h4 className="text-sm font-semibold text-stone-500 uppercase tracking-wide mb-2">Answer</h4>
                                            <p className="text-stone-700 whitespace-pre-line">{q.answer}</p>
                                        </div>
                                        
                                        {q.tips && q.tips.length > 0 && (
                                            <div className="mt-4 bg-amber-50 border border-amber-200 rounded-lg p-4">
                                                <div className="flex items-center gap-2 mb-2">
                                                    <Lightbulb className="w-4 h-4 text-amber-600" />
                                                    <h4 className="text-sm font-semibold text-amber-800">Interview Tips</h4>
                                                </div>
                                                <ul className="space-y-1">
                                                    {q.tips.map((tip, i) => (
                                                        <li key={i} className="text-sm text-amber-900 flex items-start gap-2">
                                                            <span className="text-amber-500">•</span>
                                                            {tip}
                                                        </li>
                                                    ))}
                                                </ul>
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>
                        );
                    })}
                </div>
            </div>
        </div>
    );
};

