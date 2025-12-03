import React, { useState } from 'react';
import { Search, BookOpen } from 'lucide-react';

interface Term {
    term: string;
    definition: string;
    category: string;
}

const glossaryTerms: Term[] = [
    // Core Concepts
    { term: 'LLM (Large Language Model)', definition: 'Neural networks trained on massive text datasets that can generate, understand, and manipulate human language. Examples: GPT-4, Claude, Llama.', category: 'Core Concepts' },
    { term: 'Foundation Model', definition: 'Large AI models trained on broad data that can be adapted to many downstream tasks. They serve as a "foundation" for building specialized applications.', category: 'Core Concepts' },
    { term: 'Transformer', definition: 'Neural network architecture using self-attention mechanisms. The basis for modern LLMs, introduced in "Attention Is All You Need" (2017).', category: 'Core Concepts' },
    { term: 'Token', definition: 'The basic unit of text processing in LLMs. Can be a word, subword, or character. "Hello world" might be 2 tokens, while "tokenization" might be 3.', category: 'Core Concepts' },
    { term: 'Context Window', definition: 'Maximum number of tokens an LLM can process in a single request. Ranges from 4K (older models) to 1M+ (latest models).', category: 'Core Concepts' },
    { term: 'Inference', definition: 'The process of running a trained model to generate predictions or outputs. What happens when you send a prompt to an API.', category: 'Core Concepts' },
    { term: 'Latency', definition: 'Time between sending a request and receiving a response. Critical metric for real-time applications.', category: 'Core Concepts' },
    { term: 'Throughput', definition: 'Number of requests or tokens a system can process per unit time. Important for high-volume applications.', category: 'Core Concepts' },
    
    // Prompting
    { term: 'Prompt', definition: 'The input text sent to an LLM to elicit a response. Includes system prompts, user messages, and context.', category: 'Prompting' },
    { term: 'System Prompt', definition: 'Instructions that define the AI\'s behavior, persona, and constraints. Set once at the beginning of a conversation.', category: 'Prompting' },
    { term: 'Few-Shot Learning', definition: 'Providing examples in the prompt to guide the model\'s output format and behavior without fine-tuning.', category: 'Prompting' },
    { term: 'Zero-Shot', definition: 'Asking a model to perform a task without providing any examples, relying on its pre-trained knowledge.', category: 'Prompting' },
    { term: 'Chain of Thought (CoT)', definition: 'Prompting technique that encourages step-by-step reasoning, improving performance on complex tasks.', category: 'Prompting' },
    { term: 'Prompt Injection', definition: 'Attack where malicious input tricks an LLM into ignoring its instructions or revealing sensitive information.', category: 'Prompting' },
    { term: 'Temperature', definition: 'Parameter controlling randomness in outputs. 0 = deterministic, 1+ = more creative/random.', category: 'Prompting' },
    { term: 'Top-p (Nucleus Sampling)', definition: 'Sampling parameter that considers only tokens whose cumulative probability exceeds p. Alternative to temperature.', category: 'Prompting' },
    
    // RAG & Retrieval
    { term: 'RAG (Retrieval-Augmented Generation)', definition: 'Architecture that retrieves relevant documents and includes them in the prompt to ground LLM responses in factual data.', category: 'RAG & Retrieval' },
    { term: 'Embedding', definition: 'Dense vector representation of text that captures semantic meaning. Similar texts have similar embeddings.', category: 'RAG & Retrieval' },
    { term: 'Vector Database', definition: 'Database optimized for storing and querying high-dimensional vectors. Used to find semantically similar content.', category: 'RAG & Retrieval' },
    { term: 'Semantic Search', definition: 'Search based on meaning rather than keywords. Uses embeddings to find conceptually related content.', category: 'RAG & Retrieval' },
    { term: 'Chunking', definition: 'Splitting documents into smaller pieces for embedding and retrieval. Strategy affects RAG quality significantly.', category: 'RAG & Retrieval' },
    { term: 'Reranking', definition: 'Second-stage retrieval that reorders initial results using a more sophisticated model for better relevance.', category: 'RAG & Retrieval' },
    { term: 'Hybrid Search', definition: 'Combining semantic (vector) search with keyword (BM25) search for better retrieval accuracy.', category: 'RAG & Retrieval' },
    { term: 'Cosine Similarity', definition: 'Metric measuring the angle between two vectors. Used to compare embedding similarity (1 = identical, 0 = unrelated).', category: 'RAG & Retrieval' },
    
    // Agents & Tools
    { term: 'Agent', definition: 'AI system that can reason, plan, and take actions using tools to accomplish goals autonomously.', category: 'Agents & Tools' },
    { term: 'Function Calling', definition: 'LLM capability to output structured requests for external functions/APIs, enabling tool use.', category: 'Agents & Tools' },
    { term: 'Tool Use', definition: 'Enabling LLMs to interact with external systems like APIs, databases, or code interpreters.', category: 'Agents & Tools' },
    { term: 'ReAct', definition: 'Reasoning + Acting pattern where agents alternate between thinking and taking actions.', category: 'Agents & Tools' },
    { term: 'Planning', definition: 'Agent capability to break down complex goals into subtasks and determine execution order.', category: 'Agents & Tools' },
    { term: 'Memory', definition: 'Mechanisms for agents to retain information across interactions: short-term (context), long-term (database).', category: 'Agents & Tools' },
    { term: 'Multi-Agent', definition: 'Systems with multiple specialized AI agents collaborating or competing to solve problems.', category: 'Agents & Tools' },
    
    // Training & Fine-Tuning
    { term: 'Fine-Tuning', definition: 'Further training a pre-trained model on task-specific data to improve performance on that task.', category: 'Training' },
    { term: 'LoRA (Low-Rank Adaptation)', definition: 'Parameter-efficient fine-tuning that trains small adapter layers instead of full model weights. Much cheaper than full fine-tuning.', category: 'Training' },
    { term: 'QLoRA', definition: 'Quantized LoRA - combines 4-bit quantization with LoRA for even more memory-efficient fine-tuning.', category: 'Training' },
    { term: 'RLHF', definition: 'Reinforcement Learning from Human Feedback - training models using human preference data to align with human values.', category: 'Training' },
    { term: 'DPO (Direct Preference Optimization)', definition: 'Simpler alternative to RLHF that directly optimizes on preference data without a separate reward model.', category: 'Training' },
    { term: 'SFT (Supervised Fine-Tuning)', definition: 'Training on input-output pairs where the model learns to produce specific outputs for given inputs.', category: 'Training' },
    { term: 'Instruction Tuning', definition: 'Fine-tuning models to follow instructions better using datasets of instruction-response pairs.', category: 'Training' },
    { term: 'Catastrophic Forgetting', definition: 'When fine-tuning causes a model to lose previously learned capabilities.', category: 'Training' },
    
    // Architecture & Optimization
    { term: 'Attention', definition: 'Mechanism allowing models to weigh the importance of different parts of the input when generating output.', category: 'Architecture' },
    { term: 'Self-Attention', definition: 'Attention applied within a single sequence, allowing each token to attend to all other tokens.', category: 'Architecture' },
    { term: 'MoE (Mixture of Experts)', definition: 'Architecture using multiple specialized sub-networks (experts) with a router selecting which to activate. Enables larger models with less compute.', category: 'Architecture' },
    { term: 'Quantization', definition: 'Reducing model precision (e.g., 16-bit to 4-bit) to decrease memory usage and increase speed with minimal quality loss.', category: 'Architecture' },
    { term: 'KV Cache', definition: 'Caching key-value pairs from attention layers to avoid recomputation during autoregressive generation.', category: 'Architecture' },
    { term: 'Speculative Decoding', definition: 'Using a smaller draft model to propose tokens, then verifying with the main model. Speeds up inference.', category: 'Architecture' },
    { term: 'Continuous Batching', definition: 'Dynamically adding/removing requests from a batch during inference for better GPU utilization.', category: 'Architecture' },
    
    // Evaluation
    { term: 'Hallucination', definition: 'When an LLM generates plausible-sounding but factually incorrect or fabricated information.', category: 'Evaluation' },
    { term: 'Grounding', definition: 'Anchoring LLM outputs to factual sources (via RAG or citations) to reduce hallucinations.', category: 'Evaluation' },
    { term: 'LLM-as-Judge', definition: 'Using an LLM to evaluate outputs from another LLM, common for subjective quality assessment.', category: 'Evaluation' },
    { term: 'BLEU', definition: 'Metric comparing generated text to reference text based on n-gram overlap. Common in translation.', category: 'Evaluation' },
    { term: 'ROUGE', definition: 'Recall-oriented metric for summarization comparing generated summaries to references.', category: 'Evaluation' },
    { term: 'Perplexity', definition: 'Measure of how well a model predicts text. Lower = better. Used to evaluate language model quality.', category: 'Evaluation' },
    { term: 'Benchmark', definition: 'Standardized test suite for comparing model capabilities (e.g., MMLU, HumanEval, GSM8K).', category: 'Evaluation' },
    
    // Production
    { term: 'Guardrails', definition: 'Safety mechanisms that filter, validate, or modify LLM inputs/outputs to prevent harmful content.', category: 'Production' },
    { term: 'Rate Limiting', definition: 'Restricting the number of API requests per time period to prevent abuse and manage costs.', category: 'Production' },
    { term: 'Semantic Caching', definition: 'Caching LLM responses based on semantic similarity of queries, not exact matches.', category: 'Production' },
    { term: 'Streaming', definition: 'Returning LLM output token-by-token as generated, rather than waiting for complete response.', category: 'Production' },
    { term: 'Fallback', definition: 'Backup strategy when primary model fails (e.g., switch to different provider or simpler model).', category: 'Production' },
    { term: 'Observability', definition: 'Monitoring, logging, and tracing LLM applications to understand behavior and debug issues.', category: 'Production' },
    { term: 'Prompt Versioning', definition: 'Tracking changes to prompts over time, similar to code version control.', category: 'Production' },
];

const categories = [...new Set(glossaryTerms.map(t => t.category))];

export const GlossaryView = () => {
    const [searchQuery, setSearchQuery] = useState('');
    const [activeCategory, setActiveCategory] = useState<string | null>(null);

    const filteredTerms = glossaryTerms.filter(t => {
        const matchesSearch = searchQuery === '' || 
            t.term.toLowerCase().includes(searchQuery.toLowerCase()) ||
            t.definition.toLowerCase().includes(searchQuery.toLowerCase());
        const matchesCategory = activeCategory === null || t.category === activeCategory;
        return matchesSearch && matchesCategory;
    });

    const groupedTerms = filteredTerms.reduce((acc, term) => {
        if (!acc[term.category]) acc[term.category] = [];
        acc[term.category].push(term);
        return acc;
    }, {} as Record<string, Term[]>);

    return (
        <div className="min-h-screen bg-stone-50">
            <div className="bg-stone-900 text-white py-12 px-6">
                <div className="max-w-4xl mx-auto">
                    <div className="flex items-center gap-3 mb-2">
                        <BookOpen className="w-8 h-8 text-brand-400" />
                        <h1 className="text-3xl font-bold">AI Engineering Glossary</h1>
                    </div>
                    <p className="text-stone-400 mb-6">Essential terminology for AI engineers</p>
                    
                    <div className="relative">
                        <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-stone-500" />
                        <input
                            type="text"
                            placeholder="Search terms..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            className="w-full max-w-md pl-12 pr-4 py-3 bg-stone-800 border border-stone-700 rounded-xl text-white placeholder-stone-500 focus:outline-none focus:border-brand-500"
                        />
                    </div>
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
                        All
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
                            {cat}
                        </button>
                    ))}
                </div>

                {/* Terms */}
                {Object.entries(groupedTerms).map(([category, terms]) => (
                    <div key={category} className="mb-8">
                        <h2 className="text-lg font-bold text-stone-900 mb-4 pb-2 border-b border-stone-200">{category}</h2>
                        <div className="space-y-3">
                            {terms.map(t => (
                                <div key={t.term} className="bg-white border border-stone-200 rounded-xl p-4">
                                    <h3 className="font-bold text-stone-900 mb-1">{t.term}</h3>
                                    <p className="text-stone-600 text-sm">{t.definition}</p>
                                </div>
                            ))}
                        </div>
                    </div>
                ))}

                {filteredTerms.length === 0 && (
                    <div className="text-center py-12 text-stone-500">
                        No terms found matching "{searchQuery}"
                    </div>
                )}
            </div>
        </div>
    );
};

