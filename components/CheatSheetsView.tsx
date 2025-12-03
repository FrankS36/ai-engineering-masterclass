import React, { useState } from 'react';
import { FileText, Copy, Check } from 'lucide-react';

type SheetId = 'prompting' | 'models' | 'rag' | 'errors' | 'costs';

interface CheatSheet {
    id: SheetId;
    title: string;
    sections: { title: string; items: { label: string; value: string; code?: string }[] }[];
}

const cheatSheets: CheatSheet[] = [
    {
        id: 'prompting',
        title: 'Prompt Engineering',
        sections: [
            {
                title: 'Prompting Patterns',
                items: [
                    { label: 'Zero-Shot', value: 'Direct instruction without examples', code: 'Classify the sentiment: "I love this product"' },
                    { label: 'Few-Shot', value: 'Provide 2-5 examples before the task', code: 'Positive: "Great!" → positive\nNegative: "Terrible" → negative\nClassify: "Amazing service"' },
                    { label: 'Chain of Thought', value: 'Add "Let\'s think step by step"', code: 'Solve this math problem. Let\'s think step by step:\n[problem]' },
                    { label: 'Role Playing', value: 'Assign a persona/expertise', code: 'You are an expert Python developer. Review this code...' },
                    { label: 'Output Format', value: 'Specify exact output structure', code: 'Respond in JSON format:\n{"sentiment": "positive|negative", "confidence": 0.0-1.0}' },
                ]
            },
            {
                title: 'System Prompt Structure',
                items: [
                    { label: 'Identity', value: 'Who the AI is', code: 'You are a helpful customer support agent for Acme Inc.' },
                    { label: 'Task', value: 'What it should do', code: 'Help users troubleshoot product issues and process returns.' },
                    { label: 'Constraints', value: 'What it should NOT do', code: 'Never share internal pricing. Don\'t make promises about refunds.' },
                    { label: 'Format', value: 'How to structure responses', code: 'Keep responses under 3 sentences. Use bullet points for steps.' },
                    { label: 'Examples', value: 'Sample interactions', code: 'User: "My order is late"\nAssistant: "I\'ll check that for you. Order #?"' },
                ]
            },
            {
                title: 'Temperature Guide',
                items: [
                    { label: '0.0', value: 'Deterministic - same output every time. Best for: factual Q&A, classification, code' },
                    { label: '0.3-0.5', value: 'Low creativity - slight variation. Best for: summarization, extraction' },
                    { label: '0.7-0.8', value: 'Balanced - good default. Best for: general chat, explanations' },
                    { label: '1.0+', value: 'High creativity - unpredictable. Best for: brainstorming, creative writing' },
                ]
            },
        ]
    },
    {
        id: 'models',
        title: 'Model Comparison',
        sections: [
            {
                title: 'Frontier Models (Nov 2024)',
                items: [
                    { label: 'OpenAI Flagship', value: 'OpenAI | 128K+ context | Multimodal, tools, reasoning' },
                    { label: 'OpenAI Budget', value: 'OpenAI | 128K+ context | Cost-effective, fast' },
                    { label: 'Claude Flagship', value: 'Anthropic | 200K context | Best for code, long context' },
                    { label: 'Claude Budget', value: 'Anthropic | 200K context | Fast, cost-effective' },
                    { label: 'Gemini Flagship', value: 'Google | 1M+ context | Longest context, multimodal' },
                    { label: 'Gemini Budget', value: 'Google | 1M+ context | Fast, very affordable' },
                ]
            },
            {
                title: 'Open Source Models',
                items: [
                    { label: 'Llama (Large)', value: 'Meta | 128K context | Best open model, needs multi-GPU' },
                    { label: 'Llama (Medium)', value: 'Meta | 128K context | Great balance, runs on 1-2 GPUs' },
                    { label: 'Llama (Small)', value: 'Meta | 128K context | Fast, runs on single GPU' },
                    { label: 'Mixtral', value: 'Mistral | 64K+ context | MoE, efficient inference' },
                    { label: 'Qwen', value: 'Alibaba | 128K context | Strong multilingual' },
                    { label: 'DeepSeek', value: 'DeepSeek | 128K context | MoE, strong reasoning' },
                ]
            },
            {
                title: 'Use Case Recommendations',
                items: [
                    { label: 'Code Generation', value: 'Claude flagship > OpenAI flagship > DeepSeek' },
                    { label: 'Long Documents', value: 'Gemini (1M+) > Claude (200K) > OpenAI (128K)' },
                    { label: 'Reasoning', value: 'OpenAI reasoning models > Claude > Gemini' },
                    { label: 'Speed Critical', value: 'Groq > Budget models > Gemini Flash' },
                    { label: 'Budget Constrained', value: 'Budget models > Open source via inference providers' },
                    { label: 'Privacy/On-Prem', value: 'Llama > Mixtral > Qwen' },
                ]
            },
        ]
    },
    {
        id: 'rag',
        title: 'RAG Pipeline',
        sections: [
            {
                title: 'Chunking Strategies',
                items: [
                    { label: 'Fixed Size', value: '512-1024 tokens with 10-20% overlap. Simple, works for most cases.' },
                    { label: 'Semantic', value: 'Split on paragraph/section boundaries. Better coherence.' },
                    { label: 'Recursive', value: 'Try multiple separators (\\n\\n, \\n, .). LangChain default.' },
                    { label: 'Document-Aware', value: 'Use document structure (headers, sections). Best for structured docs.' },
                ]
            },
            {
                title: 'Embedding Models',
                items: [
                    { label: 'OpenAI (large)', value: 'OpenAI | High dims | Best quality, higher cost' },
                    { label: 'OpenAI (small)', value: 'OpenAI | Medium dims | Good balance, affordable' },
                    { label: 'Cohere Embed', value: 'Cohere | 1024 dims | Multilingual, compression' },
                    { label: 'Voyage', value: 'Voyage | 1024 dims | Domain-specific options' },
                    { label: 'BGE/E5 (open)', value: 'Open source | 1024 dims | Free, self-hosted' },
                ]
            },
            {
                title: 'Retrieval Tips',
                items: [
                    { label: 'Top-K', value: 'Start with k=5, adjust based on context window and relevance' },
                    { label: 'Hybrid Search', value: 'Combine vector (semantic) + BM25 (keyword) for best results' },
                    { label: 'Reranking', value: 'Use Cohere Rerank or cross-encoder after initial retrieval' },
                    { label: 'Query Expansion', value: 'Generate multiple query variations, retrieve for each, dedupe' },
                    { label: 'Metadata Filtering', value: 'Pre-filter by date, source, category before vector search' },
                ]
            },
        ]
    },
    {
        id: 'errors',
        title: 'Common Errors & Fixes',
        sections: [
            {
                title: 'API Errors',
                items: [
                    { label: '400 Bad Request', value: 'Invalid request format. Check JSON structure, required fields.' },
                    { label: '401 Unauthorized', value: 'Invalid API key. Check key is correct and not expired.' },
                    { label: '403 Forbidden', value: 'Key lacks permissions. Check org settings, model access.' },
                    { label: '429 Rate Limited', value: 'Too many requests. Implement exponential backoff, request limit increase.' },
                    { label: '500 Server Error', value: 'Provider issue. Retry with backoff, check status page.' },
                    { label: '503 Overloaded', value: 'High demand. Retry later, consider fallback model.' },
                ]
            },
            {
                title: 'Context Length Errors',
                items: [
                    { label: 'Token limit exceeded', value: 'Reduce prompt size, truncate context, use longer-context model' },
                    { label: 'Max tokens too high', value: 'Set max_tokens ≤ (context_limit - prompt_tokens)' },
                ]
            },
            {
                title: 'Output Issues',
                items: [
                    { label: 'Hallucinations', value: 'Add RAG, lower temperature, ask for citations, use grounding' },
                    { label: 'Wrong format', value: 'Use JSON mode, structured outputs, or Instructor library' },
                    { label: 'Inconsistent outputs', value: 'Set temperature=0, use seed parameter if available' },
                    { label: 'Truncated response', value: 'Increase max_tokens, check for stop sequences' },
                    { label: 'Refuses to answer', value: 'Rephrase prompt, check for false positive safety triggers' },
                ]
            },
        ]
    },
    {
        id: 'costs',
        title: 'Cost Optimization',
        sections: [
            {
                title: 'Quick Cost Estimates',
                items: [
                    { label: '1 page of text', value: '~500 tokens' },
                    { label: '1 book (300 pages)', value: '~150K tokens' },
                    { label: '1 hour of transcript', value: '~10K tokens' },
                    { label: '1M API calls (short)', value: '~500M tokens' },
                ]
            },
            {
                title: 'Cost Reduction Strategies',
                items: [
                    { label: 'Model Tiering', value: 'Route simple queries to budget models, complex to flagship' },
                    { label: 'Prompt Caching', value: 'Use Anthropic/OpenAI prompt caching for repeated system prompts (50-90% off)' },
                    { label: 'Semantic Caching', value: 'Cache responses for similar queries. 100% savings on cache hits.' },
                    { label: 'Prompt Compression', value: 'Remove redundant context, use concise instructions' },
                    { label: 'Batching', value: 'Use batch APIs for non-real-time workloads (50% discount)' },
                    { label: 'Output Limits', value: 'Set appropriate max_tokens to avoid paying for unused capacity' },
                ]
            },
            {
                title: 'Monitoring',
                items: [
                    { label: 'Track per-request', value: 'Log tokens, latency, cost for every request' },
                    { label: 'Set budgets', value: 'Use provider spending limits, alerts at thresholds' },
                    { label: 'Analyze patterns', value: 'Find expensive queries, optimize or cache them' },
                ]
            },
        ]
    },
];

export const CheatSheetsView = () => {
    const [activeSheet, setActiveSheet] = useState<SheetId>('prompting');
    const [copiedCode, setCopiedCode] = useState<string | null>(null);

    const currentSheet = cheatSheets.find(s => s.id === activeSheet)!;

    const copyToClipboard = (code: string) => {
        navigator.clipboard.writeText(code);
        setCopiedCode(code);
        setTimeout(() => setCopiedCode(null), 2000);
    };

    return (
        <div className="min-h-screen bg-stone-50">
            <div className="bg-stone-900 text-white py-12 px-6">
                <div className="max-w-4xl mx-auto">
                    <div className="flex items-center gap-3 mb-2">
                        <FileText className="w-8 h-8 text-brand-400" />
                        <h1 className="text-3xl font-bold">Cheat Sheets</h1>
                    </div>
                    <p className="text-stone-400">Quick reference guides for AI engineering</p>
                </div>
            </div>

            <div className="max-w-4xl mx-auto px-6 py-8">
                {/* Sheet tabs */}
                <div className="flex flex-wrap gap-2 mb-8">
                    {cheatSheets.map(sheet => (
                        <button
                            key={sheet.id}
                            onClick={() => setActiveSheet(sheet.id)}
                            className={`px-4 py-2 rounded-xl font-medium transition-all ${
                                activeSheet === sheet.id 
                                    ? 'bg-stone-900 text-white' 
                                    : 'bg-white text-stone-600 border border-stone-200 hover:bg-stone-100'
                            }`}
                        >
                            {sheet.title}
                        </button>
                    ))}
                </div>

                {/* Sheet content */}
                <div className="space-y-8">
                    {currentSheet.sections.map(section => (
                        <div key={section.title}>
                            <h2 className="text-xl font-bold text-stone-900 mb-4">{section.title}</h2>
                            <div className="space-y-3">
                                {section.items.map(item => (
                                    <div key={item.label} className="bg-white border border-stone-200 rounded-xl p-4">
                                        <div className="flex items-start justify-between gap-4">
                                            <div className="flex-1">
                                                <h3 className="font-bold text-stone-900 mb-1">{item.label}</h3>
                                                <p className="text-stone-600 text-sm">{item.value}</p>
                                            </div>
                                        </div>
                                        {item.code && (
                                            <div className="mt-3 relative">
                                                <pre className="bg-stone-900 text-stone-300 text-sm p-3 rounded-lg overflow-x-auto">
                                                    <code>{item.code}</code>
                                                </pre>
                                                <button
                                                    onClick={() => copyToClipboard(item.code!)}
                                                    className="absolute top-2 right-2 p-1.5 bg-stone-700 hover:bg-stone-600 rounded text-stone-400 hover:text-white transition-colors"
                                                >
                                                    {copiedCode === item.code ? <Check size={14} /> : <Copy size={14} />}
                                                </button>
                                            </div>
                                        )}
                                    </div>
                                ))}
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};

