import React, { useState } from 'react';
import { Layout, ChevronRight, Server, Database, Shield, Zap, DollarSign, Clock, Users, MessageSquare, FileSearch, Code, Filter, Layers } from 'lucide-react';

type CaseStudyId = 'chatbot' | 'rag' | 'code-review' | 'moderation' | 'multi-tenant';

interface ArchitectureComponent {
    name: string;
    description: string;
    icon: React.ReactNode;
}

interface CaseStudy {
    id: CaseStudyId;
    title: string;
    description: string;
    icon: React.ReactNode;
    requirements: string[];
    components: ArchitectureComponent[];
    dataFlow: string[];
    keyDecisions: { decision: string; reasoning: string }[];
    scalingConsiderations: string[];
    estimations: { metric: string; value: string }[];
}

const caseStudies: CaseStudy[] = [
    {
        id: 'chatbot',
        title: 'Customer Support Chatbot',
        description: 'Design an AI-powered customer support system for an e-commerce platform handling 100K daily conversations.',
        icon: <MessageSquare className="w-6 h-6" />,
        requirements: [
            'Handle FAQs, order status, returns, and escalations',
            '< 2 second response time for 95% of queries',
            '24/7 availability with 99.9% uptime',
            'Seamless handoff to human agents',
            'Multi-language support',
        ],
        components: [
            { name: 'API Gateway', description: 'Rate limiting, auth, request routing', icon: <Server className="w-4 h-4" /> },
            { name: 'Intent Classifier', description: 'Route to FAQ, order, returns, or escalation flow', icon: <Filter className="w-4 h-4" /> },
            { name: 'RAG System', description: 'Knowledge base for FAQs and policies', icon: <Database className="w-4 h-4" /> },
            { name: 'Order Service', description: 'Integration with order management API', icon: <Layers className="w-4 h-4" /> },
            { name: 'LLM Orchestrator', description: 'Budget model for responses, fallback to flagship', icon: <Zap className="w-4 h-4" /> },
            { name: 'Conversation Memory', description: 'Redis for session state', icon: <Database className="w-4 h-4" /> },
            { name: 'Guardrails', description: 'Input/output filtering, PII detection', icon: <Shield className="w-4 h-4" /> },
            { name: 'Human Handoff', description: 'Queue system for agent escalation', icon: <Users className="w-4 h-4" /> },
        ],
        dataFlow: [
            'User message → API Gateway (auth, rate limit)',
            'Intent Classifier determines conversation type',
            'If FAQ → RAG retrieval → LLM generates response',
            'If Order → Fetch order data → LLM summarizes status',
            'Guardrails filter response → Return to user',
            'If confidence low or user requests → Human handoff',
        ],
        keyDecisions: [
            { decision: 'Budget model as primary', reasoning: 'Cost-effective for simple queries (80% of traffic), with flagship fallback for complex cases' },
            { decision: 'Intent classification before RAG', reasoning: 'Reduces unnecessary retrieval, improves latency, enables specialized handling' },
            { decision: 'Redis for conversation state', reasoning: 'Sub-millisecond reads, TTL for automatic cleanup, handles 100K+ concurrent sessions' },
            { decision: 'Async human handoff queue', reasoning: 'Prevents blocking, allows graceful degradation during peak times' },
        ],
        scalingConsiderations: [
            'Horizontal scaling of API servers behind load balancer',
            'Semantic cache for common queries (50%+ hit rate expected)',
            'Model tiering: route simple queries to smaller/faster models',
            'Queue-based processing for non-real-time features (analytics, training data)',
        ],
        estimations: [
            { metric: 'Daily conversations', value: '100,000' },
            { metric: 'Avg messages per conversation', value: '6' },
            { metric: 'Avg tokens per message', value: '150 in, 200 out' },
            { metric: 'Daily token usage', value: '~210M tokens' },
            { metric: 'Estimated daily cost (budget model)', value: '~$50-80' },
            { metric: 'P95 latency target', value: '< 2 seconds' },
        ],
    },
    {
        id: 'rag',
        title: 'Document Q&A System',
        description: 'Design a RAG system for a legal firm to search and answer questions across 1M+ documents.',
        icon: <FileSearch className="w-6 h-6" />,
        requirements: [
            'Search across 1M+ legal documents (contracts, cases, memos)',
            'Accurate answers with source citations',
            'Support complex multi-hop questions',
            'Access control based on user permissions',
            'Audit trail for compliance',
        ],
        components: [
            { name: 'Document Ingestion', description: 'PDF parsing, OCR, chunking pipeline', icon: <Layers className="w-4 h-4" /> },
            { name: 'Embedding Service', description: 'High-quality embeddings for document search', icon: <Zap className="w-4 h-4" /> },
            { name: 'Vector Database', description: 'Pinecone with metadata filtering', icon: <Database className="w-4 h-4" /> },
            { name: 'Hybrid Search', description: 'Vector + BM25 keyword search', icon: <Filter className="w-4 h-4" /> },
            { name: 'Reranker', description: 'Cohere Rerank for result ordering', icon: <Layers className="w-4 h-4" /> },
            { name: 'Query Engine', description: 'Query decomposition for complex questions', icon: <Zap className="w-4 h-4" /> },
            { name: 'LLM (Long Context)', description: 'Model with 100K+ context, strong reasoning', icon: <Zap className="w-4 h-4" /> },
            { name: 'Access Control', description: 'Document-level permissions in metadata', icon: <Shield className="w-4 h-4" /> },
        ],
        dataFlow: [
            'User query → Query decomposition (if complex)',
            'Each sub-query → Hybrid search (vector + keyword)',
            'Filter by user permissions in metadata',
            'Rerank top-50 → Select top-10',
            'Construct prompt with retrieved chunks',
            'LLM generates answer with citations',
            'Log query and response for audit',
        ],
        keyDecisions: [
            { decision: 'Hybrid search (vector + BM25)', reasoning: 'Legal docs have specific terminology; keyword search catches exact matches vectors might miss' },
            { decision: 'Long-context model', reasoning: 'Large context window handles more retrieved docs, strong at following citation instructions' },
            { decision: 'Document-level access in vector DB metadata', reasoning: 'Filter at retrieval time, not post-processing; faster and more secure' },
            { decision: 'Query decomposition for complex questions', reasoning: 'Multi-hop questions like "Compare clause X in contracts A and B" need separate retrievals' },
        ],
        scalingConsiderations: [
            'Batch ingestion pipeline with queue for new documents',
            'Incremental index updates (don\'t rebuild entire index)',
            'Cache frequent queries and their retrieved contexts',
            'Separate read replicas for search vs. ingestion',
        ],
        estimations: [
            { metric: 'Total documents', value: '1,000,000' },
            { metric: 'Avg document size', value: '10 pages (~5K tokens)' },
            { metric: 'Total chunks (512 tokens each)', value: '~10M chunks' },
            { metric: 'Vector DB storage', value: '~30GB (3072-dim embeddings)' },
            { metric: 'Daily queries', value: '10,000' },
            { metric: 'Embedding cost (one-time)', value: '~$650' },
        ],
    },
    {
        id: 'code-review',
        title: 'AI Code Review Assistant',
        description: 'Design a GitHub bot that automatically reviews PRs and suggests improvements.',
        icon: <Code className="w-6 h-6" />,
        requirements: [
            'Trigger on PR creation and updates',
            'Analyze code changes for bugs, style, security',
            'Post inline comments with suggestions',
            'Learn from accepted/rejected suggestions',
            'Support multiple languages and frameworks',
        ],
        components: [
            { name: 'GitHub Webhook Handler', description: 'Receive PR events, validate signatures', icon: <Server className="w-4 h-4" /> },
            { name: 'Diff Parser', description: 'Extract changed files and hunks', icon: <Layers className="w-4 h-4" /> },
            { name: 'Context Gatherer', description: 'Fetch related files, docs, past reviews', icon: <FileSearch className="w-4 h-4" /> },
            { name: 'RAG over Codebase', description: 'Style guides, patterns, past decisions', icon: <Database className="w-4 h-4" /> },
            { name: 'LLM (Code-focused)', description: 'Best for code understanding', icon: <Zap className="w-4 h-4" /> },
            { name: 'Comment Formatter', description: 'Structure suggestions as GitHub comments', icon: <MessageSquare className="w-4 h-4" /> },
            { name: 'Feedback Collector', description: 'Track accepted/rejected suggestions', icon: <Users className="w-4 h-4" /> },
            { name: 'Rate Limiter', description: 'Prevent spam, manage API costs', icon: <Shield className="w-4 h-4" /> },
        ],
        dataFlow: [
            'GitHub webhook → PR event received',
            'Fetch diff and changed files',
            'For each file: gather context (imports, related files)',
            'RAG: retrieve relevant style guides and patterns',
            'LLM analyzes changes, generates suggestions',
            'Filter low-confidence suggestions',
            'Post comments via GitHub API',
            'Track feedback on suggestions',
        ],
        keyDecisions: [
            { decision: 'Code-capable flagship model', reasoning: 'Best-in-class code understanding, handles large diffs well' },
            { decision: 'Confidence threshold for comments', reasoning: 'Only post suggestions above 0.8 confidence to avoid noise' },
            { decision: 'Incremental review on PR updates', reasoning: 'Only review new changes, not entire PR again' },
            { decision: 'Async processing with status updates', reasoning: 'Large PRs take time; show "Reviewing..." status' },
        ],
        scalingConsiderations: [
            'Queue-based processing for large PRs',
            'Parallel analysis of independent files',
            'Cache embeddings of unchanged files',
            'Rate limit per repository to manage costs',
        ],
        estimations: [
            { metric: 'PRs per day', value: '500' },
            { metric: 'Avg files changed per PR', value: '8' },
            { metric: 'Avg tokens per file review', value: '2K in, 500 out' },
            { metric: 'Daily token usage', value: '~10M tokens' },
            { metric: 'Estimated daily cost', value: '~$40-60' },
            { metric: 'Review time per PR', value: '30-90 seconds' },
        ],
    },
    {
        id: 'moderation',
        title: 'Content Moderation Pipeline',
        description: 'Design a system to moderate user-generated content at scale for a social platform.',
        icon: <Shield className="w-6 h-6" />,
        requirements: [
            'Process 10M+ posts per day',
            'Detect hate speech, spam, misinformation, NSFW',
            'Sub-second latency for real-time moderation',
            'Appeals process with human review',
            'Minimize false positives (user experience)',
        ],
        components: [
            { name: 'Ingestion Queue', description: 'Kafka for high-throughput content stream', icon: <Layers className="w-4 h-4" /> },
            { name: 'Fast Classifier', description: 'Lightweight model for initial triage', icon: <Zap className="w-4 h-4" /> },
            { name: 'LLM Analyzer', description: 'Budget model for nuanced cases', icon: <Zap className="w-4 h-4" /> },
            { name: 'Image/Video Analyzer', description: 'Vision model for media content', icon: <Filter className="w-4 h-4" /> },
            { name: 'Decision Engine', description: 'Combine signals, apply policies', icon: <Layers className="w-4 h-4" /> },
            { name: 'Action Service', description: 'Remove, warn, shadowban, or approve', icon: <Shield className="w-4 h-4" /> },
            { name: 'Appeals Queue', description: 'Human review for contested decisions', icon: <Users className="w-4 h-4" /> },
            { name: 'Feedback Loop', description: 'Retrain models on corrections', icon: <Database className="w-4 h-4" /> },
        ],
        dataFlow: [
            'New content → Ingestion queue',
            'Fast classifier (< 50ms) for obvious cases',
            'If uncertain → LLM analysis for nuance',
            'If media → Vision model analysis',
            'Decision engine combines all signals',
            'Action taken (approve/remove/flag)',
            'If appealed → Human review queue',
            'Corrections fed back for model improvement',
        ],
        keyDecisions: [
            { decision: 'Two-tier classification (fast + LLM)', reasoning: '90% of content is clearly safe/unsafe; only 10% needs expensive LLM analysis' },
            { decision: 'Async processing with optimistic approval', reasoning: 'Most content is fine; review async, remove later if needed' },
            { decision: 'Separate queues by content type', reasoning: 'Text, image, video have different latency profiles and models' },
            { decision: 'Human-in-the-loop for edge cases', reasoning: 'LLMs make mistakes on cultural context; humans handle appeals' },
        ],
        scalingConsiderations: [
            'Horizontal scaling of classifier workers',
            'Batch LLM calls for non-real-time review',
            'Cache embeddings for similar content detection',
            'Regional deployment for latency',
        ],
        estimations: [
            { metric: 'Daily posts', value: '10,000,000' },
            { metric: 'Fast classifier throughput', value: '50K/sec per instance' },
            { metric: 'LLM analysis rate', value: '~1M/day (10%)' },
            { metric: 'LLM cost per day', value: '~$200-400' },
            { metric: 'Human review queue', value: '~10K/day (0.1%)' },
            { metric: 'Target false positive rate', value: '< 0.1%' },
        ],
    },
    {
        id: 'multi-tenant',
        title: 'Multi-Tenant AI Platform',
        description: 'Design a platform that allows multiple customers to deploy custom AI assistants.',
        icon: <Users className="w-6 h-6" />,
        requirements: [
            'Each tenant has isolated data and custom prompts',
            'Support for custom knowledge bases (RAG)',
            'Usage-based billing per tenant',
            'Admin dashboard for configuration',
            'API access for integration',
        ],
        components: [
            { name: 'API Gateway', description: 'Tenant auth, routing, rate limiting', icon: <Server className="w-4 h-4" /> },
            { name: 'Tenant Service', description: 'Configuration, billing, usage tracking', icon: <Users className="w-4 h-4" /> },
            { name: 'Prompt Registry', description: 'Versioned prompts per tenant', icon: <Layers className="w-4 h-4" /> },
            { name: 'Vector DB (Multi-tenant)', description: 'Namespace isolation per tenant', icon: <Database className="w-4 h-4" /> },
            { name: 'LLM Router', description: 'Route to tenant-configured model', icon: <Zap className="w-4 h-4" /> },
            { name: 'Usage Metering', description: 'Track tokens, requests per tenant', icon: <DollarSign className="w-4 h-4" /> },
            { name: 'Admin Dashboard', description: 'Self-serve configuration UI', icon: <Layout className="w-4 h-4" /> },
            { name: 'Billing Service', description: 'Stripe integration, invoicing', icon: <DollarSign className="w-4 h-4" /> },
        ],
        dataFlow: [
            'Request → API Gateway (tenant auth via API key)',
            'Load tenant config (model, prompts, limits)',
            'If RAG enabled → Search tenant namespace',
            'Construct prompt from tenant template',
            'Route to configured LLM provider',
            'Meter usage (tokens, latency)',
            'Return response, update billing',
        ],
        keyDecisions: [
            { decision: 'Namespace isolation in vector DB', reasoning: 'Pinecone/Qdrant support namespaces; simpler than separate DBs per tenant' },
            { decision: 'Bring-your-own-key option', reasoning: 'Enterprise customers want to use their own API keys for compliance' },
            { decision: 'Usage-based + base fee pricing', reasoning: 'Aligns costs with value; base fee covers platform overhead' },
            { decision: 'Async webhook for long operations', reasoning: 'Document ingestion can take minutes; don\'t block API calls' },
        ],
        scalingConsiderations: [
            'Tenant-aware rate limiting (fair usage)',
            'Noisy neighbor protection (isolate heavy users)',
            'Shared infrastructure with logical isolation',
            'Regional deployment for data residency requirements',
        ],
        estimations: [
            { metric: 'Number of tenants', value: '500' },
            { metric: 'Avg requests per tenant/day', value: '5,000' },
            { metric: 'Total daily requests', value: '2.5M' },
            { metric: 'Avg margin per tenant', value: '30-40% on LLM costs' },
            { metric: 'Infrastructure cost', value: '~$5K/month base' },
            { metric: 'Break-even tenants', value: '~50 at $100/month avg' },
        ],
    },
];

const architecturePatterns = [
    {
        name: 'Simple Chat',
        description: 'Direct LLM call with conversation history',
        diagram: 'User → API → LLM → Response',
        useCase: 'Basic chatbots, Q&A interfaces',
        complexity: 'Low',
    },
    {
        name: 'RAG Pipeline',
        description: 'Retrieve relevant context before generation',
        diagram: 'User → Embed → Vector Search → Augment Prompt → LLM → Response',
        useCase: 'Knowledge bases, document Q&A, support bots',
        complexity: 'Medium',
    },
    {
        name: 'Agent Loop',
        description: 'LLM decides actions, executes tools, iterates',
        diagram: 'User → LLM (Plan) → Tool → Observe → LLM (Decide) → ... → Response',
        useCase: 'Research assistants, coding agents, automation',
        complexity: 'High',
    },
    {
        name: 'Router Pattern',
        description: 'Classify intent, route to specialized handlers',
        diagram: 'User → Classifier → [Handler A | Handler B | Handler C] → Response',
        useCase: 'Multi-function assistants, cost optimization',
        complexity: 'Medium',
    },
    {
        name: 'Evaluation Loop',
        description: 'Generate, evaluate, refine until quality threshold',
        diagram: 'User → LLM (Generate) → LLM (Evaluate) → [Refine | Accept] → Response',
        useCase: 'Content generation, code review, quality-critical apps',
        complexity: 'Medium',
    },
];

const productionConcerns = [
    { category: 'Reliability', items: ['Retry with exponential backoff', 'Fallback models/providers', 'Circuit breakers', 'Graceful degradation', 'Health checks'] },
    { category: 'Performance', items: ['Streaming responses', 'Semantic caching', 'Model tiering (cheap → expensive)', 'Async processing', 'Connection pooling'] },
    { category: 'Cost', items: ['Token usage tracking', 'Budget alerts', 'Prompt compression', 'Batch APIs for non-real-time', 'Cache hit optimization'] },
    { category: 'Security', items: ['Input sanitization', 'Output filtering', 'PII detection/redaction', 'Prompt injection defense', 'Rate limiting'] },
    { category: 'Observability', items: ['Request/response logging', 'Latency tracking (p50/p95/p99)', 'Error rate monitoring', 'Cost per request', 'Quality metrics'] },
];

export const SystemDesignView = () => {
    const [activeTab, setActiveTab] = useState<'patterns' | 'cases' | 'production'>('cases');
    const [selectedCase, setSelectedCase] = useState<CaseStudyId | null>(null);

    const activeCase = caseStudies.find(c => c.id === selectedCase);

    return (
        <div className="min-h-screen bg-stone-50">
            <div className="bg-stone-900 text-white py-12 px-6">
                <div className="max-w-5xl mx-auto">
                    <div className="flex items-center gap-3 mb-2">
                        <Layout className="w-8 h-8 text-brand-400" />
                        <h1 className="text-3xl font-bold">System Design</h1>
                    </div>
                    <p className="text-stone-400">Architecture patterns and case studies for production LLM applications</p>
                </div>
            </div>

            <div className="max-w-5xl mx-auto px-6 py-8">
                {/* Tabs */}
                <div className="flex gap-2 mb-8">
                    {[
                        { id: 'cases', label: 'Case Studies' },
                        { id: 'patterns', label: 'Architecture Patterns' },
                        { id: 'production', label: 'Production Checklist' },
                    ].map(tab => (
                        <button
                            key={tab.id}
                            onClick={() => { setActiveTab(tab.id as any); setSelectedCase(null); }}
                            className={`px-4 py-2 rounded-xl font-medium transition-all ${
                                activeTab === tab.id 
                                    ? 'bg-stone-900 text-white' 
                                    : 'bg-white text-stone-600 border border-stone-200 hover:bg-stone-100'
                            }`}
                        >
                            {tab.label}
                        </button>
                    ))}
                </div>

                {/* Case Studies */}
                {activeTab === 'cases' && !selectedCase && (
                    <div className="grid md:grid-cols-2 gap-4">
                        {caseStudies.map(study => (
                            <button
                                key={study.id}
                                onClick={() => setSelectedCase(study.id)}
                                className="bg-white border border-stone-200 rounded-2xl p-6 text-left hover:shadow-lg hover:border-brand-200 transition-all group"
                            >
                                <div className="flex items-start gap-4">
                                    <div className="w-12 h-12 rounded-xl bg-brand-100 text-brand-600 flex items-center justify-center shrink-0">
                                        {study.icon}
                                    </div>
                                    <div className="flex-1 min-w-0">
                                        <h3 className="font-bold text-stone-900 mb-1 group-hover:text-brand-600 transition-colors">
                                            {study.title}
                                        </h3>
                                        <p className="text-stone-600 text-sm line-clamp-2">{study.description}</p>
                                    </div>
                                    <ChevronRight className="w-5 h-5 text-stone-400 group-hover:text-brand-500 transition-colors shrink-0" />
                                </div>
                            </button>
                        ))}
                    </div>
                )}

                {/* Selected Case Study */}
                {activeTab === 'cases' && selectedCase && activeCase && (
                    <div>
                        <button
                            onClick={() => setSelectedCase(null)}
                            className="text-brand-600 hover:text-brand-700 font-medium mb-6 flex items-center gap-1"
                        >
                            ← Back to case studies
                        </button>

                        <div className="bg-white border border-stone-200 rounded-2xl p-6 mb-6">
                            <div className="flex items-start gap-4 mb-4">
                                <div className="w-14 h-14 rounded-xl bg-brand-100 text-brand-600 flex items-center justify-center">
                                    {activeCase.icon}
                                </div>
                                <div>
                                    <h2 className="text-2xl font-bold text-stone-900">{activeCase.title}</h2>
                                    <p className="text-stone-600">{activeCase.description}</p>
                                </div>
                            </div>
                        </div>

                        {/* Requirements */}
                        <div className="bg-white border border-stone-200 rounded-2xl p-6 mb-6">
                            <h3 className="font-bold text-stone-900 mb-4">Requirements</h3>
                            <ul className="space-y-2">
                                {activeCase.requirements.map((req, i) => (
                                    <li key={i} className="flex items-start gap-2 text-stone-700">
                                        <span className="text-brand-500 mt-1">•</span>
                                        {req}
                                    </li>
                                ))}
                            </ul>
                        </div>

                        {/* Components */}
                        <div className="bg-white border border-stone-200 rounded-2xl p-6 mb-6">
                            <h3 className="font-bold text-stone-900 mb-4">System Components</h3>
                            <div className="grid sm:grid-cols-2 gap-3">
                                {activeCase.components.map((comp, i) => (
                                    <div key={i} className="flex items-start gap-3 p-3 bg-stone-50 rounded-xl">
                                        <div className="w-8 h-8 rounded-lg bg-brand-100 text-brand-600 flex items-center justify-center shrink-0">
                                            {comp.icon}
                                        </div>
                                        <div>
                                            <h4 className="font-semibold text-stone-900 text-sm">{comp.name}</h4>
                                            <p className="text-stone-600 text-xs">{comp.description}</p>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Data Flow */}
                        <div className="bg-white border border-stone-200 rounded-2xl p-6 mb-6">
                            <h3 className="font-bold text-stone-900 mb-4">Data Flow</h3>
                            <div className="space-y-2">
                                {activeCase.dataFlow.map((step, i) => (
                                    <div key={i} className="flex items-center gap-3">
                                        <span className="w-6 h-6 rounded-full bg-brand-500 text-white text-xs font-bold flex items-center justify-center shrink-0">
                                            {i + 1}
                                        </span>
                                        <span className="text-stone-700 text-sm">{step}</span>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Key Decisions */}
                        <div className="bg-white border border-stone-200 rounded-2xl p-6 mb-6">
                            <h3 className="font-bold text-stone-900 mb-4">Key Design Decisions</h3>
                            <div className="space-y-4">
                                {activeCase.keyDecisions.map((decision, i) => (
                                    <div key={i} className="border-l-2 border-brand-500 pl-4">
                                        <h4 className="font-semibold text-stone-900">{decision.decision}</h4>
                                        <p className="text-stone-600 text-sm">{decision.reasoning}</p>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Estimations */}
                        <div className="bg-white border border-stone-200 rounded-2xl p-6 mb-6">
                            <h3 className="font-bold text-stone-900 mb-4">Capacity Estimations</h3>
                            <div className="grid sm:grid-cols-2 md:grid-cols-3 gap-4">
                                {activeCase.estimations.map((est, i) => (
                                    <div key={i} className="text-center p-4 bg-stone-50 rounded-xl">
                                        <div className="text-2xl font-bold text-brand-600">{est.value}</div>
                                        <div className="text-stone-600 text-sm">{est.metric}</div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Scaling */}
                        <div className="bg-stone-900 text-white rounded-2xl p-6">
                            <h3 className="font-bold mb-4">Scaling Considerations</h3>
                            <ul className="space-y-2">
                                {activeCase.scalingConsiderations.map((item, i) => (
                                    <li key={i} className="flex items-start gap-2 text-stone-300">
                                        <Zap className="w-4 h-4 text-brand-400 mt-0.5 shrink-0" />
                                        {item}
                                    </li>
                                ))}
                            </ul>
                        </div>
                    </div>
                )}

                {/* Architecture Patterns */}
                {activeTab === 'patterns' && (
                    <div className="space-y-4">
                        {architecturePatterns.map((pattern, i) => (
                            <div key={i} className="bg-white border border-stone-200 rounded-2xl p-6">
                                <div className="flex items-start justify-between gap-4 mb-4">
                                    <div>
                                        <h3 className="font-bold text-stone-900 text-lg">{pattern.name}</h3>
                                        <p className="text-stone-600">{pattern.description}</p>
                                    </div>
                                    <span className={`text-xs font-medium px-2 py-1 rounded ${
                                        pattern.complexity === 'Low' ? 'bg-green-100 text-green-700' :
                                        pattern.complexity === 'Medium' ? 'bg-amber-100 text-amber-700' :
                                        'bg-red-100 text-red-700'
                                    }`}>
                                        {pattern.complexity} Complexity
                                    </span>
                                </div>
                                <div className="bg-stone-900 text-stone-300 rounded-xl p-4 font-mono text-sm mb-4 overflow-x-auto">
                                    {pattern.diagram}
                                </div>
                                <div className="text-sm text-stone-600">
                                    <span className="font-medium text-stone-900">Best for:</span> {pattern.useCase}
                                </div>
                            </div>
                        ))}
                    </div>
                )}

                {/* Production Checklist */}
                {activeTab === 'production' && (
                    <div className="grid md:grid-cols-2 gap-6">
                        {productionConcerns.map((concern, i) => (
                            <div key={i} className="bg-white border border-stone-200 rounded-2xl p-6">
                                <h3 className="font-bold text-stone-900 mb-4 flex items-center gap-2">
                                    {concern.category === 'Reliability' && <Shield className="w-5 h-5 text-brand-500" />}
                                    {concern.category === 'Performance' && <Zap className="w-5 h-5 text-brand-500" />}
                                    {concern.category === 'Cost' && <DollarSign className="w-5 h-5 text-brand-500" />}
                                    {concern.category === 'Security' && <Shield className="w-5 h-5 text-brand-500" />}
                                    {concern.category === 'Observability' && <Clock className="w-5 h-5 text-brand-500" />}
                                    {concern.category}
                                </h3>
                                <ul className="space-y-2">
                                    {concern.items.map((item, j) => (
                                        <li key={j} className="flex items-center gap-2 text-stone-700 text-sm">
                                            <div className="w-4 h-4 rounded border border-stone-300" />
                                            {item}
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
};

