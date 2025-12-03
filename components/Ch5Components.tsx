import React, { useState } from 'react';
import { Database, Search, FileText, Cpu, ArrowRight, CheckCircle, AlertTriangle, Zap } from 'lucide-react';

export const RAGMotivation = () => {
    const [activeReason, setActiveReason] = useState<string>('cutoff');

    const reasons = [
        { id: 'cutoff', title: 'Knowledge Cutoff', icon: <Database className="w-5 h-5" />, problem: 'Models only know what was in training data (often months/years old)', solution: 'RAG retrieves current documents at query time', example: '"What were last quarter\'s sales?" → Retrieves latest financial reports' },
        { id: 'hallucination', title: 'Hallucination', icon: <AlertTriangle className="w-5 h-5" />, problem: 'Models confidently make up plausible-sounding but false information', solution: 'RAG grounds responses in real documents you control', example: '"What\'s our refund policy?" → Retrieves actual policy document' },
        { id: 'domain', title: 'Domain Knowledge', icon: <FileText className="w-5 h-5" />, problem: 'Models lack your proprietary data, internal docs, specialized knowledge', solution: 'RAG gives access to your private knowledge base', example: '"How do I configure the X feature?" → Retrieves internal documentation' },
        { id: 'citation', title: 'Citations', icon: <CheckCircle className="w-5 h-5" />, problem: 'Users want to verify answers and see sources', solution: 'RAG enables traceable, verifiable responses', example: 'Answer includes: "Source: Product Manual v2.3, page 45"' },
    ];

    const active = reasons.find(r => r.id === activeReason)!;

    return (
        <div className="my-12">
            <div className="grid lg:grid-cols-4 gap-3 mb-6">
                {reasons.map(r => (
                    <button
                        key={r.id}
                        onClick={() => setActiveReason(r.id)}
                        className={`p-4 rounded-xl text-left transition-all ${activeReason === r.id ? 'bg-stone-900 text-white' : 'bg-stone-100 text-stone-700 hover:bg-stone-200'}`}
                    >
                        <div className={`mb-2 ${activeReason === r.id ? 'text-brand-400' : 'text-stone-400'}`}>{r.icon}</div>
                        <span className="font-medium text-sm">{r.title}</span>
                    </button>
                ))}
            </div>

            <div className="bg-white border border-stone-200 rounded-2xl p-6">
                <h4 className="font-bold text-stone-900 text-lg mb-4 flex items-center gap-2">
                    {active.icon} {active.title}
                </h4>
                <div className="grid md:grid-cols-2 gap-6">
                    <div className="p-4 bg-red-50 border border-red-200 rounded-xl">
                        <span className="text-xs font-medium text-red-600 uppercase">Problem</span>
                        <p className="text-red-800 mt-1">{active.problem}</p>
                    </div>
                    <div className="p-4 bg-green-50 border border-green-200 rounded-xl">
                        <span className="text-xs font-medium text-green-600 uppercase">RAG Solution</span>
                        <p className="text-green-800 mt-1">{active.solution}</p>
                    </div>
                </div>
                <div className="mt-4 p-4 bg-stone-100 rounded-xl">
                    <span className="text-xs font-medium text-stone-500 uppercase">Example</span>
                    <p className="text-stone-700 mt-1 font-mono text-sm">{active.example}</p>
                </div>
            </div>
        </div>
    );
};

export const RAGPipeline = () => {
    const [activeStep, setActiveStep] = useState(0);

    const steps = [
        { name: 'Ingest', icon: <FileText className="w-5 h-5" />, desc: 'Load and parse documents from various sources', detail: 'PDFs, web pages, databases, APIs → Clean text' },
        { name: 'Chunk', icon: <Database className="w-5 h-5" />, desc: 'Split documents into smaller retrievable pieces', detail: '200-1000 tokens per chunk with overlap' },
        { name: 'Embed', icon: <Cpu className="w-5 h-5" />, desc: 'Convert chunks to semantic vectors', detail: 'embedding model → [0.02, -0.15, ...]' },
        { name: 'Store', icon: <Database className="w-5 h-5" />, desc: 'Save embeddings in vector database', detail: 'Pinecone, Chroma, pgvector, etc.' },
        { name: 'Retrieve', icon: <Search className="w-5 h-5" />, desc: 'Find relevant chunks for user query', detail: 'Embed query → Vector similarity search → Top K' },
        { name: 'Generate', icon: <Zap className="w-5 h-5" />, desc: 'LLM answers using retrieved context', detail: 'Context + Question → Grounded answer' },
    ];

    return (
        <div className="my-12">
            <div className="bg-stone-900 rounded-2xl p-6">
                {/* Pipeline visualization */}
                <div className="flex items-center justify-between mb-8 overflow-x-auto pb-4">
                    {steps.map((step, i) => (
                        <React.Fragment key={step.name}>
                            <button
                                onClick={() => setActiveStep(i)}
                                className={`flex flex-col items-center gap-2 p-4 rounded-xl transition-all min-w-[100px] ${activeStep === i ? 'bg-brand-500 text-white' : 'bg-stone-800 text-stone-400 hover:bg-stone-700'}`}
                            >
                                {step.icon}
                                <span className="text-sm font-medium">{step.name}</span>
                            </button>
                            {i < steps.length - 1 && (
                                <ArrowRight className="w-5 h-5 text-stone-600 shrink-0 mx-2" />
                            )}
                        </React.Fragment>
                    ))}
                </div>

                {/* Step detail */}
                <div className="p-6 bg-stone-800 rounded-xl">
                    <div className="flex items-center gap-3 mb-3">
                        <span className="text-brand-400">{steps[activeStep].icon}</span>
                        <h4 className="text-white font-bold text-lg">{steps[activeStep].name}</h4>
                    </div>
                    <p className="text-stone-300 mb-4">{steps[activeStep].desc}</p>
                    <div className="p-3 bg-stone-900 rounded-lg">
                        <code className="text-brand-400 text-sm">{steps[activeStep].detail}</code>
                    </div>
                </div>
            </div>
        </div>
    );
};

export const RetrievalStrategies = () => {
    const [activeStrategy, setActiveStrategy] = useState<string>('semantic');

    const strategies = [
        { 
            id: 'semantic', 
            name: 'Semantic Search', 
            desc: 'Find by meaning using vector similarity',
            pros: ['Understands synonyms', 'Handles paraphrases', 'Concept matching'],
            cons: ['May miss exact terms', 'Requires good embeddings'],
            example: 'Query: "car maintenance" → Finds: "automobile servicing guide"'
        },
        { 
            id: 'keyword', 
            name: 'Keyword (BM25)', 
            desc: 'Traditional text search by term frequency',
            pros: ['Exact matches', 'Good for names/IDs', 'Fast and simple'],
            cons: ['Misses synonyms', 'No semantic understanding'],
            example: 'Query: "error code E-401" → Finds: docs containing "E-401"'
        },
        { 
            id: 'hybrid', 
            name: 'Hybrid Search', 
            desc: 'Combine semantic and keyword for best coverage',
            pros: ['Best of both', 'Catches more relevant docs', 'Robust'],
            cons: ['More complex', 'Needs score fusion'],
            example: 'Combines vector + BM25 results with reciprocal rank fusion'
        },
        { 
            id: 'rerank', 
            name: 'Two-Stage + Rerank', 
            desc: 'Fast retrieval then precise reranking',
            pros: ['High precision', 'Scalable', 'Best quality'],
            cons: ['Slower', 'Additional cost'],
            example: 'Retrieve 50 → Rerank → Return top 5'
        },
    ];

    const active = strategies.find(s => s.id === activeStrategy)!;

    return (
        <div className="my-12">
            <div className="flex flex-wrap gap-2 mb-6">
                {strategies.map(s => (
                    <button
                        key={s.id}
                        onClick={() => setActiveStrategy(s.id)}
                        className={`px-4 py-2 rounded-xl font-medium transition-all ${activeStrategy === s.id ? 'bg-stone-900 text-white' : 'bg-stone-100 text-stone-600 hover:bg-stone-200'}`}
                    >
                        {s.name}
                    </button>
                ))}
            </div>

            <div className="bg-white border border-stone-200 rounded-2xl p-6">
                <h4 className="font-bold text-stone-900 text-lg mb-2">{active.name}</h4>
                <p className="text-stone-600 mb-6">{active.desc}</p>

                <div className="grid md:grid-cols-2 gap-4 mb-6">
                    <div className="p-4 bg-green-50 border border-green-200 rounded-xl">
                        <span className="text-xs font-medium text-green-600 uppercase">Pros</span>
                        <ul className="mt-2 space-y-1">
                            {active.pros.map((p, i) => (
                                <li key={i} className="text-green-800 text-sm flex items-center gap-2">
                                    <CheckCircle className="w-4 h-4" /> {p}
                                </li>
                            ))}
                        </ul>
                    </div>
                    <div className="p-4 bg-amber-50 border border-amber-200 rounded-xl">
                        <span className="text-xs font-medium text-amber-600 uppercase">Cons</span>
                        <ul className="mt-2 space-y-1">
                            {active.cons.map((c, i) => (
                                <li key={i} className="text-amber-800 text-sm flex items-center gap-2">
                                    <AlertTriangle className="w-4 h-4" /> {c}
                                </li>
                            ))}
                        </ul>
                    </div>
                </div>

                <div className="p-4 bg-stone-100 rounded-xl">
                    <span className="text-xs font-medium text-stone-500 uppercase">Example</span>
                    <p className="text-stone-700 mt-1 font-mono text-sm">{active.example}</p>
                </div>
            </div>
        </div>
    );
};

export const RAGEval = () => {
    const [activeMetric, setActiveMetric] = useState<'retrieval' | 'generation'>('retrieval');

    const retrievalMetrics = [
        { name: 'Recall@K', desc: '% of relevant docs in top K', formula: 'relevant_in_K / total_relevant', good: '> 0.8' },
        { name: 'Precision@K', desc: '% of top K that are relevant', formula: 'relevant_in_K / K', good: '> 0.6' },
        { name: 'MRR', desc: 'Avg reciprocal rank of first relevant', formula: '1/rank_of_first_relevant', good: '> 0.5' },
        { name: 'NDCG', desc: 'Quality considering position', formula: 'DCG / IDCG', good: '> 0.7' },
    ];

    const generationMetrics = [
        { name: 'Faithfulness', desc: 'Does answer reflect context?', method: 'LLM-as-judge or NLI', good: '> 0.9' },
        { name: 'Answer Relevance', desc: 'Is answer relevant to question?', method: 'LLM-as-judge', good: '> 0.8' },
        { name: 'Correctness', desc: 'Is the answer factually correct?', method: 'Ground truth comparison', good: '> 0.85' },
        { name: 'Completeness', desc: 'Does it fully answer the question?', method: 'Human eval or LLM', good: '> 0.75' },
    ];

    const metrics = activeMetric === 'retrieval' ? retrievalMetrics : generationMetrics;

    return (
        <div className="my-12">
            <div className="flex gap-2 mb-6">
                <button
                    onClick={() => setActiveMetric('retrieval')}
                    className={`px-4 py-2 rounded-lg font-medium transition-all ${activeMetric === 'retrieval' ? 'bg-stone-900 text-white' : 'bg-stone-100 text-stone-600'}`}
                >
                    <Search className="w-4 h-4 inline mr-2" />Retrieval Metrics
                </button>
                <button
                    onClick={() => setActiveMetric('generation')}
                    className={`px-4 py-2 rounded-lg font-medium transition-all ${activeMetric === 'generation' ? 'bg-stone-900 text-white' : 'bg-stone-100 text-stone-600'}`}
                >
                    <Zap className="w-4 h-4 inline mr-2" />Generation Metrics
                </button>
            </div>

            <div className="bg-white border border-stone-200 rounded-2xl overflow-hidden">
                <table className="w-full">
                    <thead className="bg-stone-100">
                        <tr>
                            <th className="text-left p-4 font-medium text-stone-700">Metric</th>
                            <th className="text-left p-4 font-medium text-stone-700">Description</th>
                            <th className="text-left p-4 font-medium text-stone-700">{activeMetric === 'retrieval' ? 'Formula' : 'Method'}</th>
                            <th className="text-left p-4 font-medium text-stone-700">Target</th>
                        </tr>
                    </thead>
                    <tbody>
                        {metrics.map((m, i) => (
                            <tr key={m.name} className={i % 2 === 0 ? 'bg-white' : 'bg-stone-50'}>
                                <td className="p-4 font-medium text-stone-900">{m.name}</td>
                                <td className="p-4 text-stone-600">{m.desc}</td>
                                <td className="p-4 font-mono text-sm text-stone-600">{'formula' in m ? m.formula : m.method}</td>
                                <td className="p-4">
                                    <span className="px-2 py-1 bg-green-100 text-green-700 rounded text-sm font-medium">{m.good}</span>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            <div className="mt-6 p-4 bg-brand-50 border border-brand-200 rounded-xl">
                <p className="text-sm text-brand-800">
                    <strong>Tip:</strong> Start with retrieval metrics. If retrieval is poor, no amount of prompt engineering will fix generation quality.
                </p>
            </div>
        </div>
    );
};

export const Ch5Summary = () => {
    const takeaways = [
        { number: '01', title: 'RAG Before Fine-Tuning', summary: 'RAG is faster to implement, easier to update, and provides citations. Try it first.', action: 'Only consider fine-tuning after RAG hits its limits.' },
        { number: '02', title: 'Chunking Is Critical', summary: 'Chunk size affects everything. Too small loses context; too large dilutes relevance.', action: 'Start with 500 tokens, 10% overlap. Experiment based on your content.' },
        { number: '03', title: 'Hybrid Search Wins', summary: 'Combining semantic and keyword search catches more relevant documents than either alone.', action: 'Implement hybrid search with reciprocal rank fusion.' },
        { number: '04', title: 'Reranking Improves Quality', summary: 'A two-stage approach (fast retrieval + precise reranking) gives best results.', action: 'Retrieve 20-50 candidates, rerank to top 5.' },
        { number: '05', title: 'Evaluate Systematically', summary: 'Build eval sets with real questions. Measure retrieval and generation separately.', action: 'Track Recall@K for retrieval, faithfulness for generation.' },
        { number: '06', title: 'Debug Retrieval First', summary: 'Most RAG failures are retrieval failures. Fix retrieval before tweaking prompts.', action: 'When answers are wrong, check what was retrieved before blaming the LLM.' },
    ];

    return (
        <div className="my-12">
            <div className="text-center mb-10">
                <h3 className="text-2xl font-bold text-stone-900 mb-2">Chapter 5 Key Takeaways</h3>
                <p className="text-stone-500">Master these concepts to build effective knowledge systems</p>
            </div>
            <div className="space-y-4">
                {takeaways.map((item) => (
                    <div key={item.number} className="bg-white border border-stone-200 rounded-2xl overflow-hidden hover:shadow-lg transition-shadow">
                        <div className="flex">
                            <div className="w-20 shrink-0 bg-stone-900 flex items-center justify-center">
                                <span className="text-2xl font-bold text-brand-500">{item.number}</span>
                            </div>
                            <div className="flex-1 p-5">
                                <h4 className="font-bold text-stone-900 text-lg mb-2">{item.title}</h4>
                                <p className="text-stone-600 mb-3">{item.summary}</p>
                                <div className="flex items-start gap-2 p-3 bg-brand-50 rounded-xl border border-brand-100">
                                    <span className="text-brand-600 font-bold shrink-0">→</span>
                                    <p className="text-sm text-brand-700 font-medium">{item.action}</p>
                                </div>
                            </div>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

