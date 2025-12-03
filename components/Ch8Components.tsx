import React, { useState } from 'react';
import { AlertTriangle, CheckCircle, Activity, Shield, Server, Zap } from 'lucide-react';

export const ProductionGap = () => {
    const gaps = [
        { prototype: 'Single user', production: 'Thousands concurrent', icon: '👤' },
        { prototype: 'Happy path', production: 'Edge cases everywhere', icon: '🛤️' },
        { prototype: 'Cost doesn\'t matter', production: 'Every token counts', icon: '💰' },
        { prototype: 'Latency flexible', production: 'Users expect < 2s', icon: '⏱️' },
        { prototype: 'Failures acceptable', production: '99.9% uptime required', icon: '🔧' },
    ];

    return (
        <div className="my-12">
            <div className="bg-stone-900 rounded-2xl p-6">
                <h4 className="text-white font-bold text-lg mb-6 text-center">The Production Gap</h4>
                <div className="grid md:grid-cols-2 gap-4">
                    <div className="bg-amber-900/30 rounded-xl p-4">
                        <h5 className="text-amber-400 font-medium mb-4 text-center">Prototype</h5>
                        <div className="space-y-3">
                            {gaps.map((gap, i) => (
                                <div key={i} className="flex items-center gap-3 text-amber-200">
                                    <span>{gap.icon}</span>
                                    <span>{gap.prototype}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                    <div className="bg-green-900/30 rounded-xl p-4">
                        <h5 className="text-green-400 font-medium mb-4 text-center">Production</h5>
                        <div className="space-y-3">
                            {gaps.map((gap, i) => (
                                <div key={i} className="flex items-center gap-3 text-green-200">
                                    <CheckCircle className="w-4 h-4 text-green-500" />
                                    <span>{gap.production}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export const Observability = () => {
    const [activeTab, setActiveTab] = useState<'logs' | 'metrics' | 'traces'>('logs');

    const content = {
        logs: {
            title: 'What to Log',
            items: [
                { category: 'Inputs', examples: ['User query (sanitized)', 'System prompt version', 'Model & parameters', 'Retrieved context'] },
                { category: 'Outputs', examples: ['Model response', 'Token counts', 'Latency breakdown', 'Tool calls made'] },
                { category: 'Metadata', examples: ['Request ID', 'User ID', 'Timestamp', 'Error details'] },
            ]
        },
        metrics: {
            title: 'Key Metrics',
            items: [
                { category: 'Latency', examples: ['P50, P95, P99 response time', 'Time to first token'] },
                { category: 'Throughput', examples: ['Requests per second', 'Concurrent users'] },
                { category: 'Errors', examples: ['Error rate by type', 'Timeout rate'] },
                { category: 'Cost', examples: ['Tokens per request', 'Cost per user'] },
            ]
        },
        traces: {
            title: 'Distributed Tracing',
            items: [
                { category: 'Example Trace', examples: ['API Gateway: 5ms', 'RAG Retrieval: 150ms', 'LLM Call: 2.3s', 'Total: 2.5s'] },
            ]
        }
    };

    const active = content[activeTab];

    return (
        <div className="my-12">
            <div className="flex gap-2 mb-6">
                <button onClick={() => setActiveTab('logs')} className={`px-4 py-2 rounded-lg font-medium transition-all ${activeTab === 'logs' ? 'bg-stone-900 text-white' : 'bg-stone-100 text-stone-600'}`}>
                    Logging
                </button>
                <button onClick={() => setActiveTab('metrics')} className={`px-4 py-2 rounded-lg font-medium transition-all ${activeTab === 'metrics' ? 'bg-stone-900 text-white' : 'bg-stone-100 text-stone-600'}`}>
                    <Activity className="w-4 h-4 inline mr-1" /> Metrics
                </button>
                <button onClick={() => setActiveTab('traces')} className={`px-4 py-2 rounded-lg font-medium transition-all ${activeTab === 'traces' ? 'bg-stone-900 text-white' : 'bg-stone-100 text-stone-600'}`}>
                    Tracing
                </button>
            </div>

            <div className="bg-white border border-stone-200 rounded-2xl p-6">
                <h4 className="font-bold text-stone-900 text-lg mb-6">{active.title}</h4>
                <div className="space-y-4">
                    {active.items.map((item) => (
                        <div key={item.category} className="p-4 bg-stone-100 rounded-xl">
                            <h5 className="font-medium text-stone-900 mb-2">{item.category}</h5>
                            <div className="flex flex-wrap gap-2">
                                {item.examples.map((ex, i) => (
                                    <span key={i} className="px-3 py-1 bg-white rounded-full text-sm text-stone-700 border border-stone-200">
                                        {ex}
                                    </span>
                                ))}
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};

export const Ch8Summary = () => {
    const takeaways = [
        { number: '01', title: 'Observability Is Essential', summary: 'Log inputs, outputs, latency, and costs. You can\'t fix what you can\'t see.', action: 'Set up structured logging and distributed tracing from day one.' },
        { number: '02', title: 'Plan for Failure', summary: 'Use retries, circuit breakers, and fallbacks. Everything fails eventually.', action: 'Implement exponential backoff and have a fallback model ready.' },
        { number: '03', title: 'Control Costs Proactively', summary: 'Model tiering, caching, and budget limits. Costs can explode without controls.', action: 'Set up per-user limits and alerts at 50%, 80%, 100% of budget.' },
        { number: '04', title: 'Security Is Non-Negotiable', summary: 'Validate inputs, filter outputs, manage secrets properly.', action: 'Use a secrets manager, never commit keys, implement rate limiting.' },
        { number: '05', title: 'Deploy Gradually', summary: 'Canary releases and feature flags catch issues before they affect everyone.', action: 'Start with 5% traffic to new versions, monitor before increasing.' },
        { number: '06', title: 'Monitor What Matters', summary: 'P95 latency, error rates, costs, user satisfaction. Set alerts.', action: 'Build dashboards for key metrics, page on-call for critical issues.' },
    ];

    return (
        <div className="my-12">
            <div className="text-center mb-10">
                <h3 className="text-2xl font-bold text-stone-900 mb-2">Chapter 8 Key Takeaways</h3>
                <p className="text-stone-500">Master these concepts to run reliable LLM applications in production</p>
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


