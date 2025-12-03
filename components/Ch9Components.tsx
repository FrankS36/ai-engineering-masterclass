import React, { useState } from 'react';
import { CheckCircle, AlertTriangle, Database, Cpu, ArrowRight } from 'lucide-react';

export const FinetuneDecision = () => {
    const [activeTab, setActiveTab] = useState<'good' | 'bad'>('good');

    const goodReasons = [
        'Consistent style/tone that\'s hard to prompt',
        'Domain-specific terminology or formats',
        'Reducing prompt length (bake examples into weights)',
        'Latency-critical applications (shorter prompts = faster)',
        'Proprietary behavior you don\'t want in prompts',
    ];

    const badReasons = [
        'Adding new knowledge (use RAG instead)',
        'One-off tasks (just prompt better)',
        'Experimenting (too slow for iteration)',
        '"It feels more AI-y" (not a real reason)',
    ];

    return (
        <div className="my-12">
            <div className="flex gap-2 mb-6">
                <button
                    onClick={() => setActiveTab('good')}
                    className={`px-4 py-2 rounded-lg font-medium transition-all ${activeTab === 'good' ? 'bg-stone-900 text-white' : 'bg-stone-100 text-stone-600'}`}
                >
                    <CheckCircle className="w-4 h-4 inline mr-2" />Good Reasons
                </button>
                <button
                    onClick={() => setActiveTab('bad')}
                    className={`px-4 py-2 rounded-lg font-medium transition-all ${activeTab === 'bad' ? 'bg-stone-900 text-white' : 'bg-stone-100 text-stone-600'}`}
                >
                    <AlertTriangle className="w-4 h-4 inline mr-2" />Bad Reasons
                </button>
            </div>

            <div className={`rounded-2xl p-6 ${activeTab === 'good' ? 'bg-brand-50 border border-brand-200' : 'bg-stone-100 border border-stone-200'}`}>
                <h4 className={`font-bold mb-4 ${activeTab === 'good' ? 'text-brand-900' : 'text-stone-900'}`}>
                    {activeTab === 'good' ? 'When Fine-Tuning Makes Sense' : 'When NOT to Fine-Tune'}
                </h4>
                <ul className="space-y-3">
                    {(activeTab === 'good' ? goodReasons : badReasons).map((reason, i) => (
                        <li key={i} className={`flex items-start gap-3 ${activeTab === 'good' ? 'text-brand-800' : 'text-stone-700'}`}>
                            {activeTab === 'good' ? (
                                <CheckCircle className="w-5 h-5 text-brand-500 shrink-0 mt-0.5" />
                            ) : (
                                <AlertTriangle className="w-5 h-5 text-stone-400 shrink-0 mt-0.5" />
                            )}
                            {reason}
                        </li>
                    ))}
                </ul>
            </div>

            <div className="mt-6 bg-stone-900 rounded-2xl p-6">
                <h4 className="text-white font-bold mb-4">Decision Framework</h4>
                <div className="space-y-3 text-stone-300 text-sm font-mono">
                    <div className="p-3 bg-stone-800 rounded-lg">
                        Is the issue <span className="text-brand-400">KNOWLEDGE</span> (facts, data)? → Use RAG
                    </div>
                    <div className="p-3 bg-stone-800 rounded-lg">
                        Is the issue <span className="text-brand-400">BEHAVIOR</span> (style, format)? → Try prompting first → Fine-tune if prompts too long
                    </div>
                    <div className="p-3 bg-stone-800 rounded-lg">
                        Is the issue <span className="text-brand-400">CAPABILITY</span> (can't do task)? → Try better base model
                    </div>
                </div>
            </div>
        </div>
    );
};

export const DataPrep = () => {
    const [activeStep, setActiveStep] = useState(0);

    const steps = [
        { name: 'Quality', desc: '100 high-quality examples often beat 10,000 noisy ones', tip: 'Each example should demonstrate exactly what you want' },
        { name: 'Format', desc: 'Conversation format with messages array (system, user, assistant)', tip: 'Match the format your API expects' },
        { name: 'Sources', desc: 'Production logs, expert annotation, synthetic generation, public datasets', tip: 'Real production data is often best' },
        { name: 'Cleaning', desc: 'Remove duplicates, filter low-quality, balance categories, validate format', tip: 'Check for PII and sensitive content' },
    ];

    return (
        <div className="my-12">
            <div className="grid lg:grid-cols-4 gap-3 mb-6">
                {steps.map((step, i) => (
                    <button
                        key={step.name}
                        onClick={() => setActiveStep(i)}
                        className={`p-4 rounded-xl text-left transition-all ${activeStep === i ? 'bg-stone-900 text-white' : 'bg-stone-100 text-stone-700 hover:bg-stone-200'}`}
                    >
                        <div className="flex items-center gap-2 mb-1">
                            <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${activeStep === i ? 'bg-brand-500 text-white' : 'bg-stone-300 text-stone-600'}`}>
                                {i + 1}
                            </span>
                            <span className="font-medium">{step.name}</span>
                        </div>
                    </button>
                ))}
            </div>

            <div className="bg-white border border-stone-200 rounded-2xl p-6">
                <h4 className="font-bold text-stone-900 text-lg mb-2">{steps[activeStep].name}</h4>
                <p className="text-stone-600 mb-4">{steps[activeStep].desc}</p>
                <div className="p-4 bg-brand-50 border border-brand-200 rounded-xl">
                    <span className="text-xs font-medium text-brand-600 uppercase">Tip</span>
                    <p className="text-brand-800 mt-1">{steps[activeStep].tip}</p>
                </div>
            </div>
        </div>
    );
};

export const Ch9Summary = () => {
    const takeaways = [
        { number: '01', title: 'Fine-Tune for Behavior, Not Knowledge', summary: 'Use RAG for facts and data. Fine-tuning is for consistent style, format, and behavior.', action: 'Ask: "Is this a knowledge problem or a behavior problem?"' },
        { number: '02', title: 'Quality Over Quantity', summary: '100 high-quality examples often outperform 10,000 noisy ones.', action: 'Invest in expert annotation rather than volume.' },
        { number: '03', title: 'Use PEFT Methods', summary: 'LoRA and QLoRA make fine-tuning practical. Full fine-tuning is rarely needed.', action: 'Start with LoRA rank 16. Increase only if needed.' },
        { number: '04', title: 'Prevent Catastrophic Forgetting', summary: 'Models can lose general capabilities while learning new tasks.', action: 'Include diverse examples and evaluate on general benchmarks.' },
        { number: '05', title: 'Evaluate Thoroughly', summary: 'Compare to base model, check for regression, test edge cases.', action: 'Build an eval set before you start fine-tuning.' },
        { number: '06', title: 'Consider the ROI', summary: 'Fine-tuning has upfront costs. Calculate if shorter prompts justify the investment.', action: 'Model the cost savings from reduced prompt length at scale.' },
    ];

    return (
        <div className="my-12">
            <div className="text-center mb-10">
                <h3 className="text-2xl font-bold text-stone-900 mb-2">Chapter 9 Key Takeaways</h3>
                <p className="text-stone-500">Master these concepts to fine-tune effectively</p>
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


