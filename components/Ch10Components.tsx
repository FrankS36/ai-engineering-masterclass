import React, { useState } from 'react';
import { CheckCircle, AlertTriangle, Zap, Shield, TrendingUp, Users } from 'lucide-react';

export const UseCaseEval = () => {
    const [activeTab, setActiveTab] = useState<'good' | 'red'>('good');

    const goodIndicators = [
        { title: 'Tolerance for imperfection', desc: 'Tasks where 90% accuracy is valuable', examples: ['Content suggestions', 'Draft generation', 'Search/discovery'] },
        { title: 'High volume, low stakes', desc: 'Many small decisions', examples: ['Email categorization', 'Content moderation', 'Lead scoring'] },
        { title: 'Augmentation over automation', desc: 'AI assists humans', examples: ['Writing assistance', 'Code completion', 'Research summarization'] },
    ];

    const redFlags = [
        { title: 'Zero tolerance for errors', desc: 'Without human review', examples: ['Legal documents', 'Medical diagnoses'] },
        { title: 'Deterministic requirements', desc: 'Same input must give same output', examples: ['Financial calculations', 'Compliance checks'] },
        { title: 'Simple rules suffice', desc: 'If-then logic would work', examples: ['Basic routing', 'Simple validation'] },
    ];

    const items = activeTab === 'good' ? goodIndicators : redFlags;

    return (
        <div className="my-12">
            <div className="flex gap-2 mb-6">
                <button
                    onClick={() => setActiveTab('good')}
                    className={`px-4 py-2 rounded-lg font-medium transition-all ${activeTab === 'good' ? 'bg-stone-900 text-white' : 'bg-stone-100 text-stone-600'}`}
                >
                    <CheckCircle className="w-4 h-4 inline mr-2" />Good Indicators
                </button>
                <button
                    onClick={() => setActiveTab('red')}
                    className={`px-4 py-2 rounded-lg font-medium transition-all ${activeTab === 'red' ? 'bg-stone-900 text-white' : 'bg-stone-100 text-stone-600'}`}
                >
                    <AlertTriangle className="w-4 h-4 inline mr-2" />Red Flags
                </button>
            </div>

            <div className="grid md:grid-cols-3 gap-4">
                {items.map((item, i) => (
                    <div key={i} className={`rounded-2xl p-5 ${activeTab === 'good' ? 'bg-brand-50 border border-brand-200' : 'bg-stone-100 border border-stone-200'}`}>
                        <div className={`w-10 h-10 rounded-xl flex items-center justify-center mb-3 ${activeTab === 'good' ? 'bg-brand-100 text-brand-600' : 'bg-stone-200 text-stone-500'}`}>
                            {activeTab === 'good' ? <CheckCircle className="w-5 h-5" /> : <AlertTriangle className="w-5 h-5" />}
                        </div>
                        <h4 className={`font-bold mb-1 ${activeTab === 'good' ? 'text-brand-900' : 'text-stone-900'}`}>{item.title}</h4>
                        <p className={`text-sm mb-3 ${activeTab === 'good' ? 'text-brand-700' : 'text-stone-600'}`}>{item.desc}</p>
                        <div className="flex flex-wrap gap-1">
                            {item.examples.map((ex, j) => (
                                <span key={j} className={`text-xs px-2 py-1 rounded ${activeTab === 'good' ? 'bg-brand-100 text-brand-700' : 'bg-stone-200 text-stone-600'}`}>
                                    {ex}
                                </span>
                            ))}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

export const BuildBuy = () => {
    const [activeOption, setActiveOption] = useState<string>('api');

    const options = [
        {
            id: 'api',
            name: 'Use APIs',
            when: ['Speed to market matters', 'Use case is general', 'Scale is uncertain', 'ML expertise is limited'],
            tradeoffs: ['Ongoing costs', 'Data leaves your infrastructure', 'Provider dependency', 'Limited customization']
        },
        {
            id: 'opensource',
            name: 'Open Source',
            when: ['Data privacy is critical', 'Need full control', 'High volume makes API costly', 'Specific customization needed'],
            tradeoffs: ['Infrastructure complexity', 'ML expertise required', 'Responsible for updates', 'Higher upfront investment']
        },
        {
            id: 'custom',
            name: 'Build Custom',
            when: ['Unique task with no solution', 'Competitive differentiation', 'Massive scale justifies cost', 'Strong ML team'],
            tradeoffs: ['Highest cost and time', 'Ongoing maintenance', 'Risk of failure', 'Opportunity cost']
        },
    ];

    const active = options.find(o => o.id === activeOption)!;

    return (
        <div className="my-12">
            <div className="flex gap-2 mb-6">
                {options.map(o => (
                    <button
                        key={o.id}
                        onClick={() => setActiveOption(o.id)}
                        className={`px-4 py-2 rounded-xl font-medium transition-all ${activeOption === o.id ? 'bg-stone-900 text-white' : 'bg-stone-100 text-stone-600 hover:bg-stone-200'}`}
                    >
                        {o.name}
                    </button>
                ))}
            </div>

            <div className="grid md:grid-cols-2 gap-6">
                <div className="bg-brand-50 border border-brand-200 rounded-2xl p-6">
                    <h4 className="font-bold text-brand-900 mb-4">When to Choose</h4>
                    <ul className="space-y-2">
                        {active.when.map((item, i) => (
                            <li key={i} className="flex items-start gap-2 text-brand-800">
                                <CheckCircle className="w-4 h-4 text-brand-500 shrink-0 mt-0.5" />
                                {item}
                            </li>
                        ))}
                    </ul>
                </div>
                <div className="bg-stone-100 border border-stone-200 rounded-2xl p-6">
                    <h4 className="font-bold text-stone-900 mb-4">Trade-offs</h4>
                    <ul className="space-y-2">
                        {active.tradeoffs.map((item, i) => (
                            <li key={i} className="flex items-start gap-2 text-stone-700">
                                <AlertTriangle className="w-4 h-4 text-stone-400 shrink-0 mt-0.5" />
                                {item}
                            </li>
                        ))}
                    </ul>
                </div>
            </div>
        </div>
    );
};

export const Ch10Summary = () => {
    const takeaways = [
        { number: '01', title: 'Not Every Problem Needs AI', summary: 'Good AI use cases tolerate imperfection, involve high volume, and augment humans.', action: 'Ask: "Would 90% accuracy be valuable here?"' },
        { number: '02', title: 'Start with APIs', summary: 'Unless you have specific reasons for open source or custom, APIs get you to market fastest.', action: 'Prototype with APIs. Optimize later if needed.' },
        { number: '03', title: 'Plan for Costs', summary: 'AI costs can be 50-80% of feature revenue. Build monitoring from day one.', action: 'Model costs at scale before committing to a pricing model.' },
        { number: '04', title: 'Set Honest Expectations', summary: 'Users expect more than AI can deliver. Educate about capabilities AND limitations.', action: 'Show examples of both good and bad use cases in onboarding.' },
        { number: '05', title: 'Build Defensibility', summary: 'AI features are easy to copy. Moats come from data, feedback loops, and integration.', action: 'Design for compounding advantage, not just features.' },
        { number: '06', title: 'Plan for Change', summary: 'Models improve fast. Abstract your model selection and version your prompts.', action: 'Build flexibility to swap models and adopt new capabilities.' },
    ];

    return (
        <div className="my-12">
            <div className="text-center mb-10">
                <h3 className="text-2xl font-bold text-stone-900 mb-2">Chapter 10 Key Takeaways</h3>
                <p className="text-stone-500">Master these concepts to make smart AI product decisions</p>
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


