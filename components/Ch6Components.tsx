import React, { useState } from 'react';
import { Brain, Wrench, Eye, RotateCcw, Users, Shield, ArrowRight, CheckCircle, Play } from 'lucide-react';

export const AgentDefinition = () => {
    const capabilities = [
        { id: 'reason', icon: <Brain className="w-6 h-6" />, title: 'Reason', desc: 'Think about the problem and what steps are needed' },
        { id: 'plan', icon: <CheckCircle className="w-6 h-6" />, title: 'Plan', desc: 'Create a sequence of actions to accomplish the goal' },
        { id: 'act', icon: <Wrench className="w-6 h-6" />, title: 'Act', desc: 'Execute tools and APIs to interact with the world' },
        { id: 'observe', icon: <Eye className="w-6 h-6" />, title: 'Observe', desc: 'See results and adjust approach accordingly' },
    ];

    return (
        <div className="my-12">
            <div className="bg-stone-900 rounded-2xl p-8">
                <h4 className="text-white font-bold text-xl mb-2 text-center">What Makes an Agent?</h4>
                <p className="text-stone-400 text-center mb-8">An LLM that operates in a loop, making decisions based on results</p>

                <div className="grid md:grid-cols-4 gap-4">
                    {capabilities.map((cap, i) => (
                        <div key={cap.id} className="relative">
                            <div className="bg-stone-800 rounded-xl p-6 text-center">
                                <div className="w-12 h-12 bg-brand-500/20 rounded-xl flex items-center justify-center mx-auto mb-4 text-brand-400">
                                    {cap.icon}
                                </div>
                                <h5 className="text-white font-bold mb-2">{cap.title}</h5>
                                <p className="text-stone-400 text-sm">{cap.desc}</p>
                            </div>
                            {i < capabilities.length - 1 && (
                                <div className="hidden md:block absolute top-1/2 -right-2 transform -translate-y-1/2 z-10">
                                    <ArrowRight className="w-4 h-4 text-stone-600" />
                                </div>
                            )}
                        </div>
                    ))}
                </div>

                <div className="mt-8 flex justify-center">
                    <div className="flex items-center gap-2 text-stone-500">
                        <RotateCcw className="w-5 h-5" />
                        <span className="text-sm">Loop until task complete</span>
                    </div>
                </div>
            </div>
        </div>
    );
};

export const ReActLoop = () => {
    const [step, setStep] = useState(0);

    const trace = [
        { type: 'user', content: 'What\'s the weather in Paris and should I bring an umbrella?' },
        { type: 'think', content: 'I need to check the weather in Paris to answer this question.' },
        { type: 'act', content: 'get_weather(location="Paris")' },
        { type: 'observe', content: '{"temp": 15, "conditions": "Rain", "humidity": 85}' },
        { type: 'think', content: 'It\'s raining in Paris with 85% humidity. I should recommend an umbrella.' },
        { type: 'act', content: 'respond("It\'s 15°C and raining in Paris. Yes, bring an umbrella!")' },
    ];

    const typeColors: Record<string, string> = {
        user: 'bg-stone-100 border-stone-200 text-stone-800',
        think: 'bg-brand-100 border-brand-200 text-brand-800',
        act: 'bg-stone-200 border-stone-300 text-stone-800',
        observe: 'bg-stone-50 border-stone-200 text-stone-700',
    };

    const typeLabels: Record<string, string> = {
        user: 'USER',
        think: 'THINK',
        act: 'ACT',
        observe: 'OBSERVE',
    };

    return (
        <div className="my-12">
            <div className="bg-white border border-stone-200 rounded-2xl p-6">
                <div className="flex items-center justify-between mb-6">
                    <h4 className="font-bold text-stone-900">ReAct Trace Example</h4>
                    <div className="flex gap-2">
                        <button
                            onClick={() => setStep(Math.max(0, step - 1))}
                            disabled={step === 0}
                            className="px-3 py-1 rounded-lg bg-stone-100 text-stone-600 disabled:opacity-50"
                        >
                            ← Back
                        </button>
                        <button
                            onClick={() => setStep(Math.min(trace.length - 1, step + 1))}
                            disabled={step === trace.length - 1}
                            className="px-3 py-1 rounded-lg bg-stone-900 text-white disabled:opacity-50 flex items-center gap-1"
                        >
                            <Play className="w-3 h-3" /> Next
                        </button>
                    </div>
                </div>

                <div className="space-y-3">
                    {trace.slice(0, step + 1).map((item, i) => (
                        <div
                            key={i}
                            className={`p-4 rounded-xl border ${typeColors[item.type]} ${i === step ? 'ring-2 ring-brand-500' : ''}`}
                        >
                            <span className="text-xs font-bold uppercase opacity-70">{typeLabels[item.type]}</span>
                            <p className="mt-1 font-mono text-sm">{item.content}</p>
                        </div>
                    ))}
                </div>

                <div className="mt-6 flex justify-center gap-1">
                    {trace.map((_, i) => (
                        <button
                            key={i}
                            onClick={() => setStep(i)}
                            className={`w-2 h-2 rounded-full transition-all ${i === step ? 'bg-brand-500 w-4' : 'bg-stone-300'}`}
                        />
                    ))}
                </div>
            </div>
        </div>
    );
};

export const PlanningStrategies = () => {
    const [activeStrategy, setActiveStrategy] = useState<string>('direct');

    const strategies = [
        {
            id: 'direct',
            name: 'No Planning',
            desc: 'Just start executing. Works for simple tasks.',
            example: 'User: "What time is it in Tokyo?"\n→ get_time(timezone="Asia/Tokyo")\n→ "It\'s 3:45 PM in Tokyo"',
            pros: ['Fast', 'Simple', 'Low overhead'],
            cons: ['Fails on complex tasks', 'No error recovery']
        },
        {
            id: 'plan-execute',
            name: 'Plan-then-Execute',
            desc: 'Generate full plan upfront, then execute all steps.',
            example: 'PLAN:\n1. Search flights\n2. Select best option\n3. Search hotels\n4. Book both\n\nEXECUTE: [run each step]',
            pros: ['Clear structure', 'Easy to debug', 'Predictable'],
            cons: ['Can\'t adapt', 'Wasted work if plan wrong']
        },
        {
            id: 'iterative',
            name: 'Iterative Planning',
            desc: 'Plan a few steps, execute, replan based on results.',
            example: 'PLAN: [steps 1-3]\nEXECUTE: [steps 1-3]\nOBSERVE: Prices high\nREPLAN: Check alt dates',
            pros: ['Adaptive', 'Handles surprises', 'Efficient'],
            cons: ['More complex', 'May lose coherence']
        },
    ];

    const active = strategies.find(s => s.id === activeStrategy)!;

    return (
        <div className="my-12">
            <div className="flex gap-2 mb-6">
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

            <div className="grid lg:grid-cols-2 gap-6">
                <div className="bg-stone-900 rounded-2xl p-6">
                    <h4 className="text-white font-bold mb-2">{active.name}</h4>
                    <p className="text-stone-400 mb-4">{active.desc}</p>
                    <pre className="p-4 bg-stone-800 rounded-xl text-sm text-stone-300 font-mono whitespace-pre-wrap">
                        {active.example}
                    </pre>
                </div>

                <div className="space-y-4">
                    <div className="bg-green-50 border border-green-200 rounded-xl p-4">
                        <span className="text-xs font-medium text-green-600 uppercase">Pros</span>
                        <ul className="mt-2 space-y-1">
                            {active.pros.map((p, i) => (
                                <li key={i} className="text-green-800 text-sm flex items-center gap-2">
                                    <CheckCircle className="w-4 h-4" /> {p}
                                </li>
                            ))}
                        </ul>
                    </div>
                    <div className="bg-amber-50 border border-amber-200 rounded-xl p-4">
                        <span className="text-xs font-medium text-amber-600 uppercase">Cons</span>
                        <ul className="mt-2 space-y-1">
                            {active.cons.map((c, i) => (
                                <li key={i} className="text-amber-800 text-sm">• {c}</li>
                            ))}
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    );
};

export const MultiAgentPatterns = () => {
    const [activePattern, setActivePattern] = useState<string>('supervisor');

    const patterns = [
        {
            id: 'supervisor',
            name: 'Supervisor',
            desc: 'One agent coordinates and delegates to specialists',
            visual: ['Supervisor', '↓', '[Research] [Writing] [Review]'],
            useCase: 'Complex tasks requiring diverse expertise'
        },
        {
            id: 'debate',
            name: 'Debate',
            desc: 'Agents argue different perspectives',
            visual: ['Agent A (Pro)', '↔', 'Agent B (Con)', '↓', 'Synthesis'],
            useCase: 'Decision making, exploring trade-offs'
        },
        {
            id: 'pipeline',
            name: 'Pipeline',
            desc: 'Agents process in sequence',
            visual: ['Planner → Executor → Reviewer → Output'],
            useCase: 'Multi-stage workflows with handoffs'
        },
        {
            id: 'swarm',
            name: 'Swarm',
            desc: 'Agents work in parallel on subtasks',
            visual: ['Task', '↓', '[Agent 1] [Agent 2] [Agent 3]', '↓', 'Merge'],
            useCase: 'Parallelizable work, speed optimization'
        },
    ];

    const active = patterns.find(p => p.id === activePattern)!;

    return (
        <div className="my-12">
            <div className="grid lg:grid-cols-4 gap-3 mb-6">
                {patterns.map(p => (
                    <button
                        key={p.id}
                        onClick={() => setActivePattern(p.id)}
                        className={`p-4 rounded-xl text-left transition-all ${activePattern === p.id ? 'bg-stone-900 text-white' : 'bg-stone-100 text-stone-700 hover:bg-stone-200'}`}
                    >
                        <Users className={`w-5 h-5 mb-2 ${activePattern === p.id ? 'text-brand-400' : 'text-stone-400'}`} />
                        <span className="font-medium">{p.name}</span>
                    </button>
                ))}
            </div>

            <div className="bg-white border border-stone-200 rounded-2xl p-6">
                <h4 className="font-bold text-stone-900 text-lg mb-2">{active.name} Pattern</h4>
                <p className="text-stone-600 mb-6">{active.desc}</p>

                <div className="bg-stone-100 rounded-xl p-6 text-center mb-6">
                    {active.visual.map((line, i) => (
                        <div key={i} className="font-mono text-stone-700 py-1">{line}</div>
                    ))}
                </div>

                <div className="p-4 bg-brand-50 border border-brand-200 rounded-xl">
                    <span className="text-xs font-medium text-brand-600 uppercase">Best For</span>
                    <p className="text-brand-800 mt-1">{active.useCase}</p>
                </div>
            </div>
        </div>
    );
};

export const Ch6Summary = () => {
    const takeaways = [
        { number: '01', title: 'Agents Are Loops', summary: 'The key difference: agents reason, act, observe, and repeat until done.', action: 'Start with the ReAct pattern. Add complexity only when needed.' },
        { number: '02', title: 'Tool Design Is Critical', summary: 'Clear descriptions, focused scope, predictable outputs. The model chooses tools based on descriptions.', action: 'Write tool descriptions as if explaining to a new team member.' },
        { number: '03', title: 'Match Planning to Complexity', summary: 'Simple tasks need no planning. Complex tasks need iterative replanning.', action: 'Start with direct execution, add planning when you see failures.' },
        { number: '04', title: 'Memory Enables Continuity', summary: 'Short-term for session context, long-term for persistence, working memory for current task.', action: 'Implement summarization early—context windows fill fast.' },
        { number: '05', title: 'Safety Is Non-Negotiable', summary: 'Agents take real actions. Use least privilege, confirmation, sandboxing, and logging.', action: 'Require human approval for any action that can\'t be undone.' },
        { number: '06', title: 'Simpler Is Often Better', summary: 'Don\'t use agents when prompting or RAG would work. Agents add complexity.', action: 'Ask: "Could I solve this with a single LLM call?" If yes, do that.' },
    ];

    return (
        <div className="my-12">
            <div className="text-center mb-10">
                <h3 className="text-2xl font-bold text-stone-900 mb-2">Chapter 6 Key Takeaways</h3>
                <p className="text-stone-500">Master these concepts to build effective, safe agents</p>
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

