import React, { useState, useEffect } from 'react';
import { Play, Pause, RefreshCw, AlertTriangle, CheckCircle, Clock, Zap, Server, Code } from 'lucide-react';

export const APIAnatomy = () => {
    const [activeTab, setActiveTab] = useState<'request' | 'response'>('request');

    const requestExample = `{
  "model": "model-name",
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is 2+2?"}
  ],
  "temperature": 0.7,
  "max_tokens": 100,
  "stream": false
}`;

    const responseExample = `{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1699000000,
  "model": "model-name",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "2 + 2 equals 4."
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 8,
    "total_tokens": 33
  }
}`;

    const annotations = {
        request: [
            { key: 'model', desc: 'Which model to use (provider-specific)' },
            { key: 'messages', desc: 'Conversation history with roles' },
            { key: 'temperature', desc: 'Randomness: 0=deterministic, 1=creative' },
            { key: 'max_tokens', desc: 'Maximum response length' },
            { key: 'stream', desc: 'Whether to stream the response' }
        ],
        response: [
            { key: 'id', desc: 'Unique identifier for this completion' },
            { key: 'choices', desc: 'Array of completions (usually 1)' },
            { key: 'message', desc: 'The assistant\'s response' },
            { key: 'finish_reason', desc: 'Why generation stopped (stop/length/tool_calls)' },
            { key: 'usage', desc: 'Token counts for billing' }
        ]
    };

    return (
        <div className="my-12">
            <div className="flex gap-2 mb-6">
                <button onClick={() => setActiveTab('request')} className={`px-4 py-2 rounded-lg font-medium transition-all ${activeTab === 'request' ? 'bg-stone-900 text-white' : 'bg-stone-100 text-stone-600 hover:bg-stone-200'}`}>
                    <Server className="w-4 h-4 inline mr-2" />Request
                </button>
                <button onClick={() => setActiveTab('response')} className={`px-4 py-2 rounded-lg font-medium transition-all ${activeTab === 'response' ? 'bg-stone-900 text-white' : 'bg-stone-100 text-stone-600 hover:bg-stone-200'}`}>
                    <Code className="w-4 h-4 inline mr-2" />Response
                </button>
            </div>

            <div className="grid lg:grid-cols-2 gap-6">
                <div className="bg-stone-900 rounded-2xl p-6">
                    <h4 className="text-white font-bold mb-4">{activeTab === 'request' ? 'Request Body' : 'Response Body'}</h4>
                    <pre className="p-4 bg-stone-800 rounded-xl text-sm text-stone-300 font-mono whitespace-pre overflow-x-auto">
                        {activeTab === 'request' ? requestExample : responseExample}
                    </pre>
                </div>

                <div className="bg-white border border-stone-200 rounded-2xl p-6">
                    <h4 className="font-bold text-stone-900 mb-4">Key Fields</h4>
                    <div className="space-y-3">
                        {annotations[activeTab].map((item) => (
                            <div key={item.key} className="flex gap-3">
                                <code className="px-2 py-1 bg-brand-100 text-brand-700 rounded text-sm font-mono shrink-0">{item.key}</code>
                                <span className="text-stone-600 text-sm">{item.desc}</span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
};

export const StreamingDemo = () => {
    const [isStreaming, setIsStreaming] = useState(false);
    const [streamedText, setStreamedText] = useState('');
    const [nonStreamedText, setNonStreamedText] = useState('');
    const [streamTime, setStreamTime] = useState(0);
    const [nonStreamTime, setNonStreamTime] = useState(0);

    const fullText = "Streaming responses dramatically improve user experience. Instead of waiting for the entire response, users see tokens appear in real-time, making the application feel responsive and fast.";
    const words = fullText.split(' ');

    const runDemo = () => {
        setIsStreaming(true);
        setStreamedText('');
        setNonStreamedText('');
        setStreamTime(0);
        setNonStreamTime(0);

        // Simulate streaming - show words one at a time
        let wordIndex = 0;
        const streamStart = Date.now();
        const streamInterval = setInterval(() => {
            if (wordIndex < words.length) {
                setStreamedText(prev => prev + (wordIndex > 0 ? ' ' : '') + words[wordIndex]);
                setStreamTime(Date.now() - streamStart);
                wordIndex++;
            } else {
                clearInterval(streamInterval);
            }
        }, 80);

        // Simulate non-streaming - wait then show all at once
        const totalTime = words.length * 80;
        setTimeout(() => {
            setNonStreamedText(fullText);
            setNonStreamTime(totalTime);
            setIsStreaming(false);
        }, totalTime);
    };

    return (
        <div className="my-12">
            <div className="flex justify-center mb-6">
                <button
                    onClick={runDemo}
                    disabled={isStreaming}
                    className="px-6 py-3 bg-stone-900 text-white rounded-xl font-medium disabled:opacity-50 flex items-center gap-2"
                >
                    {isStreaming ? <><Pause className="w-4 h-4" /> Running...</> : <><Play className="w-4 h-4" /> Run Comparison</>}
                </button>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
                {/* Streaming */}
                <div className="bg-green-50 border border-green-200 rounded-2xl p-6">
                    <div className="flex items-center justify-between mb-4">
                        <h4 className="font-bold text-green-900 flex items-center gap-2">
                            <Zap className="w-5 h-5" /> Streaming
                        </h4>
                        {streamTime > 0 && (
                            <span className="text-sm text-green-700">First token: ~80ms</span>
                        )}
                    </div>
                    <div className="min-h-[120px] p-4 bg-white rounded-xl border border-green-200">
                        <p className="text-stone-800">{streamedText}<span className="animate-pulse">|</span></p>
                    </div>
                </div>

                {/* Non-streaming */}
                <div className="bg-stone-100 border border-stone-200 rounded-2xl p-6">
                    <div className="flex items-center justify-between mb-4">
                        <h4 className="font-bold text-stone-900 flex items-center gap-2">
                            <Clock className="w-5 h-5" /> Non-Streaming
                        </h4>
                        {nonStreamTime > 0 && (
                            <span className="text-sm text-stone-600">Wait: {nonStreamTime}ms</span>
                        )}
                    </div>
                    <div className="min-h-[120px] p-4 bg-white rounded-xl border border-stone-200">
                        {nonStreamedText ? (
                            <p className="text-stone-800">{nonStreamedText}</p>
                        ) : (
                            isStreaming && (
                                <div className="flex items-center justify-center h-full text-stone-400">
                                    <RefreshCw className="w-5 h-5 animate-spin mr-2" /> Waiting for complete response...
                                </div>
                            )
                        )}
                    </div>
                </div>
            </div>

            <div className="mt-6 p-4 bg-brand-50 border border-brand-200 rounded-xl">
                <p className="text-sm text-brand-800">
                    <strong>Key insight:</strong> Both take the same total time, but streaming shows output immediately. 
                    Users perceive streaming as much faster because they see progress from the start.
                </p>
            </div>
        </div>
    );
};

export const FunctionCallingVisual = () => {
    const [step, setStep] = useState(0);

    const steps = [
        { title: 'Define Tools', desc: 'You specify available functions with names, descriptions, and parameter schemas', code: '{ name: "get_weather", parameters: { location: string } }' },
        { title: 'Send Request', desc: 'User asks a question that might need a tool', code: '"What\'s the weather in Paris?"' },
        { title: 'Model Decides', desc: 'Model chooses to call a tool (or respond directly)', code: 'tool_calls: [{ name: "get_weather", args: { location: "Paris" } }]' },
        { title: 'Execute Function', desc: 'You run the actual function with provided arguments', code: 'get_weather("Paris") → { temp: 18, conditions: "Sunny" }' },
        { title: 'Return Result', desc: 'Send function result back to the model', code: '{ role: "tool", content: "18°C, Sunny" }' },
        { title: 'Final Response', desc: 'Model generates response using the tool result', code: '"The weather in Paris is 18°C and sunny!"' }
    ];

    return (
        <div className="my-12">
            <div className="bg-stone-900 rounded-2xl p-6 text-white">
                <h4 className="font-bold mb-6">The Tool Use Loop</h4>

                {/* Step indicators */}
                <div className="flex gap-2 mb-8 overflow-x-auto pb-2">
                    {steps.map((s, i) => (
                        <button
                            key={i}
                            onClick={() => setStep(i)}
                            className={`flex items-center gap-2 px-4 py-2 rounded-lg whitespace-nowrap transition-all ${
                                step === i ? 'bg-brand-500 text-white' : 'bg-stone-800 text-stone-400 hover:bg-stone-700'
                            }`}
                        >
                            <span className="w-6 h-6 rounded-full bg-stone-700 flex items-center justify-center text-xs font-bold">
                                {i + 1}
                            </span>
                            {s.title}
                        </button>
                    ))}
                </div>

                {/* Current step detail */}
                <div className="p-6 bg-stone-800 rounded-xl">
                    <h5 className="text-lg font-bold mb-2">{steps[step].title}</h5>
                    <p className="text-stone-400 mb-4">{steps[step].desc}</p>
                    <pre className="p-4 bg-stone-900 rounded-lg text-sm font-mono text-brand-400">
                        {steps[step].code}
                    </pre>
                </div>

                {/* Navigation */}
                <div className="flex justify-between mt-6">
                    <button
                        onClick={() => setStep(Math.max(0, step - 1))}
                        disabled={step === 0}
                        className="px-4 py-2 rounded-lg bg-stone-800 text-stone-300 disabled:opacity-50"
                    >
                        ← Previous
                    </button>
                    <button
                        onClick={() => setStep(Math.min(steps.length - 1, step + 1))}
                        disabled={step === steps.length - 1}
                        className="px-4 py-2 rounded-lg bg-brand-500 text-white disabled:opacity-50"
                    >
                        Next →
                    </button>
                </div>
            </div>
        </div>
    );
};

export const ErrorPatterns = () => {
    const [selectedError, setSelectedError] = useState<string>('rate_limit');

    const errors = [
        { id: 'rate_limit', code: '429', name: 'Rate Limit', cause: 'Too many requests per minute', solution: 'Exponential backoff, request queuing, upgrade tier', icon: <Clock className="w-5 h-5" /> },
        { id: 'context', code: '400', name: 'Context Length', cause: 'Input exceeds model limit', solution: 'Truncate messages, summarize history, use longer-context model', icon: <AlertTriangle className="w-5 h-5" /> },
        { id: 'server', code: '5xx', name: 'Server Error', cause: 'Provider infrastructure issues', solution: 'Retry with backoff, provider fallback, circuit breaker', icon: <Server className="w-5 h-5" /> },
        { id: 'timeout', code: 'Timeout', name: 'Request Timeout', cause: 'Response took too long', solution: 'Set appropriate timeouts, use streaming, reduce max_tokens', icon: <Clock className="w-5 h-5" /> },
    ];

    const active = errors.find(e => e.id === selectedError)!;

    const retryCode = `async function callWithRetry(fn, maxRetries = 3) {
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      if (error.status === 429) {
        // Exponential backoff: 1s, 2s, 4s...
        await sleep(Math.pow(2, attempt) * 1000);
      } else if (error.status >= 500) {
        await sleep(1000);
      } else {
        throw error; // Don't retry client errors
      }
    }
  }
  throw new Error("Max retries exceeded");
}`;

    return (
        <div className="my-12">
            <div className="grid lg:grid-cols-3 gap-6">
                {/* Error selector */}
                <div className="space-y-2">
                    {errors.map(e => (
                        <button
                            key={e.id}
                            onClick={() => setSelectedError(e.id)}
                            className={`w-full p-4 rounded-xl text-left transition-all flex items-center gap-3 ${
                                selectedError === e.id ? 'bg-red-100 border border-red-200 text-red-900' : 'bg-stone-100 text-stone-700 hover:bg-stone-200'
                            }`}
                        >
                            <span className={`${selectedError === e.id ? 'text-red-500' : 'text-stone-400'}`}>{e.icon}</span>
                            <div>
                                <span className="font-medium">{e.name}</span>
                                <span className="text-xs ml-2 opacity-60">({e.code})</span>
                            </div>
                        </button>
                    ))}
                </div>

                {/* Error detail */}
                <div className="lg:col-span-2 space-y-4">
                    <div className="bg-red-50 border border-red-200 rounded-2xl p-6">
                        <div className="flex items-center gap-3 mb-4">
                            <span className="text-red-500">{active.icon}</span>
                            <h4 className="font-bold text-red-900">{active.name} ({active.code})</h4>
                        </div>
                        <div className="space-y-3">
                            <div>
                                <span className="text-sm font-medium text-red-700">Cause:</span>
                                <p className="text-red-800">{active.cause}</p>
                            </div>
                            <div>
                                <span className="text-sm font-medium text-red-700">Solution:</span>
                                <p className="text-red-800">{active.solution}</p>
                            </div>
                        </div>
                    </div>

                    <div className="bg-stone-900 rounded-2xl p-6">
                        <h4 className="text-white font-bold mb-4">Retry Pattern</h4>
                        <pre className="p-4 bg-stone-800 rounded-xl text-xs text-stone-300 font-mono whitespace-pre overflow-x-auto">
                            {retryCode}
                        </pre>
                    </div>
                </div>
            </div>
        </div>
    );
};

export const Ch4Summary = () => {
    const takeaways = [
        { number: '01', title: 'APIs Share Common Patterns', summary: 'All providers use messages arrays, roles, and similar parameters. Learn the pattern once, apply everywhere.', action: 'Build abstractions that work across providers to avoid lock-in.' },
        { number: '02', title: 'Streaming Is Essential for UX', summary: 'Users perceive streaming as dramatically faster even when total time is the same.', action: 'Always use streaming for user-facing applications.' },
        { number: '03', title: 'Function Calling Enables Reliability', summary: 'Tool use gives you structured outputs, agent behaviors, and guaranteed schemas.', action: 'Use function calling for any output that needs to be parsed by code.' },
        { number: '04', title: 'Handle Errors Gracefully', summary: 'Rate limits, timeouts, and server errors are inevitable. Plan for them.', action: 'Implement exponential backoff, circuit breakers, and fallback strategies.' },
        { number: '05', title: 'Manage Context Proactively', summary: 'Context windows fill up fast. Have a strategy before you hit limits.', action: 'Track token usage and implement truncation/summarization early.' },
        { number: '06', title: 'Control Costs from Day One', summary: 'LLM costs can explode without monitoring. Model tiering and caching are essential.', action: 'Set budgets, implement caching, and route simple tasks to cheaper models.' },
    ];

    return (
        <div className="my-12">
            <div className="text-center mb-10">
                <h3 className="text-2xl font-bold text-stone-900 mb-2">Chapter 4 Key Takeaways</h3>
                <p className="text-stone-500">Master these patterns to build production-ready LLM applications</p>
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


