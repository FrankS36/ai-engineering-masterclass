import React, { useState } from 'react';
import { Copy, Check, Search } from 'lucide-react';

interface LookUpPromptProps {
    title?: string;
    prompt: string;
    why?: string;
    sources?: { label: string; href: string }[];
}

/**
 * Durable "go look this up" box. Use anywhere a fact ages quickly
 * (model IDs, list prices, context windows, leaderboard ranks).
 */
export const LookUpPrompt: React.FC<LookUpPromptProps> = ({
    title = 'Look up now',
    prompt,
    why,
    sources,
}) => {
    const [copied, setCopied] = useState(false);

    const copy = async () => {
        try {
            await navigator.clipboard.writeText(prompt);
            setCopied(true);
            setTimeout(() => setCopied(false), 1600);
        } catch {
            setCopied(false);
        }
    };

    return (
        <div className="my-6 rounded-2xl border border-brand-200 bg-gradient-to-br from-brand-50 to-stone-50 p-5">
            <div className="flex items-start gap-3">
                <div className="shrink-0 mt-0.5 p-2 rounded-lg bg-brand-100 text-brand-700">
                    <Search className="w-4 h-4" />
                </div>
                <div className="flex-1 min-w-0">
                    <p className="text-xs font-bold uppercase tracking-wider text-brand-700 mb-1">{title}</p>
                    {why && <p className="text-sm text-stone-600 mb-3">{why}</p>}
                    <blockquote className="text-sm text-stone-800 leading-relaxed font-medium border-l-2 border-brand-400 pl-3">
                        {prompt}
                    </blockquote>
                    {sources && sources.length > 0 && (
                        <div className="flex flex-wrap gap-2 mt-3">
                            {sources.map((s) => (
                                <a
                                    key={s.href}
                                    href={s.href}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="text-xs font-medium text-brand-700 hover:text-brand-800 underline underline-offset-2"
                                >
                                    {s.label}
                                </a>
                            ))}
                        </div>
                    )}
                    <button
                        type="button"
                        onClick={copy}
                        className="mt-3 inline-flex items-center gap-1.5 text-xs font-semibold text-stone-600 hover:text-stone-900"
                    >
                        {copied ? <Check className="w-3.5 h-3.5 text-green-600" /> : <Copy className="w-3.5 h-3.5" />}
                        {copied ? 'Copied' : 'Copy prompt'}
                    </button>
                </div>
            </div>
        </div>
    );
};
