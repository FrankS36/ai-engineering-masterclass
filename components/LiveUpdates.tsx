import React, { useState, useEffect } from 'react';
import { getLiveUpdates, SearchResult } from '../services/geminiService';
import { Radio, Loader2, Globe, ExternalLink, Newspaper } from 'lucide-react';

export const LiveUpdates = () => {
  const [data, setData] = useState<SearchResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [hasLoaded, setHasLoaded] = useState(false);

  const fetchUpdates = async () => {
    setLoading(true);
    const result = await getLiveUpdates("Large Language Models and AI Engineering");
    setData(result);
    setLoading(false);
    setHasLoaded(true);
  };

  // Helper to parse Gemini's markdown response into clean news items
  // We expect Gemini to return a list, but often it returns header/bullets.
  // We strip the markdown syntax for a cleaner UI.
  const parseNewsItems = (text: string) => {
      // Split by double newline or bullet points
      const rawItems = text.split(/(?:\r\n|\r|\n)(?:[\*\-]|\d+\.)\s+/);
      
      // Filter empty and header lines (lines starting with #)
      return rawItems
        .filter(item => item.trim().length > 10 && !item.trim().startsWith('#'))
        .map(item => {
             // Remove bold markdown if present around title-like structures
             const cleanText = item.replace(/\*\*(.*?)\*\*/, '$1')
                                   .replace(/^[\s\*\-\u2022]+/, '') // STRICT REGEX TO REMOVE BULLETS
                                   .trim();
                                   
             // Split into headline / body if possible (heuristic: first sentence is headline)
             const parts = cleanText.split(/[:\.]\s+/, 2);
             if (parts.length > 1 && parts[0].length < 100) {
                 return { title: parts[0], body: cleanText.substring(parts[0].length + 1) || cleanText };
             }
             return { title: "Update", body: cleanText };
        });
  };

  return (
    <div className="my-10 bg-white rounded-xl overflow-hidden shadow-lg border border-brand-100 ring-4 ring-brand-50/50">
      <div className="p-5 bg-gradient-to-r from-stone-800 to-stone-900 flex justify-between items-center text-white">
        <div className="flex items-center gap-3">
            <div className={`p-2 rounded-full bg-white/10 ${loading ? 'animate-pulse' : ''}`}>
                <Radio size={18} className="text-brand-500" />
            </div>
            <div>
                <h3 className="font-bold text-base leading-tight">Industry Pulse</h3>
                <p className="text-[11px] text-stone-400 opacity-80">Live Grounding via Google Search</p>
            </div>
        </div>
        {!hasLoaded && !loading && (
            <button 
                onClick={fetchUpdates}
                className="bg-brand-600 text-white px-3 py-1.5 rounded-lg text-xs font-bold uppercase tracking-wider transition-transform hover:scale-105 active:scale-95 shadow-sm hover:bg-brand-500"
            >
                Scan Now
            </button>
        )}
      </div>

      <div className="p-6 min-h-[120px]">
        {!hasLoaded && !loading && (
            <div className="text-center py-8">
                <Globe size={48} className="mx-auto text-stone-200 mb-3" />
                <p className="text-stone-500 text-sm">Tap "Scan Now" to fetch the latest AI Engineering news.</p>
            </div>
        )}

        {loading && (
            <div className="space-y-4">
                <div className="flex items-center gap-3 text-brand-600 text-sm font-medium justify-center py-8">
                    <Loader2 size={20} className="animate-spin" />
                    Accessing real-time knowledge graph...
                </div>
            </div>
        )}

        {hasLoaded && data && (
            <div className="animate-fade-in space-y-4">
                {parseNewsItems(data.text).map((news, idx) => (
                    <div key={idx} className="flex gap-4 p-4 rounded-xl bg-stone-50 border border-stone-100 hover:border-brand-200 transition-colors group">
                        <div className="shrink-0 mt-1">
                            <div className="w-8 h-8 rounded-lg bg-white border border-stone-200 flex items-center justify-center text-stone-400 shadow-sm group-hover:text-brand-600 group-hover:shadow-md transition-all">
                                <Newspaper size={16} />
                            </div>
                        </div>
                        <div>
                            <h4 className="font-bold text-stone-800 text-sm mb-1">{news.title}</h4>
                            <p className="text-stone-600 text-sm leading-relaxed">{news.body}</p>
                        </div>
                    </div>
                ))}
                
                {data.sources.length > 0 && (
                    <div className="mt-6 pt-4 border-t border-stone-100 flex flex-wrap gap-2">
                        {data.sources.map((source, idx) => (
                            <a 
                                key={idx} 
                                href={source.uri} 
                                target="_blank" 
                                rel="noopener noreferrer"
                                className="inline-flex items-center gap-1.5 px-2 py-1 bg-white border border-stone-200 rounded text-[10px] font-medium text-stone-500 hover:text-brand-600 hover:border-brand-200 transition-colors"
                            >
                                <span className="truncate max-w-[120px]">{source.title}</span>
                                <ExternalLink size={10} />
                            </a>
                        ))}
                    </div>
                )}
            </div>
        )}
      </div>
    </div>
  );
};