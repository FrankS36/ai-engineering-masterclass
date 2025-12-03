import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Search, X, BookOpen, FileText, HelpCircle, Lightbulb, Layout, Command } from 'lucide-react';
import { chapters } from '../constants';
import { ViewMode } from '../types';

interface SearchResult {
    type: 'chapter' | 'glossary' | 'quiz' | 'flashcard' | 'section';
    title: string;
    subtitle?: string;
    preview: string;
    chapterId?: string;
    viewMode: ViewMode;
    icon: React.ReactNode;
}

interface GlobalSearchProps {
    onNavigate: (chapterId: string, viewMode: ViewMode) => void;
}

// Glossary terms for search
const glossaryTerms = [
    { term: 'LLM', definition: 'Large Language Model - AI models trained on vast text data' },
    { term: 'RAG', definition: 'Retrieval-Augmented Generation - combining retrieval with generation' },
    { term: 'Embedding', definition: 'Dense vector representation of text' },
    { term: 'Token', definition: 'Basic unit of text processing in LLMs' },
    { term: 'Fine-tuning', definition: 'Adapting a pre-trained model to specific tasks' },
    { term: 'Prompt Engineering', definition: 'Crafting effective prompts for LLMs' },
    { term: 'Context Window', definition: 'Maximum tokens an LLM can process at once' },
    { term: 'Hallucination', definition: 'When LLMs generate false information' },
    { term: 'Vector Database', definition: 'Database optimized for similarity search' },
    { term: 'Chunking', definition: 'Splitting documents into smaller pieces' },
    { term: 'Temperature', definition: 'Parameter controlling output randomness' },
    { term: 'Agent', definition: 'LLM that can reason and use tools autonomously' },
    { term: 'Function Calling', definition: 'LLM capability to invoke external functions' },
    { term: 'LoRA', definition: 'Low-Rank Adaptation for efficient fine-tuning' },
    { term: 'Transformer', definition: 'Neural network architecture using attention' },
    { term: 'Attention', definition: 'Mechanism allowing models to focus on relevant parts' },
    { term: 'Inference', definition: 'Running a trained model to get predictions' },
    { term: 'Latency', definition: 'Time delay in model response' },
    { term: 'TTFT', definition: 'Time to First Token - initial response latency' },
    { term: 'Guardrails', definition: 'Safety mechanisms for LLM outputs' },
];

export const GlobalSearch: React.FC<GlobalSearchProps> = ({ onNavigate }) => {
    const [isOpen, setIsOpen] = useState(false);
    const [query, setQuery] = useState('');
    const [selectedIndex, setSelectedIndex] = useState(0);
    const inputRef = useRef<HTMLInputElement>(null);
    const resultsRef = useRef<HTMLDivElement>(null);

    // Build search index
    const searchResults = useMemo(() => {
        if (!query.trim()) return [];
        
        const q = query.toLowerCase();
        const results: SearchResult[] = [];

        // Search chapters
        chapters.forEach(chapter => {
            const titleMatch = chapter.title.toLowerCase().includes(q);
            const contentMatch = chapter.content.toLowerCase().includes(q);
            
            if (titleMatch || contentMatch) {
                let preview = '';
                if (contentMatch && !titleMatch) {
                    const idx = chapter.content.toLowerCase().indexOf(q);
                    const start = Math.max(0, idx - 50);
                    const end = Math.min(chapter.content.length, idx + 100);
                    preview = '...' + chapter.content.slice(start, end).replace(/[#*`]/g, '') + '...';
                } else {
                    preview = chapter.content.slice(0, 150).replace(/[#*`\n]/g, ' ').trim() + '...';
                }
                
                results.push({
                    type: 'chapter',
                    title: chapter.title,
                    preview,
                    chapterId: chapter.id,
                    viewMode: ViewMode.NOTES,
                    icon: <BookOpen size={16} className="text-brand-500" />
                });
            }

            // Search quizzes
            chapter.quizzes.forEach(quiz => {
                if (quiz.question.toLowerCase().includes(q)) {
                    results.push({
                        type: 'quiz',
                        title: quiz.question.slice(0, 80) + (quiz.question.length > 80 ? '...' : ''),
                        subtitle: chapter.title,
                        preview: quiz.explanation.slice(0, 100) + '...',
                        chapterId: chapter.id,
                        viewMode: ViewMode.QUIZ,
                        icon: <HelpCircle size={16} className="text-green-500" />
                    });
                }
            });

            // Search flashcards
            chapter.flashcards.forEach(card => {
                if (card.front.toLowerCase().includes(q) || card.back.toLowerCase().includes(q)) {
                    results.push({
                        type: 'flashcard',
                        title: card.front.slice(0, 80) + (card.front.length > 80 ? '...' : ''),
                        subtitle: chapter.title,
                        preview: card.back.slice(0, 100) + '...',
                        chapterId: chapter.id,
                        viewMode: ViewMode.FLASHCARDS,
                        icon: <Lightbulb size={16} className="text-amber-500" />
                    });
                }
            });
        });

        // Search glossary
        glossaryTerms.forEach(item => {
            if (item.term.toLowerCase().includes(q) || item.definition.toLowerCase().includes(q)) {
                results.push({
                    type: 'glossary',
                    title: item.term,
                    preview: item.definition,
                    viewMode: ViewMode.GLOSSARY,
                    icon: <FileText size={16} className="text-purple-500" />
                });
            }
        });

        return results.slice(0, 15);
    }, [query]);

    // Keyboard shortcut to open
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
                e.preventDefault();
                setIsOpen(true);
            }
            if (e.key === 'Escape') {
                setIsOpen(false);
            }
        };
        document.addEventListener('keydown', handleKeyDown);
        return () => document.removeEventListener('keydown', handleKeyDown);
    }, []);

    // Focus input when opened
    useEffect(() => {
        if (isOpen && inputRef.current) {
            inputRef.current.focus();
        }
    }, [isOpen]);

    // Keyboard navigation
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (!isOpen) return;
            
            if (e.key === 'ArrowDown') {
                e.preventDefault();
                setSelectedIndex(i => Math.min(i + 1, searchResults.length - 1));
            } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                setSelectedIndex(i => Math.max(i - 1, 0));
            } else if (e.key === 'Enter' && searchResults[selectedIndex]) {
                e.preventDefault();
                handleSelect(searchResults[selectedIndex]);
            }
        };
        document.addEventListener('keydown', handleKeyDown);
        return () => document.removeEventListener('keydown', handleKeyDown);
    }, [isOpen, searchResults, selectedIndex]);

    // Reset selection when results change
    useEffect(() => {
        setSelectedIndex(0);
    }, [searchResults]);

    const handleSelect = (result: SearchResult) => {
        onNavigate(result.chapterId || chapters[0].id, result.viewMode);
        setIsOpen(false);
        setQuery('');
    };

    if (!isOpen) {
        return (
            <button
                onClick={() => setIsOpen(true)}
                className="flex items-center gap-2 px-3 py-1.5 bg-stone-100 hover:bg-stone-200 rounded-lg text-stone-500 text-sm transition-colors"
            >
                <Search size={14} />
                <span className="hidden md:inline">Search</span>
                <kbd className="hidden md:flex items-center gap-0.5 px-1.5 py-0.5 bg-white rounded text-xs text-stone-400 border border-stone-200">
                    <Command size={10} />K
                </kbd>
            </button>
        );
    }

    return (
        <div className="fixed inset-0 z-[100] flex items-start justify-center pt-[15vh]">
            {/* Backdrop */}
            <div 
                className="absolute inset-0 bg-stone-900/50 backdrop-blur-sm"
                onClick={() => setIsOpen(false)}
            />
            
            {/* Modal */}
            <div className="relative w-full max-w-2xl mx-4 bg-white rounded-2xl shadow-2xl overflow-hidden">
                {/* Search input */}
                <div className="flex items-center gap-3 px-4 py-3 border-b border-stone-200">
                    <Search size={20} className="text-stone-400" />
                    <input
                        ref={inputRef}
                        type="text"
                        value={query}
                        onChange={e => setQuery(e.target.value)}
                        placeholder="Search chapters, quizzes, flashcards, glossary..."
                        className="flex-1 text-lg outline-none placeholder:text-stone-400"
                    />
                    <button 
                        onClick={() => setIsOpen(false)}
                        className="p-1 hover:bg-stone-100 rounded"
                    >
                        <X size={18} className="text-stone-400" />
                    </button>
                </div>

                {/* Results */}
                <div ref={resultsRef} className="max-h-[60vh] overflow-y-auto">
                    {query && searchResults.length === 0 && (
                        <div className="px-4 py-8 text-center text-stone-500">
                            No results found for "{query}"
                        </div>
                    )}
                    
                    {searchResults.map((result, idx) => (
                        <button
                            key={`${result.type}-${result.title}-${idx}`}
                            onClick={() => handleSelect(result)}
                            className={`w-full px-4 py-3 flex items-start gap-3 text-left transition-colors ${
                                idx === selectedIndex ? 'bg-brand-50' : 'hover:bg-stone-50'
                            }`}
                        >
                            <div className="mt-0.5">{result.icon}</div>
                            <div className="flex-1 min-w-0">
                                <div className="font-medium text-stone-900 truncate">{result.title}</div>
                                {result.subtitle && (
                                    <div className="text-xs text-stone-500 mb-1">{result.subtitle}</div>
                                )}
                                <div className="text-sm text-stone-500 line-clamp-2">{result.preview}</div>
                            </div>
                            <span className="text-xs text-stone-400 bg-stone-100 px-2 py-0.5 rounded capitalize">
                                {result.type}
                            </span>
                        </button>
                    ))}

                    {!query && (
                        <div className="px-4 py-6 text-center text-stone-500">
                            <p className="mb-2">Start typing to search across all content</p>
                            <p className="text-sm text-stone-400">
                                Searches chapters, quizzes, flashcards, and glossary
                            </p>
                        </div>
                    )}
                </div>

                {/* Footer */}
                <div className="px-4 py-2 bg-stone-50 border-t border-stone-200 flex items-center gap-4 text-xs text-stone-500">
                    <span className="flex items-center gap-1">
                        <kbd className="px-1.5 py-0.5 bg-white rounded border border-stone-200">↑↓</kbd>
                        Navigate
                    </span>
                    <span className="flex items-center gap-1">
                        <kbd className="px-1.5 py-0.5 bg-white rounded border border-stone-200">Enter</kbd>
                        Select
                    </span>
                    <span className="flex items-center gap-1">
                        <kbd className="px-1.5 py-0.5 bg-white rounded border border-stone-200">Esc</kbd>
                        Close
                    </span>
                </div>
            </div>
        </div>
    );
};


