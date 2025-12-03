import React, { useState } from 'react';
import { Rocket, Clock, Zap, Trophy, ChevronRight } from 'lucide-react';

type Difficulty = 'beginner' | 'intermediate' | 'advanced';

interface Project {
    title: string;
    description: string;
    difficulty: Difficulty;
    timeEstimate: string;
    skills: string[];
    stack: string[];
    features: string[];
    extensions: string[];
}

const projects: Project[] = [
    // Beginner
    {
        title: 'Personal Knowledge Base Q&A',
        description: 'Build a RAG chatbot that answers questions about your own documents (notes, PDFs, bookmarks).',
        difficulty: 'beginner',
        timeEstimate: '1-2 days',
        skills: ['RAG basics', 'Embeddings', 'Vector search', 'Prompt engineering'],
        stack: ['OpenAI API', 'Chroma', 'LangChain', 'Streamlit'],
        features: [
            'Upload PDF/text files',
            'Chunk and embed documents',
            'Semantic search over your knowledge base',
            'Chat interface with source citations',
        ],
        extensions: ['Add web scraping for URLs', 'Implement hybrid search', 'Add conversation memory'],
    },
    {
        title: 'AI Writing Assistant',
        description: 'Create a tool that helps improve writing with suggestions for clarity, tone, and grammar.',
        difficulty: 'beginner',
        timeEstimate: '1-2 days',
        skills: ['Prompt engineering', 'Streaming', 'UI/UX for AI'],
        stack: ['OpenAI API', 'Next.js', 'Vercel AI SDK'],
        features: [
            'Text input with real-time suggestions',
            'Multiple improvement modes (formal, casual, concise)',
            'Track changes view showing edits',
            'Explanation for each suggestion',
        ],
        extensions: ['Add tone detection', 'Support multiple languages', 'Browser extension'],
    },
    {
        title: 'Flashcard Generator',
        description: 'Automatically generate study flashcards from any text, lecture notes, or textbook chapters.',
        difficulty: 'beginner',
        timeEstimate: '1 day',
        skills: ['Structured outputs', 'Prompt engineering', 'JSON mode'],
        stack: ['OpenAI API', 'React', 'Instructor'],
        features: [
            'Paste text or upload document',
            'Generate Q&A flashcards automatically',
            'Edit and refine generated cards',
            'Export to Anki format',
        ],
        extensions: ['Add spaced repetition', 'Generate from YouTube transcripts', 'Multi-language support'],
    },
    {
        title: 'Code Explainer',
        description: 'Build a tool that explains code snippets in plain English, line by line.',
        difficulty: 'beginner',
        timeEstimate: '1 day',
        skills: ['Prompt engineering', 'Code understanding', 'Streaming'],
        stack: ['Claude API', 'React', 'Monaco Editor'],
        features: [
            'Paste code in any language',
            'Get line-by-line explanations',
            'Adjust explanation detail level',
            'Highlight related code sections',
        ],
        extensions: ['Add code improvement suggestions', 'Generate documentation', 'Support entire repos'],
    },

    // Intermediate
    {
        title: 'Multi-Document Research Assistant',
        description: 'Build an assistant that can synthesize information across multiple documents to answer complex research questions.',
        difficulty: 'intermediate',
        timeEstimate: '3-5 days',
        skills: ['Advanced RAG', 'Query decomposition', 'Multi-step reasoning'],
        stack: ['OpenAI/Claude API', 'Pinecone', 'LlamaIndex', 'FastAPI'],
        features: [
            'Upload multiple documents (PDF, web pages)',
            'Decompose complex queries into sub-questions',
            'Retrieve and synthesize from multiple sources',
            'Generate reports with citations',
        ],
        extensions: ['Add fact verification', 'Implement iterative refinement', 'Export to various formats'],
    },
    {
        title: 'AI Code Review Bot',
        description: 'Create a GitHub bot that automatically reviews pull requests and suggests improvements.',
        difficulty: 'intermediate',
        timeEstimate: '3-5 days',
        skills: ['Function calling', 'GitHub API', 'Code analysis', 'Webhooks'],
        stack: ['Claude API', 'GitHub API', 'Node.js', 'Vercel'],
        features: [
            'Trigger on PR creation/update',
            'Analyze diff for issues',
            'Post inline comments with suggestions',
            'Summarize overall PR quality',
        ],
        extensions: ['Learn from feedback', 'Custom rule configuration', 'Security vulnerability detection'],
    },
    {
        title: 'Conversational Data Analyst',
        description: 'Build a natural language interface to query and visualize data from databases or CSV files.',
        difficulty: 'intermediate',
        timeEstimate: '3-5 days',
        skills: ['Text-to-SQL', 'Function calling', 'Data visualization'],
        stack: ['OpenAI API', 'SQLite/Postgres', 'Plotly', 'Streamlit'],
        features: [
            'Natural language to SQL conversion',
            'Execute queries safely',
            'Auto-generate visualizations',
            'Explain results in plain English',
        ],
        extensions: ['Support multiple data sources', 'Save and share dashboards', 'Scheduled reports'],
    },
    {
        title: 'Meeting Summarizer & Action Tracker',
        description: 'Transcribe meetings and automatically extract summaries, action items, and decisions.',
        difficulty: 'intermediate',
        timeEstimate: '2-3 days',
        skills: ['Audio transcription', 'Structured extraction', 'Summarization'],
        stack: ['Whisper API', 'OpenAI API', 'Instructor', 'React'],
        features: [
            'Upload audio/video or connect to Zoom',
            'Transcribe with speaker diarization',
            'Extract key decisions and action items',
            'Generate shareable summary document',
        ],
        extensions: ['Calendar integration', 'Follow-up reminders', 'Search across meetings'],
    },
    {
        title: 'Semantic Search Engine',
        description: 'Build a search engine that understands meaning, not just keywords, for a specific domain.',
        difficulty: 'intermediate',
        timeEstimate: '3-4 days',
        skills: ['Embeddings', 'Hybrid search', 'Reranking', 'Search UX'],
        stack: ['Cohere Embed + Rerank', 'Weaviate', 'FastAPI', 'React'],
        features: [
            'Index documents with embeddings',
            'Hybrid search (semantic + keyword)',
            'Rerank results for relevance',
            'Snippet highlighting and faceted filters',
        ],
        extensions: ['Query suggestions', 'Personalized ranking', 'Analytics dashboard'],
    },

    // Advanced
    {
        title: 'Autonomous Research Agent',
        description: 'Build an agent that can autonomously research topics by searching the web, reading papers, and synthesizing findings.',
        difficulty: 'advanced',
        timeEstimate: '1-2 weeks',
        skills: ['Agent architecture', 'Tool use', 'Planning', 'Web scraping'],
        stack: ['LangGraph', 'Tavily/Serper API', 'Browserbase', 'Claude API'],
        features: [
            'Accept research questions',
            'Plan research strategy',
            'Search web and academic sources',
            'Read and extract from sources',
            'Synthesize into comprehensive report',
        ],
        extensions: ['Add human-in-the-loop checkpoints', 'Citation verification', 'Iterative refinement based on feedback'],
    },
    {
        title: 'Multi-Agent Customer Support System',
        description: 'Create a team of specialized agents that collaborate to handle complex customer inquiries.',
        difficulty: 'advanced',
        timeEstimate: '1-2 weeks',
        skills: ['Multi-agent systems', 'Orchestration', 'Tool integration', 'State management'],
        stack: ['AutoGen/CrewAI', 'OpenAI API', 'Redis', 'FastAPI'],
        features: [
            'Triage agent for classification',
            'Specialist agents (billing, technical, sales)',
            'Supervisor agent for escalation',
            'Seamless handoffs between agents',
            'Human escalation when needed',
        ],
        extensions: ['Add learning from resolutions', 'Sentiment-based routing', 'Proactive outreach'],
    },
    {
        title: 'AI Coding Assistant (Copilot Clone)',
        description: 'Build an IDE extension that provides intelligent code completions and chat-based coding help.',
        difficulty: 'advanced',
        timeEstimate: '2-3 weeks',
        skills: ['Code completion', 'Context management', 'IDE extension development'],
        stack: ['Claude/DeepSeek API', 'VS Code Extension API', 'Tree-sitter'],
        features: [
            'Inline code completions',
            'Chat panel for questions',
            'Codebase-aware context',
            'Multi-file editing suggestions',
        ],
        extensions: ['Add code generation from comments', 'Test generation', 'Documentation generation'],
    },
    {
        title: 'RAG Evaluation Framework',
        description: 'Build a comprehensive framework to evaluate and improve RAG system quality.',
        difficulty: 'advanced',
        timeEstimate: '1-2 weeks',
        skills: ['Evaluation metrics', 'LLM-as-judge', 'Data generation', 'MLOps'],
        stack: ['Ragas', 'Langfuse', 'OpenAI API', 'Pytest'],
        features: [
            'Generate synthetic test questions',
            'Evaluate retrieval (recall, precision, MRR)',
            'Evaluate generation (faithfulness, relevance)',
            'A/B testing framework',
            'Regression detection in CI/CD',
        ],
        extensions: ['Add human evaluation workflows', 'Automated prompt optimization', 'Cost-quality tradeoff analysis'],
    },
    {
        title: 'Fine-Tuning Pipeline',
        description: 'Build an end-to-end pipeline for fine-tuning LLMs on custom data with evaluation.',
        difficulty: 'advanced',
        timeEstimate: '1-2 weeks',
        skills: ['Fine-tuning', 'Data preparation', 'Evaluation', 'MLOps'],
        stack: ['Axolotl/Unsloth', 'Hugging Face', 'Weights & Biases', 'Modal'],
        features: [
            'Data ingestion and cleaning',
            'Conversation formatting',
            'LoRA fine-tuning on GPU',
            'Evaluation on held-out set',
            'Model deployment and A/B testing',
        ],
        extensions: ['Add DPO/RLHF', 'Automated hyperparameter tuning', 'Continuous fine-tuning pipeline'],
    },
    {
        title: 'LLM Gateway & Observability Platform',
        description: 'Build a proxy that routes LLM requests with caching, fallbacks, and comprehensive logging.',
        difficulty: 'advanced',
        timeEstimate: '2-3 weeks',
        skills: ['API design', 'Caching', 'Observability', 'System design'],
        stack: ['FastAPI', 'Redis', 'PostgreSQL', 'Grafana'],
        features: [
            'Unified API for multiple providers',
            'Semantic caching',
            'Automatic fallbacks on failure',
            'Rate limiting and cost tracking',
            'Request/response logging',
            'Analytics dashboard',
        ],
        extensions: ['Add prompt versioning', 'A/B testing built-in', 'PII redaction'],
    },
];

const difficultyConfig = {
    beginner: { label: 'Beginner', color: 'text-green-600 bg-green-50 border-green-200', icon: '🌱' },
    intermediate: { label: 'Intermediate', color: 'text-amber-600 bg-amber-50 border-amber-200', icon: '🔥' },
    advanced: { label: 'Advanced', color: 'text-red-600 bg-red-50 border-red-200', icon: '🚀' },
};

export const ProjectIdeasView = () => {
    const [activeDifficulty, setActiveDifficulty] = useState<Difficulty | null>(null);
    const [expandedProject, setExpandedProject] = useState<string | null>(null);

    const filteredProjects = activeDifficulty 
        ? projects.filter(p => p.difficulty === activeDifficulty)
        : projects;

    return (
        <div className="min-h-screen bg-stone-50">
            <div className="bg-stone-900 text-white py-12 px-6">
                <div className="max-w-4xl mx-auto">
                    <div className="flex items-center gap-3 mb-2">
                        <Rocket className="w-8 h-8 text-brand-400" />
                        <h1 className="text-3xl font-bold">Project Ideas</h1>
                    </div>
                    <p className="text-stone-400">Hands-on projects to build your AI engineering portfolio</p>
                </div>
            </div>

            <div className="max-w-4xl mx-auto px-6 py-8">
                {/* Difficulty filters */}
                <div className="flex flex-wrap gap-2 mb-8">
                    <button
                        onClick={() => setActiveDifficulty(null)}
                        className={`px-4 py-2 rounded-xl font-medium transition-all ${
                            activeDifficulty === null 
                                ? 'bg-stone-900 text-white' 
                                : 'bg-white text-stone-600 border border-stone-200 hover:bg-stone-100'
                        }`}
                    >
                        All Projects ({projects.length})
                    </button>
                    {(['beginner', 'intermediate', 'advanced'] as Difficulty[]).map(diff => (
                        <button
                            key={diff}
                            onClick={() => setActiveDifficulty(diff)}
                            className={`px-4 py-2 rounded-xl font-medium transition-all flex items-center gap-2 ${
                                activeDifficulty === diff 
                                    ? 'bg-stone-900 text-white' 
                                    : 'bg-white text-stone-600 border border-stone-200 hover:bg-stone-100'
                            }`}
                        >
                            <span>{difficultyConfig[diff].icon}</span>
                            {difficultyConfig[diff].label} ({projects.filter(p => p.difficulty === diff).length})
                        </button>
                    ))}
                </div>

                {/* Projects */}
                <div className="space-y-4">
                    {filteredProjects.map((project) => {
                        const isExpanded = expandedProject === project.title;
                        const config = difficultyConfig[project.difficulty];
                        
                        return (
                            <div key={project.title} className="bg-white border border-stone-200 rounded-xl overflow-hidden">
                                <button
                                    onClick={() => setExpandedProject(isExpanded ? null : project.title)}
                                    className="w-full p-5 text-left hover:bg-stone-50 transition-colors"
                                >
                                    <div className="flex items-start justify-between gap-4">
                                        <div className="flex-1">
                                            <div className="flex items-center gap-2 mb-2">
                                                <span className={`text-xs font-medium px-2 py-0.5 rounded border ${config.color}`}>
                                                    {config.icon} {config.label}
                                                </span>
                                                <span className="text-xs text-stone-500 flex items-center gap-1">
                                                    <Clock size={12} />
                                                    {project.timeEstimate}
                                                </span>
                                            </div>
                                            <h3 className="font-bold text-stone-900 text-lg mb-1">{project.title}</h3>
                                            <p className="text-stone-600 text-sm">{project.description}</p>
                                        </div>
                                        <ChevronRight className={`w-5 h-5 text-stone-400 shrink-0 transition-transform ${isExpanded ? 'rotate-90' : ''}`} />
                                    </div>
                                </button>
                                
                                {isExpanded && (
                                    <div className="px-5 pb-5 border-t border-stone-100 pt-4 space-y-4">
                                        {/* Skills */}
                                        <div>
                                            <h4 className="text-sm font-semibold text-stone-500 uppercase tracking-wide mb-2 flex items-center gap-2">
                                                <Zap size={14} />
                                                Skills You'll Learn
                                            </h4>
                                            <div className="flex flex-wrap gap-2">
                                                {project.skills.map(skill => (
                                                    <span key={skill} className="text-sm px-2 py-1 bg-brand-50 text-brand-700 rounded">
                                                        {skill}
                                                    </span>
                                                ))}
                                            </div>
                                        </div>

                                        {/* Tech Stack */}
                                        <div>
                                            <h4 className="text-sm font-semibold text-stone-500 uppercase tracking-wide mb-2">
                                                Recommended Stack
                                            </h4>
                                            <div className="flex flex-wrap gap-2">
                                                {project.stack.map(tech => (
                                                    <span key={tech} className="text-sm px-2 py-1 bg-stone-100 text-stone-700 rounded">
                                                        {tech}
                                                    </span>
                                                ))}
                                            </div>
                                        </div>

                                        {/* Features */}
                                        <div>
                                            <h4 className="text-sm font-semibold text-stone-500 uppercase tracking-wide mb-2">
                                                Core Features
                                            </h4>
                                            <ul className="space-y-1">
                                                {project.features.map((feature, i) => (
                                                    <li key={i} className="text-sm text-stone-700 flex items-start gap-2">
                                                        <span className="text-brand-500 mt-1">✓</span>
                                                        {feature}
                                                    </li>
                                                ))}
                                            </ul>
                                        </div>

                                        {/* Extensions */}
                                        <div className="bg-stone-50 rounded-lg p-4">
                                            <h4 className="text-sm font-semibold text-stone-500 uppercase tracking-wide mb-2 flex items-center gap-2">
                                                <Trophy size={14} />
                                                Stretch Goals
                                            </h4>
                                            <ul className="space-y-1">
                                                {project.extensions.map((ext, i) => (
                                                    <li key={i} className="text-sm text-stone-600 flex items-start gap-2">
                                                        <span className="text-stone-400">→</span>
                                                        {ext}
                                                    </li>
                                                ))}
                                            </ul>
                                        </div>
                                    </div>
                                )}
                            </div>
                        );
                    })}
                </div>
            </div>
        </div>
    );
};


