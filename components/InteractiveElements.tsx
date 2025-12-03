import React, { useState, useRef, useEffect } from 'react';
import { 
  ArrowRight, 
  Terminal, 
  Layers, 
  Rocket, 
  Server, 
  Bot, 
  Play, 
  Zap, 
  Search, 
  PenTool, 
  ArrowLeftRight,
  Code2,
  Image as ImageIcon,
  Feather,
  GraduationCap,
  AlertTriangle,
  Clock,
  CheckCircle2,
  ChevronRight,
  ChevronLeft,
  Layout,
  Cpu,
  GitBranch,
  Gauge,
  Target,
  Shield,
  Coins,
  Brain,
  MessageSquare,
  DollarSign,
  Maximize,
  Lock,
  Filter,
  RefreshCw,
  Database,
  Lightbulb,
  X,
  Eye,
  TrendingUp,
  Calculator,
  Scale,
  MapPin,
  BarChart3,
  BookOpen,
  Package,
  HardDrive,
  Binary,
  Sparkles,
  Users,
  Network,
  Boxes,
  Mic,
  Video,
  FileText
} from 'lucide-react';
import { runQuickPrompt } from '../services/geminiService';

// --- REUSABLE BASE COMPONENTS ---

interface CardSelectorItem {
    id: number;
    name: string;
    label?: string;
    tagline: string;
    capability: string;
    examples: string[];
    description: string;
    characteristics: string[];
    insight: string;
}

interface CardSelectorProps {
    items: CardSelectorItem[];
    defaultSelected?: number;
    characteristicsTitle?: string;
    insightTitle?: string;
    insightIcon?: React.ReactNode;
}

export const CardSelector: React.FC<CardSelectorProps> = ({ 
    items, 
    defaultSelected = 0,
    characteristicsTitle = 'Key Characteristics',
    insightTitle = 'Key Insight',
    insightIcon
}) => {
    const [selected, setSelected] = useState<number>(defaultSelected);
    const selectedItem = items.find(item => item.id === selected)!;

    return (
        <div className="my-12">
            {/* Item selector */}
            <div className="flex gap-2 mb-8 overflow-x-auto pb-2">
                {items.map((item) => (
                    <button
                        key={item.id}
                        onClick={() => setSelected(item.id)}
                        className={`
                            flex-1 min-w-[200px] p-4 rounded-xl border-2 transition-all text-left
                            ${selected === item.id 
                                ? 'border-brand-500 bg-brand-50 shadow-lg' 
                                : 'border-stone-200 bg-white hover:border-stone-300'}
                        `}
                    >
                        <div className="flex items-center justify-between mb-1">
                            <span className="text-xs font-mono text-stone-400">{item.label || item.name}</span>
                            {selected === item.id && (
                                <span className="px-2 py-0.5 bg-brand-500 text-white text-[10px] font-bold rounded">SELECTED</span>
                            )}
                        </div>
                        <h4 className={`font-bold ${selected === item.id ? 'text-brand-700' : 'text-stone-800'}`}>
                            {item.name}
                        </h4>
                        <p className="text-sm text-stone-500 mt-1">{item.tagline}</p>
                    </button>
                ))}
            </div>
            
            {/* Selected item detail */}
            <div className="bg-white border border-stone-200 rounded-2xl overflow-hidden shadow-lg">
                {/* Header - always stone-900 */}
                <div className="p-6 bg-stone-900 text-white">
                    <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-mono opacity-70">{selectedItem.label || selectedItem.name}</span>
                        <div className="flex gap-2">
                            {selectedItem.examples.map(ex => (
                                <span key={ex} className="px-2 py-1 bg-white/20 rounded text-xs font-medium">{ex}</span>
                            ))}
                        </div>
                    </div>
                    <h3 className="text-2xl font-bold mb-1">{selectedItem.name}</h3>
                    <p className="text-xl text-white/80">{selectedItem.capability}</p>
                </div>
                
                {/* Content */}
                <div className="p-6">
                    <p className="text-stone-600 leading-relaxed mb-6">{selectedItem.description}</p>
                    
                    <div className="grid md:grid-cols-2 gap-6">
                        {/* Characteristics */}
                        <div>
                            <h4 className="font-bold text-stone-900 mb-3 flex items-center gap-2">
                                <CheckCircle2 size={16} className="text-brand-500" />
                                {characteristicsTitle}
                            </h4>
                            <div className="space-y-2">
                                {selectedItem.characteristics.map((char, i) => (
                                    <div key={i} className="flex gap-2 text-sm">
                                        <span className="text-brand-500">✓</span>
                                        <span className="text-stone-600">{char}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                        
                        {/* Insight */}
                        <div>
                            <h4 className="font-bold text-stone-900 mb-3 flex items-center gap-2">
                                {insightIcon || <Lightbulb size={16} className="text-brand-500" />}
                                {insightTitle}
                            </h4>
                            <p className="text-sm text-stone-600 bg-brand-50 border border-brand-200 rounded-xl p-4">
                                {selectedItem.insight}
                            </p>
                        </div>
                    </div>
                </div>
            </div>
            
            {/* Progress indicator */}
            <div className="flex items-center justify-center gap-2 mt-6">
                {items.map((item) => (
                    <button
                        key={item.id}
                        onClick={() => setSelected(item.id)}
                        className={`w-2 h-2 rounded-full transition-all ${
                            selected === item.id ? 'w-8 bg-brand-500' : 'bg-stone-300 hover:bg-stone-400'
                        }`}
                    />
                ))}
            </div>
        </div>
    );
};

// --- EXISTING COMPONENTS ---

export const WorkflowCompare = () => {
  const [mode, setMode] = useState<'traditional' | 'ai'>('ai');

  return (
    <div className="my-12 bg-white border border-stone-200 rounded-2xl overflow-hidden shadow-sm ring-1 ring-stone-100">
      <div className="bg-stone-50/80 backdrop-blur border-b border-stone-200 p-1.5 flex gap-1">
        <button
          onClick={() => setMode('traditional')}
          className={`flex-1 py-3 text-sm font-semibold rounded-xl transition-all duration-200 flex items-center justify-center gap-2 ${
            mode === 'traditional' 
                ? 'bg-white shadow-sm text-stone-800 ring-1 ring-black/5' 
                : 'text-stone-500 hover:text-stone-700 hover:bg-stone-100/50'
          }`}
        >
            <Server size={16} />
            Legacy ML Engineering
        </button>
        <button
          onClick={() => setMode('ai')}
          className={`flex-1 py-3 text-sm font-semibold rounded-xl transition-all duration-200 flex items-center justify-center gap-2 ${
            mode === 'ai' 
                ? 'bg-white shadow-sm text-brand-600 ring-1 ring-black/5' 
                : 'text-stone-500 hover:text-stone-700 hover:bg-stone-100/50'
          }`}
        >
          <Bot size={16} />
          Modern AI Engineering
        </button>
      </div>

      <div className="p-8 md:p-16 min-h-[360px] flex flex-col justify-center bg-white">
        {mode === 'traditional' ? (
          <div className="animate-fade-in space-y-10">
             <div className="flex flex-col md:flex-row gap-8 items-center justify-center relative">
                <Step icon={<Server size={24} />} label="Data Pipelines" sub="ETL & Cleaning" />
                <Arrow />
                <Step icon={<Layers size={24} />} label="Train Model" sub="Pytorch / TF" />
                <Arrow />
                <Step icon={<Rocket size={24} />} label="Deploy" sub="Inference API" />
             </div>
             
             <div className="bg-stone-50 rounded-xl p-6 border border-stone-200 text-sm text-stone-600 flex gap-5 items-center max-w-2xl mx-auto shadow-sm">
                <div className="font-mono text-[10px] uppercase tracking-wider text-stone-400 rotate-180" style={{ writingMode: 'vertical-rl' }}>Key Focus</div>
                <div className="h-full w-px bg-stone-200"></div>
                <p className="leading-relaxed">Differentiation is created by <span className="font-semibold text-stone-900">training a better model</span> than competitors. This requires massive compute resources, PhD-level talent, and months of iteration.</p>
             </div>
          </div>
        ) : (
          <div className="animate-fade-in space-y-10">
             <div className="flex flex-col md:flex-row gap-8 items-center justify-center relative">
                <Step icon={<Bot size={24} />} label="Foundation Model" sub="Pre-trained" active />
                <Arrow active />
                <Step icon={<Terminal size={24} />} label="AI Engineering" sub="Prompt / RAG / Agents" active />
                <Arrow active />
                <Step icon={<Rocket size={24} />} label="Product" sub="User Experience" active />
             </div>

             <div className="bg-brand-50/50 rounded-xl p-6 border border-brand-100 text-sm text-brand-900 flex gap-5 items-center max-w-2xl mx-auto shadow-sm">
                 <div className="font-mono text-[10px] uppercase tracking-wider text-brand-400 rotate-180" style={{ writingMode: 'vertical-rl' }}>Key Focus</div>
                 <div className="h-full w-px bg-brand-200"></div>
                <p className="leading-relaxed">Differentiation is created by <span className="font-semibold text-brand-700">the application layer</span>. The model is a commoditized utility. Speed to market and UX are the key metrics.</p>
             </div>
          </div>
        )}
      </div>
    </div>
  );
};

const Step = ({ icon, label, sub, active = false }: { icon: React.ReactNode, label: string, sub: string, active?: boolean }) => (
    <div className={`
        flex flex-col items-center gap-4 p-6 rounded-2xl border w-48 text-center transition-all duration-500 relative z-10
        ${active 
            ? 'bg-white border-brand-200 text-brand-700 shadow-[0_8px_30px_rgb(0,0,0,0.04)] ring-1 ring-brand-100' 
            : 'bg-white border-stone-200 text-stone-500 shadow-sm opacity-80'}
    `}>
        <div className={`p-3 rounded-xl ${active ? 'bg-brand-50 text-brand-600' : 'bg-stone-50 text-stone-400'}`}>
            {icon}
        </div>
        <div>
            <span className={`block text-sm font-bold ${active ? 'text-brand-900' : 'text-stone-700'}`}>{label}</span>
            <span className="text-[11px] uppercase tracking-wider font-semibold opacity-60 mt-1 block">{sub}</span>
        </div>
    </div>
);

const Arrow = ({ active = false }: { active?: boolean }) => (
    <div className="hidden md:block">
        <ArrowRight className={`w-6 h-6 ${active ? 'text-brand-300' : 'text-stone-200'}`} />
    </div>
);

export const ModelTypes = () => {
    const eras: CardSelectorItem[] = [
        { 
            id: 0, 
            name: "Language Models", 
            label: "2018-2020",
            tagline: "Single-Task Specialists",
            capability: "One model, one job",
            examples: ["BERT", "RoBERTa", "DistilBERT"],
            description: "Early language models were trained for specific tasks. Need sentiment analysis? Train a sentiment model. Need NER? Train a different model. Each use case required its own fine-tuned model.",
            characteristics: [
                "Task-specific fine-tuning required",
                "Limited to trained capabilities",
                "Smaller models (110M-340M params)",
                "Fast inference, narrow scope"
            ],
            insight: "You couldn't ask BERT to write a poem—it wasn't trained for that."
        },
        { 
            id: 1, 
            name: "Large Language Models", 
            label: "2020-2023",
            tagline: "Multi-Task Generalists",
            capability: "One model, many jobs",
            examples: ["GPT-3", "PaLM", "LLaMA"],
            description: "Scale unlocked emergent abilities. GPT-3 showed that a single model could write, summarize, translate, code, and reason—without task-specific training. Prompting replaced fine-tuning.",
            characteristics: [
                "In-context learning via prompts",
                "Emergent abilities at scale",
                "Massive models (175B+ params)",
                "API-first deployment"
            ],
            insight: "Text-only. Couldn't see images, hear audio, or maintain long context."
        },
        { 
            id: 2, 
            name: "Foundation Models", 
            label: "2023+",
            tagline: "General-Purpose Intelligence",
            capability: "One model, any modality, any task",
            examples: ["GPT", "Gemini", "Claude"],
            description: "The current frontier. Native multimodality (text, image, audio, video), million-token context windows, sophisticated reasoning, and tool use. These models can be steered to do almost anything through prompts.",
            characteristics: [
                "Native multimodal understanding",
                "1M+ token context windows",
                "Chain-of-thought reasoning",
                "Function calling & tool use"
            ],
            insight: "Cost, latency, and the need for careful evaluation at scale."
        }
    ];

    return (
        <CardSelector 
            items={eras} 
            defaultSelected={2}
            insightTitle="Limitations"
            insightIcon={<AlertTriangle size={16} className="text-amber-500" />}
        />
    );
};

export const UseCaseCarousel = () => {
    const scrollRef = useRef<HTMLDivElement>(null);

    const cases = [
        {
            title: "Coding Applications",
            desc: "Code generation, bug detection, test case creation, and documentation.",
            limit: "Requires expert code review and security scanning.",
            icon: <Code2 size={24} />,
            color: "blue"
        },
        {
            title: "Image & Video",
            desc: "Text-to-image generation, style transfer, storyboarding, and editing.",
            limit: "Consistency challenges and high computational cost.",
            icon: <ImageIcon size={24} />,
            color: "purple"
        },
        {
            title: "Writing Tasks",
            desc: "Content generation, editing, tone adjustment, and summarization.",
            limit: "Hallucination risk; requires fact-checking.",
            icon: <Feather size={24} />,
            color: "emerald"
        },
        {
            title: "Education",
            desc: "Personalized tutoring, assessment generation, and adaptive learning.",
            limit: "Concerns around academic integrity and over-reliance.",
            icon: <GraduationCap size={24} />,
            color: "amber"
        }
    ];

    const scroll = (direction: 'left' | 'right') => {
        if (scrollRef.current) {
            const scrollAmount = 300;
            scrollRef.current.scrollBy({
                left: direction === 'left' ? -scrollAmount : scrollAmount,
                behavior: 'smooth'
            });
        }
    };

    return (
        <div className="my-12 relative group">
            {/* Scroll Container */}
            <div 
                ref={scrollRef}
                className="flex gap-6 overflow-x-auto pb-8 snap-x snap-mandatory hide-scrollbar"
                style={{ scrollbarWidth: 'none', msOverflowStyle: 'none' }}
            >
                {cases.map((c, i) => (
                    <div 
                        key={i} 
                        className="snap-center shrink-0 w-[300px] bg-white rounded-2xl border border-stone-200 p-6 shadow-sm hover:shadow-md transition-all flex flex-col"
                    >
                        <div className={`w-12 h-12 rounded-xl flex items-center justify-center mb-4 bg-stone-50 text-brand-600`}>
                            {c.icon}
                        </div>
                        <h3 className="font-bold text-lg text-stone-900 mb-2">{c.title}</h3>
                        <p className="text-stone-600 text-sm leading-relaxed mb-6 flex-1">
                            {c.desc}
                        </p>
                        
                        <div className="bg-red-50 rounded-lg p-3 border border-red-100 flex gap-2 items-start">
                            <AlertTriangle size={14} className="text-red-500 shrink-0 mt-0.5" />
                            <p className="text-xs text-red-700 leading-tight">
                                <span className="font-semibold block mb-0.5">Limitation:</span>
                                {c.limit}
                            </p>
                        </div>
                    </div>
                ))}
                
                {/* Padding div for easier scrolling to end */}
                <div className="w-4 shrink-0" />
            </div>
            
             {/* Controls */}
             <div className="absolute top-1/2 -translate-y-1/2 w-full justify-between flex pointer-events-none px-2 opacity-0 group-hover:opacity-100 transition-opacity">
                 <button onClick={() => scroll('left')} className="pointer-events-auto p-3 rounded-full bg-white/90 backdrop-blur border border-stone-200 text-stone-600 hover:text-brand-600 shadow-md">
                     <ChevronLeft size={20} />
                 </button>
                 <button onClick={() => scroll('right')} className="pointer-events-auto p-3 rounded-full bg-white/90 backdrop-blur border border-stone-200 text-stone-600 hover:text-brand-600 shadow-md">
                     <ChevronRight size={20} />
                 </button>
             </div>
        </div>
    );
};

export const MilestoneTimeline = () => {
    const [selectedPhase, setSelectedPhase] = useState<number | null>(null);
    
    const phases = [
        {
            title: "Phase 1: Proof of Concept",
            duration: "2-4 weeks",
            goal: "Validate that AI can solve the core problem",
            items: ["Basic functionality", "Initial prompt engineering", "Rough accuracy assessment"],
            details: {
                description: "The PoC phase is about proving feasibility quickly. You're not building a product—you're answering the question: 'Can an LLM actually do this task well enough to be useful?'",
                keyActivities: [
                    "Select 2-3 representative use cases from your problem space",
                    "Write initial prompts and test against 20-50 examples",
                    "Measure baseline accuracy (aim for >70% to proceed)",
                    "Identify the hardest edge cases early"
                ],
                deliverables: ["Working prompt template", "Initial accuracy metrics", "Go/no-go recommendation"],
                commonMistakes: ["Over-engineering the solution", "Testing on too few examples", "Ignoring latency requirements"]
            }
        },
        {
            title: "Phase 2: Prototype",
            duration: "4-8 weeks",
            goal: "Build a functional system with evaluation infrastructure",
            items: ["Core features", "Evaluation pipeline setup", "Edge case identification"],
            details: {
                description: "The prototype phase transforms your PoC into something testable at scale. The key investment here is building robust evaluation—without it, you're flying blind.",
                keyActivities: [
                    "Build automated evaluation pipeline with 200+ test cases",
                    "Implement prompt versioning and A/B testing infrastructure",
                    "Create golden datasets for regression testing",
                    "Set up logging for all LLM interactions"
                ],
                deliverables: ["Evaluation dashboard", "Prompt library", "Edge case documentation", "Cost projections"],
                commonMistakes: ["Skipping evaluation infrastructure", "Not tracking prompt versions", "Underestimating edge cases"]
            }
        },
        {
            title: "Phase 3: Alpha",
            duration: "8-12 weeks",
            goal: "Production-ready code with comprehensive testing",
            items: ["Production-ready code", "Comprehensive testing", "Performance optimization"],
            details: {
                description: "Alpha is where engineering rigor meets AI experimentation. You're hardening the system for real-world conditions while maintaining the flexibility to iterate on prompts.",
                keyActivities: [
                    "Implement retry logic, fallbacks, and graceful degradation",
                    "Add caching layer for common queries (can reduce costs 40-60%)",
                    "Set up monitoring and alerting for model performance drift",
                    "Conduct load testing and optimize for latency targets"
                ],
                deliverables: ["Production codebase", "Runbook for operations", "Performance benchmarks", "Security review"],
                commonMistakes: ["No fallback when API fails", "Missing rate limit handling", "Ignoring cost monitoring"]
            }
        },
        {
            title: "Phase 4: Beta",
            duration: "4-8 weeks",
            goal: "Validate with real users in controlled environment",
            items: ["Limited user rollout", "Monitoring & logging", "Iterative improvements"],
            details: {
                description: "Beta puts your system in front of real users for the first time. The goal is learning, not perfection. Expect surprises—users will find ways to break your system you never imagined.",
                keyActivities: [
                    "Roll out to 5-10% of users with feature flags",
                    "Implement user feedback collection (thumbs up/down, corrections)",
                    "Set up real-time dashboards for key metrics",
                    "Create rapid response process for critical issues"
                ],
                deliverables: ["User feedback analysis", "Updated prompts based on real usage", "Refined success metrics"],
                commonMistakes: ["Rolling out too broadly too fast", "Not collecting structured feedback", "Ignoring negative signals"]
            }
        },
        {
            title: "Phase 5: Production Launch",
            duration: "Ongoing",
            goal: "Full deployment with continuous improvement",
            items: ["Full deployment", "Continuous monitoring", "Incident response"],
            details: {
                description: "Production is not the end—it's the beginning of continuous improvement. AI systems require ongoing attention as models update, user behavior shifts, and new edge cases emerge.",
                keyActivities: [
                    "Gradual rollout to 100% with monitoring at each stage",
                    "Implement model performance drift detection",
                    "Create playbooks for common incident types",
                    "Establish regular prompt review and optimization cycles"
                ],
                deliverables: ["SLA documentation", "Incident response runbook", "Continuous improvement roadmap"],
                commonMistakes: ["Treating launch as 'done'", "No process for prompt updates", "Missing cost anomaly detection"]
            }
        }
    ];

    const selected = selectedPhase !== null ? phases[selectedPhase] : null;

    return (
        <div className="my-12">
            {/* Timeline */}
            <div className="flex gap-2 mb-6 overflow-x-auto pb-2">
            {phases.map((phase, i) => (
                    <button
                        key={i}
                        onClick={() => setSelectedPhase(selectedPhase === i ? null : i)}
                        className={`
                            flex items-center gap-3 px-4 py-3 rounded-xl border-2 transition-all shrink-0
                            ${selectedPhase === i 
                                ? 'border-brand-500 bg-brand-50 shadow-md' 
                                : 'border-stone-200 bg-white hover:border-stone-300 hover:shadow-sm'}
                        `}
                    >
                        <span className={`
                            w-8 h-8 rounded-lg flex items-center justify-center text-sm font-bold
                            ${selectedPhase === i ? 'bg-brand-500 text-white' : 'bg-stone-100 text-stone-600'}
                    `}>
                        {i + 1}
                        </span>
                        <div className="text-left">
                            <p className={`font-semibold text-sm ${selectedPhase === i ? 'text-brand-700' : 'text-stone-800'}`}>
                                {phase.title.replace(/Phase \d: /, '')}
                            </p>
                            <p className="text-xs text-stone-500 flex items-center gap-1">
                                <Clock size={10} /> {phase.duration}
                            </p>
                        </div>
                    </button>
                ))}
                    </div>
                    
            {/* Expanded detail panel */}
            {selected && (
                <div className="bg-white border border-stone-200 rounded-2xl overflow-hidden shadow-lg animate-fade-in">
                    {/* Header */}
                    <div className="bg-stone-900 text-white p-6">
                    <div className="flex items-center gap-3 mb-2">
                            <span className="w-10 h-10 rounded-xl bg-brand-500 flex items-center justify-center font-bold text-lg">
                                {selectedPhase! + 1}
                        </span>
                            <div>
                                <h3 className="text-xl font-bold">{selected.title}</h3>
                                <p className="text-stone-400 text-sm">{selected.duration}</p>
                            </div>
                        </div>
                        <p className="text-brand-300 font-medium mt-3">{selected.goal}</p>
                    </div>

                    {/* Content */}
                    <div className="p-6">
                        <p className="text-stone-600 leading-relaxed mb-6">{selected.details.description}</p>
                        
                        <div className="grid md:grid-cols-2 gap-6">
                            {/* Key Activities */}
                            <div>
                                <h4 className="font-bold text-stone-900 mb-3 flex items-center gap-2">
                                    <CheckCircle2 size={16} className="text-brand-500" />
                                    Key Activities
                                </h4>
                                <div className="space-y-2">
                                    {selected.details.keyActivities.map((activity, i) => (
                                        <div key={i} className="flex gap-3 text-sm">
                                            <span className="text-brand-500 font-mono">{i + 1}.</span>
                                            <span className="text-stone-600">{activity}</span>
                                        </div>
                                    ))}
                                </div>
                    </div>

                            {/* Deliverables */}
                            <div>
                                <h4 className="font-bold text-stone-900 mb-3 flex items-center gap-2">
                                    <Target size={16} className="text-brand-500" />
                                    Deliverables
                                </h4>
                                <div className="flex flex-wrap gap-2 mb-6">
                                    {selected.details.deliverables.map((item, i) => (
                                        <span key={i} className="px-3 py-1.5 bg-stone-50 text-stone-700 text-sm rounded-lg border border-stone-200">
                                {item}
                            </span>
                        ))}
                    </div>
                                
                                <h4 className="font-bold text-stone-900 mb-3 flex items-center gap-2">
                                    <AlertTriangle size={16} className="text-amber-500" />
                                    Common Mistakes
                                </h4>
                                <div className="space-y-2">
                                    {selected.details.commonMistakes.map((mistake, i) => (
                                        <div key={i} className="flex gap-2 text-sm text-stone-600">
                                            <span className="text-amber-500">⚠</span>
                                            {mistake}
                </div>
            ))}
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    {/* Close button */}
                    <div className="border-t border-stone-100 p-4 flex justify-end">
                        <button 
                            onClick={() => setSelectedPhase(null)}
                            className="px-4 py-2 text-sm text-stone-500 hover:text-stone-700 hover:bg-stone-100 rounded-lg transition-colors"
                        >
                            Close
                        </button>
                    </div>
                </div>
            )}
            
            {/* Hint when nothing selected */}
            {selectedPhase === null && (
                <p className="text-center text-sm text-stone-400">Click a phase above to learn more</p>
            )}
        </div>
    );
};

export const PromptSandbox = () => {
    const [prompt, setPrompt] = useState('Explain "Attention Mechanism" to a 5 year old.');
    const [response, setResponse] = useState('');
    const [loading, setLoading] = useState(false);

    const handleRun = async () => {
        setLoading(true);
        setResponse('');
        const res = await runQuickPrompt(prompt);
        setResponse(res);
        setLoading(false);
    };

    return (
        <div className="my-12 rounded-2xl border border-stone-800 bg-[#0c0a09] overflow-hidden shadow-2xl ring-1 ring-stone-800">
            {/* Window Controls */}
            <div className="bg-[#1c1917] px-4 py-3 flex items-center justify-between border-b border-stone-800/50">
                <div className="flex gap-2">
                    <div className="w-3 h-3 rounded-full bg-[#EF4444] border border-[#B91C1C] opacity-80"></div>
                    <div className="w-3 h-3 rounded-full bg-[#F59E0B] border border-[#B45309] opacity-80"></div>
                    <div className="w-3 h-3 rounded-full bg-[#10B981] border border-[#047857] opacity-80"></div>
                </div>
                <div className="text-xs font-mono text-stone-500 flex items-center gap-2 select-none">
                    <Terminal size={12} />
                    prompt_engineering.ts
                </div>
                <div className="w-12"></div>
            </div>

            <div className="flex flex-col md:flex-row h-[480px]">
                {/* Input Area */}
                <div className="flex-1 flex flex-col border-r border-stone-800/50 relative group">
                    <div className="flex-1 p-6 relative">
                        <label className="text-[10px] font-bold text-stone-600 uppercase tracking-widest mb-4 block select-none">System // User Input</label>
                        <textarea 
                            className="w-full h-[calc(100%-40px)] bg-transparent text-stone-300 font-mono text-sm focus:outline-none resize-none placeholder-stone-700 leading-relaxed selection:bg-brand-500/30"
                            value={prompt}
                            onChange={(e) => setPrompt(e.target.value)}
                            placeholder="Enter your prompt here..."
                            spellCheck={false}
                        />
                    </div>
                    
                    <div className="p-4 border-t border-stone-800/50 bg-[#1c1917]/50 flex justify-between items-center backdrop-blur-sm">
                         <div className="flex gap-2">
                            <button 
                                onClick={() => setPrompt('Explain "Attention Mechanism" to a 5 year old.')}
                                className="px-3 py-1.5 text-xs font-mono text-stone-400 bg-stone-800/50 rounded hover:bg-stone-700/50 transition-colors"
                            >
                                Simple
                            </button>
                            <button 
                                onClick={() => setPrompt('Write a haiku about machine learning.')}
                                className="px-3 py-1.5 text-xs font-mono text-stone-400 bg-stone-800/50 rounded hover:bg-stone-700/50 transition-colors"
                            >
                                Creative
                            </button>
                        </div>
                        <button 
                            onClick={handleRun}
                            disabled={loading}
                            className="px-4 py-2 bg-stone-800 hover:bg-stone-700 disabled:bg-stone-700 text-white text-sm font-medium rounded-lg flex items-center gap-2 transition-colors"
                        >
                            {loading ? (
                                <>
                                    <RefreshCw size={14} className="animate-spin" />
                                    Running...
                                </>
                            ) : (
                                <>
                                    <Play size={14} />
                                    Run
                                </>
                            )}
                        </button>
                    </div>
                </div>

                {/* Output Area */}
                <div className="flex-1 flex flex-col relative">
                    <div className="flex-1 p-6 overflow-auto">
                        <label className="text-[10px] font-bold text-stone-600 uppercase tracking-widest mb-4 block select-none">Model // Response</label>
                        <div className="text-stone-300 font-mono text-sm leading-relaxed whitespace-pre-wrap">
                            {response || <span className="text-stone-600 italic">Response will appear here...</span>}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export const EvaluationVisual = () => {
    const [isRunning, setIsRunning] = useState(false);
    const [deterministicOutputs, setDeterministicOutputs] = useState<string[]>([]);
    const [probabilisticOutputs, setProbabilisticOutputs] = useState<string[]>([]);
    
    const probabilisticVariants = [
        "The capital of France is Paris.",
        "Paris is the capital of France.",
        "France's capital city is Paris.",
        "The capital city of France is Paris, known as the City of Light.",
        "Paris serves as the capital of France.",
    ];
    
    const runComparison = async () => {
        if (isRunning) return;
        setIsRunning(true);
        setDeterministicOutputs([]);
        setProbabilisticOutputs([]);
        
        // Run 5 iterations with delays
        for (let i = 0; i < 5; i++) {
            await new Promise(resolve => setTimeout(resolve, 600));
            setDeterministicOutputs(prev => [...prev, "Paris"]);
            // Pick a different variant each time (cycle through them)
            setProbabilisticOutputs(prev => [...prev, probabilisticVariants[i]]);
        }
        
        setIsRunning(false);
    };
    
    const reset = () => {
        setDeterministicOutputs([]);
        setProbabilisticOutputs([]);
        setIsRunning(false);
    };

    return (
        <div className="my-12">
            {/* Header */}
            <div className="text-center mb-8">
                <h3 className="text-2xl font-bold text-stone-900 mb-2">The Probabilistic Gap</h3>
                <p className="text-stone-500 max-w-2xl mx-auto">
                    Traditional software is deterministic—same input, same output, every time. 
                    AI systems are probabilistic—same input can produce different outputs.
                </p>
            </div>
            
            {/* Comparison panels */}
            <div className="grid md:grid-cols-2 gap-6 mb-6">
                {/* Deterministic */}
                <div className="rounded-2xl border-2 border-stone-200 bg-stone-50/50 overflow-hidden">
                    <div className="bg-stone-900 text-white px-5 py-3 flex items-center justify-between">
                        <div className="flex items-center gap-2">
                            <Lock size={18} />
                            <span className="font-bold">Deterministic</span>
                        </div>
                        <span className="text-stone-300 text-sm">Traditional Software</span>
                    </div>
                    
                    <div className="p-5">
                        {/* Code block */}
                        <div className="bg-stone-900 rounded-xl p-4 mb-4 font-mono text-sm">
                            <div className="text-stone-500 mb-1">// Function call</div>
                            <div className="text-brand-400">getCapital<span className="text-white">(</span><span className="text-brand-400">"France"</span><span className="text-white">)</span></div>
                        </div>
                        
                        {/* Outputs */}
                        <div className="space-y-2 min-h-[120px]">
                            {deterministicOutputs.length === 0 ? (
                                <div className="text-stone-400 text-sm italic text-center py-8">
                                    Click "Run" to see outputs
                                </div>
                            ) : (
                                deterministicOutputs.map((output, i) => (
                                    <div key={i} className="flex items-center gap-2 text-sm">
                                        <span className="text-stone-400 font-mono w-8">#{i + 1}</span>
                                        <span className="px-3 py-1.5 bg-stone-100 text-stone-800 rounded-lg font-mono">
                                            "{output}"
                                        </span>
                                        {i > 0 && (
                                            <span className="text-stone-600 text-xs flex items-center gap-1">
                                                <CheckCircle2 size={12} /> Same
                                            </span>
                                        )}
                                    </div>
                                ))
                            )}
                        </div>
                        
                        {/* Insight */}
                        {deterministicOutputs.length >= 3 && (
                            <div className="mt-4 p-3 bg-stone-100 rounded-xl text-sm text-stone-800">
                                <strong>✓ Predictable:</strong> Unit tests work. Assert output === expected.
                            </div>
                        )}
                    </div>
                </div>
                
                {/* Probabilistic */}
                <div className="rounded-2xl border-2 border-brand-200 bg-brand-50/50 overflow-hidden">
                    <div className="bg-brand-600 text-white px-5 py-3 flex items-center justify-between">
                        <div className="flex items-center gap-2">
                            <RefreshCw size={18} />
                            <span className="font-bold">Probabilistic</span>
                        </div>
                        <span className="text-brand-200 text-sm">AI / LLM Systems</span>
                    </div>
                    
                    <div className="p-5">
                        {/* Code block */}
                        <div className="bg-stone-900 rounded-xl p-4 mb-4 font-mono text-sm">
                            <div className="text-stone-500 mb-1">// LLM call</div>
                            <div className="text-brand-400">llm.generate<span className="text-white">(</span><span className="text-brand-400">"What is the capital of France?"</span><span className="text-white">)</span></div>
                        </div>
                        
                        {/* Outputs */}
                        <div className="space-y-2 min-h-[120px]">
                            {probabilisticOutputs.length === 0 ? (
                                <div className="text-stone-400 text-sm italic text-center py-8">
                                    Click "Run" to see outputs
                                </div>
                            ) : (
                                probabilisticOutputs.map((output, i) => (
                                    <div key={i} className="flex items-start gap-2 text-sm">
                                        <span className="text-stone-400 font-mono w-8 shrink-0">#{i + 1}</span>
                                        <span className="px-3 py-1.5 bg-brand-100 text-brand-800 rounded-lg">
                                            "{output}"
                                        </span>
                                        {i > 0 && probabilisticOutputs[i] !== probabilisticOutputs[i-1] && (
                                            <span className="text-amber-600 text-xs flex items-center gap-1 shrink-0">
                                                <AlertTriangle size={12} /> Different!
                                            </span>
                                        )}
                                    </div>
                                ))
                            )}
                        </div>
                        
                        {/* Insight */}
                        {probabilisticOutputs.length >= 3 && (
                            <div className="mt-4 p-3 bg-amber-100 rounded-xl text-sm text-amber-800">
                                <strong>⚠ Variable:</strong> Same meaning, different words. Traditional tests fail.
                            </div>
                        )}
                    </div>
                </div>
            </div>
            
            {/* Controls */}
            <div className="flex justify-center gap-3">
                <button
                    onClick={runComparison}
                    disabled={isRunning}
                    className={`px-6 py-3 rounded-xl font-semibold transition-colors flex items-center gap-2 ${
                        isRunning 
                            ? 'bg-stone-300 text-stone-500 cursor-not-allowed' 
                            : 'bg-stone-900 text-white hover:bg-stone-800'
                    }`}
                >
                    {isRunning ? (
                        <>
                            <RefreshCw size={18} className="animate-spin" />
                            Running...
                        </>
                    ) : (
                        <>
                            <Play size={18} />
                            Run Comparison
                        </>
                    )}
                </button>
                {deterministicOutputs.length > 0 && !isRunning && (
                    <button
                        onClick={reset}
                        className="px-4 py-3 bg-stone-100 text-stone-600 rounded-xl font-medium hover:bg-stone-200 transition-colors"
                    >
                        Reset
                    </button>
                )}
            </div>
            
            {/* Key implications */}
            {deterministicOutputs.length >= 5 && (
                <div className="mt-10 grid md:grid-cols-3 gap-4">
                    <div className="p-5 rounded-2xl bg-white border border-stone-200">
                        <div className="w-10 h-10 rounded-xl bg-red-100 text-red-600 flex items-center justify-center mb-3">
                            <AlertTriangle size={20} />
                        </div>
                        <h4 className="font-bold text-stone-900 mb-1">Testing Challenge</h4>
                        <p className="text-sm text-stone-500">Can't use assertEquals(). Need semantic similarity, LLM-as-judge, or human eval.</p>
                    </div>
                    <div className="p-5 rounded-2xl bg-white border border-stone-200">
                        <div className="w-10 h-10 rounded-xl bg-amber-100 text-amber-600 flex items-center justify-center mb-3">
                            <Gauge size={20} />
                        </div>
                        <h4 className="font-bold text-stone-900 mb-1">Metrics Shift</h4>
                        <p className="text-sm text-stone-500">From "pass/fail" to "accuracy %", "similarity score", "preference rate".</p>
                    </div>
                    <div className="p-5 rounded-2xl bg-white border border-stone-200">
                        <div className="w-10 h-10 rounded-xl bg-stone-100 text-stone-600 flex items-center justify-center mb-3">
                            <Target size={20} />
                        </div>
                        <h4 className="font-bold text-stone-900 mb-1">New Skillset</h4>
                        <p className="text-sm text-stone-500">Evaluation design becomes a core competency. Build golden datasets, not unit tests.</p>
                    </div>
                </div>
            )}
        </div>
    );
};

export const PlanningFramework = () => {
    const questions = [
        { category: 'Business Value', items: ['What problem does this solve?', 'Who are the users?', 'What is the ROI?'] },
        { category: 'Technical Feasibility', items: ['Can existing models handle this?', 'What are latency requirements?', 'Data availability?'] },
        { category: 'Risk Assessment', items: ['What happens if it fails?', 'Privacy/compliance concerns?', 'Reputational risk?'] },
    ];

    return (
        <div className="my-8 grid md:grid-cols-3 gap-4">
            {questions.map((q) => (
                <div key={q.category} className="p-5 bg-white border border-stone-200 rounded-xl">
                    <h4 className="font-bold text-stone-800 mb-3">{q.category}</h4>
                    <div className="space-y-2">
                        {q.items.map((item, i) => (
                            <p key={i} className="text-sm text-stone-600">{item}</p>
                        ))}
                    </div>
                </div>
            ))}
        </div>
    );
};

export const TechStack = () => {
    const [selectedLayer, setSelectedLayer] = useState<number | null>(null);
    
    const layers = [
        { 
            name: 'Application Layer', 
            icon: 'layout',
            description: 'The user-facing components and business logic that orchestrate AI capabilities',
            items: [
                { name: 'UI/UX', desc: 'Chat interfaces, streaming responses, loading states for AI latency' },
                { name: 'API Gateway', desc: 'Rate limiting, authentication, request routing to AI services' },
                { name: 'Business Logic', desc: 'Orchestration, workflow management, response post-processing' },
                { name: 'Session Management', desc: 'Conversation history, context windowing, user preferences' },
            ],
            tools: ['Next.js', 'FastAPI', 'Express', 'Vercel AI SDK'],
            keyConsiderations: [
                'Design for latency—AI calls take 1-30 seconds, not milliseconds',
                'Implement streaming to improve perceived performance',
                'Build feedback loops (thumbs up/down) into the UI from day one'
            ]
        },
        { 
            name: 'AI Layer', 
            icon: 'brain',
            description: 'The intelligence layer where prompts, retrieval, and model interactions happen',
            items: [
                { name: 'Prompt Engineering', desc: 'System prompts, few-shot examples, output formatting' },
                { name: 'RAG Pipeline', desc: 'Document chunking, embedding, retrieval, context injection' },
                { name: 'Fine-tuning', desc: 'Custom model training for domain-specific tasks' },
                { name: 'Agents & Tools', desc: 'Function calling, multi-step reasoning, external API integration' },
            ],
            tools: ['LangChain', 'LlamaIndex', 'Instructor', 'DSPy'],
            keyConsiderations: [
                'Start with prompting—fine-tuning is expensive and often unnecessary',
                'RAG quality depends 80% on chunking strategy and retrieval',
                'Version control your prompts like code'
            ]
        },
        { 
            name: 'Infrastructure', 
            icon: 'cpu',
            description: 'The foundational services that power AI applications at scale',
            items: [
                { name: 'Model APIs', desc: 'OpenAI, Anthropic, Google, or self-hosted models' },
                { name: 'Vector Databases', desc: 'Pinecone, Weaviate, Chroma, pgvector for embeddings' },
                { name: 'Caching Layer', desc: 'Redis, semantic caching to reduce costs and latency' },
                { name: 'Observability', desc: 'LLM-specific logging, tracing, cost tracking' },
            ],
            tools: ['Pinecone', 'Weaviate', 'Redis', 'LangSmith', 'Weights & Biases'],
            keyConsiderations: [
                'Multi-provider strategy—don\'t lock into one model vendor',
                'Caching can reduce costs by 40-60% for common queries',
                'Log everything—you\'ll need it for debugging and evaluation'
            ]
        },
    ];

    const selected = selectedLayer !== null ? layers[selectedLayer] : null;

    return (
        <div className="my-12">
            {/* Layer buttons */}
            <div className="flex flex-col sm:flex-row gap-3 mb-6">
                {layers.map((layer, i) => (
                    <button
                        key={layer.name}
                        onClick={() => setSelectedLayer(selectedLayer === i ? null : i)}
                        className={`
                            flex-1 flex items-center gap-4 p-4 rounded-xl border-2 transition-all text-left
                            ${selectedLayer === i 
                                ? 'border-brand-500 bg-brand-50 shadow-lg' 
                                : 'border-stone-200 bg-white hover:border-stone-300 hover:shadow-md'}
                        `}
                    >
                        <span className="text-3xl">{layer.icon}</span>
                        <div>
                            <div className="flex items-center gap-2">
                                <span className="text-xs font-mono text-stone-400">L{i + 1}</span>
                                <h4 className={`font-bold ${selectedLayer === i ? 'text-brand-700' : 'text-stone-800'}`}>
                                    {layer.name}
                                </h4>
                            </div>
                            <p className="text-xs text-stone-500 mt-0.5">{layer.items.length} components</p>
                        </div>
                    </button>
                ))}
            </div>

            {/* Expanded detail panel */}
            {selected && (
                <div className="bg-white border border-stone-200 rounded-2xl overflow-hidden shadow-lg">
                    {/* Header */}
                    <div className="bg-stone-900 text-white p-6">
                        <div className="flex items-center gap-4 mb-2">
                            <div className="w-12 h-12 rounded-xl bg-brand-500 flex items-center justify-center">
                                <IconRenderer icon={selected.icon} size={24} className="text-white" />
                            </div>
                            <div>
                                <h3 className="text-xl font-bold">{selected.name}</h3>
                                <p className="text-stone-400 text-sm">Layer {selectedLayer! + 1} of the stack</p>
                            </div>
                        </div>
                        <p className="text-stone-300 mt-3">{selected.description}</p>
                    </div>
                    
                    {/* Content */}
                    <div className="p-6">
                        {/* Components grid */}
                        <h4 className="font-bold text-stone-900 mb-4 flex items-center gap-2">
                            <Layers size={16} className="text-brand-500" />
                            Components
                        </h4>
                        <div className="grid sm:grid-cols-2 gap-3 mb-8">
                            {selected.items.map((item) => (
                                <div key={item.name} className="p-4 rounded-xl bg-stone-50 border border-stone-100">
                                    <h5 className="font-semibold text-stone-900 mb-1">{item.name}</h5>
                                    <p className="text-sm text-stone-500">{item.desc}</p>
                                </div>
                            ))}
                        </div>
                        
                        <div className="grid md:grid-cols-2 gap-6">
                            {/* Tools */}
                            <div>
                                <h4 className="font-bold text-stone-900 mb-3 flex items-center gap-2">
                                    <Code2 size={16} className="text-stone-500" />
                                    Popular Tools
                                </h4>
                                <div className="flex flex-wrap gap-2">
                                    {selected.tools.map((tool) => (
                                        <span key={tool} className="px-3 py-1.5 bg-stone-100 text-stone-700 text-sm rounded-lg border border-stone-200 font-medium">
                                            {tool}
                                        </span>
                                    ))}
                                </div>
                            </div>
                            
                            {/* Key Considerations */}
                            <div>
                                <h4 className="font-bold text-stone-900 mb-3 flex items-center gap-2">
                                    <Lightbulb size={16} className="text-brand-500" />
                                    Key Considerations
                                </h4>
                                <div className="space-y-2">
                                    {selected.keyConsiderations.map((consideration, i) => (
                                        <div key={i} className="flex gap-2 text-sm">
                                            <span className="text-amber-500 shrink-0">→</span>
                                            <span className="text-stone-600">{consideration}</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    {/* Close button */}
                    <div className="border-t border-stone-100 p-4 flex justify-end">
                        <button 
                            onClick={() => setSelectedLayer(null)}
                            className="px-4 py-2 text-sm text-stone-500 hover:text-stone-700 hover:bg-stone-100 rounded-lg transition-colors"
                        >
                            Close
                        </button>
                    </div>
                </div>
            )}
            
            {/* Hint when nothing selected */}
            {selectedLayer === null && (
                <p className="text-center text-sm text-stone-400">Click a layer to explore its components</p>
            )}
        </div>
    );
};

export const RoleComparison = () => {
    const [selectedRole, setSelectedRole] = useState<string | null>(null);
    
    const roles = [
        { 
            id: 'ml',
            title: 'ML Engineer', 
            focus: 'Model Training',
            icon: 'search',
            color: 'blue',
            description: 'ML Engineers focus on the science of models—training, fine-tuning, and optimizing neural networks from scratch or adapting existing architectures.',
            skills: ['PyTorch/TensorFlow', 'Data Pipelines', 'Feature Engineering', 'Model Architecture', 'Distributed Training'],
            dayToDay: [
                'Designing and training custom model architectures',
                'Building data preprocessing and augmentation pipelines',
                'Running experiments and hyperparameter tuning',
                'Optimizing model performance and reducing inference costs',
                'Publishing research and contributing to open-source'
            ],
            background: ['PhD or MS in ML/CS (often)', 'Strong math foundation (linear algebra, calculus, statistics)', 'Research experience'],
            salary: '$150K - $300K+',
            demand: 'High but specialized'
        },
        { 
            id: 'ai',
            title: 'AI Engineer', 
            focus: 'Application Building',
            icon: 'target',
            color: 'brand',
            description: 'AI Engineers build products using pre-trained models as building blocks. They focus on integration, prompt engineering, and delivering user value—not training models.',
            skills: ['Prompt Engineering', 'API Integration', 'RAG Systems', 'Vector Databases', 'Evaluation Design'],
            dayToDay: [
                'Designing prompts and testing against evaluation datasets',
                'Building RAG pipelines with vector search',
                'Integrating LLM APIs into production applications',
                'Setting up observability and monitoring for AI features',
                'Optimizing for cost, latency, and quality tradeoffs'
            ],
            background: ['Software engineering experience', 'Understanding of LLM capabilities/limitations', 'Product mindset'],
            salary: '$130K - $250K+',
            demand: 'Exploding—every company wants this'
        },
    ];

    const selected = roles.find(r => r.id === selectedRole);

    return (
        <div className="my-12">
            {/* Role selector */}
            <div className="grid md:grid-cols-2 gap-4 mb-6">
                {roles.map((role) => (
                    <button
                        key={role.id}
                        onClick={() => setSelectedRole(selectedRole === role.id ? null : role.id)}
                        className={`
                            p-5 rounded-xl border-2 text-left transition-all
                            ${selectedRole === role.id 
                                ? 'border-brand-500 bg-brand-50 shadow-lg' 
                                : 'border-stone-200 bg-white hover:border-stone-300 hover:shadow-md'}
                        `}
                    >
                        <div className="flex items-center gap-3 mb-2">
                            <span className="text-3xl">{role.icon}</span>
                            <div>
                                <h4 className="font-bold text-stone-900">{role.title}</h4>
                                <p className="text-sm text-brand-600">{role.focus}</p>
                            </div>
                        </div>
                        <div className="flex flex-wrap gap-2 mt-3">
                            {role.skills.slice(0, 3).map((skill) => (
                                <span key={skill} className="px-2 py-1 text-xs bg-stone-100 rounded text-stone-600">{skill}</span>
                            ))}
                        </div>
                    </button>
                ))}
            </div>

            {/* Expanded detail */}
            {selected && (
                <div className="bg-white border border-stone-200 rounded-2xl overflow-hidden shadow-lg">
                    <div className="p-6 bg-stone-900 text-white">
                        <div className="flex items-center gap-4">
                            <div className="w-14 h-14 rounded-xl bg-brand-500 flex items-center justify-center">
                                <IconRenderer icon={selected.icon} size={28} className="text-white" />
                            </div>
                            <div>
                                <h3 className="text-2xl font-bold">{selected.title}</h3>
                                <p className="text-white/80">{selected.focus}</p>
                            </div>
                        </div>
                        <p className="mt-4 text-white/90 leading-relaxed">{selected.description}</p>
                    </div>
                    
                    <div className="p-6">
                        <div className="grid md:grid-cols-2 gap-6">
                            {/* Day to Day */}
                            <div>
                                <h4 className="font-bold text-stone-900 mb-3 flex items-center gap-2">
                                    <Clock size={16} className="text-stone-400" />
                                    Day-to-Day Work
                                </h4>
                                <div className="space-y-2">
                                    {selected.dayToDay.map((task, i) => (
                                        <div key={i} className="flex gap-2 text-sm">
                                            <span className="text-brand-500">→</span>
                                            <span className="text-stone-600">{task}</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                            
                            {/* Skills & Background */}
                            <div>
                                <h4 className="font-bold text-stone-900 mb-3 flex items-center gap-2">
                                    <GraduationCap size={16} className="text-stone-400" />
                                    Background
                                </h4>
                                <div className="space-y-2 mb-6">
                                    {selected.background.map((item, i) => (
                                        <p key={i} className="text-sm text-stone-600 flex gap-2">
                                            <span className="text-stone-400">•</span>
                                            {item}
                                        </p>
                                    ))}
                                </div>
                                
                                <h4 className="font-bold text-stone-900 mb-3 flex items-center gap-2">
                                    <DollarSign size={16} className="text-brand-500" />
                                    Compensation
                                </h4>
                                <p className="text-lg font-semibold text-stone-600">{selected.salary}</p>
                                <p className="text-sm text-stone-500 mt-1">Demand: {selected.demand}</p>
                            </div>
                        </div>
                        
                        {/* All skills */}
                        <div className="mt-6 pt-6 border-t border-stone-100">
                            <h4 className="font-bold text-stone-900 mb-3">Key Skills</h4>
                            <div className="flex flex-wrap gap-2">
                                {selected.skills.map((skill) => (
                                    <span key={skill} className={`px-3 py-1.5 rounded-lg text-sm font-medium ${
                                        selected.color === 'brand' 
                                            ? 'bg-brand-100 text-brand-700 border border-brand-200' 
                                            : 'bg-stone-100 text-stone-700 border border-stone-200'
                                    }`}>
                                        {skill}
                                    </span>
                                ))}
                            </div>
                        </div>
                    </div>
                    
                    <div className="border-t border-stone-100 p-4 flex justify-end">
                        <button 
                            onClick={() => setSelectedRole(null)}
                            className="px-4 py-2 text-sm text-stone-500 hover:text-stone-700 hover:bg-stone-100 rounded-lg transition-colors"
                        >
                            Close
                        </button>
                    </div>
                </div>
            )}
            
            {!selectedRole && (
                <p className="text-center text-sm text-stone-400">Click a role to compare in detail</p>
            )}
        </div>
    );
};

export const PromptPatterns = () => {
    const [selectedPattern, setSelectedPattern] = useState<string>('zero-shot');
    
    const patterns = [
        { 
            id: 'zero-shot',
            name: 'Zero-shot', 
            desc: 'Direct instruction without examples',
            prompt: `Classify the sentiment of this review as positive, negative, or neutral.

Review: "The battery life is amazing but the screen is too dim."

Sentiment:`,
            highlights: [
                { text: 'Direct task instruction', color: 'bg-stone-100 text-stone-700' },
                { text: 'No examples provided', color: 'bg-stone-100 text-stone-600' },
            ],
            pros: ['Simple and fast', 'Works well for common tasks', 'Minimal token usage'],
            cons: ['Less control over output format', 'May fail on complex tasks'],
            response: 'Neutral'
        },
        { 
            id: 'few-shot',
            name: 'Few-shot', 
            desc: 'Provide examples to guide output format',
            prompt: `Classify the sentiment of reviews as positive, negative, or neutral.

Review: "Best purchase I've ever made!"
Sentiment: positive

Review: "Broke after one week. Total waste of money."
Sentiment: negative

Review: "It's okay, nothing special."
Sentiment: neutral

Review: "The battery life is amazing but the screen is too dim."
Sentiment:`,
            highlights: [
                { text: '3 labeled examples', color: 'bg-stone-100 text-stone-700' },
                { text: 'Consistent format demonstrated', color: 'bg-stone-100 text-stone-700' },
            ],
            pros: ['Controls output format precisely', 'Teaches edge cases', 'No fine-tuning needed'],
            cons: ['Uses more tokens', 'Examples must be high quality'],
            response: 'neutral'
        },
        { 
            id: 'chain-of-thought',
            name: 'Chain of Thought', 
            desc: 'Ask model to reason step-by-step',
            prompt: `Classify the sentiment of this review. Think through it step by step.

Review: "The battery life is amazing but the screen is too dim."

Let's analyze this step by step:
1. First, identify the positive aspects mentioned
2. Then, identify the negative aspects mentioned  
3. Weigh the overall sentiment
4. Provide final classification

Analysis:`,
            highlights: [
                { text: '"Think step by step"', color: 'bg-stone-100 text-stone-700' },
                { text: 'Structured reasoning template', color: 'bg-brand-100 text-brand-700' },
            ],
            pros: ['Better accuracy on complex tasks', 'Explainable reasoning', 'Catches nuance'],
            cons: ['Much higher token usage', 'Slower responses'],
            response: `1. Positive aspects: "battery life is amazing" - strong positive sentiment
2. Negative aspects: "screen is too dim" - moderate negative sentiment
3. Weighing: One strong positive, one moderate negative - balanced
4. Final classification: neutral (mixed sentiment with both praise and criticism)`
        },
        { 
            id: 'role-playing',
            name: 'Role Playing', 
            desc: 'Assign an expert persona',
            prompt: `You are a senior product analyst at a consumer electronics company with 10 years of experience analyzing customer feedback. Your job is to classify sentiment and extract actionable insights.

Analyze this review:
"The battery life is amazing but the screen is too dim."

Provide:
- Sentiment classification
- Key insights for the product team`,
            highlights: [
                { text: 'Expert persona assigned', color: 'bg-rose-100 text-rose-700' },
                { text: 'Domain context provided', color: 'bg-stone-100 text-stone-700' },
            ],
            pros: ['Domain-appropriate responses', 'More professional tone', 'Richer analysis'],
            cons: ['Persona can drift', 'May be overconfident'],
            response: `**Sentiment Classification:** Mixed/Neutral

**Key Insights for Product Team:**
- Battery: Strong positive signal. This is a competitive advantage worth highlighting in marketing.
- Display: Brightness is a pain point. Recommend investigating:
  - Auto-brightness calibration issues
  - Maximum nits comparison vs competitors
  - Consider this for next hardware revision

**Priority:** Medium - the screen issue doesn't appear to be a dealbreaker but could impact outdoor usability reviews.`
        },
    ];

    const selected = patterns.find(p => p.id === selectedPattern)!;

    return (
        <div className="my-12">
            {/* Pattern selector */}
            <div className="flex flex-wrap gap-2 mb-6">
                {patterns.map((p) => (
                    <button
                        key={p.id}
                        onClick={() => setSelectedPattern(p.id)}
                        className={`
                            px-4 py-2.5 rounded-xl font-medium transition-all
                            ${selectedPattern === p.id 
                                ? 'bg-stone-900 text-white shadow-lg' 
                                : 'bg-white border border-stone-200 text-stone-600 hover:border-stone-300 hover:shadow-sm'}
                        `}
                    >
                        {p.name}
                    </button>
                ))}
            </div>

            {/* Main comparison view */}
            <div className="grid lg:grid-cols-2 gap-6">
                {/* Prompt panel */}
                <div className="bg-stone-900 rounded-2xl overflow-hidden">
                    <div className="px-4 py-3 bg-stone-800 flex items-center justify-between">
                        <div className="flex items-center gap-2">
                            <div className="w-3 h-3 rounded-full bg-red-500/80" />
                            <div className="w-3 h-3 rounded-full bg-yellow-500/80" />
                            <div className="w-3 h-3 rounded-full bg-green-500/80" />
                        </div>
                        <span className="text-xs font-mono text-stone-500">prompt.txt</span>
                    </div>
                    <div className="p-5">
                        <pre className="text-sm text-stone-300 font-mono whitespace-pre-wrap leading-relaxed">
                            {selected.prompt}
                        </pre>
                    </div>
                    {/* Highlights */}
                    <div className="px-5 pb-5 flex flex-wrap gap-2">
                        {selected.highlights.map((h, i) => (
                            <span key={i} className={`px-2 py-1 rounded text-xs font-medium ${h.color}`}>
                                {h.text}
                            </span>
                        ))}
                    </div>
                </div>

                {/* Response panel */}
                <div className="bg-white border border-stone-200 rounded-2xl overflow-hidden">
                    <div className="px-5 py-3 bg-stone-50 border-b border-stone-200 flex items-center justify-between">
                        <span className="text-sm font-semibold text-stone-700">Model Response</span>
                        <span className="text-xs text-stone-400">Simulated output</span>
                    </div>
                    <div className="p-5">
                        <pre className="text-sm text-stone-700 whitespace-pre-wrap leading-relaxed">
                            {selected.response}
                        </pre>
                    </div>
                </div>
            </div>

            {/* Pros/Cons */}
            <div className="grid md:grid-cols-2 gap-4 mt-6">
                <div className="p-4 rounded-xl bg-stone-50 border border-stone-200">
                    <h4 className="font-bold text-stone-800 mb-2 flex items-center gap-2">
                        <CheckCircle2 size={16} />
                        When to use {selected.name}
                    </h4>
                    <div className="space-y-1">
                        {selected.pros.map((pro, i) => (
                            <p key={i} className="text-sm text-stone-700 flex items-center gap-2">
                                <span className="text-brand-500">✓</span> {pro}
                            </p>
                        ))}
                    </div>
                </div>
                <div className="p-4 rounded-xl bg-amber-50 border border-amber-200">
                    <h4 className="font-bold text-amber-800 mb-2 flex items-center gap-2">
                        <AlertTriangle size={16} />
                        Watch out for
                    </h4>
                    <div className="space-y-1">
                        {selected.cons.map((con, i) => (
                            <p key={i} className="text-sm text-amber-700 flex items-center gap-2">
                                <span className="text-amber-500">⚠</span> {con}
                            </p>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
};

export const SummaryGrid = () => {
    const takeaways = [
        { 
            number: '01',
            title: 'The Paradigm Shift', 
            summary: 'AI Engineering is fundamentally different from ML Engineering. You\'re not training models—you\'re composing applications using pre-trained intelligence as building blocks.',
            action: 'Stop thinking "how do I train this?" Start thinking "how do I prompt this?"'
        },
        { 
            number: '02',
            title: 'The Stack Has Changed', 
            summary: 'The new stack is prompts, RAG, vector databases, and evaluation pipelines. Traditional ML skills (PyTorch, feature engineering) matter less than API integration and prompt design.',
            action: 'Learn LangChain, understand embeddings, master prompt engineering.'
        },
        { 
            number: '03',
            title: 'Evaluation is Everything', 
            summary: 'You can\'t unit test probabilistic systems. Building robust evaluation—golden datasets, LLM-as-judge, human feedback loops—is the core competency that separates amateurs from professionals.',
            action: 'Build your eval pipeline before you build your product.'
        },
        { 
            number: '04',
            title: 'Cost & Latency Are Features', 
            summary: 'Model selection isn\'t just about capability—it\'s about the tradeoff between quality, speed, and cost. The best model is often the smallest one that meets your quality bar.',
            action: 'Start with the cheapest model. Only upgrade when evals prove you need to.'
        },
        { 
            number: '05',
            title: 'Ship Fast, Iterate Faster', 
            summary: 'The 5-phase development cycle (PoC → Prototype → Alpha → Beta → Production) compresses what used to take years into months. The barrier to entry has never been lower.',
            action: 'Get a working demo in 2 weeks. Perfect it over the next 2 months.'
        },
    ];

    return (
        <div className="my-12">
            <div className="text-center mb-10">
                <h3 className="text-2xl font-bold text-stone-900 mb-2">Chapter 1 Key Takeaways</h3>
                <p className="text-stone-500">The essential concepts you need to internalize</p>
            </div>
            
            <div className="space-y-4">
                {takeaways.map((item) => (
                    <div key={item.number} className="bg-white border border-stone-200 rounded-2xl overflow-hidden hover:shadow-lg transition-shadow">
                        <div className="flex">
                            {/* Number */}
                            <div className="w-20 shrink-0 bg-stone-900 flex items-center justify-center">
                                <span className="text-2xl font-bold text-brand-500">{item.number}</span>
                            </div>
                            
                            {/* Content */}
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

export const ModelComparison = () => {
    const [selectedModels, setSelectedModels] = useState<string[]>(['gpt-4-1', 'claude-4-sonnet', 'gemini-2-5-pro']);
    
    const models = [
        // OpenAI Models (2025)
        {
            id: 'gpt-4-1',
            name: 'GPT-4.1',
            provider: 'OpenAI',
            tier: 'flagship',
            inputCost: 2.00,
            outputCost: 8.00,
            contextWindow: 1000000,
            latency: 'Medium',
            latencyMs: 600,
            capabilities: { reasoning: 96, coding: 97, creative: 92, instruction: 96 },
            bestFor: ['Complex coding', 'Long context', 'Agentic workflows'],
            avoid: ['Cost-sensitive high-volume']
        },
        {
            id: 'gpt-4-1-mini',
            name: 'GPT-4.1 Mini',
            provider: 'OpenAI',
            tier: 'efficient',
            inputCost: 0.40,
            outputCost: 1.60,
            contextWindow: 1000000,
            latency: 'Fast',
            latencyMs: 250,
            capabilities: { reasoning: 85, coding: 88, creative: 82, instruction: 90 },
            bestFor: ['Balanced cost/quality', 'Long context on budget'],
            avoid: ['Tasks needing top-tier reasoning']
        },
        {
            id: 'gpt-4-1-nano',
            name: 'GPT-4.1 Nano',
            provider: 'OpenAI',
            tier: 'efficient',
            inputCost: 0.10,
            outputCost: 0.40,
            contextWindow: 1000000,
            latency: 'Very Fast',
            latencyMs: 100,
            capabilities: { reasoning: 75, coding: 78, creative: 72, instruction: 85 },
            bestFor: ['High-volume classification', 'Simple extraction'],
            avoid: ['Complex reasoning', 'Creative tasks']
        },
        {
            id: 'o3',
            name: 'o3',
            provider: 'OpenAI',
            tier: 'reasoning',
            inputCost: 10.00,
            outputCost: 40.00,
            contextWindow: 200000,
            latency: 'Slow',
            latencyMs: 5000,
            capabilities: { reasoning: 99, coding: 98, creative: 85, instruction: 95 },
            bestFor: ['PhD-level reasoning', 'Math proofs', 'Complex analysis'],
            avoid: ['Real-time apps', 'Simple tasks', 'Cost-sensitive']
        },
        {
            id: 'o4-mini',
            name: 'o4-mini',
            provider: 'OpenAI',
            tier: 'reasoning',
            inputCost: 1.10,
            outputCost: 4.40,
            contextWindow: 200000,
            latency: 'Medium',
            latencyMs: 1500,
            capabilities: { reasoning: 92, coding: 94, creative: 80, instruction: 90 },
            bestFor: ['STEM reasoning', 'Code debugging', 'Multi-step logic'],
            avoid: ['Latency-critical', 'Creative writing']
        },
        // Anthropic Models (2025)
        {
            id: 'claude-4-opus',
            name: 'Claude 4 Opus',
            provider: 'Anthropic',
            tier: 'flagship',
            inputCost: 15.00,
            outputCost: 75.00,
            contextWindow: 200000,
            latency: 'Slow',
            latencyMs: 2000,
            capabilities: { reasoning: 98, coding: 97, creative: 98, instruction: 98 },
            bestFor: ['Highest quality', 'Complex creative', 'Research'],
            avoid: ['Cost-sensitive', 'High-volume', 'Real-time']
        },
        {
            id: 'claude-4-sonnet',
            name: 'Claude 4 Sonnet',
            provider: 'Anthropic',
            tier: 'flagship',
            inputCost: 3.00,
            outputCost: 15.00,
            contextWindow: 200000,
            latency: 'Medium',
            latencyMs: 700,
            capabilities: { reasoning: 96, coding: 98, creative: 95, instruction: 96 },
            bestFor: ['Production workloads', 'Coding', 'Extended thinking'],
            avoid: ['Extreme cost sensitivity']
        },
        {
            id: 'claude-3-5-haiku',
            name: 'Claude (Budget)',
            provider: 'Anthropic',
            tier: 'efficient',
            inputCost: 0.80,
            outputCost: 4.00,
            contextWindow: 200000,
            latency: 'Fast',
            latencyMs: 300,
            capabilities: { reasoning: 82, coding: 85, creative: 78, instruction: 88 },
            bestFor: ['Fast classification', 'Chat', 'Summarization'],
            avoid: ['Complex multi-step reasoning']
        },
        // Google Models (2025)
        {
            id: 'gemini-flagship',
            name: 'Gemini (Flagship)',
            provider: 'Google',
            tier: 'flagship',
            inputCost: 1.25,
            outputCost: 10.00,
            contextWindow: 1000000,
            latency: 'Medium',
            latencyMs: 800,
            capabilities: { reasoning: 95, coding: 95, creative: 88, instruction: 93 },
            bestFor: ['Massive context', 'Multimodal', 'Thinking mode'],
            avoid: ['Latency-critical real-time']
        },
        {
            id: 'gemini-2-5-flash',
            name: 'Gemini (Fast)',
            provider: 'Google',
            tier: 'efficient',
            inputCost: 0.15,
            outputCost: 0.60,
            contextWindow: 1000000,
            latency: 'Very Fast',
            latencyMs: 150,
            capabilities: { reasoning: 85, coding: 88, creative: 80, instruction: 88 },
            bestFor: ['Speed + quality balance', 'Agentic tasks', 'High volume'],
            avoid: ['Highest reasoning needs']
        },
        {
            id: 'gemini-2-0-flash',
            name: 'Gemini (Budget)',
            provider: 'Google',
            tier: 'efficient',
            inputCost: 0.10,
            outputCost: 0.40,
            contextWindow: 1000000,
            latency: 'Very Fast',
            latencyMs: 120,
            capabilities: { reasoning: 80, coding: 82, creative: 75, instruction: 85 },
            bestFor: ['Real-time apps', 'Low latency', 'Native tool use'],
            avoid: ['Complex reasoning']
        },
    ];

    const toggleModel = (id: string) => {
        setSelectedModels(prev => 
            prev.includes(id) 
                ? prev.filter(m => m !== id)
                : [...prev, id].slice(-4) // Max 4 models
        );
    };

    const selected = models.filter(m => selectedModels.includes(m.id));

    return (
        <div className="my-12">
            {/* Model selector */}
            <div className="mb-6">
                <p className="text-sm text-stone-500 mb-3">Select up to 4 models to compare:</p>
                <div className="flex flex-wrap gap-2">
                    {models.map(model => (
                        <button
                            key={model.id}
                            onClick={() => toggleModel(model.id)}
                            className={`px-3 py-2 rounded-lg text-sm font-medium transition-all ${
                                selectedModels.includes(model.id)
                                    ? 'bg-stone-900 text-white'
                                    : 'bg-stone-100 text-stone-600 hover:bg-stone-200'
                            }`}
                        >
                            {model.name}
                        </button>
                    ))}
                </div>
            </div>

            {/* Comparison table */}
            {selected.length > 0 && (
                <div className="overflow-x-auto">
                    <div className="inline-flex gap-4 min-w-full pb-4">
                        {selected.map(model => (
                            <div key={model.id} className="w-72 shrink-0 bg-white border border-stone-200 rounded-2xl overflow-hidden">
                                {/* Header */}
                                <div className={`p-4 ${
                                    model.tier === 'flagship' ? 'bg-stone-900 text-white' : 
                                    model.tier === 'reasoning' ? 'bg-stone-900 text-white' : 
                                    'bg-stone-100'
                                }`}>
                                    <div className="flex items-center justify-between mb-1">
                                        <span className={`text-xs font-mono ${
                                            model.tier === 'flagship' || model.tier === 'reasoning' ? 'text-stone-400' : 'text-stone-500'
                                        }`}>
                                            {model.provider}
                                        </span>
                                        <span className={`text-xs px-2 py-0.5 rounded ${
                                            model.tier === 'flagship' 
                                                ? 'bg-brand-500 text-white' 
                                                : model.tier === 'reasoning'
                                                ? 'bg-stone-700 text-white'
                                                : 'bg-stone-100 text-stone-700'
                                        }`}>
                                            {model.tier === 'flagship' ? 'Flagship' : model.tier === 'reasoning' ? 'Reasoning' : 'Efficient'}
                                        </span>
                                    </div>
                                    <h4 className="font-bold text-lg">{model.name}</h4>
                                </div>
                                
                                {/* Stats */}
                                <div className="p-4 space-y-4">
                                    {/* Cost */}
                                    <div>
                                        <p className="text-xs text-stone-500 uppercase tracking-wide mb-1">Cost per 1M tokens</p>
                                        <div className="flex gap-4">
                                            <div>
                                                <span className="text-lg font-bold text-stone-900">${model.inputCost.toFixed(2)}</span>
                                                <span className="text-xs text-stone-400 ml-1">in</span>
                                            </div>
                                            <div>
                                                <span className="text-lg font-bold text-stone-900">${model.outputCost.toFixed(2)}</span>
                                                <span className="text-xs text-stone-400 ml-1">out</span>
                                            </div>
                                        </div>
                                    </div>
                                    
                                    {/* Context */}
                                    <div>
                                        <p className="text-xs text-stone-500 uppercase tracking-wide mb-1">Context Window</p>
                                        <p className="text-lg font-bold text-stone-900">
                                            {model.contextWindow >= 1000000 
                                                ? `${(model.contextWindow / 1000000).toFixed(1)}M` 
                                                : `${model.contextWindow / 1000}K`} tokens
                                        </p>
                                    </div>
                                    
                                    {/* Latency */}
                                    <div>
                                        <p className="text-xs text-stone-500 uppercase tracking-wide mb-1">Latency</p>
                                        <div className="flex items-center gap-2">
                                            <span className={`text-sm font-semibold ${
                                                model.latencyMs < 300 ? 'text-stone-600' :
                                                model.latencyMs < 700 ? 'text-amber-600' : 'text-stone-600'
                                            }`}>{model.latency}</span>
                                            <span className="text-xs text-stone-400">~{model.latencyMs}ms</span>
                                        </div>
                                    </div>
                                    
                                    {/* Capabilities */}
                                    <div>
                                        <p className="text-xs text-stone-500 uppercase tracking-wide mb-2">Capabilities</p>
                                        <div className="space-y-1.5">
                                            {Object.entries(model.capabilities).map(([key, value]) => (
                                                <div key={key} className="flex items-center gap-2">
                                                    <span className="text-xs text-stone-500 w-16 capitalize">{key}</span>
                                                    <div className="flex-1 h-1.5 bg-stone-100 rounded-full overflow-hidden">
                                                        <div 
                                                            className={`h-full rounded-full ${
                                                                value >= 90 ? 'bg-stone-700' :
                                                                value >= 80 ? 'bg-stone-700' : 'bg-stone-400'
                                                            }`}
                                                            style={{ width: `${value}%` }}
                                                        />
                                                    </div>
                                                    <span className="text-xs font-mono text-stone-400 w-6">{value}</span>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                    
                                    {/* Best for */}
                                    <div>
                                        <p className="text-xs text-stone-500 uppercase tracking-wide mb-2">Best For</p>
                                        <div className="flex flex-wrap gap-1">
                                            {model.bestFor.map(use => (
                                                <span key={use} className="text-xs px-2 py-1 bg-stone-50 text-stone-700 rounded">
                                                    {use}
                                                </span>
                                            ))}
                                        </div>
                                    </div>
                                    
                                    {/* Avoid */}
                                    <div>
                                        <p className="text-xs text-stone-500 uppercase tracking-wide mb-2">Avoid For</p>
                                        <div className="flex flex-wrap gap-1">
                                            {model.avoid.map(use => (
                                                <span key={use} className="text-xs px-2 py-1 bg-red-50 text-red-700 rounded">
                                                    {use}
                                                </span>
                                            ))}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}
            
            {/* Decision framework */}
            <div className="mt-8 p-6 bg-stone-50 rounded-2xl border border-stone-200">
                <h4 className="font-bold text-stone-900 mb-4">2025 Model Selection Guide</h4>
                <div className="grid md:grid-cols-2 gap-4 text-sm">
                    <div className="p-4 bg-white rounded-xl border border-stone-200">
                        <p className="font-semibold text-stone-800 mb-2">Start with Flash/Nano tiers</p>
                        <p className="text-stone-600">Use budget models for most tasks. Upgrade to flagship only when evals prove you need it.</p>
                    </div>
                    <div className="p-4 bg-white rounded-xl border border-stone-200">
                        <p className="font-semibold text-stone-800 mb-2">Reasoning models for STEM</p>
                        <p className="text-stone-600">o3/o4-mini excel at math, logic, and complex code. Higher latency but dramatically better accuracy on hard problems.</p>
                    </div>
                    <div className="p-4 bg-white rounded-xl border border-stone-200">
                        <p className="font-semibold text-stone-800 mb-2">1M context is the new baseline</p>
                        <p className="text-stone-600">Most 2025 models support 1M+ tokens. RAG chunking complexity is often unnecessary now.</p>
                    </div>
                    <div className="p-4 bg-white rounded-xl border border-stone-200">
                        <p className="font-semibold text-stone-800 mb-2">Multi-provider is essential</p>
                        <p className="text-stone-600">Abstract your LLM calls. Models evolve fast—what's best today may not be tomorrow.</p>
                    </div>
                </div>
            </div>
        </div>
    );
};

export const CostOptimization = () => {
    const [selectedStrategy, setSelectedStrategy] = useState<string | null>(null);
    
    const strategies = [
        {
            id: 'model-selection',
            name: 'Model Tiering',
            impact: '40-70%',
            difficulty: 'Easy',
            description: 'Route simple tasks to cheap models, complex tasks to expensive ones.',
            before: { model: 'Flagship model', cost: '$10.00/1M out', monthly: '$30,000' },
            after: { model: '90% Budget, 10% Flagship', cost: '$1.54/1M avg', monthly: '$4,620' },
            implementation: [
                'Classify incoming requests by complexity',
                'Use budget models for classification, summarization, simple Q&A',
                'Reserve flagship models for reasoning, code gen, complex analysis',
                'Build a router that scores task complexity and picks the model'
            ],
            codeExample: `// Simple router example
const model = taskComplexity > 0.7 
  ? "gpt-4o" 
  : "gpt-4o-mini";`
        },
        {
            id: 'caching',
            name: 'Semantic Caching',
            impact: '30-60%',
            difficulty: 'Medium',
            description: 'Cache responses for similar queries. "What\'s the weather?" and "How\'s the weather?" should hit the same cache.',
            before: { model: 'Every request', cost: '1M API calls', monthly: '$10,000' },
            after: { model: '40% cache hits', cost: '600K API calls', monthly: '$6,000' },
            implementation: [
                'Embed incoming queries with a cheap embedding model',
                'Check vector DB for similar past queries (cosine similarity > 0.95)',
                'Return cached response if found, otherwise call LLM and cache',
                'Set TTL based on how dynamic the content is'
            ],
            codeExample: `// Semantic cache check
const embedding = await embed(query);
const cached = await vectorDB.search(embedding, { threshold: 0.95 });
if (cached) return cached.response;`
        },
        {
            id: 'prompt-optimization',
            name: 'Prompt Compression',
            impact: '20-40%',
            difficulty: 'Easy',
            description: 'Shorter prompts = fewer tokens = lower cost. Remove fluff, use abbreviations in system prompts.',
            before: { model: 'Verbose prompt', cost: '2,000 tokens/call', monthly: '$15,000' },
            after: { model: 'Optimized prompt', cost: '800 tokens/call', monthly: '$6,000' },
            implementation: [
                'Audit your prompts—remove redundant instructions',
                'Use concise examples in few-shot (quality > quantity)',
                'Move static context to system prompt (often cheaper)',
                'Consider prompt compression libraries for long contexts'
            ],
            codeExample: `// Before: 500 tokens
"Please analyze the following text and provide a detailed summary..."

// After: 50 tokens  
"Summarize in 3 bullets:"`
        },
        {
            id: 'batching',
            name: 'Request Batching',
            impact: '15-30%',
            difficulty: 'Medium',
            description: 'Process multiple items in one API call instead of separate calls.',
            before: { model: '100 separate calls', cost: '100 API calls', monthly: '$5,000' },
            after: { model: '10 batched calls', cost: '10 API calls', monthly: '$1,500' },
            implementation: [
                'Group similar tasks that arrive within a time window',
                'Send as a single prompt: "Classify these 10 reviews: [...]"',
                'Parse the batched response back to individual results',
                'Balance batch size vs latency requirements'
            ],
            codeExample: `// Batch 10 classifications in one call
const prompt = \`Classify sentiment for each:
1. "\${reviews[0]}"
2. "\${reviews[1]}"
...
Return JSON array of sentiments.\`;`
        },
        {
            id: 'streaming',
            name: 'Smart Streaming',
            impact: '0% cost, 50%+ UX',
            difficulty: 'Easy',
            description: 'Stream responses to improve perceived latency. Users see tokens as they generate.',
            before: { model: 'Wait for full response', cost: '3s perceived wait', monthly: '😤 Frustrating' },
            after: { model: 'Stream tokens', cost: '0.3s to first token', monthly: '😊 Delightful' },
            implementation: [
                'Enable streaming in your API calls',
                'Render tokens as they arrive in your UI',
                'Show typing indicator during generation',
                'Handle partial responses gracefully'
            ],
            codeExample: `// OpenAI streaming
const stream = await openai.chat.completions.create({
  model: "gpt-4o",
  messages: [...],
  stream: true
});
for await (const chunk of stream) {
  process.stdout.write(chunk.choices[0]?.delta?.content || "");
}`
        },
    ];

    const selected = strategies.find(s => s.id === selectedStrategy);

    return (
        <div className="my-12">
            {/* Strategy cards */}
            <div className="grid md:grid-cols-5 gap-3 mb-6">
                {strategies.map(strategy => (
                    <button
                        key={strategy.id}
                        onClick={() => setSelectedStrategy(selectedStrategy === strategy.id ? null : strategy.id)}
                        className={`p-4 rounded-xl border-2 text-left transition-all ${
                            selectedStrategy === strategy.id
                                ? 'border-brand-500 bg-brand-50 shadow-lg'
                                : 'border-stone-200 bg-white hover:border-stone-300'
                        }`}
                    >
                        <div className="flex items-center justify-between mb-2">
                            <span className={`text-xs font-bold px-2 py-0.5 rounded ${
                                strategy.impact.startsWith('40') || strategy.impact.startsWith('30') 
                                    ? 'bg-stone-100 text-stone-700'
                                    : 'bg-stone-100 text-stone-700'
                            }`}>
                                {strategy.impact} savings
                            </span>
                        </div>
                        <h4 className="font-bold text-stone-900 text-sm">{strategy.name}</h4>
                        <p className="text-xs text-stone-500 mt-1">{strategy.difficulty}</p>
                    </button>
                ))}
            </div>

            {/* Expanded detail */}
            {selected && (
                <div className="bg-white border border-stone-200 rounded-2xl overflow-hidden shadow-lg">
                    <div className="bg-stone-900 text-white p-6">
                        <div className="flex items-center justify-between mb-2">
                            <h3 className="text-xl font-bold">{selected.name}</h3>
                            <span className="px-3 py-1 bg-stone-700 rounded-full text-sm font-bold">
                                {selected.impact} cost reduction
                            </span>
                        </div>
                        <p className="text-stone-300">{selected.description}</p>
                    </div>
                    
                    <div className="p-6">
                        {/* Before/After */}
                        <div className="grid md:grid-cols-2 gap-4 mb-6">
                            <div className="p-4 bg-red-50 rounded-xl border border-red-200">
                                <p className="text-xs font-bold text-red-600 uppercase mb-2">Before</p>
                                <p className="text-sm text-stone-700">{selected.before.model}</p>
                                <p className="text-lg font-bold text-red-700 mt-1">{selected.before.monthly}{selected.id !== 'streaming' && '/mo'}</p>
                            </div>
                            <div className="p-4 bg-stone-50 rounded-xl border border-stone-200">
                                <p className="text-xs font-bold text-stone-600 uppercase mb-2">After</p>
                                <p className="text-sm text-stone-700">{selected.after.model}</p>
                                <p className="text-lg font-bold text-stone-700 mt-1">{selected.after.monthly}{selected.id !== 'streaming' && '/mo'}</p>
                            </div>
                        </div>
                        
                        {/* Implementation steps */}
                        <h4 className="font-bold text-stone-900 mb-3">How to implement</h4>
                        <div className="space-y-2 mb-6">
                            {selected.implementation.map((step, i) => (
                                <div key={i} className="flex gap-3 text-sm">
                                    <span className="w-6 h-6 rounded-full bg-brand-100 text-brand-600 flex items-center justify-center text-xs font-bold shrink-0">
                                        {i + 1}
                                    </span>
                                    <span className="text-stone-600">{step}</span>
                                </div>
                            ))}
                        </div>
                        
                        {/* Code example */}
                        <h4 className="font-bold text-stone-900 mb-3">Code snippet</h4>
                        <div className="bg-stone-900 rounded-xl p-4 overflow-x-auto">
                            <pre className="text-sm text-stone-300 font-mono whitespace-pre-wrap">{selected.codeExample}</pre>
                        </div>
                    </div>
                    
                    <div className="border-t border-stone-100 p-4 flex justify-end">
                        <button 
                            onClick={() => setSelectedStrategy(null)}
                            className="px-4 py-2 text-sm text-stone-500 hover:text-stone-700 hover:bg-stone-100 rounded-lg transition-colors"
                        >
                            Close
                        </button>
                    </div>
                </div>
            )}
            
            {!selectedStrategy && (
                <p className="text-center text-sm text-stone-400">Click a strategy to see implementation details</p>
            )}
        </div>
    );
};

export const ConvergenceForces = () => {
    const forces = [
        { 
            title: 'Model Capability', 
            year: '2020',
            metric: 'GPT-3 → GPT-4',
            description: 'Emergent abilities at scale',
        },
        { 
            title: 'API Access', 
            year: '2022',
            metric: '1 line of code',
            description: 'Frontier models via REST',
        },
        { 
            title: 'Cost Collapse', 
            year: '2023',
            metric: '100x cheaper',
            description: '$100 → $1 per task',
        },
        { 
            title: 'Context Windows', 
            year: '2024',
            metric: '4K → 1M+ tokens',
            description: 'Process entire codebases',
        },
        { 
            title: 'Tooling Ecosystem', 
            year: '2023',
            metric: 'LangChain, Vector DBs',
            description: 'Infrastructure abstraction',
        },
        { 
            title: 'Enterprise Demand', 
            year: '2024',
            metric: '$B+ investment',
            description: 'Every company wants AI',
        },
    ];

    return (
        <div className="my-12 -mx-6 md:-mx-8">
            {/* Horizontal scrolling timeline */}
            <div className="relative">
                {/* Timeline line */}
                <div className="absolute top-[52px] left-0 right-0 h-0.5 bg-stone-200" />
                
                {/* Scrollable container */}
                <div className="overflow-x-auto pb-4 scrollbar-hide">
                    <div className="flex gap-4 px-6 md:px-8 min-w-max">
                        {forces.map((force, i) => (
                            <div 
                                key={force.title}
                                className="relative flex flex-col items-center"
                            >
                                {/* Year marker */}
                                <div className="mb-2 px-3 py-1 rounded-full bg-stone-900 text-white text-xs font-mono">
                                    {force.year}
                                </div>
                                
                                {/* Timeline dot */}
                                <div className="w-4 h-4 rounded-full bg-brand-500 border-4 border-white shadow-md z-10 mb-4" />
                                
                                {/* Card */}
                                <div className="w-64 p-5 rounded-2xl bg-white border border-stone-200 hover:border-brand-300 hover:shadow-lg transition-all">
                                    <h4 className="font-bold text-stone-900 mb-1">{force.title}</h4>
                                    <p className="text-lg font-semibold text-brand-600 mb-2">{force.metric}</p>
                                    <p className="text-sm text-stone-500">{force.description}</p>
                                </div>
                            </div>
                        ))}
                        
                        {/* End marker */}
                        <div className="relative flex flex-col items-center">
                            <div className="mb-2 px-3 py-1 rounded-full bg-brand-500 text-white text-xs font-bold">
                                NOW
                            </div>
                            <div className="w-4 h-4 rounded-full bg-brand-500 border-4 border-white shadow-md z-10 mb-4" />
                            <div className="w-48 p-5 rounded-2xl bg-stone-900 text-white text-center">
                                <p className="font-bold">AI Engineering Era</p>
                                <p className="text-sm text-stone-400 mt-1">The perfect storm</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            {/* Scroll hint */}
            <p className="text-center text-xs text-stone-400 mt-2">← Scroll to explore →</p>
        </div>
    );
};

// ============================================
// CHAPTER 2 INTERACTIVE COMPONENTS
// ============================================

export const FoundationPillars = () => {
    const pillars: CardSelectorItem[] = [
        {
            id: 0,
            name: 'Scale',
            label: 'PILLAR 1',
            tagline: 'Trained on internet-scale data',
            capability: 'Trillions of tokens, billions of images',
            examples: ['Web crawls', 'Books', 'Code'],
            description: 'Foundation models are trained on trillions of tokens—essentially the entire written knowledge of humanity. This massive scale is what enables their broad capabilities without task-specific training.',
            characteristics: [
                'Training data: 10-100+ trillion tokens',
                'Model size: 100B-2T parameters',
                'Training cost: $50-100M+',
                'Data sources: web, books, code, images, audio'
            ],
            insight: 'Scale is not just "more data"—it unlocks emergent capabilities that smaller models simply cannot achieve.'
        },
        {
            id: 1,
            name: 'Generality',
            label: 'PILLAR 2',
            tagline: 'One model, infinite tasks',
            capability: 'Not designed for any single task',
            examples: ['Code', 'Translation', 'Reasoning'],
            description: 'Unlike traditional ML models built for specific tasks, foundation models can perform thousands of different tasks without task-specific training. The same model writes code, analyzes images, and solves math.',
            characteristics: [
                'Supports 1000s of task types',
                'Works across 100+ languages',
                'Handles 5+ modalities natively',
                'Zero-shot and few-shot capable'
            ],
            insight: 'You don\'t train a foundation model for your task—you prompt it. This is the paradigm shift.'
        },
        {
            id: 2,
            name: 'Adaptability',
            label: 'PILLAR 3',
            tagline: 'Specialized without retraining',
            capability: 'Customized through prompting and fine-tuning',
            examples: ['Prompts', 'LoRA', 'RAG'],
            description: 'Foundation models can be adapted to specific use cases through prompting, fine-tuning, or retrieval—without touching the base model weights. This makes them practical building blocks.',
            characteristics: [
                'Prompt engineering: minutes to implement',
                'Fine-tuning (LoRA): hours, not weeks',
                'RAG: add knowledge without retraining',
                'Full retraining: never needed'
            ],
            insight: 'The base model stays frozen. All customization happens through prompts, retrieval, or lightweight adapters.'
        }
    ];

    return <CardSelector items={pillars} defaultSelected={0} />;
};

export const ScalingLaws = () => {
    const [activeView, setActiveView] = useState<'chart' | 'milestones' | 'emergent'>('chart');
    
    const milestones = [
        { year: '2018', model: 'BERT', params: '340M', innovation: 'Bidirectional pre-training', impact: 'Proved transfer learning works for NLP' },
        { year: '2019', model: 'GPT-2', params: '1.5B', innovation: 'Showed generalization', impact: 'First hints of emergent capabilities' },
        { year: '2020', model: 'GPT-3', params: '175B', innovation: 'In-context learning', impact: 'No fine-tuning needed for new tasks' },
        { year: '2022', model: 'Chinchilla', params: '70B', innovation: 'Optimal scaling', impact: 'Quality over size—20 tokens/param' },
        { year: '2023', model: 'GPT-4', params: '~1T (MoE)', innovation: 'Multimodal + reasoning', impact: 'Human-level on many benchmarks' },
        { year: '2024+', model: 'Latest Gen', params: '~1T+', innovation: 'Long context + safety', impact: '1M+ tokens, better alignment' },
        { year: '2025', model: 'o3/Claude 4', params: '~2T+', innovation: 'Deep reasoning', impact: 'PhD-level problem solving' },
    ];

    const emergentCapabilities = [
        { 
            scale: '~10B', 
            level: 1,
            capabilities: [
                { name: 'Basic instruction following', desc: 'Can understand and follow simple commands like "summarize this" or "translate to Spanish"' },
                { name: 'Simple Q&A', desc: 'Answers straightforward factual questions from training data' },
                { name: 'Text completion', desc: 'Continues text in a coherent style, the original GPT task' },
            ]
        },
        { 
            scale: '~100B', 
            level: 2,
            capabilities: [
                { name: 'Chain of thought reasoning', desc: 'Can break problems into steps: "Let me think step by step..."' },
                { name: 'Code generation', desc: 'Writes functional code in multiple languages from natural language descriptions' },
                { name: 'Few-shot learning', desc: 'Learns new tasks from just 2-5 examples in the prompt' },
                { name: 'Translation', desc: 'High-quality translation between 100+ language pairs' },
            ]
        },
        { 
            scale: '~500B+', 
            level: 3,
            capabilities: [
                { name: 'Complex multi-step reasoning', desc: 'Solves problems requiring 10+ logical steps without losing track' },
                { name: 'Theory of mind', desc: 'Understands that other agents have different beliefs and knowledge' },
                { name: 'Advanced math', desc: 'Solves calculus, linear algebra, and competition math problems' },
                { name: 'Scientific reasoning', desc: 'Applies scientific method, forms hypotheses, interprets data' },
            ]
        },
        { 
            scale: '~1T+ (reasoning)', 
            level: 4,
            capabilities: [
                { name: 'PhD-level problem solving', desc: 'Matches expert performance on graduate-level STEM problems' },
                { name: 'Novel research', desc: 'Can generate genuinely new insights and research directions' },
                { name: 'Self-correction', desc: 'Detects and fixes its own errors through reflection' },
                { name: 'Long-horizon planning', desc: 'Creates and executes multi-step plans over extended contexts' },
            ]
        },
    ];

    return (
        <div className="my-12">
            <div className="flex flex-wrap gap-2 mb-6">
                <button
                    onClick={() => setActiveView('chart')}
                    className={`px-4 py-2 rounded-lg font-medium transition-all ${
                        activeView === 'chart' ? 'bg-stone-900 text-white' : 'bg-stone-100 text-stone-600 hover:bg-stone-200'
                    }`}
                >
                    Scaling Formula
                </button>
                <button
                    onClick={() => setActiveView('milestones')}
                    className={`px-4 py-2 rounded-lg font-medium transition-all ${
                        activeView === 'milestones' ? 'bg-stone-900 text-white' : 'bg-stone-100 text-stone-600 hover:bg-stone-200'
                    }`}
                >
                    Model Timeline
                </button>
                <button
                    onClick={() => setActiveView('emergent')}
                    className={`px-4 py-2 rounded-lg font-medium transition-all ${
                        activeView === 'emergent' ? 'bg-stone-900 text-white' : 'bg-stone-100 text-stone-600 hover:bg-stone-200'
                    }`}
                >
                    Emergent Capabilities
                </button>
            </div>

            {activeView === 'chart' && (
                <div className="bg-stone-900 rounded-2xl p-8 text-white">
                    <h4 className="text-lg font-bold mb-6">The Scaling Laws Formula</h4>
                    
                    <div className="grid md:grid-cols-3 gap-6 mb-8">
                        <div className="p-4 bg-stone-800 rounded-xl text-center">
                            <div className="w-12 h-12 rounded-xl bg-stone-700 flex items-center justify-center mx-auto mb-2">
                                <BarChart3 size={24} className="text-brand-400" />
                            </div>
                            <p className="text-2xl font-bold text-brand-400">Parameters</p>
                            <p className="text-stone-400 text-sm mt-1">Model size (weights)</p>
                        </div>
                        <div className="p-4 bg-stone-800 rounded-xl text-center">
                            <div className="w-12 h-12 rounded-xl bg-stone-700 flex items-center justify-center mx-auto mb-2">
                                <BookOpen size={24} className="text-brand-400" />
                            </div>
                            <p className="text-2xl font-bold text-brand-400">Data</p>
                            <p className="text-stone-400 text-sm mt-1">Training tokens</p>
                        </div>
                        <div className="p-4 bg-stone-800 rounded-xl text-center">
                            <div className="w-12 h-12 rounded-xl bg-stone-700 flex items-center justify-center mx-auto mb-2">
                                <Zap size={24} className="text-brand-400" />
                            </div>
                            <p className="text-2xl font-bold text-brand-400">Compute</p>
                            <p className="text-stone-400 text-sm mt-1">FLOPS used</p>
                        </div>
                    </div>
                    
                    <div className="p-6 bg-stone-800 rounded-xl mb-6">
                        <p className="text-center font-mono text-lg">
                            Loss ≈ <span className="text-brand-400">N</span><sup className="text-stone-400">-0.076</sup> + 
                            <span className="text-brand-400"> D</span><sup className="text-stone-400">-0.095</sup> + 
                            <span className="text-brand-400"> C</span><sup className="text-stone-400">-0.050</sup>
                        </p>
                        <p className="text-center text-stone-400 text-sm mt-3">
                            Performance improves as a power law with each factor
                        </p>
                    </div>
                    
                    <div className="p-4 bg-stone-700/20 border border-stone-500/40 rounded-xl">
                        <p className="text-stone-300 font-medium">
                            <strong>Chinchilla Insight:</strong> Most models were undertrained. 
                            The optimal ratio is ~20 tokens per parameter. A 70B model with enough data beats a 280B model without.
                        </p>
                    </div>
                </div>
            )}

            {activeView === 'milestones' && (
                <div className="-mx-6 md:-mx-8">
                    {/* Horizontal scrolling timeline */}
                    <div className="relative">
                        {/* Timeline line */}
                        <div className="absolute top-[52px] left-0 right-0 h-0.5 bg-stone-200" />
                        
                        {/* Scrollable container */}
                        <div className="overflow-x-auto pb-4 scrollbar-hide">
                            <div className="flex gap-4 px-6 md:px-8 min-w-max">
                                {milestones.map((m) => (
                                    <div key={m.model} className="relative flex flex-col items-center">
                                        {/* Year marker */}
                                        <div className="mb-2 px-3 py-1 rounded-full bg-stone-900 text-white text-xs font-mono">
                                            {m.year}
                                        </div>
                                        
                                        {/* Timeline dot */}
                                        <div className="w-4 h-4 rounded-full bg-brand-500 border-4 border-white shadow-md z-10 mb-4" />
                                        
                                        {/* Card */}
                                        <div className="w-56 p-4 rounded-2xl bg-white border border-stone-200 hover:border-brand-300 hover:shadow-lg transition-all">
                                            <div className="flex items-center justify-between mb-2">
                                                <h4 className="font-bold text-stone-900">{m.model}</h4>
                                                <span className="px-2 py-0.5 bg-stone-100 text-stone-700 rounded text-xs font-mono">{m.params}</span>
                                            </div>
                                            <p className="text-brand-600 font-medium text-sm mb-1">{m.innovation}</p>
                                            <p className="text-stone-500 text-xs">{m.impact}</p>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                    
                    {/* Scroll hint */}
                    <p className="text-center text-stone-400 text-sm mt-2">← Scroll to explore →</p>
                </div>
            )}

            {activeView === 'emergent' && (
                <div className="space-y-4">
                    <p className="text-stone-600 mb-6">
                        Emergent capabilities appear <strong>suddenly</strong> at certain scale thresholds—they're not gradually learned but seem to "switch on" once the model is large enough. <span className="text-stone-400">(Hover over capabilities for details)</span>
                    </p>
                    {emergentCapabilities.map((tier) => {
                        // Progressive opacity based on level
                        const opacity = 0.4 + (tier.level * 0.15);
                        return (
                            <div key={tier.scale} className="p-5 rounded-2xl border-2 border-stone-200 bg-white hover:shadow-md transition-shadow">
                                <div className="flex items-center gap-4 mb-4">
                                    <span 
                                        className="px-3 py-1.5 rounded-lg text-white font-bold text-sm"
                                        style={{ backgroundColor: `rgba(234, 88, 12, ${opacity})` }}
                                    >
                                        {tier.scale}
                                    </span>
                                    <div className="flex-1 h-px bg-stone-200" />
                                    <span className="text-xs text-stone-400 font-mono">Level {tier.level}</span>
                                </div>
                                <div className="flex flex-wrap gap-2">
                                    {tier.capabilities.map(cap => (
                                        <div key={cap.name} className="group relative">
                                            <span className="px-3 py-1.5 bg-stone-50 rounded-lg text-sm text-stone-700 border border-stone-200 cursor-help hover:border-brand-300 hover:shadow-sm transition-all inline-block">
                                                {cap.name}
                                            </span>
                                            {/* Tooltip */}
                                            <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-2 bg-stone-900 text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap max-w-xs z-10 shadow-xl">
                                                <div className="relative">
                                                    {cap.desc}
                                                    {/* Arrow */}
                                                    <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-stone-900" />
                                                </div>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        );
                    })}
                    <div className="p-4 bg-amber-50 border border-amber-200 rounded-xl mt-6">
                        <p className="text-amber-800 text-sm">
                            <strong>Key insight:</strong> These capabilities aren't explicitly programmed—they emerge from the training objective. No one taught GPT-4 to solve differential equations; it learned to predict tokens that follow mathematical reasoning patterns.
                        </p>
                    </div>
                </div>
            )}
        </div>
    );
};

// Icon renderer helper
const IconRenderer = ({ icon, size = 24, className = "" }: { icon: string; size?: number; className?: string }) => {
    const iconMap: Record<string, React.ReactNode> = {
        eye: <Eye size={size} className={className} />,
        calculator: <Calculator size={size} className={className} />,
        scale: <Scale size={size} className={className} />,
        mappin: <MapPin size={size} className={className} />,
        brain: <Brain size={size} className={className} />,
        rocket: <Rocket size={size} className={className} />,
        zap: <Zap size={size} className={className} />,
        database: <Database size={size} className={className} />,
        cpu: <Cpu size={size} className={className} />,
        package: <Package size={size} className={className} />,
        harddrive: <HardDrive size={size} className={className} />,
        binary: <Binary size={size} className={className} />,
        sparkles: <Sparkles size={size} className={className} />,
        users: <Users size={size} className={className} />,
        target: <Target size={size} className={className} />,
        barchart: <BarChart3 size={size} className={className} />,
        book: <BookOpen size={size} className={className} />,
        network: <Network size={size} className={className} />,
        boxes: <Boxes size={size} className={className} />,
        layout: <Layout size={size} className={className} />,
        search: <Search size={size} className={className} />,
        messagesquare: <MessageSquare size={size} className={className} />,
        image: <ImageIcon size={size} className={className} />,
        audio: <Mic size={size} className={className} />,
        video: <Video size={size} className={className} />,
        code: <Code2 size={size} className={className} />,
    };
    return iconMap[icon] || <Sparkles size={size} className={className} />;
};

export const TransformerVisual = () => {
    const [selectedComponent, setSelectedComponent] = useState<string | null>(null);
    const [activeView, setActiveView] = useState<'why' | 'components' | 'attention'>('components');
    
    const components = [
        {
            id: 'attention',
            name: 'Self-Attention',
            icon: 'eye',
            description: 'The core innovation. Each token computes relevance scores to all other tokens, deciding what context matters.',
            formula: 'Attention(Q,K,V) = softmax(QK^T / √d) × V',
            details: [
                'Query (Q): What am I looking for?',
                'Key (K): What do I contain?',
                'Value (V): What information do I provide?',
                'Dot product Q·K gives relevance scores',
                'Softmax normalizes to probabilities',
                'Weighted sum of Values is the output'
            ]
        },
        {
            id: 'ffn',
            name: 'Feed-Forward Network',
            icon: 'calculator',
            description: 'After attention combines context, FFN transforms each position independently. This is where "knowledge" is stored.',
            formula: 'FFN(x) = GELU(xW₁ + b₁)W₂ + b₂',
            details: [
                'Typically 4x the hidden dimension',
                'Acts as a key-value memory',
                'Where factual knowledge lives',
                'Independent per position (parallel)',
                'Most of the model parameters'
            ]
        },
        {
            id: 'layernorm',
            name: 'Layer Normalization',
            icon: 'scale',
            description: 'Normalizes activations to stabilize training of very deep networks. Critical for scaling to 100+ layers.',
            formula: 'LayerNorm(x) = γ × (x - μ) / σ + β',
            details: [
                'Normalizes across features',
                'Learned scale (γ) and shift (β)',
                'Enables stable gradients',
                'Pre-norm vs post-norm variants',
                'Essential for training 100+ layers'
            ]
        },
        {
            id: 'positional',
            name: 'Positional Encoding',
            icon: 'mappin',
            description: 'Attention is position-agnostic by default. Positional encodings inject sequence order information.',
            formula: 'PE(pos, 2i) = sin(pos / 10000^(2i/d))',
            details: [
                'Original: sinusoidal functions',
                'Modern: learned embeddings',
                'RoPE: rotary position embeddings',
                'Enables length generalization',
                'Critical for understanding order'
            ]
        }
    ];

    const selected = components.find(c => c.id === selectedComponent);

    return (
        <div className="my-12">
            {/* View tabs */}
            <div className="flex flex-wrap gap-2 mb-6">
                <button
                    onClick={() => setActiveView('why')}
                    className={`px-4 py-2 rounded-lg font-medium transition-all ${
                        activeView === 'why' ? 'bg-stone-900 text-white' : 'bg-stone-100 text-stone-600 hover:bg-stone-200'
                    }`}
                >
                    Why Transformers Won
                </button>
                <button
                    onClick={() => setActiveView('components')}
                    className={`px-4 py-2 rounded-lg font-medium transition-all ${
                        activeView === 'components' ? 'bg-stone-900 text-white' : 'bg-stone-100 text-stone-600 hover:bg-stone-200'
                    }`}
                >
                    Core Components
                </button>
                <button
                    onClick={() => setActiveView('attention')}
                    className={`px-4 py-2 rounded-lg font-medium transition-all ${
                        activeView === 'attention' ? 'bg-stone-900 text-white' : 'bg-stone-100 text-stone-600 hover:bg-stone-200'
                    }`}
                >
                    Attention Deep Dive
                </button>
            </div>

            {/* Why Transformers Won */}
            {activeView === 'why' && (
                <div className="space-y-6">
                    {/* Before/After comparison */}
                    <div className="grid md:grid-cols-2 gap-4">
                        <div className="p-6 bg-stone-100 rounded-2xl border-2 border-stone-200">
                            <div className="flex items-center gap-2 mb-4">
                                <span className="px-2 py-1 bg-stone-700 text-white text-xs font-bold rounded">BEFORE</span>
                                <h4 className="font-bold text-stone-700">RNNs & LSTMs</h4>
                            </div>
                            <div className="space-y-3">
                                <div className="flex items-center gap-3 text-stone-600">
                                    <X size={18} className="text-red-500 shrink-0" />
                                    <span>Sequential processing (slow)</span>
                                </div>
                                <div className="flex items-center gap-3 text-stone-600">
                                    <X size={18} className="text-red-500 shrink-0" />
                                    <span>Vanishing gradients over long sequences</span>
                                </div>
                                <div className="flex items-center gap-3 text-stone-600">
                                    <X size={18} className="text-red-500 shrink-0" />
                                    <span>Can't parallelize during training</span>
                                </div>
                                <div className="flex items-center gap-3 text-stone-600">
                                    <X size={18} className="text-red-500 shrink-0" />
                                    <span>Limited context window (~100 tokens)</span>
                                </div>
                            </div>
                        </div>
                        <div className="p-6 bg-stone-50 rounded-2xl border-2 border-stone-200">
                            <div className="flex items-center gap-2 mb-4">
                                <span className="px-2 py-1 bg-stone-800 text-white text-xs font-bold rounded">AFTER</span>
                                <h4 className="font-bold text-stone-700">Transformers</h4>
                            </div>
                            <div className="space-y-3">
                                <div className="flex items-center gap-3 text-stone-700">
                                    <CheckCircle2 size={18} className="text-brand-500 shrink-0" />
                                    <span>Parallel processing (fast)</span>
                                </div>
                                <div className="flex items-center gap-3 text-stone-700">
                                    <CheckCircle2 size={18} className="text-brand-500 shrink-0" />
                                    <span>Direct attention to any position</span>
                                </div>
                                <div className="flex items-center gap-3 text-stone-700">
                                    <CheckCircle2 size={18} className="text-brand-500 shrink-0" />
                                    <span>Massively parallelizable on GPUs</span>
                                </div>
                                <div className="flex items-center gap-3 text-stone-700">
                                    <CheckCircle2 size={18} className="text-brand-500 shrink-0" />
                                    <span>Context windows up to 2M+ tokens</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Key advantages */}
                    <div className="p-6 bg-white border border-stone-200 rounded-2xl">
                        <h4 className="font-bold text-stone-900 mb-4">The Three Superpowers</h4>
                        <div className="grid md:grid-cols-3 gap-4">
                            <div className="p-4 bg-stone-50 rounded-xl border border-stone-200 hover:border-brand-300 transition-colors">
                                <div className="w-10 h-10 bg-stone-900 rounded-lg flex items-center justify-center text-white mb-3">
                                    <Zap size={20} />
                                </div>
                                <h5 className="font-bold text-stone-900 mb-1">Parallelizable</h5>
                                <p className="text-sm text-stone-600">Process all tokens simultaneously during training—1000x faster than sequential</p>
                            </div>
                            <div className="p-4 bg-stone-50 rounded-xl border border-stone-200 hover:border-brand-300 transition-colors">
                                <div className="w-10 h-10 bg-stone-900 rounded-lg flex items-center justify-center text-white mb-3">
                                    <Eye size={20} />
                                </div>
                                <h5 className="font-bold text-stone-900 mb-1">Attention</h5>
                                <p className="text-sm text-stone-600">Any token can directly "see" any other token—no information bottleneck</p>
                            </div>
                            <div className="p-4 bg-stone-50 rounded-xl border border-stone-200 hover:border-brand-300 transition-colors">
                                <div className="w-10 h-10 bg-stone-900 rounded-lg flex items-center justify-center text-white mb-3">
                                    <TrendingUp size={20} />
                                </div>
                                <h5 className="font-bold text-stone-900 mb-1">Scalable</h5>
                                <p className="text-sm text-stone-600">Performance improves predictably with size—just add more parameters</p>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Core Components */}
            {activeView === 'components' && (
                <>
                    <p className="text-stone-600 mb-6">
                        A transformer layer stacks four key components. Click each to understand how they work together:
                    </p>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
                        {components.map(comp => (
                            <button
                                key={comp.id}
                                onClick={() => setSelectedComponent(selectedComponent === comp.id ? null : comp.id)}
                                className={`p-4 rounded-xl border-2 text-center transition-all ${
                                    selectedComponent === comp.id
                                        ? 'border-brand-500 bg-brand-50 shadow-lg'
                                        : 'border-stone-200 bg-white hover:border-stone-300'
                                }`}
                            >
                                <div className="w-10 h-10 rounded-lg bg-stone-100 flex items-center justify-center mb-2">
                                    <IconRenderer icon={comp.icon} size={20} className="text-stone-700" />
                                </div>
                                <p className="font-medium text-stone-900 text-sm">{comp.name}</p>
                            </button>
                        ))}
                    </div>

                    {!selected && (
                        <div className="p-8 bg-stone-50 rounded-2xl border-2 border-dashed border-stone-300 text-center">
                            <p className="text-stone-500">↑ Click a component above to learn how it works</p>
                        </div>
                    )}

                    {selected && (
                        <div className="bg-white border border-stone-200 rounded-2xl overflow-hidden shadow-lg">
                            <div className="bg-stone-900 text-white p-6">
                                <div className="flex items-center gap-3 mb-2">
                                    <div className="w-10 h-10 rounded-lg bg-brand-500 flex items-center justify-center">
                                        <IconRenderer icon={selected.icon} size={20} className="text-white" />
                                    </div>
                                    <h3 className="text-xl font-bold">{selected.name}</h3>
                                </div>
                                <p className="text-stone-300">{selected.description}</p>
                            </div>
                            
                            <div className="p-6">
                                <div className="p-4 bg-stone-100 rounded-xl mb-6 font-mono text-center text-stone-800">
                                    {selected.formula}
                                </div>
                                
                                <h4 className="font-bold text-stone-900 mb-3">Key Points</h4>
                                <div className="space-y-2">
                                    {selected.details.map((detail, i) => (
                                        <div key={i} className="flex gap-2 text-sm">
                                            <span className="text-brand-500">→</span>
                                            <span className="text-stone-600">{detail}</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    )}
                </>
            )}

            {/* Attention Deep Dive */}
            {activeView === 'attention' && (
                <div className="space-y-6">
                    <div className="p-6 bg-stone-900 rounded-2xl text-white">
                        <h4 className="text-lg font-bold mb-4">The Attention Mechanism</h4>
                        <p className="text-stone-300 mb-6">
                            For each token, attention answers: "What other tokens should influence my representation?"
                        </p>
                        
                        {/* QKV explanation */}
                        <div className="grid md:grid-cols-3 gap-4 mb-6">
                            <div className="p-4 bg-stone-800 rounded-xl border border-stone-700">
                                <div className="text-2xl font-bold text-brand-400 mb-2">Q</div>
                                <div className="text-sm text-white font-medium mb-1">Query</div>
                                <p className="text-xs text-stone-400">"What am I looking for?"</p>
                            </div>
                            <div className="p-4 bg-stone-800 rounded-xl border border-stone-700">
                                <div className="text-2xl font-bold text-brand-400 mb-2">K</div>
                                <div className="text-sm text-white font-medium mb-1">Key</div>
                                <p className="text-xs text-stone-400">"What do I contain?"</p>
                            </div>
                            <div className="p-4 bg-stone-800 rounded-xl border border-stone-700">
                                <div className="text-2xl font-bold text-brand-400 mb-2">V</div>
                                <div className="text-sm text-white font-medium mb-1">Value</div>
                                <p className="text-xs text-stone-400">"What information do I provide?"</p>
                            </div>
                        </div>

                        {/* Formula */}
                        <div className="p-4 bg-stone-800 rounded-xl font-mono text-center">
                            <span className="text-brand-400">Q</span> · <span className="text-brand-400">K</span><sup>T</sup> → scores → softmax → weights × <span className="text-brand-400">V</span> → output
                        </div>
                    </div>

                    {/* Visual example */}
                    <div className="p-6 bg-white border border-stone-200 rounded-2xl">
                        <h4 className="font-bold text-stone-900 mb-4">Example: "The cat sat on the mat"</h4>
                        <p className="text-stone-600 mb-4 text-sm">
                            When processing "sat", attention computes how much each other word matters:
                        </p>
                        <div className="flex flex-wrap gap-2 justify-center">
                            {['The', 'cat', 'sat', 'on', 'the', 'mat'].map((word, i) => {
                                const weights = [0.05, 0.45, 0.3, 0.05, 0.05, 0.1];
                                const opacity = weights[i];
                                return (
                                    <div key={i} className="text-center">
                                        <div 
                                            className="px-4 py-2 rounded-lg font-medium mb-1 transition-all"
                                            style={{ 
                                                backgroundColor: `rgba(234, 88, 12, ${opacity})`,
                                                color: opacity > 0.3 ? 'white' : '#44403c'
                                            }}
                                        >
                                            {word}
                                        </div>
                                        <div className="text-xs text-stone-400">{(weights[i] * 100).toFixed(0)}%</div>
                                    </div>
                                );
                            })}
                        </div>
                        <p className="text-stone-500 text-sm text-center mt-4">
                            "cat" gets high attention because it's the subject of "sat"
                        </p>
                    </div>

                    {/* Multi-head attention */}
                    <div className="p-4 bg-amber-50 border border-amber-200 rounded-xl">
                        <p className="text-amber-800 text-sm">
                            <strong>Multi-Head Attention:</strong> Modern transformers run 32-128 attention "heads" in parallel, each learning different relationships (syntax, semantics, coreference, etc.)
                        </p>
                    </div>
                </div>
            )}
        </div>
    );
};

export const TrainingPipeline = () => {
    const [activePhase, setActivePhase] = useState(0);
    
    const phases = [
        {
            name: 'Pre-Training',
            duration: 'Months',
            cost: '$50-100M+',
            objective: 'Predict the next token',
            description: 'Train on trillions of tokens from the internet. The model learns language, facts, reasoning, and code by trying to predict what comes next.',
            inputs: ['Web crawls (Common Crawl)', 'Books & papers', 'Code (GitHub)', 'Wikipedia', 'Conversations'],
            outputs: ['Language understanding', 'World knowledge', 'Reasoning patterns', 'Code generation', 'Multilingual ability'],
            keyInsight: 'This simple objective—predict next token—is enough to learn almost everything about language and reasoning.'
        },
        {
            name: 'Supervised Fine-Tuning',
            duration: 'Days',
            cost: '$10-100K',
            objective: 'Learn to follow instructions',
            description: 'Fine-tune on (instruction, response) pairs. The model learns the FORMAT of being a helpful assistant—not new knowledge.',
            inputs: ['Human-written examples', 'Instruction datasets', 'Conversation demos', '10K-100K examples'],
            outputs: ['Instruction following', 'Conversation format', 'Helpful responses', 'Task completion'],
            keyInsight: 'Supervised fine-tuning teaches style, not knowledge. A well-tuned model applies its pre-training knowledge in a helpful format.'
        },
        {
            name: 'Preference Fine-Tuning',
            duration: 'Weeks',
            cost: '$100K-1M',
            objective: 'Align with human preferences',
            description: 'Train the model to produce outputs humans prefer. Classic RLHF uses a reward model + PPO, but newer methods like DPO (Direct Preference Optimization) skip the reward model entirely. Constitutional AI uses AI feedback instead of human feedback (RLAIF).',
            inputs: ['Human preference rankings', 'Reward model (RLHF) or direct optimization (DPO)', 'Constitutional principles (CAI)', 'AI feedback (RLAIF)'],
            outputs: ['Helpfulness', 'Harmlessness', 'Honesty', 'Reduced hallucination', 'Better reasoning'],
            keyInsight: 'DPO is simpler than RLHF (no reward model needed) and often works just as well. Constitutional AI scales alignment without human labelers.'
        },
        {
            name: 'Reasoning Training',
            duration: 'Weeks',
            cost: '$1M+',
            objective: 'Learn to think step-by-step',
            description: 'Train models to output reasoning traces and verify their own work. This is how o1/o3 achieve PhD-level performance.',
            inputs: ['Chain of thought data', 'Process reward models', 'Math/code solutions', 'Verification training'],
            outputs: ['Multi-step reasoning', 'Self-correction', 'Math ability', 'Complex problem solving', 'Test-time compute'],
            keyInsight: 'Reasoning training teaches models to "think longer" on hard problems, using more compute at inference time.'
        }
    ];

    return (
        <div className="my-12">
            {/* Phase selector - numbered steps */}
            <div className="flex gap-2 mb-6 overflow-x-auto pb-2">
                {phases.map((phase, i) => (
                    <button
                        key={phase.name}
                        onClick={() => setActivePhase(i)}
                        className={`flex items-center gap-3 px-4 py-3 rounded-xl whitespace-nowrap transition-all ${
                            activePhase === i
                                ? 'bg-stone-900 text-white shadow-lg'
                                : 'bg-stone-100 text-stone-600 hover:bg-stone-200'
                        }`}
                    >
                        <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
                            activePhase === i ? 'bg-brand-500 text-white' : 'bg-stone-300 text-stone-600'
                        }`}>{i + 1}</span>
                        <span className="font-medium">{phase.name}</span>
                    </button>
                ))}
            </div>

            {/* Active phase detail */}
            <div className="bg-white border border-stone-200 rounded-2xl overflow-hidden shadow-lg">
                {/* Header */}
                <div className="bg-stone-900 text-white p-6">
                    <div className="flex flex-wrap items-center gap-4">
                        <div className="w-12 h-12 rounded-xl bg-brand-500 flex items-center justify-center text-xl font-bold">
                            {activePhase + 1}
                        </div>
                        <div className="flex-1">
                            <h3 className="text-xl font-bold">{phases[activePhase].name}</h3>
                            <p className="text-stone-400">{phases[activePhase].objective}</p>
                        </div>
                        <div className="flex gap-3">
                            <div className="px-3 py-1.5 bg-stone-800 rounded-lg text-sm">
                                <span className="text-stone-400">Duration:</span> <span className="font-bold text-white">{phases[activePhase].duration}</span>
                            </div>
                            <div className="px-3 py-1.5 bg-stone-800 rounded-lg text-sm">
                                <span className="text-stone-400">Cost:</span> <span className="font-bold text-brand-400">{phases[activePhase].cost}</span>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Content */}
                <div className="p-6">
                    <p className="text-stone-700 mb-6">{phases[activePhase].description}</p>

                    <div className="grid md:grid-cols-2 gap-6">
                        <div>
                            <h4 className="font-bold text-stone-800 mb-3">Inputs</h4>
                            <div className="space-y-2">
                                {phases[activePhase].inputs.map(input => (
                                    <div key={input} className="flex items-center gap-2 p-2 bg-stone-50 rounded-lg text-sm">
                                        <span className="text-stone-400">→</span>
                                        <span className="text-stone-700">{input}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                        <div>
                            <h4 className="font-bold text-stone-800 mb-3">Outputs</h4>
                            <div className="space-y-2">
                            {phases[activePhase].outputs.map(output => (
                                <div key={output} className="flex items-center gap-2 p-2 bg-stone-50 rounded-lg text-sm">
                                    <span className="text-brand-500">✓</span>
                                    <span className="text-stone-700">{output}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>

                    <div className="mt-6 p-4 rounded-xl bg-brand-50 border border-brand-200">
                        <p className="font-medium text-brand-700">
                            <strong>Key Insight:</strong> {phases[activePhase].keyInsight}
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
};

export const RLHFDemo = () => {
    const [step, setStep] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);

    const steps = [
        {
            title: 'Generate Responses',
            description: 'Given a prompt, generate multiple candidate responses from the model.',
            visual: 'prompt',
        },
        {
            title: 'Human Ranking',
            description: 'Humans rank responses from best to worst based on helpfulness, accuracy, and safety.',
            visual: 'ranking',
        },
        {
            title: 'Train Reward Model',
            description: 'Train a model to predict which response humans would prefer.',
            visual: 'reward',
        },
        {
            title: 'Optimize with RL',
            description: 'Use PPO to update the LLM to maximize the reward model\'s score.',
            visual: 'optimize',
        },
    ];

    useEffect(() => {
        if (isPlaying) {
            const timer = setTimeout(() => {
                if (step < steps.length - 1) {
                    setStep(s => s + 1);
                } else {
                    setIsPlaying(false);
                }
            }, 2000);
            return () => clearTimeout(timer);
        }
    }, [isPlaying, step]);

    const startDemo = () => {
        setStep(0);
        setIsPlaying(true);
    };

    return (
        <div className="my-12">
            <div className="bg-stone-900 rounded-2xl p-6 text-white">
                <div className="flex items-center justify-between mb-6">
                    <h3 className="text-xl font-bold">RLHF Training Loop</h3>
                    <button
                        onClick={startDemo}
                        className="px-4 py-2 bg-brand-500 hover:bg-brand-600 rounded-lg font-medium transition-colors"
                    >
                        {isPlaying ? 'Running...' : 'Run Demo'}
                    </button>
                </div>

                {/* Progress bar */}
                <div className="flex gap-2 mb-8">
                    {steps.map((s, i) => (
                        <div key={i} className="flex-1">
                            <div className={`h-2 rounded-full transition-all duration-500 ${
                                i <= step ? 'bg-brand-500' : 'bg-stone-700'
                            }`} />
                            <p className={`text-xs mt-2 ${i <= step ? 'text-white' : 'text-stone-500'}`}>
                                {s.title}
                            </p>
                        </div>
                    ))}
                </div>

                {/* Current step visualization */}
                <div className="min-h-[200px] p-6 bg-stone-800 rounded-xl">
                    <h4 className="font-bold text-lg mb-2">{steps[step].title}</h4>
                    <p className="text-stone-400 mb-6">{steps[step].description}</p>

                    {step === 0 && (
                        <div className="space-y-3">
                            <div className="p-3 bg-stone-700 rounded-lg">
                                <p className="text-xs text-stone-400 mb-1">Prompt</p>
                                <p className="text-white">"Explain quantum computing simply"</p>
                            </div>
                            <div className="grid grid-cols-3 gap-3">
                                {['Response A', 'Response B', 'Response C'].map(r => (
                                    <div key={r} className="p-3 bg-stone-800 border border-stone-700 rounded-lg">
                                        <p className="text-stone-300 text-sm">{r}</p>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {step === 1 && (
                        <div className="flex items-center justify-center gap-4">
                            <div className="p-4 bg-stone-700/20 border-2 border-stone-500 rounded-lg text-center">
                                <p className="text-brand-400 font-bold">🥇 Best</p>
                                <p className="text-xs text-stone-400">Response B</p>
                            </div>
                            <div className="p-4 bg-yellow-500/20 border border-yellow-500/40 rounded-lg text-center">
                                <p className="text-yellow-400 font-bold">🥈 Good</p>
                                <p className="text-xs text-stone-400">Response A</p>
                            </div>
                            <div className="p-4 bg-red-500/20 border border-red-500/40 rounded-lg text-center">
                                <p className="text-red-400 font-bold">🥉 Worst</p>
                                <p className="text-xs text-stone-400">Response C</p>
                            </div>
                        </div>
                    )}

                    {step === 2 && (
                        <div className="text-center">
                            <div className="inline-block p-6 bg-stone-800 border border-stone-700 rounded-xl">
                                <div className="w-14 h-14 rounded-xl bg-stone-700 flex items-center justify-center mx-auto mb-3">
                                    <Brain size={28} className="text-brand-400" />
                                </div>
                                <p className="text-brand-400 font-bold">Reward Model</p>
                                <p className="text-xs text-stone-400 mt-2">Learns to score responses</p>
                                <div className="mt-4 flex gap-2 justify-center">
                                    <span className="px-2 py-1 bg-stone-700/30 text-stone-300 rounded text-xs">B: 0.92</span>
                                    <span className="px-2 py-1 bg-yellow-500/30 text-yellow-300 rounded text-xs">A: 0.71</span>
                                    <span className="px-2 py-1 bg-red-500/30 text-red-300 rounded text-xs">C: 0.34</span>
                                </div>
                            </div>
                        </div>
                    )}

                    {step === 3 && (
                        <div className="text-center">
                            <div className="flex items-center justify-center gap-6">
                                <div className="p-4 bg-stone-700 rounded-lg">
                                    <p className="text-2xl mb-1">🤖</p>
                                    <p className="text-sm text-stone-300">LLM</p>
                                </div>
                                <div className="flex flex-col items-center">
                                    <div className="text-brand-400 text-2xl">→</div>
                                    <p className="text-xs text-stone-500">Generates</p>
                                </div>
                                <div className="p-4 bg-stone-700 rounded-lg">
                                    <div className="w-8 h-8 rounded-lg bg-stone-700 flex items-center justify-center mx-auto mb-1">
                                        <FileText size={16} className="text-brand-400" />
                                    </div>
                                    <p className="text-sm text-stone-300">Response</p>
                                </div>
                                <div className="flex flex-col items-center">
                                    <div className="text-brand-400 text-2xl">→</div>
                                    <p className="text-xs text-stone-500">Scored by</p>
                                </div>
                                <div className="p-4 bg-stone-700/20 rounded-lg">
                                    <div className="w-8 h-8 rounded-lg bg-stone-700 flex items-center justify-center mx-auto mb-1">
                                        <Brain size={16} className="text-brand-400" />
                                    </div>
                                    <p className="text-sm text-stone-300">Reward</p>
                                </div>
                                <div className="flex flex-col items-center">
                                    <div className="text-brand-400 text-2xl">↩</div>
                                    <p className="text-xs text-stone-500">Updates</p>
                                </div>
                            </div>
                            <p className="mt-6 text-brand-400 font-medium">Model learns to produce higher-scoring responses</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export const TrainingData = () => {
    const [activeView, setActiveView] = useState<'sources' | 'languages' | 'timeline'>('sources');

    const dataSources = [
        { name: 'Common Crawl', percentage: 60, description: 'Web pages crawled since 2008', size: '~250TB raw' },
        { name: 'Books & Literature', percentage: 12, description: 'Books3, BookCorpus, Gutenberg', size: '~100GB' },
        { name: 'Code Repositories', percentage: 10, description: 'GitHub, StackOverflow', size: '~500GB' },
        { name: 'Wikipedia', percentage: 5, description: 'All languages, regularly updated', size: '~20GB' },
        { name: 'Scientific Papers', percentage: 5, description: 'ArXiv, PubMed, Semantic Scholar', size: '~100GB' },
        { name: 'Conversations', percentage: 5, description: 'Reddit, forums, chat logs', size: '~50GB' },
        { name: 'Other', percentage: 3, description: 'News, legal docs, patents', size: 'Varies' },
    ];

    const languages = [
        { name: 'English', percentage: 55, performance: 'Excellent' },
        { name: 'Chinese', percentage: 10, performance: 'Very Good' },
        { name: 'German', percentage: 5, performance: 'Very Good' },
        { name: 'French', percentage: 4, performance: 'Very Good' },
        { name: 'Spanish', percentage: 4, performance: 'Good' },
        { name: 'Japanese', percentage: 3, performance: 'Good' },
        { name: 'Russian', percentage: 3, performance: 'Good' },
        { name: 'Other (100+)', percentage: 16, performance: 'Variable' },
    ];

    const timeline = [
        { year: '2018', event: 'BERT trained on Wikipedia + BookCorpus (~3B tokens)' },
        { year: '2019', event: 'GPT-2 trained on WebText (~40B tokens)' },
        { year: '2020', event: 'GPT-3 trained on filtered Common Crawl (~300B tokens)' },
        { year: '2021', event: 'The Pile released (800GB curated dataset)' },
        { year: '2022', event: 'Chinchilla shows optimal data/compute ratios' },
        { year: '2023', event: 'LLaMA trained on 1.4T tokens, RedPajama released' },
        { year: '2024', event: 'Frontier models using 10-15T+ tokens, data quality > quantity' },
        { year: '2025', event: 'Synthetic data augmentation, data contamination concerns' },
    ];

    return (
        <div className="my-12">
            {/* View tabs */}
            <div className="flex gap-2 mb-6">
                <button
                    onClick={() => setActiveView('sources')}
                    className={`px-4 py-2 rounded-lg font-medium transition-all ${
                        activeView === 'sources' ? 'bg-stone-900 text-white' : 'bg-stone-100 text-stone-600 hover:bg-stone-200'
                    }`}
                >
                    Data Sources
                </button>
                <button
                    onClick={() => setActiveView('languages')}
                    className={`px-4 py-2 rounded-lg font-medium transition-all ${
                        activeView === 'languages' ? 'bg-stone-900 text-white' : 'bg-stone-100 text-stone-600 hover:bg-stone-200'
                    }`}
                >
                    Language Distribution
                </button>
                <button
                    onClick={() => setActiveView('timeline')}
                    className={`px-4 py-2 rounded-lg font-medium transition-all ${
                        activeView === 'timeline' ? 'bg-stone-900 text-white' : 'bg-stone-100 text-stone-600 hover:bg-stone-200'
                    }`}
                >
                    Data Evolution
                </button>
            </div>

            {/* Data Sources View */}
            {activeView === 'sources' && (
                <div className="bg-white border border-stone-200 rounded-2xl p-6">
                    <h4 className="font-bold text-stone-900 mb-6">Training Data Composition (Typical Frontier Model)</h4>
                    <div className="space-y-4">
                        {dataSources.map((source) => (
                            <div key={source.name} className="group">
                                <div className="flex items-center justify-between mb-1">
                                    <span className="font-medium text-stone-800">{source.name}</span>
                                    <span className="text-sm text-stone-500">{source.percentage}%</span>
                                </div>
                                <div className="h-8 bg-stone-100 rounded-lg overflow-hidden relative">
                                    <div 
                                        className="h-full bg-brand-500 rounded-lg transition-all duration-500"
                                        style={{ width: `${source.percentage}%` }}
                                    />
                                    <div className="absolute inset-0 flex items-center px-3">
                                        <span className="text-xs text-stone-600 opacity-0 group-hover:opacity-100 transition-opacity">
                                            {source.description} • {source.size}
                                        </span>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                    <div className="mt-6 p-4 bg-brand-50 border border-brand-200 rounded-xl">
                        <p className="text-sm text-brand-700">
                            <strong>Common Crawl</strong> is the backbone of most foundation models. It's a nonprofit that has crawled the web since 2008, 
                            providing free access to petabytes of web data. <a href="https://commoncrawl.org/" target="_blank" rel="noopener noreferrer" className="underline hover:text-brand-900">Learn more →</a>
                        </p>
                    </div>
                </div>
            )}

            {/* Languages View */}
            {activeView === 'languages' && (
                <div className="bg-white border border-stone-200 rounded-2xl p-6">
                    <h4 className="font-bold text-stone-900 mb-2">Language Representation in Training Data</h4>
                    <p className="text-stone-500 text-sm mb-6">Model performance correlates strongly with training data volume</p>
                    <div className="space-y-3">
                        {languages.map((lang) => {
                            const performanceColor = {
                                'Excellent': 'text-brand-600',
                                'Very Good': 'text-stone-700',
                                'Good': 'text-stone-500',
                                'Variable': 'text-amber-600',
                            }[lang.performance];
                            return (
                                <div key={lang.name} className="flex items-center gap-4">
                                    <div className="w-24 font-medium text-stone-800">{lang.name}</div>
                                    <div className="flex-1 h-6 bg-stone-100 rounded-full overflow-hidden">
                                        <div 
                                            className="h-full bg-stone-700 rounded-full transition-all duration-500"
                                            style={{ width: `${lang.percentage}%` }}
                                        />
                                    </div>
                                    <div className="w-16 text-right text-sm text-stone-500">{lang.percentage}%</div>
                                    <div className={`w-20 text-right text-sm font-medium ${performanceColor}`}>{lang.performance}</div>
                                </div>
                            );
                        })}
                    </div>
                    <div className="mt-6 p-4 bg-amber-50 border border-amber-200 rounded-xl">
                        <p className="text-sm text-amber-800">
                            <strong>Warning:</strong> Low-resource languages (most African languages, indigenous languages) have significantly worse performance. 
                            Always test thoroughly in your target language before deployment.
                        </p>
                    </div>
                </div>
            )}

            {/* Timeline View */}
            {activeView === 'timeline' && (
                <div className="bg-stone-900 rounded-2xl p-6 text-white">
                    <h4 className="font-bold mb-6">The Data Arms Race</h4>
                    <div className="space-y-4">
                        {timeline.map((item, i) => (
                            <div key={item.year} className="flex gap-4">
                                <div className="w-16 shrink-0">
                                    <span className="px-2 py-1 bg-brand-500 rounded text-xs font-bold">{item.year}</span>
                                </div>
                                <div className="flex-1 pb-4 border-l border-stone-700 pl-4">
                                    <p className="text-stone-300">{item.event}</p>
                                </div>
                            </div>
                        ))}
                    </div>
                    <div className="mt-4 p-4 bg-stone-800 rounded-xl">
                        <p className="text-sm text-stone-400">
                            <strong className="text-white">The shift:</strong> Early models competed on size. 
                            Now the focus is on data quality, curation, and avoiding contamination from AI-generated content.
                        </p>
                    </div>
                </div>
            )}
        </div>
    );
};

export const TokenizerDemo = () => {
    const [inputText, setInputText] = useState('Hello, world! 🌍 Let\'s learn about tokenization.');
    const [showDetails, setShowDetails] = useState(false);

    // Simulated tokenization (real tokenizers are more complex)
    const tokenize = (text: string): { token: string; id: number }[] => {
        const tokens: { token: string; id: number }[] = [];
        const commonTokens: Record<string, number> = {
            'Hello': 9906, ',': 11, ' world': 1917, '!': 0, ' ': 220,
            '🌍': 127757, ' Let': 6914, "'s": 338, ' learn': 4048,
            ' about': 922, ' token': 4037, 'ization': 2065, '.': 13
        };
        
        // Simple simulation - real BPE is much more complex
        let remaining = text;
        while (remaining.length > 0) {
            let found = false;
            for (const [token, id] of Object.entries(commonTokens)) {
                if (remaining.startsWith(token)) {
                    tokens.push({ token, id });
                    remaining = remaining.slice(token.length);
                    found = true;
                    break;
                }
            }
            if (!found) {
                tokens.push({ token: remaining[0], id: remaining.charCodeAt(0) });
                remaining = remaining.slice(1);
            }
        }
        return tokens;
    };

    const tokens = tokenize(inputText);

    const colors = [
        'bg-stone-100 text-stone-700 border-stone-200',
        'bg-stone-100 text-stone-700 border-stone-200',
        'bg-stone-100 text-stone-700 border-stone-200',
        'bg-brand-100 text-brand-700 border-brand-200',
        'bg-stone-200 text-stone-700 border-stone-300',
        'bg-stone-100 text-stone-600 border-stone-200',
    ];

    return (
        <div className="my-12">
            <div className="bg-white border border-stone-200 rounded-2xl overflow-hidden">
                <div className="p-6 border-b border-stone-200">
                    <label className="block text-sm font-medium text-stone-700 mb-2">
                        Enter text to tokenize:
                    </label>
                    <input
                        type="text"
                        value={inputText}
                        onChange={(e) => setInputText(e.target.value)}
                        className="w-full p-3 border border-stone-300 rounded-lg focus:ring-2 focus:ring-brand-500 focus:border-brand-500"
                    />
                </div>

                <div className="p-6 bg-stone-50">
                    <div className="flex items-center justify-between mb-4">
                        <h4 className="font-bold text-stone-900">Tokenized Output</h4>
                        <div className="flex items-center gap-4">
                            <span className="text-sm text-stone-500">
                                <strong>{tokens.length}</strong> tokens
                            </span>
                            <button
                                onClick={() => setShowDetails(!showDetails)}
                                className="text-sm text-brand-600 hover:text-brand-700"
                            >
                                {showDetails ? 'Hide IDs' : 'Show IDs'}
                            </button>
                        </div>
                    </div>

                    <div className="flex flex-wrap gap-2">
                        {tokens.map((t, i) => (
                            <div
                                key={i}
                                className={`px-2 py-1 rounded border font-mono text-sm ${colors[i % colors.length]}`}
                            >
                                <span>{t.token === ' ' ? '␣' : t.token}</span>
                                {showDetails && (
                                    <span className="ml-1 text-xs opacity-60">:{t.id}</span>
                                )}
                            </div>
                        ))}
                    </div>
                </div>

                <div className="p-6 border-t border-stone-200">
                    <h4 className="font-bold text-stone-900 mb-3">Why This Matters</h4>
                    <div className="grid md:grid-cols-3 gap-4 text-sm">
                        <div className="p-3 bg-stone-50 rounded-lg">
                            <p className="font-medium text-stone-800 flex items-center gap-2">
                                <DollarSign size={16} className="text-brand-500" />
                                Cost
                            </p>
                            <p className="text-stone-600">You pay per token, not character</p>
                        </div>
                        <div className="p-3 bg-stone-50 rounded-lg">
                            <p className="font-medium text-stone-800">📏 Context</p>
                            <p className="text-stone-600">128K limit is tokens, not chars</p>
                        </div>
                        <div className="p-3 bg-stone-50 rounded-lg">
                            <p className="font-medium text-stone-800">🔢 Math</p>
                            <p className="text-stone-600">Numbers tokenize unpredictably</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export const DecodingStrategies = () => {
    const [strategy, setStrategy] = useState('greedy');
    const [temperature, setTemperature] = useState(1.0);

    const strategies = [
        { id: 'greedy', name: 'Greedy', description: 'Always pick highest probability. Deterministic but boring.' },
        { id: 'temperature', name: 'Temperature', description: 'Scale probabilities. Higher = more random.' },
        { id: 'topk', name: 'Top-K', description: 'Only consider top K tokens. K=50 is common.' },
        { id: 'topp', name: 'Top-P', description: 'Keep tokens until cumulative prob > P. Adaptive.' },
    ];

    // Simulated probability distribution
    const baseProbs = [
        { token: 'the', prob: 0.35 },
        { token: 'a', prob: 0.20 },
        { token: 'an', prob: 0.15 },
        { token: 'this', prob: 0.10 },
        { token: 'that', prob: 0.08 },
        { token: 'one', prob: 0.05 },
        { token: 'some', prob: 0.04 },
        { token: 'many', prob: 0.03 },
    ];

    const getAdjustedProbs = () => {
        if (strategy === 'greedy') {
            return baseProbs.map((p, i) => ({ ...p, active: i === 0, prob: i === 0 ? 1 : 0 }));
        }
        if (strategy === 'temperature') {
            const scaled = baseProbs.map(p => ({ ...p, prob: Math.pow(p.prob, 1/temperature) }));
            const sum = scaled.reduce((a, b) => a + b.prob, 0);
            return scaled.map(p => ({ ...p, prob: p.prob / sum, active: true }));
        }
        if (strategy === 'topk') {
            return baseProbs.map((p, i) => ({ ...p, active: i < 5, prob: i < 5 ? p.prob : 0 }));
        }
        if (strategy === 'topp') {
            let cumulative = 0;
            return baseProbs.map(p => {
                cumulative += p.prob;
                return { ...p, active: cumulative <= 0.9, prob: cumulative <= 0.9 ? p.prob : 0 };
            });
        }
        return baseProbs.map(p => ({ ...p, active: true }));
    };

    const probs = getAdjustedProbs();

    return (
        <div className="my-12">
            <div className="flex flex-wrap gap-2 mb-6">
                {strategies.map(s => (
                    <button
                        key={s.id}
                        onClick={() => setStrategy(s.id)}
                        className={`px-4 py-2 rounded-lg font-medium transition-all ${
                            strategy === s.id
                                ? 'bg-stone-900 text-white'
                                : 'bg-stone-100 text-stone-600 hover:bg-stone-200'
                        }`}
                    >
                        {s.name}
                    </button>
                ))}
            </div>

            <div className="bg-white border border-stone-200 rounded-2xl p-6">
                <p className="text-stone-600 mb-6">
                    {strategies.find(s => s.id === strategy)?.description}
                </p>

                {strategy === 'temperature' && (
                    <div className="mb-6">
                        <label className="block text-sm font-medium text-stone-700 mb-2">
                            Temperature: {temperature.toFixed(1)}
                        </label>
                        <input
                            type="range"
                            min="0.1"
                            max="2"
                            step="0.1"
                            value={temperature}
                            onChange={(e) => setTemperature(parseFloat(e.target.value))}
                            className="w-full"
                        />
                        <div className="flex justify-between text-xs text-stone-400 mt-1">
                            <span>Deterministic</span>
                            <span>Creative</span>
                        </div>
                    </div>
                )}

                <h4 className="font-bold text-stone-900 mb-4">Token Probabilities</h4>
                <div className="space-y-2">
                    {probs.map((p, i) => (
                        <div key={p.token} className="flex items-center gap-3">
                            <span className={`w-16 font-mono text-sm ${p.active ? 'text-stone-900' : 'text-stone-300'}`}>
                                "{p.token}"
                            </span>
                            <div className="flex-1 h-6 bg-stone-100 rounded overflow-hidden">
                                <div
                                    className={`h-full rounded transition-all duration-300 ${
                                        p.active ? 'bg-brand-500' : 'bg-stone-200'
                                    }`}
                                    style={{ width: `${p.prob * 100}%` }}
                                />
                            </div>
                            <span className={`w-12 text-right text-sm ${p.active ? 'text-stone-700' : 'text-stone-300'}`}>
                                {(p.prob * 100).toFixed(1)}%
                            </span>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};

export const ContextEvolution = () => {
    const [activeView, setActiveView] = useState<'timeline' | 'chart'>('timeline');

    const models = [
        { name: 'GPT-3', year: 2020, context: 4096, deprecated: true },
        { name: 'GPT-4', year: 2023, context: 8192, deprecated: true },
        { name: 'Claude 2', year: 2023, context: 100000, deprecated: true },
        { name: 'GPT-4 Turbo', year: 2024, context: 128000, deprecated: false },
        { name: 'Claude (Flagship)', year: 2024, context: 200000, deprecated: false },
        { name: 'GPT (Flagship)', year: 2024, context: 128000, deprecated: false },
        { name: 'Gemini (Flagship)', year: 2024, context: 1000000, deprecated: false },
        { name: 'Claude (Budget)', year: 2024, context: 200000, deprecated: false },
        { name: 'Gemini (Budget)', year: 2024, context: 1000000, deprecated: false },
    ];

    // Only show current models for timeline
    const currentModels = models.filter(m => !m.deprecated);
    const maxContext = Math.max(...models.map(m => m.context));

    const formatContext = (ctx: number) => {
        if (ctx >= 1000000) return `${(ctx / 1000000).toFixed(1)}M`;
        if (ctx >= 1000) return `${(ctx / 1000).toFixed(0)}K`;
        return ctx.toString();
    };

    return (
        <div className="my-12">
            {/* View tabs */}
            <div className="flex gap-2 mb-6">
                <button
                    onClick={() => setActiveView('timeline')}
                    className={`px-4 py-2 rounded-lg font-medium transition-all ${
                        activeView === 'timeline' ? 'bg-stone-900 text-white' : 'bg-stone-100 text-stone-600 hover:bg-stone-200'
                    }`}
                >
                    Timeline View
                </button>
                <button
                    onClick={() => setActiveView('chart')}
                    className={`px-4 py-2 rounded-lg font-medium transition-all ${
                        activeView === 'chart' ? 'bg-stone-900 text-white' : 'bg-stone-100 text-stone-600 hover:bg-stone-200'
                    }`}
                >
                    Comparison Chart
                </button>
            </div>

            {/* Timeline View */}
            {activeView === 'timeline' && (
                <div className="-mx-6 md:-mx-8">
                    <div className="relative">
                        {/* Timeline line */}
                        <div className="absolute top-[52px] left-0 right-0 h-0.5 bg-stone-200" />
                        
                        {/* Scrollable container */}
                        <div className="overflow-x-auto pb-4 scrollbar-hide">
                            <div className="flex gap-4 px-6 md:px-8 min-w-max">
                                {currentModels.map((model) => (
                                    <div key={model.name} className="relative flex flex-col items-center">
                                        {/* Year marker */}
                                        <div className="mb-2 px-3 py-1 rounded-full bg-stone-900 text-white text-xs font-mono">
                                            {model.year}
                                        </div>
                                        
                                        {/* Timeline dot */}
                                        <div className="w-4 h-4 rounded-full bg-brand-500 border-4 border-white shadow-md z-10 mb-4" />
                                        
                                        {/* Card */}
                                        <div className="w-48 p-4 rounded-2xl bg-white border border-stone-200 hover:border-brand-300 hover:shadow-lg transition-all">
                                            <h4 className="font-bold text-stone-900 text-sm mb-2">{model.name}</h4>
                                            <p className="text-2xl font-bold text-brand-600 mb-1">{formatContext(model.context)}</p>
                                            <p className="text-stone-500 text-xs">tokens</p>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                    
                    {/* Scroll hint */}
                    <p className="text-center text-stone-400 text-sm mt-2">← Scroll to explore →</p>
                </div>
            )}

            {/* Chart View */}
            {activeView === 'chart' && (
                <div className="bg-stone-900 rounded-2xl p-6 text-white">
                    <h4 className="text-lg font-bold mb-6">Context Window Comparison</h4>

                    <div className="space-y-3">
                        {models.map((model) => (
                            <div
                                key={model.name}
                                className={`flex items-center gap-4 ${model.deprecated ? 'opacity-40' : ''}`}
                            >
                                <div className="w-16 text-right">
                                    <span className="text-sm text-stone-400">{model.year}</span>
                                </div>
                                <div className="w-36 font-medium flex items-center gap-2 text-sm">
                                    {model.name}
                                    {model.deprecated && <span className="text-[10px] text-stone-500">(old)</span>}
                                </div>
                                <div className="flex-1 h-6 bg-stone-800 rounded-full overflow-hidden">
                                    <div
                                        className={`h-full ${model.deprecated ? 'bg-stone-600' : 'bg-brand-500'} transition-all duration-500 rounded-full`}
                                        style={{ width: `${(Math.log10(model.context) / Math.log10(maxContext)) * 100}%` }}
                                    />
                                </div>
                                <div className="w-16 text-right font-mono text-sm">
                                    {formatContext(model.context)}
                                </div>
                            </div>
                        ))}
                    </div>

                    <div className="mt-8 grid grid-cols-3 gap-4">
                        <div className="p-4 bg-stone-800 rounded-xl text-center">
                            <p className="text-3xl font-bold text-brand-400">250x</p>
                            <p className="text-sm text-stone-400">Growth since 2020</p>
                        </div>
                        <div className="p-4 bg-stone-800 rounded-xl text-center">
                            <p className="text-3xl font-bold text-brand-400">1M</p>
                            <p className="text-sm text-stone-400">Max tokens (2025)</p>
                        </div>
                        <div className="p-4 bg-stone-800 rounded-xl text-center">
                            <p className="text-3xl font-bold text-brand-400">~3 books</p>
                            <p className="text-sm text-stone-400">Per million tokens</p>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export const MOEVisual = () => {
    const [activeExpert, setActiveExpert] = useState<number | null>(null);

    const experts = [
        { id: 0, name: 'Math Expert', specialty: 'Numbers, equations, logic', tokens: ['calculate', 'solve', '123', 'equation'] },
        { id: 1, name: 'Code Expert', specialty: 'Programming, syntax, debugging', tokens: ['function', 'class', 'import', 'return'] },
        { id: 2, name: 'Language Expert', specialty: 'Grammar, style, translation', tokens: ['the', 'however', 'translate', 'write'] },
        { id: 3, name: 'Knowledge Expert', specialty: 'Facts, dates, entities', tokens: ['Paris', '1969', 'Einstein', 'capital'] },
        { id: 4, name: 'Reasoning Expert', specialty: 'Logic, inference, planning', tokens: ['because', 'therefore', 'if', 'then'] },
        { id: 5, name: 'Creative Expert', specialty: 'Stories, ideas, metaphors', tokens: ['imagine', 'once upon', 'like a', 'dream'] },
        { id: 6, name: 'Conversation Expert', specialty: 'Dialogue, context, tone', tokens: ['I think', 'you mean', 'let me', 'sure'] },
        { id: 7, name: 'Safety Expert', specialty: 'Filtering, refusals, alignment', tokens: ['cannot', 'harmful', 'instead', 'sorry'] },
    ];

    return (
        <div className="my-12">
            <div className="bg-white border border-stone-200 rounded-2xl overflow-hidden">
                <div className="p-6 bg-stone-900 text-white">
                    <h4 className="text-lg font-bold mb-2">Mixture of Experts Architecture</h4>
                    <p className="text-stone-400">Each token is routed to 2 of 8 experts. Click an expert to see what it specializes in.</p>
                </div>

                <div className="p-6">
                    {/* Router visualization */}
                    <div className="text-center mb-6">
                        <div className="inline-block p-4 bg-brand-100 border border-brand-200 rounded-xl">
                            <p className="font-bold text-brand-700">🔀 Router Network</p>
                            <p className="text-xs text-brand-600">Decides which experts to use per token</p>
                        </div>
                    </div>

                    {/* Expert grid */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
                        {experts.map(expert => (
                            <button
                                key={expert.id}
                                onClick={() => setActiveExpert(activeExpert === expert.id ? null : expert.id)}
                                className={`p-4 rounded-xl border-2 text-left transition-all ${
                                    activeExpert === expert.id
                                        ? 'border-brand-500 bg-brand-50 shadow-lg'
                                        : 'border-stone-200 hover:border-stone-300'
                                }`}
                            >
                                <p className="font-bold text-stone-900 text-sm">{expert.name}</p>
                                <p className="text-xs text-stone-500 mt-1">{expert.specialty}</p>
                            </button>
                        ))}
                    </div>

                    {activeExpert !== null && (
                        <div className="p-4 bg-stone-50 rounded-xl">
                            <p className="font-medium text-stone-800 mb-2">Tokens this expert handles:</p>
                            <div className="flex flex-wrap gap-2">
                                {experts[activeExpert].tokens.map(token => (
                                    <span key={token} className="px-3 py-1 bg-white border border-stone-200 rounded-full text-sm font-mono">
                                        {token}
                                    </span>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Stats */}
                    <div className="mt-6 grid grid-cols-3 gap-4">
                        <div className="p-4 bg-stone-50 rounded-xl text-center">
                            <p className="text-2xl font-bold text-stone-600">8</p>
                            <p className="text-xs text-stone-500">Total Experts</p>
                        </div>
                        <div className="p-4 bg-stone-50 rounded-xl text-center">
                            <p className="text-2xl font-bold text-stone-600">2</p>
                            <p className="text-xs text-brand-500">Active per Token</p>
                        </div>
                        <div className="p-4 bg-stone-50 rounded-xl text-center">
                            <p className="text-2xl font-bold text-stone-600">4x</p>
                            <p className="text-xs text-stone-500">Efficiency Gain</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export const MultimodalFlow = () => {
    const [selectedModality, setSelectedModality] = useState<string | null>(null);

    const modalities = [
        { id: 'text', name: 'Text', icon: 'messagesquare', examples: ['Questions', 'Instructions', 'Documents'] },
        { id: 'image', name: 'Image', icon: 'image', examples: ['Photos', 'Diagrams', 'Screenshots'] },
        { id: 'audio', name: 'Audio', icon: 'audio', examples: ['Speech', 'Music', 'Sounds'] },
        { id: 'video', name: 'Video', icon: 'video', examples: ['Clips', 'Recordings', 'Streams'] },
        { id: 'code', name: 'Code', icon: 'code', examples: ['Functions', 'Classes', 'Repos'] },
    ];

    const capabilities = [
        { from: 'image', to: 'text', example: '"What\'s in this photo?" → Description' },
        { from: 'text', to: 'image', example: '"A sunset over mountains" → Generated image' },
        { from: 'audio', to: 'text', example: 'Meeting recording → Transcript' },
        { from: 'text', to: 'code', example: '"Sort this list" → Python function' },
        { from: 'video', to: 'text', example: 'Tutorial → Step-by-step summary' },
        { from: 'text', to: 'audio', example: '"Read this aloud" → Speech' },
    ];

    return (
        <div className="my-12">
            <div className="bg-stone-900 rounded-2xl p-6 text-white">
                <h4 className="text-lg font-bold mb-6">Native Multimodal: Any Input → Any Output</h4>

                {/* Modality selector */}
                <div className="flex flex-wrap justify-center gap-3 mb-8">
                    {modalities.map(m => (
                        <button
                            key={m.id}
                            onClick={() => setSelectedModality(selectedModality === m.id ? null : m.id)}
                            className={`p-4 rounded-xl transition-all ${
                                selectedModality === m.id
                                    ? 'bg-brand-500 shadow-lg'
                                    : 'bg-stone-800 hover:bg-stone-700'
                            }`}
                        >
                            <div className="text-2xl mb-1">{m.icon}</div>
                            <p className="text-sm font-medium">{m.name}</p>
                        </button>
                    ))}
                </div>

                {/* Central model */}
                <div className="text-center mb-8">
                    <div className="inline-block p-6 bg-gradient-to-br from-brand-500 to-brand-600 rounded-2xl">
                        <div className="w-12 h-12 rounded-xl bg-white/20 flex items-center justify-center mx-auto mb-2">
                            <Brain size={24} className="text-white" />
                        </div>
                        <p className="font-bold">Foundation Model</p>
                        <p className="text-sm text-brand-100">Unified representation</p>
                    </div>
                </div>

                {/* Capability examples */}
                <div className="grid md:grid-cols-2 gap-3">
                    {capabilities
                        .filter(c => !selectedModality || c.from === selectedModality || c.to === selectedModality)
                        .map((cap, i) => (
                            <div key={i} className="p-3 bg-stone-800 rounded-lg flex items-center gap-3">
                                <span className="text-xl">{modalities.find(m => m.id === cap.from)?.icon}</span>
                                <span className="text-stone-500">→</span>
                                <span className="text-xl">{modalities.find(m => m.id === cap.to)?.icon}</span>
                                <span className="text-sm text-stone-400 flex-1">{cap.example}</span>
                            </div>
                        ))
                    }
                </div>
            </div>
        </div>
    );
};

export const InferenceTechniques = () => {
    const [selectedTechnique, setSelectedTechnique] = useState<string | null>(null);

    const techniques = [
        {
            id: 'quantization',
            name: 'Quantization',
            icon: 'binary',
            speedup: '2-4x',
            description: 'Reduce precision from FP16 to INT8/INT4. Massive speedup with minimal quality loss.',
            details: ['FP16 → INT8: ~2x faster, ~1% quality drop', 'INT8 → INT4: Another 2x, more quality impact', 'Best for inference, not training', 'Tools: GPTQ, AWQ, bitsandbytes']
        },
        {
            id: 'kvcache',
            name: 'KV Caching',
            icon: 'harddrive',
            speedup: '10x+',
            description: 'Store Key/Value pairs from previous tokens instead of recomputing them.',
            details: ['Essential for autoregressive generation', 'Memory grows linearly with sequence length', 'Paged Attention optimizes memory allocation', 'Enables efficient long-context inference']
        },
        {
            id: 'speculative',
            name: 'Speculative Decoding',
            icon: 'rocket',
            speedup: '2-3x',
            description: 'Small fast model drafts tokens, big model verifies in parallel.',
            details: ['Draft model: 7B parameters, fast', 'Target model: 70B+ parameters, accurate', 'Verify multiple tokens in one forward pass', 'Output identical to target model alone']
        },
        {
            id: 'batching',
            name: 'Continuous Batching',
            icon: 'boxes',
            speedup: '3-5x throughput',
            description: 'Don\'t wait for all sequences to finish. Process new requests as slots open.',
            details: ['Traditional: Wait for slowest sequence', 'Continuous: Slot-based scheduling', 'Higher GPU utilization', 'Lower average latency']
        },
    ];

    const selected = techniques.find(t => t.id === selectedTechnique);

    return (
        <div className="my-12">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
                {techniques.map(tech => (
                    <button
                        key={tech.id}
                        onClick={() => setSelectedTechnique(selectedTechnique === tech.id ? null : tech.id)}
                        className={`p-4 rounded-xl border-2 text-center transition-all ${
                            selectedTechnique === tech.id
                                ? 'border-brand-500 bg-brand-50 shadow-lg'
                                : 'border-stone-200 bg-white hover:border-stone-300'
                        }`}
                    >
                        <div className="text-3xl mb-2">{tech.icon}</div>
                        <p className="font-bold text-stone-900 text-sm">{tech.name}</p>
                        <p className="text-xs text-stone-600 font-medium mt-1">{tech.speedup}</p>
                    </button>
                ))}
            </div>

            {selected && (
                <div className="bg-white border border-stone-200 rounded-2xl overflow-hidden shadow-lg">
                    <div className="bg-stone-900 text-white p-6">
                        <div className="flex items-center gap-3">
                            <div className="w-12 h-12 rounded-xl bg-brand-500 flex items-center justify-center">
                                <IconRenderer icon={selected.icon} size={24} className="text-white" />
                            </div>
                            <div>
                                <h3 className="text-xl font-bold">{selected.name}</h3>
                                <span className="text-brand-400 font-medium">{selected.speedup} improvement</span>
                            </div>
                        </div>
                        <p className="text-stone-300 mt-3">{selected.description}</p>
                    </div>
                    <div className="p-6">
                        <h4 className="font-bold text-stone-900 mb-3">Key Points</h4>
                        <div className="space-y-2">
                            {selected.details.map((detail, i) => (
                                <div key={i} className="flex gap-3 text-sm">
                                    <span className="w-5 h-5 rounded-full bg-brand-100 text-brand-600 flex items-center justify-center text-xs font-bold shrink-0">
                                        {i + 1}
                                    </span>
                                    <span className="text-stone-600">{detail}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            )}

            {!selected && (
                <p className="text-center text-sm text-stone-400">Click a technique to learn how it works</p>
            )}
        </div>
    );
};

export const Ch2Summary = () => {
    const takeaways = [
        {
            number: '01',
            title: 'Scale Creates Capabilities',
            summary: 'Foundation models aren\'t just bigger—they\'re qualitatively different. Emergent abilities like reasoning, in-context learning, and code generation appear at scale thresholds without explicit training.',
            action: 'Understand which capabilities emerge at which scale to choose the right model.'
        },
        {
            number: '02',
            title: 'Training Is Four Phases',
            summary: 'Pre-training gives knowledge. Supervised Fine-Tuning gives format. Reinforcement Learning from Human Feedback gives alignment. Reasoning Training enables deep thinking. Each phase is essential.',
            action: 'When a model misbehaves, identify which phase failed: knowledge, format, alignment, or reasoning.'
        },
        {
            number: '03',
            title: 'Tokens Are the Unit',
            summary: 'Everything in LLM-land is measured in tokens: cost, context, speed. Understanding tokenization explains why some content is expensive, why context fills up, and why models struggle with certain inputs.',
            action: 'Always estimate token counts. Use tiktoken or similar to check before hitting API limits.'
        },
        {
            number: '04',
            title: 'Decoding Shapes Output',
            summary: 'Temperature, top-p, and penalties aren\'t magic—they reshape the probability distribution the model samples from. Understanding this lets you control creativity vs. consistency.',
            action: 'Use T=0 for factual tasks, T=0.7-1.0 for creative tasks. Add frequency penalty to reduce repetition.'
        },
        {
            number: '05',
            title: 'Architecture Enables Everything',
            summary: 'Transformers, attention, MoE, and long context aren\'t academic details—they determine what\'s possible. MoE gives trillion-param quality at lower cost. Long context eliminates chunking complexity.',
            action: 'Pick models based on architecture fit: MoE for quality/cost, long context for RAG.'
        },
        {
            number: '06',
            title: 'Optimization Is Essential',
            summary: 'Raw model inference is too slow and expensive for production. Quantization, KV caching, and batching can give 10x improvements. This is table stakes for production AI.',
            action: 'Never deploy without quantization (INT8 minimum). Always use streaming for UX.'
        },
    ];

    return (
        <div className="my-12">
            <div className="text-center mb-10">
                <h3 className="text-2xl font-bold text-stone-900 mb-2">Chapter 2 Key Takeaways</h3>
                <p className="text-stone-500">Master these concepts to work effectively with foundation models</p>
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
