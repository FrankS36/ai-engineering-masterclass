import React from 'react';
import { Chapter } from '../types';
import { 
    FoundationModelIllustration,
    MetricsDashboardIllustration,
    CLIPTrainingIllustration
} from './Illustrations';
import { 
    WorkflowCompare, 
    ModelTypes, 
    PromptSandbox, 
    UseCaseCarousel, 
    MilestoneTimeline,
    PlanningFramework,
    TechStack,
    RoleComparison,
    EvaluationVisual,
    PromptPatterns,
    SummaryGrid,
    ConvergenceForces,
    ModelComparison,
    CostOptimization,
    // Chapter 2 components
    FoundationPillars,
    ScalingLaws,
    TransformerVisual,
    TrainingPipeline,
    RLHFDemo,
    TokenizerDemo,
    DecodingStrategies,
    ContextEvolution,
    MOEVisual,
    MultimodalFlow,
    InferenceTechniques,
    Ch2Summary,
    TrainingData
} from './InteractiveElements';
import {
    PromptAnatomy,
    PromptingStrategies,
    SystemPromptBuilder,
    InjectionDemo,
    Ch3Summary
} from './Ch3Components';
import {
    APIAnatomy,
    StreamingDemo,
    FunctionCallingVisual,
    ErrorPatterns,
    Ch4Summary
} from './Ch4Components';
import {
    RAGMotivation,
    RAGPipeline,
    RetrievalStrategies,
    RAGEval,
    Ch5Summary
} from './Ch5Components';
import {
    AgentDefinition,
    ReActLoop,
    PlanningStrategies,
    MultiAgentPatterns,
    Ch6Summary
} from './Ch6Components';
import {
    EvalImportance,
    EvalSets,
    EvalMetrics,
    RedTeam,
    Ch7Summary
} from './Ch7Components';
import {
    ProductionGap,
    Observability,
    Ch8Summary
} from './Ch8Components';
import {
    FinetuneDecision,
    DataPrep,
    Ch9Summary
} from './Ch9Components';
import {
    UseCaseEval,
    BuildBuy,
    Ch10Summary
} from './Ch10Components';
import { Sparkles, Zap, BookOpen, ArrowRight, Lightbulb, ChevronRight } from 'lucide-react';

interface ChapterViewProps {
  chapter: Chapter;
  nextChapter?: Chapter;
  onNextChapter?: () => void;
}

// --- Parser Types ---
type BlockType = 
  | 'h1' | 'h2' | 'h3' | 'h4'
  | 'p' 
  | 'ul' | 'ol' 
  | 'blockquote' 
  | 'code-block' 
  | 'hr' 
  | 'table'
  | 'interactive' 
  | 'illustration';

interface Block {
  type: BlockType;
  content: string[];
  meta?: string;
}

export const ChapterView: React.FC<ChapterViewProps> = ({ chapter, nextChapter, onNextChapter }) => {

  // --- Parser Engine ---
  const parseBlocks = (markdown: string): Block[] => {
    const lines = markdown.split('\n');
    const blocks: Block[] = [];
    let currentBlock: Block | null = null;

    const finalizeBlock = () => {
      if (currentBlock) {
        blocks.push(currentBlock);
        currentBlock = null;
      }
    };

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const trimmed = line.trim();

      // CODE BLOCKS
      if (currentBlock?.type === 'code-block') {
        if (trimmed.startsWith('```')) {
          finalizeBlock();
        } else {
          currentBlock.content.push(line); 
        }
        continue;
      }

      // LISTS
      if (currentBlock?.type === 'ul') {
        if (trimmed.startsWith('* ') || trimmed.startsWith('- ')) {
          currentBlock.content.push(trimmed.substring(2));
          continue;
        } else if (trimmed !== '' && !trimmed.startsWith('#') && !trimmed.startsWith('[') && !trimmed.startsWith('```')) {
          const lastIdx = currentBlock.content.length - 1;
          if (lastIdx >= 0) {
             currentBlock.content[lastIdx] += ' ' + trimmed;
             continue;
          }
        }
        finalizeBlock();
      }
      
      if (currentBlock?.type === 'ol') {
        if (/^\d+\./.test(trimmed)) {
          const content = trimmed.replace(/^\d+\.\s*/, '');
          currentBlock.content.push(content);
          continue;
        } else if (trimmed !== '' && !trimmed.startsWith('#') && !trimmed.startsWith('[') && !trimmed.startsWith('```')) {
           const lastIdx = currentBlock.content.length - 1;
           if (lastIdx >= 0) {
              currentBlock.content[lastIdx] += ' ' + trimmed;
              continue;
           }
        }
        finalizeBlock();
      }

      // BLOCKQUOTES
      if (currentBlock?.type === 'blockquote') {
        if (trimmed.startsWith('> ')) {
          currentBlock.content.push(trimmed.substring(2));
          continue;
        } else if (trimmed !== '' && !trimmed.startsWith('#')) {
           currentBlock.content.push(trimmed);
           continue;
        }
        finalizeBlock();
      }

      // TABLE
      if (currentBlock?.type === 'table') {
        if (trimmed.startsWith('|')) {
           currentBlock.content.push(trimmed);
           continue;
        }
        finalizeBlock();
      }

      // --- Start New Block Detection ---

      if (trimmed === '') {
        finalizeBlock();
        continue;
      }

      // Interactive
      if (trimmed.startsWith('[INTERACTIVE:')) {
        finalizeBlock();
        const id = trimmed.match(/\[INTERACTIVE:\s*(.*?)\]/)?.[1] || '';
        blocks.push({ type: 'interactive', content: [], meta: id });
        continue;
      }

      // Illustrations
      if (trimmed.startsWith('[ILLUSTRATION:')) {
        finalizeBlock();
        const id = trimmed.match(/\[ILLUSTRATION:\s*(.*?)\]/)?.[1] || '';
        blocks.push({ type: 'illustration', content: [], meta: id });
        continue;
      }

      // Headers
      const headerMatch = trimmed.match(/^(#{1,6})\s+(.*)$/);
      if (headerMatch) {
        finalizeBlock();
        const level = headerMatch[1].length;
        const text = headerMatch[2];
        blocks.push({ type: `h${level > 4 ? 4 : level}` as any, content: [text] });
        continue;
      }

      // Code Block Start
      if (trimmed.startsWith('```')) {
        finalizeBlock();
        const lang = trimmed.substring(3).trim();
        currentBlock = { type: 'code-block', content: [], meta: lang };
        continue;
      }

      // Blockquotes
      if (trimmed.startsWith('> ')) {
        finalizeBlock();
        currentBlock = { type: 'blockquote', content: [trimmed.substring(2)] };
        continue;
      }

      // Tables
      if (trimmed.startsWith('|')) {
        finalizeBlock();
        currentBlock = { type: 'table', content: [trimmed] };
        continue;
      }

      // Lists
      if (trimmed.startsWith('* ') || trimmed.startsWith('- ')) {
        finalizeBlock();
        currentBlock = { type: 'ul', content: [trimmed.substring(2)] };
        continue;
      }
      if (/^\d+\./.test(trimmed)) {
        finalizeBlock();
        const content = trimmed.replace(/^\d+\.\s*/, '');
        currentBlock = { type: 'ol', content: [content] };
        continue;
      }

      // Horizontal Rule
      if (trimmed === '---' || trimmed === '***') {
        finalizeBlock();
        blocks.push({ type: 'hr', content: [] });
        continue;
      }

      // Paragraphs
      if (currentBlock?.type === 'p') {
        currentBlock.content.push(trimmed);
      } else {
        finalizeBlock();
        currentBlock = { type: 'p', content: [trimmed] };
      }
    }
    
    finalizeBlock();
    return blocks;
  };

  // --- Inline Formatting Helper ---
  const formatInline = (text: string) => {
    return text
      .replace(/`([^`]+)`/g, '<code class="font-mono text-[0.85em] bg-stone-100 text-stone-800 px-1.5 py-0.5 rounded border border-stone-200/80">$1</code>')
      .replace(/\*\*(.*?)\*\*/g, '<strong class="font-semibold text-stone-900">$1</strong>')
      .replace(/(?<!\*)\*(?!\*)(.*?)\*/g, '<em class="italic text-stone-700">$1</em>')
      .replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2" target="_blank" class="text-brand-600 font-medium underline decoration-brand-300 hover:decoration-brand-500 underline-offset-2 transition-all">$1</a>');
  };

  // --- CONCEPT CARDS: Replaces bullet lists with definition-style items ---
  const ConceptCards = ({ items }: { items: string[] }) => {
    return (
      <div className="my-10 grid gap-4">
        {items.map((item, i) => {
          // Parse "**Term**: Description" format
          const match = item.match(/^\*\*(.*?)\*\*[:\s]+(.*)$/);
          const term = match ? match[1] : null;
          const description = match ? match[2] : item;
          
          return (
            <div 
              key={i} 
              className="group relative bg-gradient-to-br from-white to-stone-50/50 rounded-2xl border border-stone-200/80 overflow-hidden hover:border-stone-300 hover:shadow-lg transition-all duration-300"
            >
              <div className="absolute top-0 left-0 w-1 h-full bg-gradient-to-b from-brand-400 to-brand-600 opacity-0 group-hover:opacity-100 transition-opacity" />
              <div className="p-6 flex items-start gap-5">
                {term && (
                  <div className="shrink-0 w-52">
                    <span className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-stone-900 text-white text-sm font-semibold tracking-tight">
                      <Zap size={14} className="text-brand-400" />
                      {term}
                    </span>
                  </div>
                )}
                <p 
                  className="flex-1 text-stone-600 leading-relaxed text-[16px]"
                  dangerouslySetInnerHTML={{ __html: formatInline(description) }}
                />
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  // --- PROCESS STEPS: For ordered lists ---
  const ProcessSteps = ({ items }: { items: string[] }) => {
    return (
      <div className="my-10 relative">
        {/* Connecting line */}
        <div className="absolute left-6 top-8 bottom-8 w-px bg-gradient-to-b from-brand-300 via-brand-400 to-brand-300" />
        
        <div className="space-y-6">
          {items.map((item, i) => {
            const match = item.match(/^\*\*(.*?)\*\*[:\s]*(.*)$/);
            const title = match ? match[1] : `Step ${i + 1}`;
            const description = match ? match[2] : item;
            
            return (
              <div key={i} className="relative flex gap-6 group">
                {/* Step number */}
                <div className="relative z-10 w-12 h-12 rounded-xl bg-stone-900 text-white flex items-center justify-center font-bold text-lg shadow-lg group-hover:bg-brand-600 transition-colors">
                  {i + 1}
                </div>
                
                {/* Content */}
                <div className="flex-1 pt-1">
                  <h4 className="text-lg font-bold text-stone-900 mb-2">{title}</h4>
                  <p 
                    className="text-stone-600 leading-relaxed"
                    dangerouslySetInnerHTML={{ __html: formatInline(description) }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  // --- FEATURE GRID: For simple unordered lists without definitions ---
  const FeatureGrid = ({ items }: { items: string[] }) => {
    return (
      <div className="my-10 grid md:grid-cols-2 lg:grid-cols-3 gap-4">
        {items.map((item, i) => {
          const match = item.match(/^\*\*(.*?)\*\*[:\s]*(.*)$/);
          const title = match ? match[1] : null;
          const description = match ? match[2] : item;
          
          return (
            <div 
              key={i} 
              className="p-5 rounded-xl bg-white border border-stone-200 hover:border-brand-300 hover:shadow-md transition-all group"
            >
              {title ? (
                <>
                  <div className="flex items-center gap-2 mb-3">
                    <div className="w-2 h-2 rounded-full bg-brand-500 group-hover:scale-125 transition-transform" />
                    <h4 className="font-bold text-stone-900">{title}</h4>
                  </div>
                  <p 
                    className="text-sm text-stone-600 leading-relaxed"
                    dangerouslySetInnerHTML={{ __html: formatInline(description) }}
                  />
                </>
              ) : (
                <p 
                  className="text-stone-700 leading-relaxed"
                  dangerouslySetInnerHTML={{ __html: formatInline(description) }}
                />
              )}
            </div>
          );
        })}
      </div>
    );
  };

  // --- Smart list renderer ---
  const renderList = (block: Block) => {
    const items = block.content;
    const isOrdered = block.type === 'ol';
    
    // Check if items have "**Term**: Description" format
    const hasDefinitions = items.some(item => /^\*\*(.*?)\*\*[:\-]/.test(item));
    
    if (isOrdered) {
      return <ProcessSteps items={items} />;
    }
    
    if (hasDefinitions) {
      return <ConceptCards items={items} />;
    }
    
    return <FeatureGrid items={items} />;
  };

  // --- Table Renderer ---
  const renderTable = (rows: string[]) => {
    const headers = rows[0].split('|').filter(c => c.trim());
    const data = rows.slice(2).map(r => r.split('|').filter(c => c.trim()));

    return (
      <div className="my-10 overflow-hidden rounded-2xl border border-stone-200 shadow-sm">
        <table className="w-full">
          <thead>
            <tr className="bg-stone-900 text-white">
              {headers.map((h, i) => (
                <th key={i} className="px-6 py-4 text-left text-sm font-semibold tracking-wide">
                  {h.trim()}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-stone-100">
            {data.map((row, i) => (
              <tr key={i} className="bg-white hover:bg-stone-50 transition-colors">
                {row.map((cell, j) => (
                  <td key={j} className="px-6 py-4 text-stone-600">
                    {j === 0 ? (
                      <span className="font-medium text-stone-900">{cell.trim()}</span>
                    ) : (
                      cell.trim()
                    )}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  // --- Main Render ---
  const blocks = parseBlocks(chapter.content);

  return (
    <div className="bg-gradient-to-b from-stone-50 to-white min-h-screen">
      <div className="max-w-4xl mx-auto px-6 md:px-8 py-12 md:py-20">
        
        {blocks.map((block, index) => {
          switch (block.type) {
            // --- H1: Chapter Title - Full-width hero style ---
            case 'h1':
              return (
                <div key={index} className="mb-16 -mx-6 md:-mx-8 px-6 md:px-8 py-12 bg-stone-900 rounded-3xl relative overflow-hidden">
                  {/* Background pattern */}
                  <div className="absolute inset-0 opacity-10">
                    <div className="absolute top-0 right-0 w-96 h-96 bg-brand-500 rounded-full blur-3xl" />
                    <div className="absolute bottom-0 left-0 w-64 h-64 bg-brand-400 rounded-full blur-3xl" />
                  </div>
                  
                  <div className="relative">
                    <div className="flex items-center gap-3 mb-4">
                      <div className="p-2 rounded-lg bg-brand-500/20 border border-brand-500/30">
                        <BookOpen size={20} className="text-brand-400" />
                      </div>
                      <span className="text-brand-400 text-sm font-mono tracking-wider uppercase">
                        Chapter Notes
                      </span>
                    </div>
                    <h1 className="text-3xl md:text-4xl lg:text-5xl font-extrabold text-white tracking-tight leading-tight">
                      {block.content[0].replace(/^.*?:\s*/, '').replace(' - Complete Notes', '')}
                    </h1>
                  </div>
                </div>
              );

            // --- H2: Section headers ---
            case 'h2':
              return (
                <div key={index} className="mt-20 mb-8 pt-8 border-t border-stone-200">
                  <div className="flex items-start gap-4">
                    <div className="shrink-0 w-10 h-10 rounded-xl bg-gradient-to-br from-brand-500 to-brand-600 flex items-center justify-center shadow-lg shadow-brand-500/20">
                      <Sparkles size={18} className="text-white" />
                    </div>
                    <h2 className="text-2xl md:text-3xl font-bold text-stone-900 tracking-tight leading-snug">
                      {block.content[0]}
                    </h2>
                  </div>
                </div>
              );

            // --- H3: Subsection headers ---
            case 'h3':
              return (
                <div key={index} className="mt-12 mb-6">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-0.5 bg-brand-400 rounded-full" />
                    <h3 className="text-xl font-bold text-stone-800">
                      {block.content[0]}
                    </h3>
                  </div>
                </div>
              );

            // --- H4: Minor headers ---
            case 'h4':
              return (
                <h4 key={index} className="text-lg font-semibold text-stone-700 mt-8 mb-4 flex items-center gap-2">
                  <ArrowRight size={16} className="text-brand-500" />
                  {block.content[0]}
                </h4>
              );

            // --- Paragraphs ---
            case 'p':
              return (
                <p 
                  key={index} 
                  className="text-[17px] leading-8 text-stone-600 mb-6"
                  dangerouslySetInnerHTML={{ __html: formatInline(block.content.join(' ')) }}
                />
              );

            // --- Lists (no bullets!) ---
            case 'ul':
            case 'ol':
              return <div key={index}>{renderList(block)}</div>;

            // --- Blockquotes ---
            case 'blockquote':
              return (
                <div key={index} className="my-10 relative">
                  <div className="absolute -left-4 top-0 bottom-0 w-1 bg-gradient-to-b from-brand-400 to-brand-600 rounded-full" />
                  <div className="bg-gradient-to-br from-brand-50 to-orange-50 rounded-2xl p-6 pl-8 border border-brand-100">
                    <Lightbulb className="text-brand-500 w-6 h-6 mb-3" />
                    <div className="text-lg text-stone-700 leading-relaxed">
                      {block.content.map((line, i) => (
                        <p key={i} dangerouslySetInnerHTML={{ __html: formatInline(line) }} />
                      ))}
                    </div>
                  </div>
                </div>
              );

            // --- Code blocks ---
            case 'code-block':
              return (
                <div key={index} className="my-8 rounded-2xl overflow-hidden bg-stone-900 shadow-2xl ring-1 ring-white/10">
                  <div className="flex items-center justify-between px-4 py-3 bg-stone-800/50 border-b border-stone-700/50">
                    <div className="flex gap-2">
                      <div className="w-3 h-3 rounded-full bg-red-500/80" />
                      <div className="w-3 h-3 rounded-full bg-yellow-500/80" />
                      <div className="w-3 h-3 rounded-full bg-green-500/80" />
                    </div>
                    <span className="text-xs font-mono text-stone-500 uppercase tracking-wider">
                      {block.meta || 'code'}
                    </span>
                  </div>
                  <div className="p-6 overflow-x-auto">
                    <pre className="font-mono text-sm leading-relaxed text-stone-300">
                      {block.content.join('\n')}
                    </pre>
                  </div>
                </div>
              );

            // --- Tables ---
            case 'table':
              return <div key={index}>{renderTable(block.content)}</div>;

            // --- Illustrations ---
            case 'illustration':
              if (block.meta === 'foundation_model') return <FoundationModelIllustration key={index} />;
              if (block.meta === 'metrics_dashboard') return <MetricsDashboardIllustration key={index} />;
              if (block.meta === 'clip_training') return <CLIPTrainingIllustration key={index} />;
              return null;

            // --- Interactive components ---
            case 'interactive':
              if (block.meta === 'WORKFLOW_COMPARE') return <WorkflowCompare key={index} />;
              if (block.meta === 'MODEL_TYPES') return <ModelTypes key={index} />;
              if (block.meta === 'PROMPT_SANDBOX') return <PromptSandbox key={index} />;
              if (block.meta === 'USE_CASE_CAROUSEL') return <UseCaseCarousel key={index} />;
              if (block.meta === 'MILESTONE_TIMELINE') return <MilestoneTimeline key={index} />;
              if (block.meta === 'PLANNING_FRAMEWORK') return <PlanningFramework key={index} />;
              if (block.meta === 'TECH_STACK') return <TechStack key={index} />;
              if (block.meta === 'ROLE_COMPARISON') return <RoleComparison key={index} />;
              if (block.meta === 'EVALUATION_VISUAL') return <EvaluationVisual key={index} />;
              if (block.meta === 'PROMPT_PATTERNS') return <PromptPatterns key={index} />;
              if (block.meta === 'SUMMARY_GRID') return <SummaryGrid key={index} />;
              if (block.meta === 'CONVERGENCE_FORCES') return <ConvergenceForces key={index} />;
              if (block.meta === 'MODEL_COMPARISON') return <ModelComparison key={index} />;
              if (block.meta === 'COST_OPTIMIZATION') return <CostOptimization key={index} />;
              // Chapter 2 components
              if (block.meta === 'FOUNDATION_PILLARS') return <FoundationPillars key={index} />;
              if (block.meta === 'SCALING_LAWS') return <ScalingLaws key={index} />;
              if (block.meta === 'TRANSFORMER_VISUAL') return <TransformerVisual key={index} />;
              if (block.meta === 'TRAINING_PIPELINE') return <TrainingPipeline key={index} />;
              if (block.meta === 'TRAINING_DATA') return <TrainingData key={index} />;
              if (block.meta === 'RLHF_DEMO') return <RLHFDemo key={index} />;
              if (block.meta === 'TOKENIZER_DEMO') return <TokenizerDemo key={index} />;
              if (block.meta === 'DECODING_STRATEGIES') return <DecodingStrategies key={index} />;
              if (block.meta === 'CONTEXT_EVOLUTION') return <ContextEvolution key={index} />;
              if (block.meta === 'MOE_VISUAL') return <MOEVisual key={index} />;
              if (block.meta === 'MULTIMODAL_FLOW') return <MultimodalFlow key={index} />;
              if (block.meta === 'INFERENCE_TECHNIQUES') return <InferenceTechniques key={index} />;
              if (block.meta === 'CH2_SUMMARY') return <Ch2Summary key={index} />;
              // Chapter 3 components
              if (block.meta === 'PROMPT_ANATOMY') return <PromptAnatomy key={index} />;
              if (block.meta === 'PROMPTING_STRATEGIES') return <PromptingStrategies key={index} />;
              if (block.meta === 'SYSTEM_PROMPT_BUILDER') return <SystemPromptBuilder key={index} />;
              if (block.meta === 'INJECTION_DEMO') return <InjectionDemo key={index} />;
              if (block.meta === 'CH3_SUMMARY') return <Ch3Summary key={index} />;
              // Chapter 4 components
              if (block.meta === 'API_ANATOMY') return <APIAnatomy key={index} />;
              if (block.meta === 'STREAMING_DEMO') return <StreamingDemo key={index} />;
              if (block.meta === 'FUNCTION_CALLING') return <FunctionCallingVisual key={index} />;
              if (block.meta === 'ERROR_PATTERNS') return <ErrorPatterns key={index} />;
              if (block.meta === 'CH4_SUMMARY') return <Ch4Summary key={index} />;
              // Chapter 5 components
              if (block.meta === 'RAG_MOTIVATION') return <RAGMotivation key={index} />;
              if (block.meta === 'RAG_PIPELINE') return <RAGPipeline key={index} />;
              if (block.meta === 'RETRIEVAL_STRATEGIES') return <RetrievalStrategies key={index} />;
              if (block.meta === 'RAG_EVAL') return <RAGEval key={index} />;
              if (block.meta === 'CH5_SUMMARY') return <Ch5Summary key={index} />;
              // Chapter 6 components
              if (block.meta === 'AGENT_DEFINITION') return <AgentDefinition key={index} />;
              if (block.meta === 'REACT_LOOP') return <ReActLoop key={index} />;
              if (block.meta === 'PLANNING_STRATEGIES') return <PlanningStrategies key={index} />;
              if (block.meta === 'MULTI_AGENT') return <MultiAgentPatterns key={index} />;
              if (block.meta === 'CH6_SUMMARY') return <Ch6Summary key={index} />;
              // Chapter 7 components
              if (block.meta === 'EVAL_IMPORTANCE') return <EvalImportance key={index} />;
              if (block.meta === 'EVAL_SETS') return <EvalSets key={index} />;
              if (block.meta === 'EVAL_METRICS') return <EvalMetrics key={index} />;
              if (block.meta === 'RED_TEAM') return <RedTeam key={index} />;
              if (block.meta === 'CH7_SUMMARY') return <Ch7Summary key={index} />;
              // Chapter 8 components
              if (block.meta === 'PRODUCTION_GAP') return <ProductionGap key={index} />;
              if (block.meta === 'OBSERVABILITY') return <Observability key={index} />;
              if (block.meta === 'CH8_SUMMARY') return <Ch8Summary key={index} />;
              // Chapter 9 components
              if (block.meta === 'FINETUNE_DECISION') return <FinetuneDecision key={index} />;
              if (block.meta === 'DATA_PREP') return <DataPrep key={index} />;
              if (block.meta === 'CH9_SUMMARY') return <Ch9Summary key={index} />;
              // Chapter 10 components
              if (block.meta === 'USE_CASE_EVAL') return <UseCaseEval key={index} />;
              if (block.meta === 'BUILD_BUY') return <BuildBuy key={index} />;
              if (block.meta === 'CH10_SUMMARY') return <Ch10Summary key={index} />;
              return null;

            // --- Horizontal rules ---
            case 'hr':
              return (
                <div key={index} className="my-16 flex items-center justify-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-stone-300" />
                  <div className="w-16 h-px bg-stone-200" />
                  <div className="w-2 h-2 rounded-full bg-stone-300" />
                  <div className="w-16 h-px bg-stone-200" />
                  <div className="w-2 h-2 rounded-full bg-stone-300" />
                </div>
              );

            default:
              return null;
          }
        })}

        {/* End of chapter marker */}
        <div className="mt-24 pt-12 border-t border-stone-200">
          <div className="flex flex-col items-center text-center">
            <div className="w-12 h-12 rounded-full bg-stone-100 flex items-center justify-center mb-4">
              <BookOpen size={20} className="text-stone-400" />
            </div>
            <p className="text-sm font-medium text-stone-400 uppercase tracking-widest mb-6">
              End of Chapter
            </p>
            
            {nextChapter && onNextChapter && (
              <button
                onClick={onNextChapter}
                className="group flex items-center gap-3 px-6 py-4 bg-stone-900 hover:bg-stone-800 text-white rounded-2xl transition-all hover:shadow-lg hover:shadow-stone-900/20"
              >
                <div className="text-left">
                  <p className="text-xs text-stone-400 uppercase tracking-wide">Next Chapter</p>
                  <p className="font-bold text-lg">{nextChapter.title.split(': ')[1] || nextChapter.title}</p>
                </div>
                <ChevronRight size={24} className="text-stone-400 group-hover:text-white group-hover:translate-x-1 transition-all" />
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
