'use client';

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
import { BookOpen, ArrowRight, Lightbulb, ChevronRight, Check, Bookmark, Search, Copy } from 'lucide-react';
import { useLearner } from '../context/LearnerContext';
import { SKILL_HOME } from '../lib/courseOrder';

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
  const { isComplete, isBookmarked, toggleComplete, toggleBookmark } = useLearner();
  const complete = isComplete(chapter.id);
  const bookmarked = isBookmarked(chapter.id);

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
      .replace(/`([^`]+)`/g, '<code class="font-mono text-[0.85em] bg-stone-100 text-stone-800 px-1.5 py-0.5 border border-stone-200/80">$1</code>')
      .replace(/\*\*(.*?)\*\*/g, '<strong class="font-semibold text-stone-900">$1</strong>')
      .replace(/(?<!\*)\*(?!\*)(.*?)\*/g, '<em class="italic font-serif text-stone-700">$1</em>')
      .replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2" target="_blank" class="text-brand-600 font-medium underline decoration-brand-400/50 hover:decoration-brand-500 underline-offset-2 transition-all">$1</a>');
  };

  // --- CONCEPT CARDS: Replaces bullet lists with definition-style items ---
  const ConceptCards = ({ items }: { items: string[] }) => {
    return (
      <dl className="my-10 divide-y divide-stone-200 border-y border-stone-200">
        {items.map((item, i) => {
          const match = item.match(/^\*\*(.*?)\*\*[:\s]*(?:--|—)?\s*(.*)$/);
          const term = match ? match[1] : null;
          const description = match ? match[2] : item;

          return (
            <div key={i} className="grid grid-cols-1 gap-2 py-5 sm:grid-cols-[auto_1fr] sm:gap-6">
              {term && (
                <dt className="font-serif text-lg leading-snug text-stone-900">{term}</dt>
              )}
              <dd
                className="text-pretty text-sm sm:text-base leading-relaxed text-stone-600"
                dangerouslySetInnerHTML={{ __html: formatInline(description) }}
              />
            </div>
          );
        })}
      </dl>
    );
  };

  // --- PROCESS STEPS: For ordered lists ---
  const ProcessSteps = ({ items }: { items: string[] }) => {
    return (
      <dl className="my-10 divide-y divide-stone-200 border-y border-stone-200">
        {items.map((item, i) => {
          const match = item.match(/^\*\*(.*?)\*\*[:\s]*(?:--|—)?\s*(.*)$/);
          const title = match ? match[1] : null;
          const description = match ? match[2] : item;
          const num = String(i + 1).padStart(2, '0');

          return (
            <div key={i} className="grid grid-cols-1 gap-2 py-5 sm:grid-cols-[auto_1fr] sm:gap-6">
              <dt className="flex items-baseline gap-3">
                <span className="font-mono text-xs text-brand-500">{num}</span>
                {title && (
                  <span className="font-serif text-lg leading-snug text-stone-900">{title}</span>
                )}
              </dt>
              <dd
                className="text-pretty text-sm sm:text-base leading-relaxed text-stone-600"
                dangerouslySetInnerHTML={{ __html: formatInline(description) }}
              />
            </div>
          );
        })}
      </dl>
    );
  };

  // --- FEATURE GRID: For simple unordered lists without definitions ---
  const FeatureGrid = ({ items }: { items: string[] }) => {
    return (
      <ul className="my-8 space-y-2 text-stone-800">
        {items.map((item, i) => {
          const match = item.match(/^\*\*(.*?)\*\*[:\s]*(?:--|—)?\s*(.*)$/);
          const title = match ? match[1] : null;
          const description = match ? match[2] : item;

          return (
            <li key={i} className="flex items-start gap-3 text-sm sm:text-base leading-relaxed">
              <span className="text-brand-500 mt-0.5">→</span>
              <span>
                {title && <strong className="font-semibold text-stone-900">{title} </strong>}
                <span
                  className="text-stone-600"
                  dangerouslySetInnerHTML={{ __html: formatInline(description) }}
                />
              </span>
            </li>
          );
        })}
      </ul>
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
    // Split on | and drop the leading/trailing empty strings from outer pipes
    const splitRow = (r: string) => {
      const cells = r.split('|');
      // Remove first and last (empty from leading/trailing |)
      if (cells.length > 0 && cells[0].trim() === '') cells.shift();
      if (cells.length > 0 && cells[cells.length - 1].trim() === '') cells.pop();
      return cells;
    };
    const headers = splitRow(rows[0]);
    const data = rows.slice(2).map(splitRow);

    return (
      <div className="my-10 overflow-x-auto -mx-4 px-4 sm:mx-0 sm:px-0">
        <div className="min-w-[480px] border border-stone-200 overflow-hidden">
          <table className="w-full">
            <thead>
              <tr className="bg-stone-900 text-white">
                {headers.map((h, i) => (
                  <th key={i} className="px-3 py-3 sm:px-6 sm:py-4 text-left text-xs sm:text-sm font-semibold tracking-wide" dangerouslySetInnerHTML={{ __html: formatInline(h.trim()) }} />
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-stone-100">
              {data.map((row, i) => (
                <tr key={i} className="bg-white hover:bg-stone-50 transition-colors">
                  {row.map((cell, j) => (
                    <td key={j} className="px-3 py-3 sm:px-6 sm:py-4 text-xs sm:text-sm text-stone-600">
                      {j === 0 ? (
                        <span className="font-medium text-stone-900" dangerouslySetInnerHTML={{ __html: formatInline(cell.trim()) }} />
                      ) : (
                        <span dangerouslySetInnerHTML={{ __html: formatInline(cell.trim()) }} />
                      )}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  };

  // --- Main Render ---
  const blocks = parseBlocks(chapter.content);
  const skill = SKILL_HOME[chapter.id];
  const chapterNum = chapter.id.replace('ch', '').padStart(2, '0');

  return (
    <div className="bg-stone-50 min-h-screen">
      <div className="max-w-3xl mx-auto px-6 sm:px-8 py-8 sm:py-12 md:py-16">
        
        {blocks.map((block, index) => {
          switch (block.type) {
            case 'h1':
              return (
                <div key={index} className="relative overflow-hidden -mx-6 sm:-mx-8 mb-14 bg-stone-900 text-white">
                  <div
                    className="pointer-events-none absolute inset-0 opacity-[0.07]"
                    style={{
                      backgroundImage:
                        'linear-gradient(to right, #c8a96e 1px, transparent 1px), linear-gradient(to bottom, #c8a96e 1px, transparent 1px)',
                      backgroundSize: '48px 48px',
                    }}
                  />
                  <div className="relative px-6 sm:px-8 py-16 sm:py-20">
                    <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-brand-400">
                      Chapter {chapterNum}
                      {skill ? ` · ${skill.capability}` : ''}
                    </p>
                    <h1 className="mt-6 max-w-3xl text-balance font-serif font-normal text-4xl leading-[1.05] sm:text-5xl">
                      {block.content[0].replace(/^.*?:\s*/, '').replace(' - Complete Notes', '')}
                    </h1>
                    {skill && skill.skill > 0 && (
                      <p className="mt-6 max-w-2xl text-pretty text-sm sm:text-base leading-relaxed text-white/60">
                        Skills map · Skill {skill.skill} · {skill.label}
                      </p>
                    )}
                  </div>
                </div>
              );

            case 'h2':
              return (
                <div key={index} className="mt-16 mb-6 pt-10 border-t border-stone-200">
                  <h2 className="font-serif font-normal text-3xl leading-[1.1] text-stone-900 sm:text-4xl">
                    {block.content[0]}
                  </h2>
                </div>
              );

            case 'h3':
              return (
                <h3 key={index} className="mt-12 mb-4 font-serif text-xl sm:text-2xl leading-snug text-stone-900">
                  {block.content[0]}
                </h3>
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
                  className="max-w-2xl text-pretty text-base sm:text-lg leading-relaxed text-stone-600 mb-6"
                  dangerouslySetInnerHTML={{ __html: formatInline(block.content.join(' ')) }}
                />
              );

            // --- Lists (no bullets!) ---
            case 'ul':
            case 'ol':
              return <div key={index}>{renderList(block)}</div>;

            // --- Blockquotes ---
            case 'blockquote': {
              const joined = block.content.join(' ');
              const isLookUp = /^\*\*Look up now\.\*\*/i.test(block.content[0] || '') || /Look up now/i.test(joined);
              const promptText = block.content
                .join('\n')
                .replace(/^\*\*Look up now\.\*\*\s*/i, '')
                .replace(/^Prompt:\s*/i, '')
                .replace(/^"|"$/g, '');
              return (
                <div key={index} className="my-10">
                  <blockquote className="border-l-2 border-brand-400 pl-5">
                    {isLookUp && (
                      <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-brand-600 mb-3 flex items-center gap-2">
                        <Search className="w-3.5 h-3.5" /> Look up now
                      </p>
                    )}
                    <div className={`${isLookUp ? 'text-base text-stone-700 leading-relaxed' : 'font-serif text-2xl italic leading-snug text-stone-900 sm:text-3xl'}`}>
                      {block.content.map((line, i) => (
                        <p key={i} dangerouslySetInnerHTML={{ __html: formatInline(line.replace(/^\*\*Look up now\.\*\*\s*/i, '').replace(/^Prompt:\s*/i, '')) }} />
                      ))}
                    </div>
                    {isLookUp && (
                      <button
                        type="button"
                        onClick={() => navigator.clipboard.writeText(promptText)}
                        className="mt-4 inline-flex items-center justify-center border border-stone-200 bg-white px-4 py-2 text-sm font-semibold text-stone-900 hover:bg-stone-50"
                      >
                        <Copy className="w-4 h-4 mr-2" />
                        Copy look-up prompt
                      </button>
                    )}
                  </blockquote>
                </div>
              );
            }

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

            <div className="flex flex-wrap items-center justify-center gap-3 mb-8">
              <button
                onClick={() => toggleBookmark(chapter.id)}
                className={`flex items-center gap-2 px-5 py-2.5 border text-sm font-semibold transition-colors ${
                  bookmarked
                    ? 'bg-brand-500 text-stone-900 border-brand-500'
                    : 'bg-transparent border-stone-200 text-stone-700 hover:bg-stone-100'
                }`}
              >
                <Bookmark size={16} className={bookmarked ? 'fill-brand-500 text-brand-500' : ''} />
                {bookmarked ? 'Saved' : 'Save for later'}
              </button>
              <button
                onClick={() => toggleComplete(chapter.id)}
                className={`flex items-center gap-2 px-5 py-2.5 border text-sm font-semibold transition-colors ${
                  complete
                    ? 'bg-emerald-800 text-white border-emerald-800'
                    : 'bg-transparent border-stone-200 text-stone-700 hover:bg-stone-100'
                }`}
              >
                <Check size={16} />
                {complete ? 'Completed' : 'Mark complete'}
              </button>
            </div>
            
            {nextChapter && onNextChapter && (
              <button
                onClick={() => {
                  if (!complete) toggleComplete(chapter.id);
                  onNextChapter();
                }}
                className="group flex items-center gap-3 px-6 py-3 bg-brand-400 hover:bg-brand-300 text-stone-900 transition-colors"
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
