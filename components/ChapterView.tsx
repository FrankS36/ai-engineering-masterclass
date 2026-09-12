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
import { ArrowRight, Check, Bookmark, Search, Copy } from 'lucide-react';
import { useLearner } from '../context/LearnerContext';
import { COURSE_ORDER, SKILL_HOME, chapterOrderLabel, chapterShortTitle } from '../lib/courseOrder';
import { EditorialHero, ArticleShell } from './EditorialChrome';

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
      .replace(/`([^`]+)`/g, '<code class="font-mono text-[0.85em] bg-card-bg text-foreground px-1.5 py-0.5 border border-border">$1</code>')
      .replace(/\*\*(.*?)\*\*/g, '<strong class="font-semibold text-foreground">$1</strong>')
      .replace(/(?<!\*)\*(?!\*)(.*?)\*/g, '<em class="italic">$1</em>')
      .replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2" target="_blank" class="font-medium text-accent underline decoration-accent/50 underline-offset-2 hover:decoration-accent">$1</a>');
  };

  // --- CONCEPT CARDS: Replaces bullet lists with definition-style items ---
  const ConceptCards = ({ items }: { items: string[] }) => {
    return (
      <ul className="my-4 space-y-3">
        {items.map((item, i) => {
          const match = item.match(/^\*\*(.*?)\*\*[:\s]*(?:--|—)?\s*(.*)$/);
          const term = match ? match[1] : null;
          const description = match ? match[2] : item;

          return (
            <li key={i} className="flex items-start gap-3 text-base leading-relaxed text-muted sm:text-[17px]">
              <span className="mt-2.5 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-accent" />
              <span>
                {term && <strong className="font-semibold text-foreground">{term}</strong>}
                {term ? ' — ' : ''}
                <span dangerouslySetInnerHTML={{ __html: formatInline(description) }} />
              </span>
            </li>
          );
        })}
      </ul>
    );
  };

  // --- PROCESS STEPS: For ordered lists ---
  const ProcessSteps = ({ items }: { items: string[] }) => {
    return (
      <ul className="my-4 space-y-3">
        {items.map((item, i) => {
          const match = item.match(/^\*\*(.*?)\*\*[:\s]*(?:--|—)?\s*(.*)$/);
          const title = match ? match[1] : null;
          const description = match ? match[2] : item;
          const num = String(i + 1).padStart(2, '0');

          return (
            <li key={i} className="flex items-start gap-3 text-base leading-relaxed text-muted sm:text-[17px]">
              <span className="mt-0.5 font-mono text-xs text-accent">{num}</span>
              <span>
                {title && <strong className="font-semibold text-foreground">{title}</strong>}
                {title ? ' — ' : ''}
                <span dangerouslySetInnerHTML={{ __html: formatInline(description) }} />
              </span>
            </li>
          );
        })}
      </ul>
    );
  };

  // --- FEATURE GRID: For simple unordered lists without definitions ---
  const FeatureGrid = ({ items }: { items: string[] }) => {
    return (
      <ul className="my-4 space-y-3">
        {items.map((item, i) => {
          const match = item.match(/^\*\*(.*?)\*\*[:\s]*(?:--|—)?\s*(.*)$/);
          const title = match ? match[1] : null;
          const description = match ? match[2] : item;

          return (
            <li key={i} className="flex items-start gap-3 text-base leading-relaxed text-muted sm:text-[17px]">
              <span className="mt-2.5 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-accent" />
              <span>
                {title && <strong className="font-semibold text-foreground">{title} </strong>}
                <span dangerouslySetInnerHTML={{ __html: formatInline(description) }} />
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
      <div className="my-8 overflow-x-auto">
        <table className="w-full min-w-[480px] border-t border-border">
          <thead>
            <tr className="border-b border-border">
              {headers.map((h, i) => (
                <th key={i} className="px-0 py-3 pr-6 text-left font-mono text-[11px] uppercase tracking-[0.15em] text-accent" dangerouslySetInnerHTML={{ __html: formatInline(h.trim()) }} />
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {data.map((row, i) => (
              <tr key={i}>
                {row.map((cell, j) => (
                  <td key={j} className="px-0 py-3 pr-6 text-sm leading-relaxed text-muted">
                    {j === 0 ? (
                      <span className="font-medium text-foreground" dangerouslySetInnerHTML={{ __html: formatInline(cell.trim()) }} />
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
    );
  };

  // --- Main Render ---
  const blocks = parseBlocks(chapter.content);
  const skill = SKILL_HOME[chapter.id];
  const chapterNum = chapterOrderLabel(chapter.id);
  const title = chapterShortTitle(chapter.content.match(/^#\s+(.+)$/m)?.[1] || chapter.title);
  const ledeIndex = blocks.findIndex((block, i) => i > 0 && (block.type === 'p' || block.type === 'blockquote'));
  const ledeBlock = ledeIndex >= 0 ? blocks[ledeIndex] : null;
  const lede = ledeBlock
    ? ledeBlock.content.join(' ').replace(/\*\*/g, '').replace(/^From the field —\s*/i, '')
    : undefined;
  const total = COURSE_ORDER.length;
  let headingCount = 0;
  const sectionNumbers = blocks.map((block) => (block.type === 'h2' ? ++headingCount : 0));

  return (
    <div>
      <EditorialHero
        kicker={`Masterclass · ${chapterNum} of ${String(total).padStart(2, '0')}`}
        number={chapterNum}
        title={title}
        lede={lede}
        byline={skill && skill.skill > 0
          ? `Frank Sellhausen · Skill ${skill.skill} · ${skill.capability}`
          : 'Frank Sellhausen · The four skills'}
      />

      <ArticleShell>
        {blocks.map((block, index) => {
          if (index === ledeIndex) return null;

          switch (block.type) {
            case 'h1':
              return null;

            case 'h2': {
              const num = String(sectionNumbers[index]).padStart(2, '0');
              return (
                <div key={index} className="mb-6 mt-12 scroll-mt-24 first:mt-0">
                  <p className="font-mono text-xs text-accent">{num}</p>
                  <h2 className="mt-2 font-serif text-2xl leading-snug text-foreground sm:text-3xl">
                    {block.content[0]}
                  </h2>
                </div>
              );
            }

            case 'h3':
              return (
                <h3 key={index} className="mt-10 mb-4 font-serif text-xl leading-snug text-foreground sm:text-2xl">
                  {block.content[0]}
                </h3>
              );

            case 'h4':
              return (
                <h4 key={index} className="mt-8 mb-3 flex items-center gap-2 text-base font-semibold text-foreground">
                  <ArrowRight size={14} className="text-accent" />
                  {block.content[0]}
                </h4>
              );

            case 'p':
              return (
                <p
                  key={index}
                  className="mb-5 text-base leading-relaxed text-muted sm:text-[17px]"
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
              if (isLookUp) {
                return (
                  <div key={index} className="my-6 border-l-2 border-accent bg-card-bg p-5">
                    <span className="font-mono text-[11px] uppercase tracking-[0.15em] text-accent">
                      <Search className="mr-2 inline h-3.5 w-3.5" />
                      Look up now
                    </span>
                    <div className="mt-3 space-y-3 text-base leading-relaxed text-muted sm:text-[17px]">
                      {block.content.map((line, i) => (
                        <p key={i} dangerouslySetInnerHTML={{ __html: formatInline(line.replace(/^\*\*Look up now\.\*\*\s*/i, '').replace(/^Prompt:\s*/i, '')) }} />
                      ))}
                    </div>
                    <button
                      type="button"
                      onClick={() => navigator.clipboard.writeText(promptText)}
                      className="mt-4 inline-flex items-center border border-border px-4 py-2 text-sm font-semibold text-foreground hover:border-accent hover:text-accent"
                    >
                      <Copy className="mr-2 h-4 w-4" />
                      Copy look-up prompt
                    </button>
                  </div>
                );
              }
              return (
                <blockquote key={index} className="my-8 border-l-2 border-accent pl-5 font-serif text-xl italic leading-snug text-foreground sm:text-2xl">
                  {block.content.map((line, i) => (
                    <p key={i} dangerouslySetInnerHTML={{ __html: formatInline(line) }} />
                  ))}
                </blockquote>
              );
            }

            // --- Code blocks ---
            case 'code-block':
              return (
                <div key={index} className="my-8 border border-border bg-card-bg">
                  <div className="flex items-center justify-between border-b border-border px-4 py-2">
                    <span className="font-mono text-[11px] uppercase tracking-[0.15em] text-accent">
                      {block.meta || 'code'}
                    </span>
                  </div>
                  <div className="overflow-x-auto p-5">
                    <pre className="font-mono text-sm leading-relaxed text-foreground">
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
              return <hr key={index} className="my-12 border-border" />;

            default:
              return null;
          }
        })}

        <div className="mt-16 border-t border-border pt-10">
          <div className="flex flex-wrap items-center gap-6">
            <button
              onClick={() => toggleBookmark(chapter.id)}
              className={`text-sm ${bookmarked ? 'text-accent' : 'text-muted hover:text-accent'}`}
            >
              <Bookmark size={14} className={`mr-1.5 inline ${bookmarked ? 'fill-accent' : ''}`} />
              {bookmarked ? 'Saved' : 'Save for later'}
            </button>
            <button
              onClick={() => toggleComplete(chapter.id)}
              className={`text-sm ${complete ? 'text-accent' : 'text-muted hover:text-accent'}`}
            >
              <Check size={14} className="mr-1.5 inline" />
              {complete ? 'Completed' : 'Mark complete'}
            </button>
          </div>

          {nextChapter && onNextChapter && (
            <button
              onClick={() => {
                if (!complete) toggleComplete(chapter.id);
                onNextChapter();
              }}
              className="mt-8 block text-left text-sm text-muted hover:text-accent"
            >
              Next · {chapterOrderLabel(nextChapter.id)} {chapterShortTitle(nextChapter.title)} →
            </button>
          )}
        </div>
      </ArticleShell>
    </div>
  );
};
