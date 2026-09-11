/**
 * Pedagogical order. Sidebar grouping is separate; next-chapter follows this.
 * Source of truth for the restructure: /SPINE.md — read that first in a new session.
 */
export const COURSE_ORDER = [
  'ch1',  // Landscape
  'ch2',  // Foundation models
  'ch3',  // Prompting
  'ch4',  // APIs
  'ch5',  // RAG
  'ch6',  // Agentic AI (build)
  'ch7',  // Evals
  'ch16', // Software engineering fundamentals
  'ch15', // Using coding agents
  'ch17', // Spec-driven development
  'ch8',  // Production
  'ch9',  // Security
  'ch10', // Fine-tuning
  'ch11', // Multimodal
  'ch12', // Local / edge
  'ch13', // UX
  'ch14', // Strategy
] as const;

export type CourseChapterId = (typeof COURSE_ORDER)[number];

export const COURSE_SECTIONS = [
  { label: 'Foundations', ids: ['ch1', 'ch2'] },
  { label: 'AI Applications', ids: ['ch3', 'ch4', 'ch5', 'ch6', 'ch7'] },
  { label: 'Engineering with Agents', ids: ['ch16', 'ch15', 'ch17'] },
  { label: 'Shipping', ids: ['ch8', 'ch9'] },
  { label: 'Advanced', ids: ['ch10', 'ch11', 'ch12', 'ch13'] },
  { label: 'Strategy', ids: ['ch14'] },
] as const;
