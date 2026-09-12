/**
 * Pedagogical order. Sidebar grouping is separate; next-chapter follows this.
 * Source of truth: /SPINE.md — Andrew Ng’s four skills are the spine.
 */
export const COURSE_ORDER = [
  'ch1',  // The four skills
  'ch2',  // LLM foundations
  'ch3',  // Prompting / structured output
  'ch4',  // Context and control surfaces
  'ch5',  // Grounding
  'ch6',  // Agentic systems
  'ch7',  // Evaluation-driven development
  'ch16', // Software engineering fundamentals
  'ch15', // Using coding agents
  'ch17', // Specs the agent can execute
  'ch8',  // Operating in production
  'ch9',  // Secure and reliable
  'ch10', // ML foundations
  'ch11', // Multimodal
  'ch12', // Local / self-host
  'ch13', // Full-stack UX
  'ch14', // Shaping the build
] as const;

export type CourseChapterId = (typeof COURSE_ORDER)[number];

export function chapterOrderIndex(id: string): number {
  const index = COURSE_ORDER.indexOf(id as CourseChapterId);
  return index >= 0 ? index + 1 : 0;
}

export function chapterOrderLabel(id: string): string {
  return String(chapterOrderIndex(id)).padStart(2, '0');
}

export function chapterShortTitle(title: string): string {
  return title.split(': ')[1] || title.split(':')[1]?.trim() || title;
}

export const COURSE_SECTIONS = [
  { label: 'The four skills', ids: ['ch1'] },
  { label: 'Building AI applications', ids: ['ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7'] },
  { label: 'Software engineering', ids: ['ch16'] },
  { label: 'Using coding agents', ids: ['ch15', 'ch17'] },
  { label: 'Operating in production', ids: ['ch8', 'ch9'] },
  { label: 'ML and specialized', ids: ['ch10', 'ch11', 'ch12', 'ch13'] },
  { label: 'Shaping the build', ids: ['ch14'] },
] as const;

export type SkillHome = {
  skill: 0 | 1 | 2 | 3 | 4;
  label: string;
  capability: string;
};

/** Where each chapter sits on Ng’s ingested skills map. */
export const SKILL_HOME: Record<string, SkillHome> = {
  ch1: { skill: 0, label: 'Orientation', capability: 'The four skills' },
  ch2: { skill: 1, label: 'Building and deploying AI applications', capability: 'LLM foundations' },
  ch3: { skill: 1, label: 'Building and deploying AI applications', capability: 'LLM foundations' },
  ch4: { skill: 1, label: 'Building and deploying AI applications', capability: 'LLM foundations' },
  ch5: { skill: 1, label: 'Building and deploying AI applications', capability: 'Grounding models with data' },
  ch6: { skill: 1, label: 'Building and deploying AI applications', capability: 'Building agentic systems' },
  ch7: { skill: 1, label: 'Building and deploying AI applications', capability: 'Evaluation-driven development' },
  ch8: { skill: 1, label: 'Building and deploying AI applications', capability: 'Operating in production' },
  ch9: { skill: 2, label: 'Software engineering fundamentals', capability: 'Secure and reliable' },
  ch10: { skill: 1, label: 'Building and deploying AI applications', capability: 'Machine learning foundations' },
  ch11: { skill: 1, label: 'Building and deploying AI applications', capability: 'LLM foundations — multimodal' },
  ch12: { skill: 1, label: 'Building and deploying AI applications', capability: 'LLM foundations — local / self-host' },
  ch13: { skill: 2, label: 'Software engineering fundamentals', capability: 'Full-stack applications' },
  ch14: { skill: 4, label: 'Shaping the build', capability: 'What belongs in the spec' },
  ch15: { skill: 3, label: 'Using coding agents', capability: 'Directing the workflow' },
  ch16: { skill: 2, label: 'Software engineering fundamentals', capability: 'Five literacies' },
  ch17: { skill: 3, label: 'Using coding agents', capability: 'Specs the agent can execute' },
};
