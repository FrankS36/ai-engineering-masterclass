import { ViewMode } from '../types';
import { chapters } from '../constants';

export const LEARNER_STORAGE_KEY = 'aem-learner-v1';
export const THEME_STORAGE_KEY = 'aem-theme';

export type Theme = 'light' | 'dark';

export interface QuizScore {
  score: number;
  total: number;
}

export interface LearnerState {
  completedChapterIds: string[];
  bookmarkedChapterIds: string[];
  lastChapterId: string;
  lastViewMode: ViewMode;
  quizBestScores: Record<string, QuizScore>;
}

export const defaultLearnerState = (): LearnerState => ({
  completedChapterIds: [],
  bookmarkedChapterIds: [],
  lastChapterId: chapters[0]?.id ?? 'ch1',
  lastViewMode: ViewMode.NOTES,
  quizBestScores: {},
});

const isViewMode = (value: unknown): value is ViewMode =>
  typeof value === 'string' && Object.values(ViewMode).includes(value as ViewMode);

export function loadLearnerState(): LearnerState {
  const fallback = defaultLearnerState();
  try {
    const raw = localStorage.getItem(LEARNER_STORAGE_KEY);
    if (!raw) return fallback;
    const parsed = JSON.parse(raw) as Partial<LearnerState>;
    const lastChapterId =
      typeof parsed.lastChapterId === 'string' && chapters.some(c => c.id === parsed.lastChapterId)
        ? parsed.lastChapterId
        : fallback.lastChapterId;

    return {
      completedChapterIds: Array.isArray(parsed.completedChapterIds)
        ? parsed.completedChapterIds.filter((id): id is string => typeof id === 'string')
        : [],
      bookmarkedChapterIds: Array.isArray(parsed.bookmarkedChapterIds)
        ? parsed.bookmarkedChapterIds.filter((id): id is string => typeof id === 'string')
        : [],
      lastChapterId,
      lastViewMode: isViewMode(parsed.lastViewMode) ? parsed.lastViewMode : fallback.lastViewMode,
      quizBestScores:
        parsed.quizBestScores && typeof parsed.quizBestScores === 'object'
          ? parsed.quizBestScores
          : {},
    };
  } catch {
    return fallback;
  }
}

export function saveLearnerState(state: LearnerState): void {
  try {
    localStorage.setItem(LEARNER_STORAGE_KEY, JSON.stringify(state));
  } catch {
    // Ignore quota / private-mode failures
  }
}

export function loadTheme(): Theme {
  try {
    const stored = localStorage.getItem(THEME_STORAGE_KEY);
    if (stored === 'light' || stored === 'dark') return stored;
    if (window.matchMedia('(prefers-color-scheme: dark)').matches) return 'dark';
  } catch {
    // fall through
  }
  return 'light';
}

export function saveTheme(theme: Theme): void {
  try {
    localStorage.setItem(THEME_STORAGE_KEY, theme);
  } catch {
    // Ignore quota / private-mode failures
  }
}

export function applyThemeClass(theme: Theme): void {
  document.documentElement.classList.toggle('dark', theme === 'dark');
}
