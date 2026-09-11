import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import { ViewMode } from '../types';
import { chapters } from '../constants';
import {
  LearnerState,
  QuizScore,
  Theme,
  applyThemeClass,
  loadLearnerState,
  loadTheme,
  saveLearnerState,
  saveTheme,
} from '../lib/learnerState';

interface LearnerContextValue {
  state: LearnerState;
  theme: Theme;
  completedCount: number;
  totalChapters: number;
  progressPercent: number;
  toggleTheme: () => void;
  setLastLocation: (chapterId: string, viewMode: ViewMode) => void;
  toggleComplete: (chapterId: string) => void;
  markComplete: (chapterId: string) => void;
  toggleBookmark: (chapterId: string) => void;
  recordQuizScore: (chapterId: string, score: number, total: number) => void;
  isComplete: (chapterId: string) => boolean;
  isBookmarked: (chapterId: string) => boolean;
}

const LearnerContext = createContext<LearnerContextValue | null>(null);

export const LearnerProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [state, setState] = useState<LearnerState>(() => loadLearnerState());
  const [theme, setTheme] = useState<Theme>(() => loadTheme());

  useEffect(() => {
    saveLearnerState(state);
  }, [state]);

  useEffect(() => {
    applyThemeClass(theme);
    saveTheme(theme);
  }, [theme]);

  const toggleTheme = useCallback(() => {
    setTheme(prev => (prev === 'dark' ? 'light' : 'dark'));
  }, []);

  const setLastLocation = useCallback((chapterId: string, viewMode: ViewMode) => {
    setState(prev => {
      if (prev.lastChapterId === chapterId && prev.lastViewMode === viewMode) return prev;
      return { ...prev, lastChapterId: chapterId, lastViewMode: viewMode };
    });
  }, []);

  const toggleComplete = useCallback((chapterId: string) => {
    setState(prev => {
      const exists = prev.completedChapterIds.includes(chapterId);
      return {
        ...prev,
        completedChapterIds: exists
          ? prev.completedChapterIds.filter(id => id !== chapterId)
          : [...prev.completedChapterIds, chapterId],
      };
    });
  }, []);

  const markComplete = useCallback((chapterId: string) => {
    setState(prev => {
      if (prev.completedChapterIds.includes(chapterId)) return prev;
      return { ...prev, completedChapterIds: [...prev.completedChapterIds, chapterId] };
    });
  }, []);

  const toggleBookmark = useCallback((chapterId: string) => {
    setState(prev => {
      const exists = prev.bookmarkedChapterIds.includes(chapterId);
      return {
        ...prev,
        bookmarkedChapterIds: exists
          ? prev.bookmarkedChapterIds.filter(id => id !== chapterId)
          : [...prev.bookmarkedChapterIds, chapterId],
      };
    });
  }, []);

  const recordQuizScore = useCallback((chapterId: string, score: number, total: number) => {
    setState(prev => {
      const previous: QuizScore | undefined = prev.quizBestScores[chapterId];
      const nextScore =
        !previous || score / total >= previous.score / previous.total
          ? { score, total }
          : previous;
      const passed = total > 0 && score / total >= 0.7;
      const completedChapterIds =
        passed && !prev.completedChapterIds.includes(chapterId)
          ? [...prev.completedChapterIds, chapterId]
          : prev.completedChapterIds;

      return {
        ...prev,
        quizBestScores: { ...prev.quizBestScores, [chapterId]: nextScore },
        completedChapterIds,
      };
    });
  }, []);

  const isComplete = useCallback(
    (chapterId: string) => state.completedChapterIds.includes(chapterId),
    [state.completedChapterIds]
  );

  const isBookmarked = useCallback(
    (chapterId: string) => state.bookmarkedChapterIds.includes(chapterId),
    [state.bookmarkedChapterIds]
  );

  const value = useMemo<LearnerContextValue>(() => {
    const completedCount = state.completedChapterIds.filter(id =>
      chapters.some(c => c.id === id)
    ).length;
    const totalChapters = chapters.length;
    return {
      state,
      theme,
      completedCount,
      totalChapters,
      progressPercent: totalChapters === 0 ? 0 : Math.round((completedCount / totalChapters) * 100),
      toggleTheme,
      setLastLocation,
      toggleComplete,
      markComplete,
      toggleBookmark,
      recordQuizScore,
      isComplete,
      isBookmarked,
    };
  }, [
    state,
    theme,
    toggleTheme,
    setLastLocation,
    toggleComplete,
    markComplete,
    toggleBookmark,
    recordQuizScore,
    isComplete,
    isBookmarked,
  ]);

  return <LearnerContext.Provider value={value}>{children}</LearnerContext.Provider>;
};

export function useLearner(): LearnerContextValue {
  const ctx = useContext(LearnerContext);
  if (!ctx) {
    throw new Error('useLearner must be used within LearnerProvider');
  }
  return ctx;
}
