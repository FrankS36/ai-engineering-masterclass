'use client';

import React from 'react';
import { chapters } from '../constants';
import { COURSE_SECTIONS, chapterOrderLabel, chapterShortTitle } from '../lib/courseOrder';
import { Bookmark } from 'lucide-react';
import { useLearner } from '../context/LearnerContext';

interface SidebarProps {
  activeChapterId: string;
  onSelectChapter: (id: string) => void;
  isCollapsed: boolean;
}

export const Sidebar: React.FC<SidebarProps> = ({ activeChapterId, onSelectChapter, isCollapsed }) => {
  const { isBookmarked } = useLearner();

  return (
    <nav className="flex-1 overflow-y-auto px-6 py-8">
      {!isCollapsed && (
        <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-muted">Chapters</p>
      )}

      <div className={isCollapsed ? '' : 'mt-4'}>
        {COURSE_SECTIONS.map((section) => {
          const sectionChapters = section.ids
            .map((id) => chapters.find((c) => c.id === id))
            .filter(Boolean);

          return (
            <div key={section.label} className="mb-6">
              {!isCollapsed && COURSE_SECTIONS.length > 1 && (
                <p className="mb-1 px-3 font-mono text-[10px] uppercase tracking-[0.18em] text-muted/70">
                  {section.label}
                </p>
              )}
              <ul className="space-y-1">
                {sectionChapters.map((chapter) => {
                  if (!chapter) return null;
                  const isActive = activeChapterId === chapter.id;
                  const num = chapterOrderLabel(chapter.id);
                  const bookmarked = isBookmarked(chapter.id);
                  const title = chapterShortTitle(chapter.title);

                  return (
                    <li key={chapter.id}>
                      <button
                        onClick={() => onSelectChapter(chapter.id)}
                        title={isCollapsed ? `${num} ${title}` : undefined}
                        className={`
                          group flex w-full items-start gap-3 rounded px-3 py-2.5 text-left text-sm transition-colors
                          ${isActive
                            ? 'bg-card-bg text-foreground'
                            : 'text-muted hover:bg-card-bg hover:text-foreground'}
                          ${isCollapsed ? 'justify-center px-2' : ''}
                        `}
                      >
                        <span
                          className={`font-mono text-xs ${
                            isActive ? 'text-accent' : 'text-muted group-hover:text-accent'
                          }`}
                        >
                          {num}
                        </span>
                        {!isCollapsed && (
                          <>
                            <span className="flex-1 leading-snug">{title}</span>
                            {bookmarked && (
                              <Bookmark size={12} className="mt-0.5 shrink-0 fill-accent text-accent" />
                            )}
                          </>
                        )}
                      </button>
                    </li>
                  );
                })}
              </ul>
            </div>
          );
        })}
      </div>
    </nav>
  );
};
