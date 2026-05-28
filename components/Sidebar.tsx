import React from 'react';
import { chapters } from '../constants';
import { ChevronRight } from 'lucide-react';

interface SidebarProps {
  activeChapterId: string;
  onSelectChapter: (id: string) => void;
  isCollapsed: boolean;
}

const sections = [
  { label: 'Foundations', ids: ['ch1', 'ch2'] },
  { label: 'Core Skills', ids: ['ch3', 'ch4', 'ch5', 'ch6'] },
  { label: 'Shipping', ids: ['ch7', 'ch8', 'ch9'] },
  { label: 'Advanced', ids: ['ch10', 'ch11', 'ch12'] },
  { label: 'Strategy', ids: ['ch13', 'ch14'] },
];

export const Sidebar: React.FC<SidebarProps> = ({ activeChapterId, onSelectChapter, isCollapsed }) => {
  return (
    <nav className="flex-1 flex flex-col overflow-hidden">
      <div className="flex-1 overflow-y-auto py-4">
        {sections.map((section) => {
          const sectionChapters = section.ids
            .map(id => chapters.find(c => c.id === id))
            .filter(Boolean);

          return (
            <div key={section.label} className="mb-5">
              {!isCollapsed && (
                <h3 className="px-5 text-[11px] font-semibold text-brand-400/70 uppercase tracking-wider mb-2 font-mono">
                  {section.label}
                </h3>
              )}
              <div className="space-y-1 px-3">
                {sectionChapters.map((chapter) => {
                  if (!chapter) return null;
                  const isActive = activeChapterId === chapter.id;
                  const chapterNum = chapter.id.replace('ch', '');

                  return (
                    <button
                      key={chapter.id}
                      onClick={() => onSelectChapter(chapter.id)}
                      title={isCollapsed ? chapter.title : undefined}
                      className={`
                        w-full text-left rounded-xl text-sm transition-all duration-200 flex items-center gap-3
                        ${isActive
                          ? 'bg-brand-500/10 text-brand-400'
                          : 'text-stone-400 hover:bg-stone-800 hover:text-stone-200'}
                        ${isCollapsed ? 'justify-center p-3' : 'px-3 py-2.5'}
                      `}
                    >
                      <span className={`
                        w-7 h-7 rounded-lg flex items-center justify-center text-xs font-bold shrink-0
                        ${isActive
                          ? 'bg-brand-500 text-white'
                          : 'bg-stone-800 text-stone-500'}
                      `}>
                        {chapterNum}
                      </span>

                      {!isCollapsed && (
                        <>
                          <span className="flex-1 truncate">
                            {chapter.title.split(':')[1]?.trim() || chapter.title}
                          </span>
                          {isActive && <ChevronRight size={14} className="text-brand-400 shrink-0" />}
                        </>
                      )}
                    </button>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    </nav>
  );
};
