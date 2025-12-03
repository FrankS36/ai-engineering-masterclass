import React from 'react';
import { chapters } from '../constants';
import { ChevronRight } from 'lucide-react';

interface SidebarProps {
  activeChapterId: string;
  onSelectChapter: (id: string) => void;
  isCollapsed: boolean;
}

export const Sidebar: React.FC<SidebarProps> = ({ activeChapterId, onSelectChapter, isCollapsed }) => {
  return (
    <nav className="flex-1 flex flex-col overflow-hidden">
      {/* Chapter list */}
      <div className="flex-1 overflow-y-auto py-4">
        
        {/* Section: Foundations */}
        <div className="mb-6">
          {!isCollapsed && (
            <h3 className="px-5 text-[11px] font-semibold text-stone-500 uppercase tracking-wider mb-2">
              Foundations
            </h3>
          )}
          <div className="space-y-1 px-3">
            {chapters.map((chapter) => {
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
                  {/* Chapter number badge */}
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

      </div>
    </nav>
  );
};
