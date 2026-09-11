'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { ChapterView } from './components/ChapterView';
import { QuizView } from './components/QuizView';
import { FlashcardView } from './components/FlashcardView';
import { AITutor } from './components/AITutor';
import { ResourcesView } from './components/ResourcesView';
import { GlossaryView } from './components/GlossaryView';
import { CheatSheetsView } from './components/CheatSheetsView';
import { InterviewPrepView } from './components/InterviewPrepView';
import { ProjectIdeasView } from './components/ProjectIdeasView';
import { SystemDesignView } from './components/SystemDesignView';
import { ToolkitView } from './components/ToolkitView';
import { GlobalSearch } from './components/GlobalSearch';
import { chapters } from './constants';
import { COURSE_ORDER } from './lib/courseOrder';
import { ViewMode } from './types';
import { useLearner } from './context/LearnerContext';
import { Menu, X, Wrench, BookA, FileText, Briefcase, Rocket, ChevronDown, Layout, Code, Moon, Sun } from 'lucide-react';

export default function App() {
  const { state, theme, toggleTheme, setLastLocation } = useLearner();
  const [activeChapterId, setActiveChapterId] = useState<string>(state.lastChapterId);
  const [viewMode, setViewMode] = useState<ViewMode>(state.lastViewMode);
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false);
  const [isDesktopSidebarCollapsed, setIsDesktopSidebarCollapsed] = useState(false);
  const [isResourcesDropdownOpen, setIsResourcesDropdownOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setLastLocation(activeChapterId, viewMode);
  }, [activeChapterId, viewMode, setLastLocation]);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsResourcesDropdownOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const activeChapter = chapters.find(c => c.id === activeChapterId) || chapters[0];

  const currentChapterIndex = COURSE_ORDER.indexOf(activeChapterId as typeof COURSE_ORDER[number]);
  const nextChapterId = currentChapterIndex >= 0 ? COURSE_ORDER[currentChapterIndex + 1] : undefined;
  const nextChapter = nextChapterId ? chapters.find(c => c.id === nextChapterId) : undefined;

  const goToNextChapter = () => {
    if (nextChapter) {
      setActiveChapterId(nextChapter.id);
      const mainContent = document.querySelector('main[data-course-scroll]');
      if (mainContent) mainContent.scrollTo(0, 0);
    }
  };

  const renderContent = () => {
    switch (viewMode) {
      case ViewMode.NOTES:
        return <ChapterView chapter={activeChapter} nextChapter={nextChapter} onNextChapter={goToNextChapter} />;
      case ViewMode.QUIZ:
        return <QuizView chapter={activeChapter} />;
      case ViewMode.FLASHCARDS:
        return <FlashcardView chapter={activeChapter} />;
      case ViewMode.TUTOR:
        return <AITutor chapter={activeChapter} />;
      case ViewMode.RESOURCES:
        return <ResourcesView />;
      case ViewMode.GLOSSARY:
        return <GlossaryView />;
      case ViewMode.CHEATSHEETS:
        return <CheatSheetsView />;
      case ViewMode.INTERVIEW:
        return <InterviewPrepView />;
      case ViewMode.PROJECTS:
        return <ProjectIdeasView />;
      case ViewMode.SYSTEM_DESIGN:
        return <SystemDesignView />;
      case ViewMode.TOOLKIT:
        return <ToolkitView />;
      default:
        return <ChapterView chapter={activeChapter} nextChapter={nextChapter} onNextChapter={goToNextChapter} />;
    }
  };

  const practiceActive = (mode: ViewMode) => viewMode === mode;
  const resourcesActive = [
    ViewMode.RESOURCES, ViewMode.GLOSSARY, ViewMode.CHEATSHEETS, ViewMode.INTERVIEW,
    ViewMode.PROJECTS, ViewMode.SYSTEM_DESIGN, ViewMode.TOOLKIT,
  ].includes(viewMode);

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-background font-sans">
      <header className="sticky top-0 z-50 shrink-0 border-b border-white/10 bg-surface-dark">
        <nav className="px-4 sm:px-6 lg:px-8">
          <div className="flex h-14 items-center justify-between lg:h-16">
            <div className="flex min-w-0 items-center gap-3">
              <button
                onClick={() => setIsMobileSidebarOpen(true)}
                className="border border-white/20 p-2 text-white lg:hidden"
                aria-label="Open chapters"
              >
                <Menu size={20} />
              </button>
              <a href="https://sellhausen.com" target="_blank" rel="noopener noreferrer" className="transition-opacity hover:opacity-90">
                <span className="inline-flex items-center gap-3">
                  <img src="/logo-mark.png" alt="" width={36} height={36} className="shrink-0" />
                  <span className="hidden sm:flex flex-col justify-center">
                    <span className="flex items-baseline gap-[0.4em] text-[15px] font-light uppercase leading-none tracking-[0.2em]">
                      <span className="text-white">Sellhausen</span>
                      <span className="text-accent">Masterclass</span>
                    </span>
                    <span className="mt-1 block h-px w-full bg-accent/60" />
                  </span>
                </span>
              </a>
            </div>

            <div className="flex items-center gap-5 sm:gap-6">
              <GlobalSearch
                onNavigate={(chapterId, nextView) => {
                  setActiveChapterId(chapterId);
                  setViewMode(nextView);
                  const mainContent = document.querySelector('main[data-course-scroll]');
                  if (mainContent) mainContent.scrollTo(0, 0);
                }}
              />

              <ul className="flex items-center gap-4 sm:gap-6">
                <li>
                  <NavLink active={practiceActive(ViewMode.NOTES)} onClick={() => setViewMode(ViewMode.NOTES)}>
                    Learn
                  </NavLink>
                </li>
                <li>
                  <NavLink active={practiceActive(ViewMode.FLASHCARDS)} onClick={() => setViewMode(ViewMode.FLASHCARDS)}>
                    Cards
                  </NavLink>
                </li>
                <li>
                  <NavLink active={practiceActive(ViewMode.QUIZ)} onClick={() => setViewMode(ViewMode.QUIZ)}>
                    Quiz
                  </NavLink>
                </li>
              </ul>

              <button
                onClick={toggleTheme}
                aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
                className="text-surface-dark-muted transition-colors hover:text-white"
                title={theme === 'dark' ? 'Light mode' : 'Dark mode'}
              >
                {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
              </button>

              <div className="relative" ref={dropdownRef}>
                <button
                  onClick={() => setIsResourcesDropdownOpen(!isResourcesDropdownOpen)}
                  className={`inline-flex items-center justify-center border px-4 py-2 text-sm font-semibold transition-colors ${
                    resourcesActive
                      ? 'border-white/50 bg-white/5 text-white'
                      : 'border-white/25 text-white hover:border-white/50 hover:bg-white/5'
                  }`}
                >
                  Resources
                  <ChevronDown size={14} className={`ml-1.5 transition-transform ${isResourcesDropdownOpen ? 'rotate-180' : ''}`} />
                </button>

                {isResourcesDropdownOpen && (
                  <div className="absolute right-0 top-full z-50 mt-2 w-48 border border-white/15 bg-surface-dark py-2">
                    <DropdownItem active={viewMode === ViewMode.RESOURCES} onClick={() => { setViewMode(ViewMode.RESOURCES); setIsResourcesDropdownOpen(false); }} icon={<Wrench size={16} />} label="Tools" />
                    <DropdownItem active={viewMode === ViewMode.GLOSSARY} onClick={() => { setViewMode(ViewMode.GLOSSARY); setIsResourcesDropdownOpen(false); }} icon={<BookA size={16} />} label="Glossary" />
                    <DropdownItem active={viewMode === ViewMode.CHEATSHEETS} onClick={() => { setViewMode(ViewMode.CHEATSHEETS); setIsResourcesDropdownOpen(false); }} icon={<FileText size={16} />} label="Cheat Sheets" />
                    <DropdownItem active={viewMode === ViewMode.INTERVIEW} onClick={() => { setViewMode(ViewMode.INTERVIEW); setIsResourcesDropdownOpen(false); }} icon={<Briefcase size={16} />} label="Interview Prep" />
                    <DropdownItem active={viewMode === ViewMode.PROJECTS} onClick={() => { setViewMode(ViewMode.PROJECTS); setIsResourcesDropdownOpen(false); }} icon={<Rocket size={16} />} label="Project Ideas" />
                    <DropdownItem active={viewMode === ViewMode.SYSTEM_DESIGN} onClick={() => { setViewMode(ViewMode.SYSTEM_DESIGN); setIsResourcesDropdownOpen(false); }} icon={<Layout size={16} />} label="System Design" />
                    <DropdownItem active={viewMode === ViewMode.TOOLKIT} onClick={() => { setViewMode(ViewMode.TOOLKIT); setIsResourcesDropdownOpen(false); }} icon={<Code size={16} />} label="Dev Toolkit" />
                  </div>
                )}
              </div>
            </div>
          </div>
        </nav>
      </header>

      <div className={`flex min-h-0 flex-1 ${isDesktopSidebarCollapsed ? 'lg:grid lg:grid-cols-[72px_1fr]' : 'lg:grid lg:grid-cols-[280px_1fr]'}`}>
        <div
          className={`
            fixed inset-0 z-40 bg-stone-900/60 lg:hidden
            ${isMobileSidebarOpen ? 'opacity-100' : 'pointer-events-none opacity-0'}
          `}
          onClick={() => setIsMobileSidebarOpen(false)}
        />

        <aside
          className={`
            z-[60] flex h-full flex-col border-r border-border bg-background
            fixed inset-y-0 left-0 w-[280px] transition-transform lg:static lg:z-auto lg:translate-x-0
            ${isMobileSidebarOpen ? 'translate-x-0' : '-translate-x-full'}
          `}
        >
          <div className="flex items-center justify-between border-b border-border px-6 py-4 lg:hidden">
            <p className="font-mono text-xs uppercase tracking-[0.15em] text-muted">Chapters</p>
            <button className="p-2 text-muted hover:text-foreground" onClick={() => setIsMobileSidebarOpen(false)} aria-label="Close chapters">
              <X size={18} />
            </button>
          </div>

          <Sidebar
            activeChapterId={activeChapterId}
            onSelectChapter={(id) => {
              setActiveChapterId(id);
              setViewMode(ViewMode.NOTES);
              setIsMobileSidebarOpen(false);
              const mainContent = document.querySelector('main[data-course-scroll]');
              if (mainContent) mainContent.scrollTo(0, 0);
            }}
            isCollapsed={isDesktopSidebarCollapsed}
          />

          <div className="hidden border-t border-border px-6 py-6 lg:block">
            <button
              onClick={() => setIsDesktopSidebarCollapsed(!isDesktopSidebarCollapsed)}
              className="block text-sm text-muted hover:text-accent"
            >
              {isDesktopSidebarCollapsed ? '→' : 'Collapse →'}
            </button>
          </div>
        </aside>

        <main data-course-scroll className="min-w-0 flex-1 overflow-y-auto bg-background">
          {renderContent()}
        </main>
      </div>
    </div>
  );
}

const NavLink = ({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) => (
  <button
    onClick={onClick}
    className={`text-sm transition-colors ${
      active ? 'text-white' : 'text-surface-dark-muted hover:text-white'
    }`}
  >
    {children}
  </button>
);

const DropdownItem = ({ active, onClick, icon, label }: { active: boolean; onClick: () => void; icon: React.ReactNode; label: string }) => (
  <button
    onClick={onClick}
    className={`flex w-full items-center gap-3 px-4 py-2.5 text-sm transition-colors ${
      active ? 'bg-white/10 text-accent' : 'text-white/70 hover:bg-white/5'
    }`}
  >
    {icon}
    {label}
  </button>
);
