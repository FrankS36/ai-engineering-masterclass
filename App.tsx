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
import { BookOpen, Copy, GraduationCap, Menu, X, PanelLeftClose, PanelLeftOpen, Wrench, BookA, FileText, Briefcase, Rocket, ChevronDown, Layout, Code, Moon, Sun } from 'lucide-react';

export default function App() {
  const { state, theme, toggleTheme, setLastLocation, progressPercent, completedCount, totalChapters } = useLearner();
  const [activeChapterId, setActiveChapterId] = useState<string>(state.lastChapterId);
  const [viewMode, setViewMode] = useState<ViewMode>(state.lastViewMode);
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false);
  const [isDesktopSidebarCollapsed, setIsDesktopSidebarCollapsed] = useState(false);
  const [isResourcesDropdownOpen, setIsResourcesDropdownOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setLastLocation(activeChapterId, viewMode);
  }, [activeChapterId, viewMode, setLastLocation]);

  // Close dropdown when clicking outside
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
      // Scroll the main content area to top
      const mainContent = document.querySelector('main');
      if (mainContent) {
        mainContent.scrollTo(0, 0);
      }
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

  const sidebarWidth = isDesktopSidebarCollapsed ? 'w-20' : 'w-72';

  return (
    <div className="flex h-screen bg-stone-50 overflow-hidden font-sans selection:bg-brand-200 selection:text-brand-900">
      
      {/* Mobile Overlay */}
      <div 
        className={`
          fixed inset-0 bg-stone-900/60 backdrop-blur-sm z-40 lg:hidden
          transition-opacity duration-300
          ${isMobileSidebarOpen ? 'opacity-100' : 'opacity-0 pointer-events-none'}
        `}
        onClick={() => setIsMobileSidebarOpen(false)}
      />

      {/* Sidebar - Fixed on desktop, slide-over on mobile */}
      <aside className={`
        fixed inset-y-0 left-0 z-50 flex flex-col bg-stone-900 text-white
        transition-all duration-300 ease-in-out
        lg:relative lg:translate-x-0
        ${isMobileSidebarOpen ? 'translate-x-0' : '-translate-x-full'}
        ${sidebarWidth}
      `}>
        {/* Sidebar Header */}
        <div className={`
          h-16 flex items-center border-b border-stone-800 bg-stone-900
          ${isDesktopSidebarCollapsed ? 'justify-center px-2' : 'px-5'}
        `}>
          <a href="https://sellhausen.com" target="_blank" rel="noopener noreferrer" className="flex items-center gap-3 min-w-0 hover:opacity-90 transition-opacity">
            <img src="/logo-mark.png" alt="" width={36} height={36} className="shrink-0" />
            {!isDesktopSidebarCollapsed && (
              <span className="min-w-0 hidden sm:flex flex-col justify-center">
                <span className="flex items-baseline gap-[0.4em] text-[13px] font-light tracking-[0.18em] uppercase leading-none">
                  <span className="text-white">Sellhausen</span>
                  <span className="text-brand-400">Masterclass</span>
                </span>
                <span className="mt-1 block h-px w-full bg-brand-400/60" />
              </span>
            )}
          </a>
          
          {/* Mobile close button */}
          <button 
            className="ml-auto lg:hidden p-2 text-stone-400 hover:text-white hover:bg-stone-800 rounded-lg transition-colors"
            onClick={() => setIsMobileSidebarOpen(false)}
          >
            <X size={20} />
          </button>
        </div>
        
        {/* Sidebar Content */}
        <Sidebar 
          activeChapterId={activeChapterId} 
          onSelectChapter={(id) => {
            setActiveChapterId(id);
            setViewMode(ViewMode.NOTES);
            setIsMobileSidebarOpen(false);
            // Scroll to top of content
            const mainContent = document.querySelector('main');
            if (mainContent) mainContent.scrollTo(0, 0);
          }}
          isCollapsed={isDesktopSidebarCollapsed}
        />

        {/* Collapse toggle - desktop only */}
        <div className="hidden lg:block border-t border-stone-800 p-3">
          <button
            onClick={() => setIsDesktopSidebarCollapsed(!isDesktopSidebarCollapsed)}
            className="w-full flex items-center justify-center gap-2 p-2.5 text-stone-500 hover:text-white hover:bg-stone-800 rounded-lg transition-colors"
          >
            {isDesktopSidebarCollapsed ? (
              <PanelLeftOpen size={18} />
            ) : (
              <>
                <PanelLeftClose size={18} />
                <span className="text-sm">Collapse</span>
              </>
            )}
          </button>
        </div>
      </aside>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 h-full">
        {/* Top Header */}
        <header className="h-14 lg:h-16 bg-stone-900 border-b border-white/10 flex items-center justify-between px-4 lg:px-6 shrink-0">
          <div className="flex items-center gap-3 min-w-0">
            {/* Mobile menu button */}
            <button 
              onClick={() => setIsMobileSidebarOpen(true)}
              className="lg:hidden p-2 -ml-2 text-white/80 hover:bg-white/5 transition-colors"
            >
              <Menu size={22} />
            </button>
            
            <h2 className="text-sm sm:text-base font-serif text-white tracking-tight truncate">
              {activeChapter.title.split(': ')[1] || activeChapter.title}
            </h2>
            <span className="hidden md:inline-flex items-center px-2 py-0.5 text-[11px] font-mono uppercase tracking-wider text-stone-400 border border-white/15 shrink-0">
              {completedCount}/{totalChapters} · {progressPercent}%
            </span>
          </div>

          {/* Global Search */}
          <GlobalSearch 
            onNavigate={(chapterId, viewMode) => {
              setActiveChapterId(chapterId);
              setViewMode(viewMode);
              const mainContent = document.querySelector('main');
              if (mainContent) mainContent.scrollTo(0, 0);
            }}
          />
          
          <button
            onClick={toggleTheme}
            aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
            className="p-2 text-white/70 hover:text-white hover:bg-white/5 transition-colors"
            title={theme === 'dark' ? 'Light mode' : 'Dark mode'}
          >
            {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
          </button>

          {/* View mode tabs */}
          <div className="flex border border-white/15 p-0.5">
            <NavButton 
              active={viewMode === ViewMode.NOTES} 
              onClick={() => setViewMode(ViewMode.NOTES)}
              icon={<BookOpen size={16} />}
              label="Learn"
            />
            <NavButton 
              active={viewMode === ViewMode.FLASHCARDS} 
              onClick={() => setViewMode(ViewMode.FLASHCARDS)}
              icon={<Copy size={16} />}
              label="Cards"
            />
            <NavButton 
              active={viewMode === ViewMode.QUIZ} 
              onClick={() => setViewMode(ViewMode.QUIZ)}
              icon={<GraduationCap size={16} />}
              label="Quiz"
            />
            
            {/* Resources Dropdown */}
            <div className="relative" ref={dropdownRef}>
              <button
                onClick={() => setIsResourcesDropdownOpen(!isResourcesDropdownOpen)}
                className={`
                  flex items-center gap-1 px-3 py-2 text-sm font-medium transition-colors
                  ${[ViewMode.RESOURCES, ViewMode.GLOSSARY, ViewMode.CHEATSHEETS, ViewMode.INTERVIEW, ViewMode.PROJECTS, ViewMode.SYSTEM_DESIGN, ViewMode.TOOLKIT].includes(viewMode)
                    ? 'bg-white/10 text-white' 
                    : 'text-white/60 hover:text-white'}
                `}
              >
                <Wrench size={16} />
                <span className="hidden sm:inline">Resources</span>
                <ChevronDown size={14} className={`transition-transform ${isResourcesDropdownOpen ? 'rotate-180' : ''}`} />
              </button>
              
              {isResourcesDropdownOpen && (
                <div className="absolute right-0 top-full mt-2 w-48 bg-stone-900 border border-white/15 py-2 z-50">
                  <DropdownItem 
                    active={viewMode === ViewMode.RESOURCES}
                    onClick={() => { setViewMode(ViewMode.RESOURCES); setIsResourcesDropdownOpen(false); }}
                    icon={<Wrench size={16} />}
                    label="Tools"
                  />
                  <DropdownItem 
                    active={viewMode === ViewMode.GLOSSARY}
                    onClick={() => { setViewMode(ViewMode.GLOSSARY); setIsResourcesDropdownOpen(false); }}
                    icon={<BookA size={16} />}
                    label="Glossary"
                  />
                  <DropdownItem 
                    active={viewMode === ViewMode.CHEATSHEETS}
                    onClick={() => { setViewMode(ViewMode.CHEATSHEETS); setIsResourcesDropdownOpen(false); }}
                    icon={<FileText size={16} />}
                    label="Cheat Sheets"
                  />
                  <DropdownItem 
                    active={viewMode === ViewMode.INTERVIEW}
                    onClick={() => { setViewMode(ViewMode.INTERVIEW); setIsResourcesDropdownOpen(false); }}
                    icon={<Briefcase size={16} />}
                    label="Interview Prep"
                  />
                  <DropdownItem 
                    active={viewMode === ViewMode.PROJECTS}
                    onClick={() => { setViewMode(ViewMode.PROJECTS); setIsResourcesDropdownOpen(false); }}
                    icon={<Rocket size={16} />}
                    label="Project Ideas"
                  />
<DropdownItem 
                                    active={viewMode === ViewMode.SYSTEM_DESIGN}
                                    onClick={() => { setViewMode(ViewMode.SYSTEM_DESIGN); setIsResourcesDropdownOpen(false); }}
                                    icon={<Layout size={16} />}
                                    label="System Design"
                                  />
                                  <DropdownItem 
                                    active={viewMode === ViewMode.TOOLKIT}
                                    onClick={() => { setViewMode(ViewMode.TOOLKIT); setIsResourcesDropdownOpen(false); }}
                                    icon={<Code size={16} />}
                                    label="Dev Toolkit"
                                  />
                                </div>
              )}
            </div>
          </div>
        </header>

        {/* Scrollable Content */}
        <main className="flex-1 overflow-y-auto bg-stone-50">
          {renderContent()}
        </main>
      </div>
    </div>
  );
}

const NavButton = ({ active, onClick, icon, label }: { active: boolean; onClick: () => void; icon: React.ReactNode; label: string }) => (
  <button
    onClick={onClick}
    className={`
      flex items-center gap-2 px-3 py-2 text-sm font-medium transition-colors
      ${active 
        ? 'bg-white/10 text-white' 
        : 'text-white/60 hover:text-white'}
    `}
  >
    {icon}
    <span className="hidden sm:inline">{label}</span>
  </button>
);

const DropdownItem = ({ active, onClick, icon, label }: { active: boolean; onClick: () => void; icon: React.ReactNode; label: string }) => (
  <button
    onClick={onClick}
    className={`
      w-full flex items-center gap-3 px-4 py-2.5 text-sm transition-colors
      ${active 
        ? 'bg-white/10 text-brand-400' 
        : 'text-white/70 hover:bg-white/5'}
    `}
  >
    {icon}
    {label}
  </button>
);
