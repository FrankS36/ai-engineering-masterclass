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
import { ViewMode } from './types';
import { BookOpen, Copy, GraduationCap, Sparkles, Menu, X, PanelLeftClose, PanelLeftOpen, Wrench, BookA, FileText, Briefcase, Rocket, ChevronDown, Layout, Code } from 'lucide-react';

export default function App() {
  const [activeChapterId, setActiveChapterId] = useState<string>(chapters[0].id);
  const [viewMode, setViewMode] = useState<ViewMode>(ViewMode.NOTES);
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false);
  const [isDesktopSidebarCollapsed, setIsDesktopSidebarCollapsed] = useState(false);
  const [isResourcesDropdownOpen, setIsResourcesDropdownOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

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

  const currentChapterIndex = chapters.findIndex(c => c.id === activeChapterId);
  const nextChapter = chapters[currentChapterIndex + 1];

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
          <div className="flex items-center gap-3 min-w-0">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-brand-500 to-brand-600 flex items-center justify-center shadow-lg shadow-brand-500/25 shrink-0">
              <Sparkles size={18} className="text-white" />
            </div>
            {!isDesktopSidebarCollapsed && (
              <div className="min-w-0">
                <h1 className="text-sm font-bold text-white tracking-tight truncate">
                  AI Engineering
                </h1>
                <p className="text-[10px] text-stone-500 font-mono tracking-wider uppercase">Masterclass</p>
              </div>
            )}
          </div>
          
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
        <header className="h-16 bg-white border-b border-stone-200 flex items-center justify-between px-4 lg:px-6 shrink-0">
          <div className="flex items-center gap-3 min-w-0">
            {/* Mobile menu button */}
            <button 
              onClick={() => setIsMobileSidebarOpen(true)}
              className="lg:hidden p-2 -ml-2 text-stone-600 hover:bg-stone-100 rounded-lg transition-colors"
            >
              <Menu size={22} />
            </button>
            
            <h2 className="text-lg font-bold text-stone-900 tracking-tight truncate">
              {activeChapter.title.split(': ')[1] || activeChapter.title}
            </h2>
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
          
          {/* View mode tabs */}
          <div className="flex bg-stone-100 p-1 rounded-xl">
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
                  flex items-center gap-1 px-3 py-2 rounded-lg text-sm font-medium transition-all duration-200
                  ${[ViewMode.RESOURCES, ViewMode.GLOSSARY, ViewMode.CHEATSHEETS, ViewMode.INTERVIEW, ViewMode.PROJECTS, ViewMode.SYSTEM_DESIGN, ViewMode.TOOLKIT].includes(viewMode)
                    ? 'bg-white text-brand-600 shadow-sm' 
                    : 'text-stone-500 hover:text-stone-800'}
                `}
              >
                <Wrench size={16} />
                <span className="hidden sm:inline">Resources</span>
                <ChevronDown size={14} className={`transition-transform ${isResourcesDropdownOpen ? 'rotate-180' : ''}`} />
              </button>
              
              {isResourcesDropdownOpen && (
                <div className="absolute right-0 top-full mt-2 w-48 bg-white rounded-xl shadow-lg border border-stone-200 py-2 z-50">
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
      flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all duration-200
      ${active 
        ? 'bg-white text-brand-600 shadow-sm' 
        : 'text-stone-500 hover:text-stone-800'}
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
        ? 'bg-brand-50 text-brand-600' 
        : 'text-stone-700 hover:bg-stone-50'}
    `}
  >
    {icon}
    {label}
  </button>
);
