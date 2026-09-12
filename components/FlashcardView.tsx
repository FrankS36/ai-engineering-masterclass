import React, { useState, useEffect, useCallback } from 'react';
import { Chapter, Flashcard } from '../types';
import { chapters } from '../constants';
import { ChevronLeft, ChevronRight, RotateCw, Grid, Layers, Library } from 'lucide-react';
import { chapterOrderLabel, chapterShortTitle } from '../lib/courseOrder';
import { EditorialHero, ArticleShell } from './EditorialChrome';

interface FlashcardViewProps {
  chapter: Chapter;
}

export const FlashcardView: React.FC<FlashcardViewProps> = ({ chapter }) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isFlipped, setIsFlipped] = useState(false);
  const [viewMode, setViewMode] = useState<'single' | 'all' | 'allSections'>('single');
  const [flippedCards, setFlippedCards] = useState<Set<string>>(new Set());

  // Get all flashcards across all sections
  const allFlashcards: { card: Flashcard; sectionTitle: string }[] = chapters.flatMap(ch => 
    ch.flashcards.map(card => ({ card, sectionTitle: ch.title.split(':')[1]?.trim() || ch.title }))
  );

  useEffect(() => {
    setCurrentIndex(0);
    setIsFlipped(false);
    setFlippedCards(new Set());
  }, [chapter.id]);

  const handleNext = useCallback(() => {
    if (currentIndex < chapter.flashcards.length - 1) {
      setIsFlipped(false);
      setTimeout(() => setCurrentIndex(prev => prev + 1), 150);
    }
  }, [currentIndex, chapter.flashcards.length]);

  const handlePrev = useCallback(() => {
    if (currentIndex > 0) {
      setIsFlipped(false);
      setTimeout(() => setCurrentIndex(prev => prev - 1), 150);
    }
  }, [currentIndex]);

  const handleFlip = useCallback(() => {
    setIsFlipped(prev => !prev);
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Only handle in single card view
      if (viewMode !== 'single') return;
      
      // Don't trigger if user is typing in an input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

      switch (e.key) {
        case 'ArrowRight':
        case 'l':
        case 'j':
          e.preventDefault();
          handleNext();
          break;
        case 'ArrowLeft':
        case 'h':
        case 'k':
          e.preventDefault();
          handlePrev();
          break;
        case ' ':
        case 'Enter':
        case 'ArrowUp':
        case 'ArrowDown':
          e.preventDefault();
          handleFlip();
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [viewMode, handleNext, handlePrev, handleFlip]);

  const toggleCardFlip = (id: string) => {
    const newFlipped = new Set(flippedCards);
    if (newFlipped.has(id)) {
      newFlipped.delete(id);
    } else {
      newFlipped.add(id);
    }
    setFlippedCards(newFlipped);
  };

  const card = chapter.flashcards[currentIndex];

  return (
    <div>
      <EditorialHero
        kicker={`Masterclass · Cards · ${chapterOrderLabel(chapter.id)}`}
        number={chapterOrderLabel(chapter.id)}
        title={chapterShortTitle(chapter.title)}
        lede={`${chapter.flashcards.length} cards. Space to flip. Arrows to move.`}
      />
      <ArticleShell>
        <div className="mb-10 flex items-center justify-end gap-5">
          <div className="flex items-center gap-5">
            <button
              onClick={() => setViewMode('single')}
              className={`flex items-center gap-2 text-sm transition-colors ${
                viewMode === 'single' ? 'text-foreground' : 'text-muted hover:text-foreground'
              }`}
            >
              <Layers size={14} />
              <span className="hidden sm:inline">Single</span>
            </button>
            <button
              onClick={() => setViewMode('all')}
              className={`flex items-center gap-2 text-sm transition-colors ${
                viewMode === 'all' ? 'text-foreground' : 'text-muted hover:text-foreground'
              }`}
            >
              <Grid size={14} />
              <span className="hidden sm:inline">Section</span>
            </button>
            <button
              onClick={() => setViewMode('allSections')}
              className={`flex items-center gap-2 text-sm transition-colors ${
                viewMode === 'allSections' ? 'text-foreground' : 'text-muted hover:text-foreground'
              }`}
            >
              <Library size={16} />
              <span className="hidden sm:inline">All ({allFlashcards.length})</span>
            </button>
          </div>
        </div>

        {viewMode === 'single' && (
          <div className="flex flex-col items-center">
            <p className="mb-6 font-mono text-xs uppercase tracking-[0.18em] text-accent">
              {String(currentIndex + 1).padStart(2, '0')} of {String(chapter.flashcards.length).padStart(2, '0')}
            </p>
            
            <div 
              className="relative w-full max-w-lg h-80 perspective-1000 group cursor-pointer"
              onClick={() => setIsFlipped(!isFlipped)}
            >
              <div 
                className={`relative w-full h-full duration-500 preserve-3d transition-transform ${isFlipped ? 'rotate-y-180' : ''}`}
                style={{ transformStyle: 'preserve-3d', transform: isFlipped ? 'rotateY(180deg)' : 'rotateY(0deg)' }}
              >
                {/* Front */}
                <div 
                  className="absolute inset-0 backface-hidden flex flex-col items-center justify-center border border-border bg-card-bg p-8 text-center transition-colors hover:border-accent"
                  style={{ backfaceVisibility: 'hidden' }}
                >
                  <span className="mb-4 font-mono text-[11px] uppercase tracking-[0.15em] text-accent">Term</span>
                  <h3 className="font-serif text-3xl leading-snug text-foreground">{card.front}</h3>
                  <p className="absolute bottom-6 flex items-center gap-1 text-sm text-muted">
                    <RotateCw size={14} /> Click to flip
                  </p>
                </div>

                {/* Back */}
                <div 
                  className="absolute inset-0 backface-hidden flex rotate-y-180 flex-col items-center justify-center border border-white/10 bg-surface-dark p-8 text-center text-white"
                  style={{ backfaceVisibility: 'hidden', transform: 'rotateY(180deg)' }}
                >
                  <span className="mb-4 font-mono text-[11px] uppercase tracking-[0.15em] text-accent">Definition</span>
                  <p className="font-serif text-xl leading-relaxed">{card.back}</p>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-6 mt-10">
              <button
                onClick={(e) => { e.stopPropagation(); handlePrev(); }}
                disabled={currentIndex === 0}
                className="border border-border p-3 text-muted transition-colors hover:border-accent hover:text-accent disabled:cursor-not-allowed disabled:opacity-30"
              >
                <ChevronLeft size={24} />
              </button>
              
              <div className="h-px w-32 overflow-hidden bg-border">
                <div 
                  className="h-full bg-accent transition-all duration-300" 
                  style={{ width: `${((currentIndex + 1) / chapter.flashcards.length) * 100}%` }}
                />
              </div>

              <button
                onClick={(e) => { e.stopPropagation(); handleNext(); }}
                disabled={currentIndex === chapter.flashcards.length - 1}
                className="border border-border p-3 text-muted transition-colors hover:border-accent hover:text-accent disabled:cursor-not-allowed disabled:opacity-30"
              >
                <ChevronRight size={24} />
              </button>
            </div>

            {/* Keyboard shortcuts hint */}
            <div className="mt-8 flex flex-wrap justify-center gap-4 font-mono text-xs text-muted">
              <span className="flex items-center gap-1.5">
                <kbd className="border border-border px-1.5 py-0.5">←</kbd>
                <kbd className="border border-border px-1.5 py-0.5">→</kbd>
                Navigate
              </span>
              <span className="flex items-center gap-1.5">
                <kbd className="border border-border px-1.5 py-0.5">Space</kbd>
                Flip
              </span>
            </div>
          </div>
        )}

        {/* All cards from current section */}
        {viewMode === 'all' && (
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {chapter.flashcards.map((flashcard) => (
              <div
                key={flashcard.id}
                onClick={() => toggleCardFlip(flashcard.id)}
                className="relative h-48 cursor-pointer perspective-1000"
              >
                <div 
                  className="relative w-full h-full duration-300 transition-transform"
                  style={{ 
                    transformStyle: 'preserve-3d', 
                    transform: flippedCards.has(flashcard.id) ? 'rotateY(180deg)' : 'rotateY(0deg)' 
                  }}
                >
                  {/* Front */}
                  <div 
                    className="absolute inset-0 flex flex-col items-center justify-center border border-border bg-card-bg p-4 text-center transition-colors hover:border-accent"
                    style={{ backfaceVisibility: 'hidden' }}
                  >
                    <span className="text-[10px] font-semibold tracking-widest text-brand-500 uppercase mb-2">Term</span>
                    <h3 className="text-lg font-bold text-stone-800 line-clamp-3">{flashcard.front}</h3>
                    <p className="absolute bottom-3 text-stone-400 text-xs flex items-center gap-1">
                      <RotateCw size={10} /> Tap to flip
                    </p>
                  </div>

                  {/* Back */}
                  <div 
                    className="absolute inset-0 flex flex-col items-center justify-center border border-white/10 bg-surface-dark p-4 text-center text-white"
                    style={{ backfaceVisibility: 'hidden', transform: 'rotateY(180deg)' }}
                  >
                    <span className="text-[10px] font-semibold tracking-widest text-brand-400 uppercase mb-2">Definition</span>
                    <p className="text-sm leading-relaxed line-clamp-5">{flashcard.back}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* All cards from ALL sections */}
        {viewMode === 'allSections' && (
          <div>
            <p className="text-stone-500 mb-6 text-center">{allFlashcards.length} flashcards across all sections</p>
            <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {allFlashcards.map(({ card, sectionTitle }) => (
                <div
                  key={card.id}
                  onClick={() => toggleCardFlip(card.id)}
                  className="relative h-52 cursor-pointer perspective-1000"
                >
                  <div 
                    className="relative w-full h-full duration-300 transition-transform"
                    style={{ 
                      transformStyle: 'preserve-3d', 
                      transform: flippedCards.has(card.id) ? 'rotateY(180deg)' : 'rotateY(0deg)' 
                    }}
                  >
                    {/* Front */}
                    <div 
                      className="absolute inset-0 flex flex-col items-center justify-center border border-border bg-card-bg p-4 text-center transition-colors hover:border-accent"
                      style={{ backfaceVisibility: 'hidden' }}
                    >
                      <span className="text-[9px] font-medium text-stone-400 uppercase mb-1">{sectionTitle}</span>
                      <span className="text-[10px] font-semibold tracking-widest text-brand-500 uppercase mb-2">Term</span>
                      <h3 className="text-lg font-bold text-stone-800 line-clamp-3">{card.front}</h3>
                      <p className="absolute bottom-3 text-stone-400 text-xs flex items-center gap-1">
                        <RotateCw size={10} /> Tap to flip
                      </p>
                    </div>

                    {/* Back */}
                    <div 
                      className="absolute inset-0 flex flex-col items-center justify-center border border-white/10 bg-surface-dark p-4 text-center text-white"
                      style={{ backfaceVisibility: 'hidden', transform: 'rotateY(180deg)' }}
                    >
                      <span className="text-[9px] font-medium text-stone-500 uppercase mb-1">{sectionTitle}</span>
                      <span className="text-[10px] font-semibold tracking-widest text-brand-400 uppercase mb-2">Definition</span>
                      <p className="text-sm leading-relaxed line-clamp-5">{card.back}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </ArticleShell>
    </div>
  );
};