import React, { useState, useEffect, useCallback } from 'react';
import { Chapter, Flashcard } from '../types';
import { chapters } from '../constants';
import { ChevronLeft, ChevronRight, RotateCw, Grid, Layers, Library } from 'lucide-react';

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
    <div className="min-h-[80vh] bg-stone-50 py-8">
      <div className="max-w-5xl mx-auto px-6">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h2 className="text-2xl font-bold text-stone-800 mb-1">Flashcards</h2>
            <p className="text-stone-500">{chapter.flashcards.length} cards in this section</p>
          </div>
          
          {/* View toggle */}
          <div className="flex bg-stone-200 p-1 rounded-lg">
            <button
              onClick={() => setViewMode('single')}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-all ${
                viewMode === 'single' ? 'bg-white text-stone-900 shadow-sm' : 'text-stone-600 hover:text-stone-900'
              }`}
            >
              <Layers size={16} />
              <span className="hidden sm:inline">Single</span>
            </button>
            <button
              onClick={() => setViewMode('all')}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-all ${
                viewMode === 'all' ? 'bg-white text-stone-900 shadow-sm' : 'text-stone-600 hover:text-stone-900'
              }`}
            >
              <Grid size={16} />
              <span className="hidden sm:inline">Section</span>
            </button>
            <button
              onClick={() => setViewMode('allSections')}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-all ${
                viewMode === 'allSections' ? 'bg-white text-stone-900 shadow-sm' : 'text-stone-600 hover:text-stone-900'
              }`}
            >
              <Library size={16} />
              <span className="hidden sm:inline">All ({allFlashcards.length})</span>
            </button>
          </div>
        </div>

        {/* Single card view */}
        {viewMode === 'single' && (
          <div className="flex flex-col items-center">
            <p className="text-stone-500 mb-6">Card {currentIndex + 1} of {chapter.flashcards.length}</p>
            
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
                  className="absolute inset-0 backface-hidden bg-white border-2 border-stone-200 rounded-2xl shadow-xl flex flex-col items-center justify-center p-8 text-center hover:border-brand-300 transition-colors"
                  style={{ backfaceVisibility: 'hidden' }}
                >
                  <span className="text-xs font-semibold tracking-widest text-brand-500 uppercase mb-4">Term</span>
                  <h3 className="text-3xl font-bold text-stone-800">{card.front}</h3>
                  <p className="absolute bottom-6 text-stone-400 text-sm flex items-center gap-1">
                    <RotateCw size={14} /> Click to flip
                  </p>
                </div>

                {/* Back */}
                <div 
                  className="absolute inset-0 backface-hidden bg-stone-900 text-white rounded-2xl shadow-xl flex flex-col items-center justify-center p-8 text-center rotate-y-180"
                  style={{ backfaceVisibility: 'hidden', transform: 'rotateY(180deg)' }}
                >
                  <span className="text-xs font-semibold tracking-widest text-brand-400 uppercase mb-4">Definition</span>
                  <p className="text-xl leading-relaxed">{card.back}</p>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-6 mt-10">
              <button
                onClick={(e) => { e.stopPropagation(); handlePrev(); }}
                disabled={currentIndex === 0}
                className="p-3 rounded-full bg-white border border-stone-200 text-stone-600 hover:bg-stone-50 hover:text-brand-600 disabled:opacity-30 disabled:cursor-not-allowed shadow-sm transition-all"
              >
                <ChevronLeft size={24} />
              </button>
              
              <div className="h-1 w-32 bg-stone-200 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-brand-500 transition-all duration-300" 
                  style={{ width: `${((currentIndex + 1) / chapter.flashcards.length) * 100}%` }}
                />
              </div>

              <button
                onClick={(e) => { e.stopPropagation(); handleNext(); }}
                disabled={currentIndex === chapter.flashcards.length - 1}
                className="p-3 rounded-full bg-white border border-stone-200 text-stone-600 hover:bg-stone-50 hover:text-brand-600 disabled:opacity-30 disabled:cursor-not-allowed shadow-sm transition-all"
              >
                <ChevronRight size={24} />
              </button>
            </div>

            {/* Keyboard shortcuts hint */}
            <div className="mt-8 flex flex-wrap justify-center gap-4 text-xs text-stone-400">
              <span className="flex items-center gap-1.5">
                <kbd className="px-1.5 py-0.5 bg-stone-100 rounded border border-stone-200 font-mono">←</kbd>
                <kbd className="px-1.5 py-0.5 bg-stone-100 rounded border border-stone-200 font-mono">→</kbd>
                Navigate
              </span>
              <span className="flex items-center gap-1.5">
                <kbd className="px-1.5 py-0.5 bg-stone-100 rounded border border-stone-200 font-mono">Space</kbd>
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
                    className="absolute inset-0 bg-white border border-stone-200 rounded-xl shadow-sm flex flex-col items-center justify-center p-4 text-center hover:border-brand-300 transition-colors"
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
                    className="absolute inset-0 bg-stone-900 text-white rounded-xl shadow-sm flex flex-col items-center justify-center p-4 text-center"
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
                      className="absolute inset-0 bg-white border border-stone-200 rounded-xl shadow-sm flex flex-col items-center justify-center p-4 text-center hover:border-brand-300 transition-colors"
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
                      className="absolute inset-0 bg-stone-900 text-white rounded-xl shadow-sm flex flex-col items-center justify-center p-4 text-center"
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
      </div>
    </div>
  );
};