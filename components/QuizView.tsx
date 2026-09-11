import React, { useState, useEffect } from 'react';
import { Chapter } from '../types';
import { CheckCircle, XCircle, RefreshCw } from 'lucide-react';
import { useLearner } from '../context/LearnerContext';
import { chapterOrderLabel, chapterShortTitle } from '../lib/courseOrder';
import { EditorialHero, ArticleShell } from './EditorialChrome';

interface QuizViewProps {
  chapter: Chapter;
}

export const QuizView: React.FC<QuizViewProps> = ({ chapter }) => {
  const { recordQuizScore, state } = useLearner();
  const best = state.quizBestScores[chapter.id];
  const [answers, setAnswers] = useState<Record<string, number>>({});
  const [showResults, setShowResults] = useState(false);
  const [score, setScore] = useState(0);

  // Reset state when chapter changes
  useEffect(() => {
    setAnswers({});
    setShowResults(false);
    setScore(0);
  }, [chapter.id]);

  const handleSelect = (questionId: string, optionIndex: number) => {
    if (showResults) return;
    setAnswers(prev => ({ ...prev, [questionId]: optionIndex }));
  };

  const calculateScore = () => {
    let correct = 0;
    chapter.quizzes.forEach(q => {
      if (answers[q.id] === q.correctIndex) correct++;
    });
    setScore(correct);
    setShowResults(true);
    recordQuizScore(chapter.id, correct, chapter.quizzes.length);
  };

  const resetQuiz = () => {
    setAnswers({});
    setShowResults(false);
    setScore(0);
  };

  const num = chapterOrderLabel(chapter.id);

  return (
    <div>
      <EditorialHero
        kicker={`Masterclass · Quiz · ${num}`}
        number={num}
        title={chapterShortTitle(chapter.title)}
        lede="Judgment, not SKUs. Pick the answer you would defend."
        byline={best ? `Best score ${best.score}/${best.total}` : `${chapter.quizzes.length} questions`}
      />
      <ArticleShell>
        {chapter.quizzes.map((q, index) => (
          <section key={q.id} className="mb-12">
            <p className="font-mono text-xs text-accent">{String(index + 1).padStart(2, '0')}</p>
            <h2 className="mt-2 font-serif text-2xl leading-snug text-foreground sm:text-3xl">{q.question}</h2>
            <div className="mt-6 space-y-2">
              {q.options.map((option, idx) => {
                let btnClass = 'w-full border px-4 py-3 text-left text-sm leading-relaxed transition-colors ';
                if (showResults) {
                  if (idx === q.correctIndex) btnClass += 'border-accent bg-card-bg text-foreground';
                  else if (answers[q.id] === idx) btnClass += 'border-red-500/50 text-red-400';
                  else btnClass += 'border-transparent text-muted opacity-50';
                } else if (answers[q.id] === idx) {
                  btnClass += 'border-accent bg-card-bg text-foreground';
                } else {
                  btnClass += 'border-border text-muted hover:border-accent hover:text-foreground';
                }
                return (
                  <button key={idx} onClick={() => handleSelect(q.id, idx)} disabled={showResults} className={btnClass}>
                    <span className="flex items-center justify-between gap-3">
                      <span>{option}</span>
                      {showResults && idx === q.correctIndex && <CheckCircle className="h-4 w-4 text-accent" />}
                      {showResults && answers[q.id] === idx && idx !== q.correctIndex && <XCircle className="h-4 w-4 text-red-400" />}
                    </span>
                  </button>
                );
              })}
            </div>
            {showResults && (
              <div className="mt-6 border-l-2 border-accent bg-card-bg p-5">
                <span className="font-mono text-[11px] uppercase tracking-[0.15em] text-accent">Explanation</span>
                <p className="mt-3 text-base leading-relaxed text-muted sm:text-[17px]">{q.explanation}</p>
              </div>
            )}
          </section>
        ))}

        <div className="sticky bottom-0 -mx-4 flex items-center justify-between border-t border-border bg-background/95 px-4 py-4 backdrop-blur sm:-mx-6 sm:px-6 lg:-mx-8 lg:px-8">
          {!showResults ? (
            <>
              <span className="font-mono text-xs text-muted">
                {Object.keys(answers).length} / {chapter.quizzes.length} answered
              </span>
              <button
                onClick={calculateScore}
                disabled={Object.keys(answers).length !== chapter.quizzes.length}
                className="border border-white/25 bg-surface-dark px-4 py-2 text-sm font-semibold text-white hover:border-white/50 disabled:opacity-40"
              >
                Submit answers
              </button>
            </>
          ) : (
            <>
              <span className="font-serif text-xl text-foreground">
                {score} / {chapter.quizzes.length}
              </span>
              <button onClick={resetQuiz} className="text-sm text-muted hover:text-accent">
                <RefreshCw size={14} className="mr-1.5 inline" />
                Try again
              </button>
            </>
          )}
        </div>
      </ArticleShell>
    </div>
  );
};