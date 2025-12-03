import React, { useState, useEffect } from 'react';
import { Chapter } from '../types';
import { CheckCircle, XCircle, RefreshCw } from 'lucide-react';

interface QuizViewProps {
  chapter: Chapter;
}

export const QuizView: React.FC<QuizViewProps> = ({ chapter }) => {
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
  };

  const resetQuiz = () => {
    setAnswers({});
    setShowResults(false);
    setScore(0);
  };

  return (
    <div className="space-y-8 p-8 max-w-4xl mx-auto">
      <div className="bg-brand-50 border border-brand-100 p-6 rounded-xl mb-6">
        <h2 className="text-2xl font-bold text-brand-900">Quiz: {chapter.title}</h2>
        <p className="text-brand-700 mt-2">Test your understanding of the concepts in this chapter.</p>
      </div>

      {chapter.quizzes.map((q, index) => {
        const isAnswered = answers[q.id] !== undefined;
        const isCorrect = answers[q.id] === q.correctIndex;

        return (
          <div key={q.id} className="bg-white p-6 rounded-xl shadow-sm border border-stone-200">
            <h3 className="text-lg font-semibold text-stone-800 mb-4">
              <span className="text-brand-500 mr-2">{index + 1}.</span>
              {q.question}
            </h3>
            <div className="space-y-3">
              {q.options.map((option, idx) => {
                let btnClass = "w-full text-left p-4 rounded-lg border-2 transition-all ";
                
                if (showResults) {
                  if (idx === q.correctIndex) {
                    btnClass += "bg-green-50 border-green-500 text-green-800";
                  } else if (answers[q.id] === idx) {
                    btnClass += "bg-red-50 border-red-500 text-red-800";
                  } else {
                    btnClass += "bg-stone-50 border-transparent text-stone-400 opacity-50";
                  }
                } else {
                  if (answers[q.id] === idx) {
                    btnClass += "bg-brand-50 border-brand-500 text-brand-700 shadow-md";
                  } else {
                    btnClass += "bg-white border-stone-100 hover:border-brand-200 hover:bg-stone-50 text-stone-600";
                  }
                }

                return (
                  <button
                    key={idx}
                    onClick={() => handleSelect(q.id, idx)}
                    disabled={showResults}
                    className={btnClass}
                  >
                    <div className="flex items-center justify-between">
                        <span>{option}</span>
                        {showResults && idx === q.correctIndex && <CheckCircle className="text-green-600 h-5 w-5" />}
                        {showResults && answers[q.id] === idx && idx !== q.correctIndex && <XCircle className="text-red-600 h-5 w-5" />}
                    </div>
                  </button>
                );
              })}
            </div>
            
            {showResults && (
              <div className="mt-4 p-4 bg-stone-50 rounded-lg text-sm text-stone-700">
                <span className="font-semibold block mb-1">Explanation:</span>
                {q.explanation}
              </div>
            )}
          </div>
        );
      })}

      <div className="sticky bottom-4 bg-white/90 backdrop-blur-md p-4 rounded-xl shadow-lg border border-stone-200 flex items-center justify-between max-w-2xl mx-auto">
        {!showResults ? (
          <div className="w-full flex justify-between items-center">
             <span className="text-stone-600 font-medium">{Object.keys(answers).length} / {chapter.quizzes.length} answered</span>
             <button
              onClick={calculateScore}
              disabled={Object.keys(answers).length !== chapter.quizzes.length}
              className="bg-brand-600 hover:bg-brand-700 text-white px-6 py-2 rounded-lg font-medium disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              Submit Answers
            </button>
          </div>
        ) : (
          <div className="w-full flex justify-between items-center">
            <div className="text-lg font-bold text-stone-800">
              Score: <span className={score === chapter.quizzes.length ? 'text-green-600' : 'text-brand-600'}>{score} / {chapter.quizzes.length}</span>
            </div>
            <button
              onClick={resetQuiz}
              className="flex items-center gap-2 text-stone-600 hover:text-stone-900 px-4 py-2 font-medium"
            >
              <RefreshCw size={18} />
              Try Again
            </button>
          </div>
        )}
      </div>
    </div>
  );
};