export enum ViewMode {
  NOTES = 'NOTES',
  QUIZ = 'QUIZ',
  FLASHCARDS = 'FLASHCARDS',
  TUTOR = 'TUTOR',
  RESOURCES = 'RESOURCES',
  GLOSSARY = 'GLOSSARY',
  CHEATSHEETS = 'CHEATSHEETS',
  INTERVIEW = 'INTERVIEW',
  PROJECTS = 'PROJECTS',
  SYSTEM_DESIGN = 'SYSTEM_DESIGN',
  TOOLKIT = 'TOOLKIT'
}

export interface QuizQuestion {
  id: string;
  question: string;
  options: string[];
  correctIndex: number;
  explanation: string;
}

export interface Flashcard {
  id: string;
  front: string;
  back: string;
}

export interface Chapter {
  id: string;
  title: string;
  content: string; // Markdown-like string
  quizzes: QuizQuestion[];
  flashcards: Flashcard[];
}

export interface Message {
  role: 'user' | 'model';
  text: string;
  isError?: boolean;
}
