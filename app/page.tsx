'use client';

import App from '../App';
import { LearnerProvider } from '../context/LearnerContext';

export default function HomePage() {
  return (
    <LearnerProvider>
      <App />
    </LearnerProvider>
  );
}
