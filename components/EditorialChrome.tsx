import React from 'react';

export const GoldGrid: React.FC<{ className?: string }> = ({ className = '' }) => (
  <div className={`gold-grid pointer-events-none absolute inset-0 opacity-[0.07] ${className}`} />
);

export const EditorialHero: React.FC<{
  kicker: string;
  number: string;
  title: string;
  lede?: string;
  byline?: string;
}> = ({ kicker, number, title, lede, byline }) => (
  <section className="relative overflow-hidden bg-surface-dark text-white">
    <GoldGrid />
    <div className="relative mx-auto max-w-3xl px-4 py-16 sm:px-6 lg:px-8 lg:py-20">
      <p className="font-mono text-xs uppercase tracking-[0.18em] text-secondary">{kicker}</p>
      <p className="mt-4 font-serif text-7xl text-secondary/30" aria-hidden="true">
        {number}
      </p>
      <h1 className="mt-2 max-w-2xl text-balance font-serif text-3xl leading-[1.15] sm:text-4xl lg:text-[2.75rem]">
        {title}
      </h1>
      {lede && (
        <p className="mt-6 max-w-2xl text-pretty text-base leading-relaxed text-surface-dark-muted sm:text-lg">
          {lede}
        </p>
      )}
      {byline && <p className="mt-6 font-mono text-xs text-surface-dark-muted">{byline}</p>}
    </div>
  </section>
);

export const ArticleShell: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <article className="bg-background py-12 lg:py-16">
    <div className="mx-auto max-w-3xl px-4 sm:px-6 lg:px-8">{children}</div>
  </article>
);
