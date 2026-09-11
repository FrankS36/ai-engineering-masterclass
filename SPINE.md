# Course spine

**Read `AGENTS.md` first if you are an agent.** That is the harness (how to work). This file is the curriculum (what to write next). Do not invent a third plan.

Repo: `github.com/FrankS36/ai-engineering-masterclass`  
Branch: `cursor/progress-dark-mode-9f6c`  
PR: https://github.com/FrankS36/ai-engineering-masterclass/pull/1

---

## Form factor (locked)

Same stack as sellhausen.com:

| Layer | Choice |
|---|---|
| App | **Next.js 15 App Router** (`app/layout.tsx`, `app/page.tsx`) |
| UI | **React 19** client island — the course shell is still one client app |
| CSS | **Tailwind v4** (`app/globals.css`, `@tailwindcss/postcss`) |
| Fonts | **DM Sans / Instrument Serif / Geist Mono** via `next/font/google` |
| Tokens | Parent dark-first palette: bg `#0e0e0c`, accent `#c8a96e`, paper `#f4f3ef` |
| Content | Canonical chapter text in `constants.ts`. Editable source in `chapters/*.md` |
| Order | `lib/courseOrder.ts` (`COURSE_ORDER`, `COURSE_SECTIONS`, `SKILL_HOME`) |
| Learner | `lib/learnerState.ts` — localStorage keys `aem-theme`, `aem-learner-v1` |
| Header | **Learn / Cards / Quiz** stay first-class in the top bar. Do not bury them. Resources is a quieter sibling. |

Run: `npm install && npm run dev` → http://localhost:3000  
Build: `npm run build`

Do **not** go back to Vite. Do **not** build a Field Guide hub. Do **not** touch the consulting repo.

---

## What this course is

Andrew Ng’s **AI Engineering Skills Map** is the spine. Skills, not a job title.

1. **Building and deploying AI applications** — ch2–ch8, ch10–ch12
2. **Software engineering fundamentals** — ch16, ch9, ch13
3. **Using coding agents** — ch15, ch17
4. **Shaping the build** — ch14

Supporting sources (fold in, don’t fork the map): Ng *Agentic AI*, Moroney *GenAI for Software Development*, Everitt *Spec-Driven Development*, Claude Certified Architect skills (not a cert track).

Letters:

- Part 1: https://www.deeplearning.ai/the-batch/the-ai-engineering-skills-map
- Part 2: https://www.deeplearning.ai/the-batch/he-ai-engineering-skills-map-in-detail-building-and-deploying-ai-applications
- Part 3: https://www.deeplearning.ai/the-batch/the-ai-engineering-skills-map-in-detail-software-engineering-fundamentals
- Part 4: https://www.deeplearning.ai/the-batch/the-ai-engineering-skills-map-in-detail-using-coding-agents
- Part 5 (shaping): not published yet — use Part 1 + ch14

---

## Quality bar (every chapter)

1. One job in the first screenful.
2. Principles that survive a model generation.
3. One worked example.
4. Look-ups only where facts move (price, window, current ID).
5. Quizzes test judgment, not SKUs.
6. One sitting. Cut the encyclopedia.

Chapter IDs stay `ch1`…`ch17`. After editing `chapters/chN-*.md`, copy into `constants.ts` (escape backticks).

---

## Chapter map

Pedagogical order = `COURSE_ORDER`. Sidebar = `COURSE_SECTIONS`.

| ID | Title now | Skill | Status |
|---|---|---|---|
| ch1 | The Four Skills | Orientation | **Prototype shipped** — iterate on this form |
| ch2 | How Foundation Models Work | 1 LLM foundations | Next rewrite |
| ch3 | Prompt Engineering | 1 LLM foundations | Encyclopedia |
| ch4 | Building with LLM APIs | 1 LLM foundations | Encyclopedia |
| ch5 | RAG & Knowledge Systems | 1 Grounding | Encyclopedia |
| ch6 | Agents & Tool Use | 1 Agentic systems | Partial Ng blurbs |
| ch7 | Evaluation & Testing | 1 Evals | Needs “evals as the gate” |
| ch16 | Software Engineering Fundamentals | 2 Five literacies | Wired; tighten |
| ch15 | Using Coding Agents | 3 Workflow | **Harness lesson shipped** — this repo is the worked example |
| ch17 | Spec-Driven Development | 3 Specs | Wired; tighten |
| ch8 | Production & Deployment | 1 Operate | Encyclopedia |
| ch9 | LLM Security | 2 Secure/reliable | Encyclopedia |
| ch10 | Fine-Tuning | 1 ML foundations | Encyclopedia |
| ch11 | Multimodal AI | 1 Multimodal | Encyclopedia |
| ch12 | Local & Edge AI | 1 Local/self-host | Encyclopedia |
| ch13 | AI UX Patterns | 2 Full-stack | Encyclopedia |
| ch14 | AI Product Strategy | 4 Shaping the build | Capstone; rename later |

---

## Next session (do this, in order)

1. **Iterate ch1** if the prototype form is wrong (type, chrome, length). Do not restyle the universe first.
2. **Rewrite ch2** to Skill 1 / LLM foundations — mental model only, no training-run gossip.
3. **Tighten ch16 → ch17** (Skill 2 then specs). ch15 harness lesson is in.
4. **Rewrite ch6 then ch7** (agentic systems + evals-as-gate).
5. Cut remaining encyclopedia chapters to the bar. Hide junk resource views until rewritten.
6. Rewrite or delete `CURRICULUM.md` / `GAPS.md` so they cannot contradict this file.

If you only have time for one thing after ch1: **ch2**.

---

## Session log

- Learner chrome (progress, bookmarks, dark mode).
- Perishable catalogs stripped; not sufficient by itself.
- ch16 / ch17 wired. ch15 in app.
- **Stack moved to Next.js 15 + Tailwind v4 + parent fonts/tokens** (same form as sellhausen.com).
- **ch1 rewritten** to Ng’s four skills. Sidebar labeled to the map. `SKILL_HOME` on every chapter.
- **Harness defined:** `AGENTS.md` + `scripts/inject-chapter.mjs` + `chapters/chN.assessments.json`. Do not hand-edit `constants.ts`.
- **ch15 rewritten** as the workbook lesson for that harness (worked example = this repo).
