# Harness

This file is the agent constitution. `SPINE.md` is the curriculum. Do not invent a third plan.

## Job

Teach a working engineer to ship an AI feature. The spine is Andrew Ng’s AI Engineering Skills Map — skills, not a job title. One chapter per run.

## Stack (locked)

Next.js 15 App Router · React 19 · Tailwind v4 · DM Sans / Instrument Serif / Geist Mono · parent tokens (`#0e0e0c`, `#c8a96e`).

- Run: `npm install && npm run dev` → http://localhost:3000
- Build: `npm run build`
- Do not go back to Vite.

## Files that matter

| Role | Path | Rule |
|---|---|---|
| Constitution | `AGENTS.md` | This file. Keep it one sitting. |
| Curriculum | `SPINE.md` | What to write next. Do not contradict it. |
| Order / map | `lib/courseOrder.ts` | `COURSE_ORDER`, `COURSE_SECTIONS`, `SKILL_HOME`. IDs stay `ch1`…`ch17`. |
| Chapter source | `chapters/chN-*.md` | Edit here. |
| Quizzes / cards | `chapters/chN.assessments.json` | Optional. Injected with the chapter. |
| Live text | `constants.ts` | **Do not read or hand-edit.** Inject only. |
| Chrome | `app/`, `App.tsx`, `components/ChapterView.tsx` | Parent-site editorial form. Learn / Cards / Quiz stay first-class in the header. |

## How to change a chapter

1. Edit `chapters/chN-*.md` (and assessments if the quizzes change).
2. `node scripts/inject-chapter.mjs chN`
3. `npm run build` if you touched app code.
4. Commit, push, update the PR.

Never open `constants.ts` to “quickly fix” a sentence.

## Quality bar

1. One job in the first screenful.
2. Principles that survive a model generation.
3. One worked example.
4. Look-ups only where facts move.
5. Quizzes test judgment, not SKUs.
6. One sitting.

## Out of scope

- Any AI transformation / Field Guide / sellhausen.com consulting repo.
- A Skills Map hub or parent-site marketing page.
- A Claude Certified Architect cram track. Fold skills in; label Claude-specific.
- Restyling the universe. Iterate the form on ch1, then rewrite the next chapter on the map.
- Burying Learn / Cards / Quiz. Those stay first-class in the top bar. Resources is a quieter sibling.

## How to run (Cloud Agent)

- **One chapter.** State it in the first message.
- **Grok Bot / Composer** for copy and chrome on a live page.
- **Cloud Agent** for a rewrite + inject + commit + PR.
- Do not use High Fast for title tweaks.
- Do not dump `constants.ts` into context. Grep a heading if you must confirm inject worked.

## Next chapter

See `SPINE.md` → “Next session”. After ch1, that is **ch2** (LLM foundations).
