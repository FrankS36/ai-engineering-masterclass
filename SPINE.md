# Course spine

**Read this first in any new session.** This is the source of truth for the masterclass restructure. Do not invent a parallel plan.

Repo: `github.com/FrankS36/ai-engineering-masterclass`  
Branch in flight: `cursor/progress-dark-mode-9f6c`  
PR: https://github.com/FrankS36/ai-engineering-masterclass/pull/1

---

## What we are doing

Restructure this **AI Engineering Masterclass** so it teaches modern practice, not a 2023 API tour.

The course stays vendor-neutral in the prose. Claude-specific material (Claude Code, Agent SDK, MCP, `CLAUDE.md`, Skills, hooks) is taught as *how this looks when you actually ship* — not as a cert dump and not as a Field Guide for AI transformation consulting.

---

## Guardrails (do not break)

- Work **only** in this repo. Do not touch any AI transformation / Field Guide / sellhausen.com consulting repo.
- Do not create a parent-site-style hub, skills-map landing, or work-system overlay.
- Do not rename this into a Claude cert-prep course. Fold Architect skills into existing chapters.
- Keep perishable model names and prices out of prose; link to sources.
- Chapter IDs stay stable (`ch1`…`ch17`). Pedogogical order lives in `lib/courseOrder.ts`, not in the array order inside `constants.ts`.

---

## Source materials

### 1. Andrew Ng — AI Engineering Skills Map (The Batch)

| Letter | URL | What we take |
|---|---|---|
| Part 1 — four skills | https://www.deeplearning.ai/the-batch/the-ai-engineering-skills-map | Spine: build/deploy AI apps · SE fundamentals · using coding agents · shaping the build. Skills, not a job title. |
| Part 2 — applications | https://www.deeplearning.ai/the-batch/he-ai-engineering-skills-map-in-detail-building-and-deploying-ai-applications | LLM foundations, grounding, agentic systems, eval-driven development, production, ML foundations. |
| Part 3 — fundamentals | https://www.deeplearning.ai/the-batch/the-ai-engineering-skills-map-in-detail-software-engineering-fundamentals | Full-stack literacy, data, architecture, secure/reliable, scale/operate. Steer the agent in engineering language. |
| Parts 4–5 (coding agents, shaping the build) | Watch The Batch for follow-ups | Fold in when published. |

### 2. DeepLearning.AI courses Frank is taking

| Course | Instructor | What we take | Lands in |
|---|---|---|---|
| **Agentic AI** | Andrew Ng | Four patterns: reflection, tool use, planning, multi-agent. Task decomposition. **Evals/error analysis as the #1 predictor.** Vendor-neutral, first principles, then frameworks. | ch6, ch7 |
| **Generative AI for Software Development** | Laurence Moroney | Pair-program the whole SDLC: iterative prompting, roles, testing/debugging, docs, dependencies, config-driven design, databases, design patterns. LLM as teammate; human keeps decisions. | ch16, ch15 |
| **Spec-Driven Development with Coding Agents** | Paul Everitt (JetBrains) | Vibe vs spec. Constitution. Feature spec. Plan → implement → verify. Replan. Legacy SDD. Package as a portable agent skill. Agent replaceability. | ch17, ch15 |

### 3. Claude Certified Architect (Anthropic)

Two exams. We teach the **skills**, not the proctoring trivia.

**Foundations (CCAR-F)** — design and build agent systems. Scenario-based. Domains:

| Domain | Wt | Course home | Notes |
|---|---|---|---|
| Agentic architecture & orchestration | 27% | ch6 | Single vs multi-agent, orchestrator/subagent, context isolation, Agent SDK loop, when a swarm is worse. |
| Claude Code configuration & workflows | 20% | ch15, ch17 | `CLAUDE.md` hierarchy, Skills (`SKILL.md`), slash commands, hooks, plan mode vs execute, CI (`-p`), permissions. |
| Prompt engineering & structured output | 20% | ch3, ch4 | System prompts, XML/structure, few-shot, thinking, JSON/schema, tool_choice, eval of prompts. |
| Tool design & MCP integration | 18% | ch6 | Tool vs resource vs skill, schemas, error contracts, 18-tool degradation, secure MCP. |
| Context management & reliability | 15% | ch4, ch8 | Caching, compaction, token budgets, long-horizon memory, HITL escalation, fail closed. |

CCAR-F scenario bank (use as *worked examples*, not exam clones):

1. Support agent + MCP + escalation
2. Team Claude Code: slash commands, `CLAUDE.md`, plan mode
3. Multi-agent research: coordinator + search/analysis/synthesis/report
4. Dev-productivity agent: Read/Write/Bash/Grep/Glob + MCP
5. Claude Code in CI: review, tests, PR feedback
6. Structured extraction against JSON schemas

**Professional (CCAR-P)** — own the production system. Domains to fold in (not a separate track):

| Domain | Wt | Course home |
|---|---|---|
| Integration | 19% | ch8, ch4 |
| Solution design & architecture | 17% | ch14, ch16, System Design resource |
| Evaluation, testing, optimization | 16% | ch7 |
| Governance, safety, risk | 14% | ch9 |
| Stakeholder communication & lifecycle | 14% | ch14 |
| Models, prompting, context | 13% | ch2–4 |
| Developer productivity / enablement | 7% | ch15, ch17 |

Official orientation: https://claude.com/blog/four-role-based-claude-certifications  
Prep lives in Anthropic Partner Academy (partner-gated). Public Claude docs for Agent SDK / MCP / Claude Code are fair game.

We are **not** writing a CCAR-F cram chapter. If a learner later wants exam practice, that is a follow-on, not this course.

### 4. Frank’s own repos (worked examples, cite by pattern)

| Repo | Use as |
|---|---|
| `ai-native-saas-starter` | Feature brief → AI session → services + tests. `.cursorrules`, `docs/ai-setup.md`. |
| `agentic-venture-studio` | Orchestrator, stage gates, observability. |
| `builder-advisor` | Portable `SKILL.md`, MCP tools, dual skill/server model. |
| `clean-code-principles` | `AGENTS.md`, TDD checklists, before/after. |
| `cli_project` | MCP client/server, educational eval harness. |
| `ai-pm-cards` | Probabilistic acceptance criteria, sub-agent defs — lightly, for ch7/ch14. |

---

## Target course map

Pedagogical order is `COURSE_ORDER` in `lib/courseOrder.ts`. Sidebar groups are `COURSE_SECTIONS`.

| # | ID | Title | Section | Status |
|---|---|---|---|---|
| 1 | ch1 | The AI Engineering Landscape | Foundations | **Needs pass** — organization blurb started; still old “landscape” body |
| 2 | ch2 | How Foundation Models Work | Foundations | Keep; light refresh later |
| 3 | ch3 | Prompt Engineering | AI Applications | **Needs CCA D4** — structured output, XML, schemas, instruction hierarchy |
| 4 | ch4 | Building with LLM APIs | AI Applications | **Needs CCA** — caching, compaction, token budgets, tool_choice |
| 5 | ch5 | RAG & Knowledge Systems | AI Applications | Keep; later add CCAR-P RAG/integration notes |
| 6 | ch6 | Agents & Tool Use | AI Applications | **Partial** — Ng four patterns + evals-first blurb added. Still need reflection as a first-class section, MCP vs skill vs resource, orchestrator/subagent isolation |
| 7 | ch7 | Evaluation & Testing | AI Applications | Keep core; **add** “evals as the gate before any model/arch change” (Ng + CCAR-P) |
| 8 | ch16 | Software Engineering Fundamentals | Engineering with Agents | **Drafted** in `chapters/ch16-se-fundamentals.md` — must live in `constants.ts` + quizzes |
| 9 | ch15 | Using Coding Agents | Engineering with Agents | **In app** — Moroney SDLC section added. **Needs CCA D3** — hooks, slash commands, plan mode, CI, permissions |
| 10 | ch17 | Spec-Driven Development | Engineering with Agents | **Drafted** in `chapters/ch17-spec-driven.md` — must live in `constants.ts` + quizzes |
| 11 | ch8 | Production & Deployment | Shipping | **Needs CCA** — context reliability, fail closed, HITL, Agent SDK in prod |
| 12 | ch9 | LLM Security | Shipping | **Needs CCA-P** — tool-call authorization, safety stack |
| 13 | ch10 | Fine-Tuning | Advanced | Keep |
| 14 | ch11 | Multimodal AI | Advanced | Keep |
| 15 | ch12 | Local & Edge AI | Advanced | Keep |
| 16 | ch13 | AI UX Patterns | Advanced | Keep |
| 17 | ch14 | AI Product Strategy | Strategy | Keep as capstone; **add** “what belongs in the spec” + CCAR-P solution-design / stakeholder |

Also in the app (not chapters): Interview, Toolkit, System Design, Glossary. Refresh later so they mention coding agents, SDD, and MCP.

---

## Mapping matrix (skills → chapter)

| Skill we promised to highlight | Primary chapter | Support |
|---|---|---|
| Build & deploy AI applications (Ng) | ch3–ch8 | ch2, ch10–ch13 |
| Software engineering fundamentals (Ng + Moroney) | ch16 | ch8, ch9 |
| Using coding agents (Ng + Moroney + CCA D3) | ch15 | ch16, ch17 |
| Shaping the build (Ng) | ch14 | ch17, ch1 |
| Agentic patterns + evals-first (Ng Agentic AI) | ch6, ch7 | ch8 |
| Spec-driven development (Everitt) | ch17 | ch15 |
| Architect: orchestration (CCA D1) | ch6 | — |
| Architect: MCP / tools (CCA D2) | ch6 | ch15 |
| Architect: Claude Code workflows (CCA D3) | ch15, ch17 | — |
| Architect: structured output (CCA D4) | ch3, ch4 | ch17 scenario 6 |
| Architect: context & reliability (CCA D5) | ch4, ch8 | ch6 |
| Architect Pro: evals as gates, safety, integration | ch7, ch8, ch9, ch14 | — |

---

## Implementation rules

1. **Canonical chapter text for the app is `constants.ts`.** Markdown under `chapters/` is the editable source — after changing an `.md`, copy into `constants.ts` (escape backticks) and keep quizzes/flashcards in sync.
2. **Next-chapter navigation** uses `COURSE_ORDER`, not `chapters[]` index.
3. New chapters get 4–5 quizzes and 5–8 flashcards in the same voice as existing ones.
4. Claude-specific sections are labeled as such (Claude Code, Agent SDK) and paired with the portable idea (`AGENTS.md` / Cursor rules / generic MCP).

---

## Session log

### Done

- Local progress, bookmarks, dark mode (learner state). Course-only; not a transformation hub.
- Deleted the Field Guide–style Skills Map experiment.
- ch15 *Using Coding Agents* in the app (mental model, context, verifiers, intervention, production bounds).
- ch1 “How this course is organized” (Ng + three DL.AI courses). **Update this blurb when you next edit ch1** to mention Claude Certified Architect as a fourth source.
- ch6 opening: Ng’s four patterns + evals-first.
- ch15: Moroney “pair-program the SDLC” section.
- `lib/courseOrder.ts` + sidebar sections for the 17-chapter map.
- Drafts: `chapters/ch16-se-fundamentals.md`, `chapters/ch17-spec-driven.md`.

### Not done (next sessions, in this order)

1. **Wire ch16 and ch17 into `constants.ts`** (content + quizzes + flashcards) so they appear in the sidebar.
2. **ch1 blurb** — add Claude Certified Architect as source #4 (Foundations domains + Professional fold-ins). Point here.
3. **ch15 CCA D3 pass** — `CLAUDE.md` hierarchy, Skills, slash commands, hooks, plan mode, CI, permissions. Keep it portable (same ideas as Cursor rules / agent skills).
4. **ch6 CCA/Ng pass** — first-class Reflection section; MCP vs tool vs skill vs resource; orchestrator/subagent and context isolation; when not to multi-agent.
5. **ch3/ch4 CCA D4 + D5 pass** — structured output, schemas, caching, compaction, token budgets.
6. **ch7** — evals as the gate (Ng + CCAR-P).
7. **ch8/ch9** — reliability, fail closed, tool authorization, HITL.
8. **ch14** — shaping the build + what belongs in the spec.
9. Refresh Interview / Toolkit / System Design so they match the new spine.
10. Update `CURRICULUM.md` to match this file (this file wins if they drift).

### Explicitly out of scope

- AI transformation Field Guide, work-system canvases, sellhausen.com content lifts.
- A standalone “pass CCAR-F” quiz track (unless Frank asks later).
- Rewriting every existing chapter from scratch in one sitting.

---

## How to continue in a new session

```
Read SPINE.md.
Stay on cursor/progress-dark-mode-9f6c (or the current PR branch).
Do the next unchecked item under "Not done".
Commit + push + update the PR.
Append a line to Session log when you finish a slice.
```

If you only have time for one thing: **wire ch16 and ch17 into the running app.** The spine is useless if those chapters only exist as markdown files.
