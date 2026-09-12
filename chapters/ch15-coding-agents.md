# Using Coding Agents

> **From the field.** A coding agent without a closer is just a faster way to drift. The skill is not prompting the tool. The skill is deciding what the agent is allowed to touch, what evidence counts as done, and when you take the wheel.

## What you will be able to do

1. **Name the harness** — constitution, source of truth, injector, machine, verifier, bounds.
2. **Keep the blob out of the window.** The live artifact is not the file you edit.
3. **Grant autonomy only when a verifier exists.**
4. **Pick the loop:** a tight bot for copy and chrome; a Cloud Agent for one rewrite and a PR.

This chapter is Skill 3 on Ng’s map: *using* coding agents. Chapter 6 is *building* agents as product. Confusing them is how teams ship an internal Copilot and call the job done.

## The harness

The model is not the product. The **harness** is the loop you put around the model so it can do work without rediscovering your system every session.

A harness has six parts:

1. **Constitution** — a one-page file the agent reads first (`AGENTS.md`, `CLAUDE.md`, `.cursor/rules`). Purpose, stack, landmines, how to change a file.
2. **Curriculum / spec** — what to do next, separate from how to work. In this course that is `SPINE.md`. In a product that is the feature spec (Chapter 17).
3. **Source of truth for edits** — the files humans and agents are allowed to type in.
4. **Injector** — the only path from source into the live artifact. A script, a codegen step, a compiler. Not a human copy-paste.
5. **Machine** — how the environment boots (`environment.json`: install, start). Warm beats clever.
6. **Verifier** — the command that means done. Build, tests, a screenshot. Without this, “done” is a paragraph.

If any piece is missing, the agent fills the gap with a guess. Guesses look like competence until you read the diff.

## Worked example — this workbook

This masterclass *is* the example. Steal the shape. Do not steal the stack unless it is yours.

| Part | In this repo | Rule the agent must obey |
|---|---|---|
| Constitution | `AGENTS.md` | One chapter per run. Next.js is locked. |
| Curriculum | `SPINE.md` | What to write next. Do not invent a third plan. |
| Source | `chapters/chN-*.md` and `chapters/chN.assessments.json` | Edit here. |
| Live artifact | `constants.ts` | **Do not read or hand-edit.** |
| Injector | `node scripts/inject-chapter.mjs chN` | The only way source becomes live. |
| Machine | `environment.json` | `npm install` / `npm run dev`. |
| Verifier | `npm run build` | If chrome changed. |
| Bounds | Consulting repo, Field Guide hub, cert cram | Out of scope. |

The failure this harness exists to prevent: an agent opens a 450KB `constants.ts`, “quickly fixes” a sentence, migrates the bundler, answers a billing question, and never finishes the chapter. That is context rot plus no closer.

The loop that works:

1. State the chapter. (`ch15` — this one.)
2. Edit the markdown (and assessments if quizzes change).
3. Run the injector.
4. Grep a heading if you must confirm the splice. Do not dump the blob into the thread.
5. Commit. One job.

That is Ng’s Skill 3 in a repo: customize the agent and the environment. The portable names differ (`AGENTS.md` / Cursor rules / `CLAUDE.md`). The job does not.

> **Look up now.** Prompt: "What does the current Cursor docs say an `AGENTS.md` or project rules file should contain, and what does Anthropic document for `CLAUDE.md`? Cite official docs."

## A mental model, not a vendor tour

A coding agent is an LLM in a loop with tools: read, edit, run, search. It stops when it thinks it is done, hits a limit, or you interrupt.

The loop fails in predictable ways:

- **It optimizes for looking finished.** Tests pass because it weakened the test.
- **It loses the plot as context fills.** Mid-task it “simplifies” a constraint you stated at the start.
- **It does not know your blast radius.** It will migrate a schema if the tools are wide and the prompt is vague.
- **It is confident in prose and sloppy in edges.** Empty states, permissions, the one env var nobody documented.

If you cannot name those, you are not using an agent. You are supervising a fast intern with no memory of last quarter’s incident.

## Context is the scarce resource

Dumping the repository into chat is not context management. It is noise.

**Constitution, not a novel.** One page: what this is for, what must not change, how to verify, where the landmines are.

**Point, don’t narrate.** “Failing test in `invoice.test.ts`; do not change the public API of `InvoiceService`” is a job. “Fix billing” is a wish.

**Reset when the thread is dirty.** Contradictory instructions accumulate. Context rot looks like competence until you read the diff.

**Separate privileged from unprivileged.** The agent that edits `src/` is not the agent that holds production credentials.

## Plan versus execute

Plan when the change crosses modules, you do not know the shape, or another human must review the spec. Skip the plan when you already have a failing test or the change is mechanical.

A useful plan is gradable: files in, files out, acceptance checks, rollback. A useless plan says “ensure quality.”

Chapter 17 is the full spec loop. Here, know when *not* to bother.

## Close the loop with a verifier

Ng: help the agent autonomously close loops by providing verifiers or evals.

Without a verifier, done means the model wrote a paragraph. With one: the named tests are green, types are clean, the build passed, the eval did not regress.

Write the verifier first when you can. A failing test is the best prompt you will ever write. If you cannot write a verifier, do not grant autonomy.

Same muscle as Chapter 7. Your repo is the system under test.

## How much to intervene

1. State the outcome and the bound.
2. Give a verifier.
3. Watch the first loop. Interrupt if the ontology is wrong.
4. Leave it alone through mechanical work.
5. Review the diff as a senior engineer. You own the merge.

Autonomy is a privilege per task, not a setting you leave on.

Use a **tight bot** (Grok Bot, Composer, in-editor agent) for copy and chrome on a live page. Use a **Cloud Agent** when you need a fresh machine, a branch, and a PR. Do not use High Fast for a title change. The expensive loop is for the hard rewrite.

## Orchestrating more than one agent

Useful when the work is actually parallel: independent features, a reviewer that did not write the code. Harmful when two agents share a working tree. Use worktrees, branches, or a queue.

A second agent as reviewer is often worth more than a second implementer. Different context, different incentive.

## Production is not a sandbox

Minimum bounds: read-only prod by default, no secrets in the thread, human-gated migrations, a blast radius you can explain. If you cannot name what the tools can destroy, the design is unfinished. That is Skill 2 leaking into Skill 3 on purpose.

## Evolve the workflow on purpose

Keep a short log: what you delegated, where the agent wasted you, which verifier caught it. Promote what survives. A workflow you cannot explain to a new teammate is a habit. Habits do not survive an incident.

## How this sits on the map

| Skill Ng named | What you practice here | Where else |
|---|---|---|
| Harness / customize the environment | Constitution, injector, machine, verifier | This chapter’s worked example |
| Mental model of agents | Loops, false done, context rot | Chapter 6 |
| Plan vs execute | Specs when they earn their keep | Chapter 17, Chapter 14 |
| Verifiers | Tests and evals as done | Chapter 7 |
| Intervention | Autonomy as a per-task privilege | Chapter 8, Chapter 9 |
| Production safety | Blast radius, gates, secrets | Chapter 8, Chapter 9 |

Using coding agents well does not replace software engineering. It *is* software engineering, performed through a stochastic intern you are responsible for.

## Summary

Give the agent a harness: a constitution, a source it is allowed to edit, an injector into the live artifact, a machine, and a verifier. Bound what it can touch. Interrupt when the goal is wrong. Review the diff as if your name is on the commit — because it is.

The product names will change. The loop will not.
