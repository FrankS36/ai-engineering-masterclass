# Spec-Driven Development

> Vibe coding is fast. It is also how you get software that does not match what you asked for, written by a model that has already forgotten the ask. A spec is how intent survives the session.

Paul Everitt’s DeepLearning.AI course with JetBrains names the practice: spec-driven development with coding agents. Andrew Ng’s skills map assumes it — “work with a clear spec, and know when not to bother.” This chapter is the workflow: constitution, feature spec, plan–implement–verify, replan, then carry the same loop into a legacy repo.

You already met constitutions in Chapter 15. Here they become the operating system of the project, not a tip.

## Vibe coding vs spec-driven

**Vibe coding:** You talk. The agent edits. You talk again. The thread is the spec. When the thread dies, the spec dies. Cognitive debt piles up: nobody can say what the system is *for* without reading the last 40 files.

**Spec-driven:** You write a markdown spec that defines what to build. The agent implements against it. You validate against it. When you replan, you change the spec first. Intent fidelity stays high because the source of truth is in the repo, not in a chat.

Many strong developers already work this way and did not have a name for it. The name matters because it makes the workflow teachable and portable across agents and IDEs.

Skip the spec for a one-line fix. Write the spec when the work will outlive one sitting: a feature, a migration, a public API, an MVP slice.

## The constitution

A project constitution is the spec you write once and reuse. Everitt’s course builds it with the agent, which is the right order: you decide, the agent drafts, you cut.

A constitution that works is short:

- **Mission** — what this repo is for, in one paragraph
- **Stack** — languages, frameworks, data store, test command
- **Invariants** — what must not change without a human
- **Landmines** — the modules that bill, auth, or delete data
- **Definition of done** — the verifier the agent may run
- **Out of scope** — what we are explicitly not building

Frank’s own starter repos do this as `.cursorrules`, `CLAUDE.md`, `AGENTS.md`, or a `docs/ai-setup.md` plus a feature-brief template. The filename is not the skill. The skill is: an agent that reads one page produces fewer “helpful” second HTTP clients.

Keep it in the repo. Point every new session at it. When reality disagrees, edit the constitution — do not let the code and the page drift in silence.

## Feature specification

A feature spec is smaller and sharper than a constitution. It should be enough for an agent that has never seen the thread.

Minimum fields (this matches a feature brief you can actually hand to Cursor or Claude):

- User story
- Data model changes
- API or actions
- UI states (empty, loading, error, success)
- Constraints (perf, security, “do not touch X”)
- Done criteria a machine can run

If a criterion cannot fail, it is decoration. “Code is clean” is decoration. “`npm test` is green and the service rejects a cross-tenant id” is a criterion.

Write the spec in markdown in the repo. The agent implements from the file, not from your memory of the file.

## Plan → implement → verify

Everitt’s loop is the whole method.

**Plan.** The agent proposes files, schema, tests, and what it will not touch. You accept or cut the plan *before* it edits twenty files. A useful plan is gradable. A useless plan says “ensure quality.”

**Implement.** Autonomy only after the first loop proves it understood the spec (Chapter 15). Mechanical work can run unattended. Ontology errors get an interrupt.

**Verify.** Run the done criteria. If the agent weakened a test to go green, the spec failed — treat that as a product bug, not a cute agent quirk.

Then **replan**. The second feature is where SDD pays off. You update the spec and the constitution if the MVP changed shape. You do not keep a stale plan in a closed PR and a new vibe in chat.

An MVP in this workflow is a thin slice that satisfies the spec, not a pile of agent output that looks like a product.

## Legacy codebases

Greenfield SDD is easy. The valuable version is a repo that already exists.

Everitt’s sequence: use the documentation you have (README, types, tests, tickets) to *generate* specs, then implement against those specs. You are not boiling the ocean. You are putting a fence around the next change.

Practical start:

1. Write a constitution that describes the system as it is, not as you wish it was
2. Spec only the next feature
3. Generate tests for the current behavior before you change it
4. Point the agent at those tests as the verifier

Existing patterns beat a clean-room rewrite. Tell the agent that. Frank’s starter rule is the right one: follow the patterns in the repo even when they differ slightly from the rules; explain why.

## Package the workflow

The last step in the SDD course is portability: turn your loop into an agent skill — a `SKILL.md` or custom command that another IDE can load. Constitutions and specs are already portable if they are files. Skills make the *ritual* portable: “read constitution → write/update spec → plan → wait for approval → implement → verify.”

Agent replaceability is the test. If the workflow only works in one vendor’s chat, you do not have a workflow. You have a habit.

## When not to bother

Ng’s parenthetical matters. Do not spec:

- A typo
- A rename the compiler can prove
- An experiment you will delete this afternoon

Ceremony that does not reduce error is theater. SDD is for work whose intent must survive a context window.

## How this sits in the course

| Move | Source | Where else |
|---|---|---|
| Constitution | Everitt; your `CLAUDE.md` / `.cursorrules` | Chapter 15 |
| Feature spec | Everitt; feature-brief template | Chapter 14 (what belongs in the spec) |
| Plan–implement–verify | Everitt; Ng “close the loop with a verifier” | Chapter 7, Chapter 15 |
| Legacy SDD | Everitt | Chapter 8 |
| Portable skill | Everitt; builder-advisor `SKILL.md` pattern | Chapter 6 (MCP / tools) |

Spec-driven development is how software engineering fundamentals (Chapter 16) get into the agent’s context, and how “using coding agents” (Chapter 15) stops being a vibe.

## Summary

Write a constitution for the repo. Write a spec for the feature. Plan, implement, verify, replan. Generate specs from what a legacy system already does before you change it. Package the ritual so the next agent can run it.

The chat is not the source of truth. The repo is.
