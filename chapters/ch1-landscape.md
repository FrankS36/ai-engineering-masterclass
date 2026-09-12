# The Four Skills

> **From the field — Frank Sellhausen, Sellhausen AI Systems.** AI does not introduce new failures. It speeds up the ones you already have: inventory gaps, controls you cannot demonstrate, and documentation that does not trace from intent to monitoring. Tools do not fix culture. They reproduce it faster.

## What you will be able to do

Leave this chapter able to:

1. **Name the four skills** Andrew Ng put on the [AI Engineering Skills Map](https://www.deeplearning.ai/the-batch/the-ai-engineering-skills-map) — skills, not a job title.
2. **Point at a chapter** for each capability under those skills.
3. **Pick a starting move** — read in order, or jump to the failure you have today.
4. **Choose a model tier** without memorizing this quarter’s SKUs.

The rest of the course is the map, taught as practice. This chapter is the legend.

## Skills, not a job title

AI engineering is a set of skills. Some people hold several. Teams split them. The title on the badge matters less than whether someone can do the work.

The work exists because models are good enough to *use*. You no longer have to train the intelligence to ship a feature. You compose it — and then you have to measure it, because the output is not deterministic.

That last clause is the whole job. Traditional software fails closed or it fails loud. Language models fail politely, confidently, and differently every run. Ng’s first skill exists because of that: you need building blocks *and* statistical techniques to measure, steer, and govern the system.

## The four skills

1. **Building and deploying AI applications** — ship systems whose outputs you cannot fully predict. Use evals and error analysis as the loop, not a report you write after launch.
2. **Software engineering fundamentals** — know the tradeoffs well enough to brief an agent in engineering language. Vibe-coding without this is how demos become incidents.
3. **Using coding agents** — a mental model, a plan, verifiers, and the judgment to stop. The agent types. You decide what “done” means.
4. **Shaping the build** — decide what belongs in the spec. Product sense, business context, ownership. When to ship an MVP, and when to slow down.

Read them as one practice. Skill 1 is the product. Skill 2 is the language you use to steer. Skill 3 is how the work gets typed. Skill 4 is whether you are building the right thing.

## Skill 1 — Building and deploying AI applications

Ng’s detail letter breaks this into six capabilities. That is the body of this course.

| Capability | What it is | Chapter |
|---|---|---|
| **LLM foundations** | Tokenize and generate. Context, cache, cutoff, sampling, tools. When to fine-tune or self-host. | 2, 3, 4, 11, 12 |
| **Grounding with data** | Not “buy a vector database.” Prompt vs retrieve-via-tools. Keep the data fresh. | 5 |
| **Agentic systems** | Workflow vs harness. Tools, memory, when *not* to multi-agent. | 6 |
| **Evaluation-driven development** | *The* distinguisher. Traces, error analysis, judges you can defend. | 7 |
| **Operating in production** | Observability, drift, cost, latency, statistical confidence. | 8 |
| **Machine learning foundations** | Bias/variance, error analysis, data work. Still required. | 10 |

If you only take one sentence from Skill 1: **evals and error analysis are the loop.** Guessing which component to improve is how agent projects stall.

## Skill 2 — Software engineering fundamentals

The AI core almost always lives inside a broader application. Ng’s five literacies are what you must already understand so a coding agent has something to steer toward.

1. **Full-stack applications** — where the request goes, and what the user actually sees. Chapter 13, Chapter 16.
2. **Managing data** — what is stored, who can read it, how it stays true. Chapter 16.
3. **Designing system architectures** — you provide the architecture; the agent cannot invent your constraints. Chapter 16.
4. **Secure and reliable** — fail closed, authorize tool calls, evidence a control. Chapter 9.
5. **Scaling and operating** — the path from laptop to something someone else can run. Chapter 8, Chapter 16.

You do not need to memorize syntax. You need to name the tradeoffs: latency, availability, consistency, reliability, maintainability, simplicity, cost. If you cannot name them, the agent will pick the ones that make the demo look finished.

## Skill 3 — Using coding agents

A coding agent is an LLM in a loop with tools: read, edit, run, search. Chapter 6 is *building* agents as product. This skill is *using* them to write software.

Ng’s workflow is three phases:

1. **Planning** — brainstorm, write a spec, make a plan.
2. **Execution** — build, test, verify. Choose autonomy vs oversight on purpose.
3. **Deployment and monitoring** — the same loop, in production.

The skills underneath: direct the workflow, enable autonomy without abandoning review, review the work, customize the agent and the environment (`AGENTS.md` / Cursor rules / `CLAUDE.md`), and understand the harness well enough to see its failure modes.

Chapter 15 is the practice. Chapter 17 is the artifact the agent executes against — constitution, feature spec, plan–implement–verify.

Long-horizon “hours of tokens” is overhyped. Iterative, high-skill intervention wins.

## Skill 4 — Shaping the build

Decide what belongs in the spec. That is product sense plus ownership: who is this for, what does “better” mean, what must not change, and whether this is an MVP or a system that has to survive an audit.

Chapter 14 is that skill. Chapter 17 is how the decision gets written down so an agent can execute it. Ng’s fifth letter (the detail on shaping) was not published when this chapter was written — the definition in Part 1 is enough to practice.

> The unit of a good AI feature is not a clever prompt. It is a specified outcome you can evaluate.

## How the other sources fold in

The map is Ng’s. Three other sources teach *how the work is done*. They are not a second curriculum.

- **Agentic AI** (Andrew Ng) — reflection, tool use, planning, multi-agent, evals-first. Chapters 6 and 7.
- **Generative AI for Software Development** (Laurence Moroney) — pair-program the SDLC; you keep the decisions. Chapters 16 and 15.
- **Spec-Driven Development with Coding Agents** (Paul Everitt) — vibe vs spec, constitution, plan–implement–verify, including on a legacy repo. Chapter 17.
- **Claude Certified Architect** (Anthropic) — orchestration, MCP, Claude Code workflows, structured output, production evals. Folded into chapters, labeled when it is Claude-specific. Not a cert track.

## How to use this course

Read in order when you can. The order is the map: Skill 1 core, then the SWE literacy that lets you steer, then coding agents, then production, then shaping.

Jump when you have a failure in hand:

- The demo works and production lies → Chapter 7, then Chapter 8.
- The agent writes a lot and nothing is done → Chapter 15, then Chapter 17.
- The model is “wrong” and you cannot say why → Chapter 5 or Chapter 7.
- You cannot brief the agent without waving → Chapter 16.

Facts that move — prices, context windows, current model IDs, current MCP docs — are **look-ups**, not memorization. Copy the prompt, run it against official docs, paste the answer into your notes. The principle stays. The SKU does not.

## How to pick a model tier

Do not pick “the best model.” Pick the cheapest tier that clears your eval. Escalate only when the eval says you must.

| Tier | Use when |
|---|---|
| **Cheap / fast** | Classification, routing, extraction, high volume. |
| **Mid** | Everyday chat, tools, most production traffic. |
| **Flagship** | Hard reasoning, messy documents, when mid-tier fails the eval. |
| **Reasoning** | Multi-step problems where extra test-time compute is worth the latency. |
| **Local** | Data cannot leave; cost at high volume; air-gapped. |

Capability, cost, latency, context. Run *your* eval on two or three candidates. Names and prices change. The tiering pattern does not.

> **Look up now.** Prompt: "What are the current list prices, context windows, and recommended IDs for the cheapest and most capable models from OpenAI, Anthropic, and Google? Cite official docs."

[INTERACTIVE: MODEL_COMPARISON]

At volume, the bill is a product decision. Tier the traffic. Cache repeated prefixes. Shorten prompts that do not move the eval. Batch work that is not live.

[INTERACTIVE: COST_OPTIMIZATION]

## The evaluation gap

You cannot write `assertEqual` on a paragraph. The same prompt can be right twice in different words and wrong once in confident ones. That is why Skill 1 leads with evals.

Teams that invest in evaluation infrastructure early ship products. Teams that skip it ship demos. Chapter 7 is the gate before any model or architecture change — including “let’s just switch providers.”

## Start here

You now have the map. Chapter 2 is LLM foundations: a mental model of tokenize, generate, and the knobs that actually change behavior. If you already have that, skip to the failure you have.

The field is young. The tools will change before you finish the course. The four skills will not.
