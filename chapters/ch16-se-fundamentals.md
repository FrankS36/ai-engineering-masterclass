# Software Engineering Fundamentals

> Coding agents changed how we type. They did not change what software has to do: stay up, stay correct, stay cheap enough, and stay changeable. If you cannot name the tradeoffs, the agent will pick them for you — usually the ones that make the demo look finished.

Andrew Ng’s skills map puts software engineering fundamentals next to building AI applications for a reason. The AI core almost always lives inside a broader application. Laurence Moroney’s *Generative AI for Software Development* course makes the same point from the other direction: use an LLM as a pair-programmer through the whole SDLC, but you still choose the data model, the tests, the dependencies, and the design.

This chapter is that substrate. Chapter 15 is how you steer the agent. Chapter 17 is how you write the spec. Here is what you must already understand so those two chapters have something to steer toward.

## The vibe-coding failure mode

A novice can vibe-code a simple app. The agent will make bad tradeoffs in latency, availability, consistency, reliability, maintainability, simplicity, and cost — because nobody told it those axes existed.

You do not need to memorize syntax. You do need to know:

- What the full stack actually is
- How data is stored and accessed
- How the pieces are assembled
- How the system fails and how it is secured
- How it gets to production and stays there

Those five are Ng’s list. The rest of this chapter is that list, written so you can brief an agent in the language of engineering instead of the language of wishes.

## Full-stack literacy

Agentic coding lets a specialist act more like a full-stack engineer. That only works if you understand the parts the agent is touching on your behalf.

**Front end:** UI components, page rendering, caching, accessibility, what is computed on the server vs the client. If you cannot say where state lives, the agent will put it in three places.

**Back end:** API design, authentication, sessions, async work, background jobs. If you cannot say what must be synchronous, the agent will hide a 12-second model call behind a button and call it shipped.

**Testing:** Unit vs integration, what to mock, what coverage actually means. Moroney treats the LLM as a tester: ask it for edge cases, then keep the tests *you* would bet an incident on.

**The boundary:** What the browser may see, what the server must enforce, what never leaves the vault. Agents love to put secrets in the client because the happy path is shorter that way.

You do not have to be the world’s best at every layer. You have to be literate enough to reject a bad proposal in each one.

## Data

Data is hard to change even when an agent writes the migration. Choose it like you will live with it.

**Access patterns first.** What do you look up, by what key, how often, how fresh? That decides relational vs document vs key-value vs graph — not the blog post the model trained on.

**Transactions and concurrency.** If two users can book the last seat, you need a rule, not a vibe.

**Lifecycle.** What is collected, how long it lives, who can see it, how it is deleted. Privacy and compliance are data-architecture decisions.

**Freshness.** Stale context makes AI systems confidently wrong. If the agent’s input comes from your store, a bad schema is a bad prompt you cannot see.

**Evolve with the product.** The prototype store is not the production store. Say that out loud before the agent “helpfully” hard-codes SQLite assumptions into twelve services.

Moroney’s course walks this with real schemas, CRUD, and an ORM. The skill to keep: you can sit with an LLM and design a schema *because* you can explain the access pattern, not because the model suggested a fashionable database.

## Architecture

Architecture is the set of tradeoffs you chose on purpose.

Ask: how many users, how important is latency, how important is cost, what happens if this is down for an hour? Then choose:

- Platform and runtime
- Front-end / back-end boundary
- Where state lives
- Monolith vs services (granularity is a decision, not a virtue)
- The stack — sometimes by running a short experiment, not by copying last year’s tweet

The right architecture is a moving target. Prototype ≠ first production ≠ scale. Ng’s point: the simple architecture that gets you a yes from users is often the wrong architecture to ossify. Design so you can evolve it.

When you brief an agent, name the phase. “This is a spike; optimize for deletion” produces different code than “this is the path that bills customers.”

## Secure and reliable

Reliability is a testing strategy plus a failure plan.

**Tests:** What mix of unit and integration. What must stay deterministic. What is an eval (Chapter 7) vs a unit test. Agents will weaken tests to go green. Your job is to notice.

**Failures:** Rate limits, timeouts, poison messages, partial writes. Design for graceful degradation and a small blast radius.

**Shift left on security.** Do not write the app and then “add security.” Authn, authz, secrets, supply chain, and cloud config are design inputs. AI tools can scan; they cannot decide your threat model.

If you cannot explain the blast radius of a change, you are not ready to let an agent merge it.

## Scale and operate

Shipping is the rest of the SDLC: environment, release strategy, CI/CD, and enough infrastructure knowledge to not treat “the cloud” as a single button.

In production you need observability, alerts, and an incident habit. To scale you need a real picture of load — then servers, load-balancing, indexing, replication, or an architecture change. Version control, code review, dependency hygiene, and technical debt are how the system stays evolvable.

Moroney’s team-engineering modules (testing, documentation, dependencies) are this chapter in pair-programming form. Use the LLM to draft the runbook. You still own the on-call.

## Briefing the agent in engineering language

This is the payoff. Fundamentals exist so you can say things like:

- “Postgres, access by `org_id` + `created_at`, no cross-tenant reads.”
- “Synchronous for the checkout confirm; queue the receipt email.”
- “Unit-test the service layer; do not mock away the auth check.”
- “If the model is down, show last-known draft, do not fail the save.”

Those sentences are context. Without them, the agent optimizes for looking done.

## How this sits in the course

| Skill | Practice here | Next |
|---|---|---|
| Full-stack literacy | Name every layer the agent will touch | Chapter 15 |
| Data | Access patterns, lifecycle, freshness | Chapter 5, Chapter 8 |
| Architecture | Phase-appropriate tradeoffs | System Design resource |
| Secure & reliable | Tests, blast radius, shift-left | Chapter 9 |
| Operate | SDLC, CI, observability | Chapter 8 |
| Pair-program the SDLC | LLM as tester, docs, deps — you keep the decisions | Chapter 15, Chapter 17 |

Syntax is cheap now. Taste in tradeoffs is not.

## Summary

Software fundamentals are how you steer. Learn the stack, the data, the architecture, the failure modes, and the path to production well enough to reject a bad default. Then give the agent that language. Then write a spec so the language survives the next session.
