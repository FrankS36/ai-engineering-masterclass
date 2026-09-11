# Using Coding Agents

> **From the field.** A coding agent without a closer is just a faster way to drift. The skill is not prompting the tool. The skill is deciding what the agent is allowed to touch, what evidence counts as done, and when you take the wheel.

Andrew Ng put "using coding agents" on the same level as building AI applications. That is not hype. Agents are now how a large share of production code gets written. The developers who stay valuable are the ones who can steer them — not the ones who outsource judgment and hope the diff is fine.

This chapter is about *using* agents to build software. Chapter 6 is about *building* agents as product. You need both. Confusing them is how teams ship an internal Copilot and call the job done.

## A mental model, not a vendor tour

A coding agent is an LLM in a loop with tools: read files, edit files, run commands, search the web, sometimes control a browser. It maintains context. It plans or it doesn't. It stops when it thinks it is done, when it hits a limit, or when you interrupt.

That loop fails in predictable ways:

- **It optimizes for looking finished.** Tests pass because it weakened the test. The UI matches the screenshot because it hardcoded the fixture.
- **It loses the plot as context fills.** Mid-task it "simplifies" a constraint you stated at the start.
- **It does not know your production blast radius.** It will migrate a schema, rotate a key, or rewrite an auth check if the prompt was vague and the tools were wide.
- **It is confident in prose and sloppy in edges.** Empty states, permissions, retries, and the one environment variable nobody documented.

If you cannot name those failure modes, you are not using an agent. You are supervising a very fast intern with no memory of last quarter's incident.

## Context is the scarce resource

The agent can only work with what is in the window — plus what it can retrieve. Dumping the repository into chat is not context management. It is noise.

**Give it a constitution, not a novel.** A short project file (the `CLAUDE.md` / `AGENTS.md` / `.cursor/rules` pattern) that states: what this system is for, what must not change, how to run tests, and where the landmines are. Agents that read a one-page constitution produce fewer "helpful" refactors of the payment path.

**Point, don't narrate.** Name the files, the failing test, the screenshot, the API contract. "Fix the billing bug" is a wish. "Failing test in `invoice.test.ts`; do not change the public API of `InvoiceService`" is a job.

**Reset when the thread is dirty.** Long threads accumulate contradictory instructions. Compact or start over when you change the goal. Context rot looks like competence until you read the diff.

**Separate privileged from unprivileged.** The agent that can edit `src/` is not the agent that should hold production credentials. If your harness can reach the database, you have an access-control problem, not a productivity win.

## Plan versus execute

Agents that jump to editing waste tokens and produce local maxima. Agents that only plan produce slideware.

Use planning when:

- The change crosses more than one module
- You do not yet know the shape of the fix
- You need a spec another human will review

Skip the plan when:

- The failure is localized and you already have a failing test
- You are doing a mechanical rename, migrate, or apply-the-pattern task
- The plan would be longer than the change

A useful plan is a spec the agent can be graded against: files in scope, files out of scope, acceptance checks, and the rollback if the check fails. A useless plan is a list of platitudes ("ensure code quality").

> Spec-driven development is not ceremony. It is how you keep intent stable across sessions. Agents forget. Markdown does not.

## Close the loop with a verifier

Ng's phrase is exact: help the agent autonomously close loops by providing verifiers or evals.

Without a verifier, "done" means "the model wrote a paragraph saying it is done." With a verifier, done means:

- The test suite you named is green
- The typechecker is clean
- The screenshot matches
- The eval set did not regress
- The linter you actually run in CI passed

Write the verifier first when you can. A failing test is the best prompt you will ever write. If you cannot write a verifier, you cannot safely grant autonomy. Stay in the loop and review every edit.

This is the same muscle as evaluation-driven development in Chapter 7. Coding agents make it personal: your own repo is the system under test.

## How much to intervene

Under-steering looks like: you go to lunch, come back to a 40-file rewrite, and spend the afternoon reverting. Over-steering looks like: you type every line into the agent that you could have typed into the editor.

A working rhythm:

1. **State the outcome and the bound.** What good looks like. What is out of bounds.
2. **Give a verifier.**
3. **Watch the first loop.** If the agent misunderstands the goal, interrupt immediately. Do not let a wrong ontology compound.
4. **Leave it alone through mechanical work.** Once the direction is right, hovering costs more than it saves.
5. **Review the diff as a senior engineer, not as a spectator.** You own the merge. "The agent wrote it" is not a defense in an incident review.

Autonomy is a privilege you grant per task, not a setting you leave on.

## Specs, when they help — and when they do not

Write a spec when the work will outlive one session: a feature, a migration, a public API. Put it in the repo. Let the agent implement against it. Update the spec when reality disagrees — do not let the code and the spec drift in silence.

Do not write a spec for a one-line fix. Ceremony that does not reduce error is theater.

A project constitution is the spec you write once and reuse. Mission, stack, test command, "never do this." That document is how you stop the agent from introducing a second HTTP client or a new state library because it saw a blog post in its training data.

## Orchestrating more than one agent

Multiple agents are useful when the work is actually parallel: independent features, a reviewer that did not write the code, a researcher that only reads.

They are harmful when they share a working tree without isolation. Two agents editing the same module is a merge conflict factory. Use worktrees, branches, or a queue.

A second agent as reviewer is often worth more than a second agent as implementer. Different context, different incentive: find what the first one papered over.

Do not confuse "I ran five agents" with "I shaped the build." Parallelism is a tactic. Ownership is the job.

## Production is not a sandbox

The failure Ng names — an agent messing up your production database — is not hypothetical.

Minimum bounds:

- Read-only by default against prod
- No credentials in the prompt or the thread
- Migrations are human-gated
- Secrets stay in a manager the agent cannot exfiltrate into logs
- Destructive tools require an explicit, narrow allow

If you cannot explain the blast radius of the tools you attached, you have not finished the design. This is Territory 02 leaking into Territory 03 on purpose. Fundamentals are how you steer.

## Evolve the workflow on purpose

The tools will change. The skill that does not expire: a routine for trying the new thing without betting the system on it.

Keep a short personal log: what you delegated last week, where the agent wasted you, which verifier caught it. Promote the patterns that survive. Discard the ones that only work in a greenfield demo.

A workflow you cannot explain to a new teammate is not a workflow. It is a habit. Habits do not survive an incident.

## How this sits on the map

| Skill Ng named | What you practice here | Where else it lives |
|---|---|---|
| Mental model of agents | Loops, tools, context rot, false done | Chapter 6 |
| Context management | Constitutions, pointers, resets | Chapter 3, Chapter 5 |
| Plan vs execute | Specs when they earn their keep | Chapter 14 |
| Verifiers | Tests and evals as the definition of done | Chapter 7 |
| Intervention | Autonomy as a per-task privilege | Chapter 8, Chapter 9 |
| Multi-agent | Isolation and a reviewer that did not write the code | Chapter 6 |
| Production safety | Blast radius, gates, secrets | Chapter 8, Chapter 9 |

Using coding agents well does not replace software engineering. It *is* software engineering, performed through a stochastic intern you are responsible for.

## Summary

The market now expects you to build with agents. You still own the merge.

Give the agent a constitution and a verifier. Bound what it can touch. Interrupt when the goal is wrong. Leave it alone when the work is mechanical. Review the diff as if your name is on the commit — because it is.

That is the skill. The product names will change. The loop will not.
