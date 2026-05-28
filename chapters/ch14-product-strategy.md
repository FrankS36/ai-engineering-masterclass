# Chapter 14: AI Product Strategy

> This course exists because AI made building easy, but building the right thing stayed hard. The technical material here is solid. Where I weigh in, it's from watching teams ship AI into real workflows — sometimes brilliantly, sometimes into a wall. Take the engineering skills seriously. But never forget: AI will faithfully scale the wrong thing just as eagerly as the right one. Intentionality is the whole job now.

You have spent thirteen chapters learning how to build AI systems. You know how to call models, manage tokens, build RAG pipelines, fine-tune, deploy, and evaluate. This chapter is about a different skill entirely: deciding what to build, when to build it, and how to make it survive contact with an organization.

Technology is the easy part. The hard part is everything around it — the use case selection, the pricing, the politics, the legal exposure, the gap between a working prototype and a production system that an organization can actually operate. This chapter covers the decisions that determine whether your technical skills create value or create an expensive lesson.

## Identifying Good AI Use Cases

Not every problem benefits from AI. Not every problem that benefits from AI benefits enough to justify the cost, complexity, and ongoing maintenance. The best AI use cases share specific characteristics, and the worst ones share different ones.

**Characteristics of strong AI use cases:**

- **Tolerance for imperfection.** The task allows occasional errors without catastrophic consequences. Document summarization, content drafting, product recommendations, internal search. An 85% accuracy rate means the system is useful 85% of the time and mildly annoying 15% of the time — which is often better than the status quo of doing it manually 100% of the time.
- **High volume, low individual stakes.** Processing thousands of support tickets where a misroute adds ten minutes to resolution time is a good use case. Processing one regulatory filing where an error triggers a multi-million dollar fine is not.
- **Augmentation over automation.** The AI helps a human work faster, rather than replacing the human entirely. A human in the loop catches errors and provides feedback data.
- **Existing manual process that is slow and inconsistent.** If humans already do this task with variable quality, AI does not need to be perfect — it needs to be better than the current average and faster.

**Red flags that a use case is wrong for AI:**

- **Zero error tolerance.** If any mistake is unacceptable, you either cannot use AI or you need a human review step so thorough that it negates the efficiency gain.
- **Deterministic requirements.** If the same input must always produce the exact same output, you want a rules engine, not a model.
- **Simple rules suffice.** If the logic can be expressed as a decision tree with under fifty nodes, AI adds complexity without adding capability. Write an if/else chain.
- **No feedback path.** If you cannot measure whether the AI is right or wrong, you cannot improve it. You are flying blind.

> I watched a team build an AI recommendation engine while one person on the floor had a simple Excel file that the salesmakers actually used — because he'd sat with them and built around how they already worked. The sophisticated tool was ignored. Quality isn't sophistication. The simplest thing that respects how the user actually runs their business beats the impressive thing that doesn't.

## Build vs. Buy vs. API

Every AI feature presents this decision. The right answer depends on five factors: cost at scale, control over behavior, speed to market, data privacy requirements, and how differentiated the capability needs to be.

**API (OpenAI, Anthropic, Google, etc.):** Fastest time to market. No infrastructure to manage. You are paying per request and depending on an external provider for uptime, latency, and model behavior. Model updates can change your product's behavior without warning. Best for: prototyping, features where the AI is a commodity component, teams without ML infrastructure.

**Buy (vendor platform):** Someone else has built the application layer. You configure, integrate, and deploy. Faster than building but less flexible. You inherit their UX decisions and their limitations. Best for: well-defined use cases where established vendors exist (customer support chatbots, document processing, fraud detection).

**Build (train or fine-tune your own):** Maximum control. Maximum cost. Maximum time to production. You need ML engineers, training infrastructure, evaluation pipelines, and ongoing model maintenance. Best for: core product differentiators where you have proprietary data and the AI behavior needs to be exactly right.

A practical decision framework:

| Factor | API | Buy | Build |
|---|---|---|---|
| Time to production | Days to weeks | Weeks to months | Months to quarters |
| Per-unit cost at scale | Highest | Medium | Lowest (but high fixed cost) |
| Control over behavior | Low | Medium | High |
| Data privacy | Data leaves your systems | Depends on vendor | Data stays internal |
| Maintenance burden | Lowest | Medium | Highest |
| Differentiation potential | Low | Low | High |

Most teams should start with APIs, move to buy when they hit the limits of APIs, and build only when they have proven the use case and need control or cost optimization that buy cannot provide.

> The vendor demo is not the product. The demo is a performance on curated data with expert support. The actual product is what shows up after the signatures dry, the integrations start, and the solutions engineer moves on. Ask: can we run this on an extract from your production data, right now, in this room?

## Pricing AI Features

AI features have a cost structure that traditional software does not: every inference costs money. This changes pricing fundamentals.

**The per-request cost formula:**

```
Cost per request = (input_tokens * input_price_per_token)
                 + (output_tokens * output_price_per_token)
                 + infrastructure_overhead
                 + retrieval_costs (if RAG)
```

For a typical RAG-based customer support query using a frontier model: input tokens (system prompt + retrieved context + user query) around 3,000 tokens, output around 500 tokens. At current frontier model pricing, that is roughly $0.02-0.05 per query. At 100,000 queries per month, you are spending $2,000-5,000 on inference alone — before infrastructure, storage, or engineering time.

**Pricing models:**

- **Per-use pricing.** Charge per query, per document processed, per generation. Aligns your revenue with your costs. Risk: usage anxiety. Users ration their requests, which means they use the feature less, get less value, and churn.
- **Subscription with tiers.** Fixed monthly price with usage limits per tier. Predictable for the user, but you absorb cost variance. Structure tiers around natural usage breakpoints, not arbitrary numbers.
- **Hybrid.** Base subscription with overage charges. The base tier is generous enough that most users never hit it. Power users pay more, which is fair because they cost more.
- **Value-based pricing.** Price based on the outcome, not the usage. If your AI saves a customer 20 hours of work per month, charge a fraction of those hours' cost. Hardest to implement, highest margins.

Usage anxiety is real and it kills adoption. If a user hesitates before every query because they are watching a meter tick, they will stop using the feature. Either price generously enough that anxiety disappears or make the per-use cost so transparently low that users do not care.

> The business case for AI is almost always wrong — not because people lie, but because the standard format was designed for deterministic investments. "The data will be ready, the team will stay dedicated, integration will be straightforward, users will adopt it." Every assumption breaks. An honest business case includes kill criteria.

## User Research for AI Products

User research for AI products cannot follow the same script as traditional software research. The standard approach — show a mockup, ask if they would use it — fails because people cannot predict how they will react to probabilistic outputs.

**Test with real workflows, not demos.** Give users the actual tool with their actual data and their actual tasks. Watch what happens when the AI gives a wrong answer. Do they notice? Do they know how to correct it? Do they trust the system less afterward, or do they shrug and fix it? These reactions tell you more than any survey.

**Observe the correction path.** Time how long it takes a user to fix an AI error versus doing the task from scratch. If correcting errors takes longer than manual work, the AI is making things worse regardless of its accuracy rate.

**Test the 80th percentile case, not the median.** Your model might be great on average. But users remember the bad experiences. Deliberately include edge cases in your research sessions — unusual inputs, ambiguous requests, adversarial phrasing. Watch how the user and the interface handle them.

**Measure trust calibration.** After using the system for a while, do users trust it the right amount? Over-trust is dangerous (they stop checking). Under-trust is wasteful (they check everything). The ideal is appropriate trust — high confidence in the system's strengths, healthy skepticism in its weak areas.

## Managing Expectations

The accuracy conversation is the most important conversation you will have with stakeholders, and most teams handle it badly. They either overpromise ("95% accuracy!") or hedge so aggressively that nobody funds the project.

**Set thresholds tied to business outcomes, not abstract metrics.** "95% accuracy" means nothing without context. 95% accuracy on a classification task where the 5% errors are evenly distributed across low-impact categories is fine. 95% accuracy where the 5% errors are concentrated on your highest-value customers is a disaster. Define what accuracy means for your specific use case, which error types matter most, and what the acceptable rate is for each.

**Communicate limitations honestly and specifically.** Not "AI can make mistakes" (everyone knows this, it conveys nothing). Instead: "This system works well for standard customer inquiries in English. It struggles with highly technical questions, code-switching between languages, and sarcasm. For those cases, it routes to a human agent." Specific limitations build more trust than vague disclaimers.

**Build in measurement from day one.** If you cannot measure accuracy in production, you cannot manage it. Log inputs, outputs, and user actions. Sample and review regularly. Set up alerts for quality degradation. The model that was 90% accurate at launch can degrade to 70% as the world changes around it, and you will not notice until users start complaining — or leaving.

## Handling Failures Gracefully

Failures are not edge cases in AI systems. They are a design parameter. Plan for them the way you plan for traffic spikes — not as exceptions, but as expected operating conditions.

**Error budgets.** Borrow the concept from site reliability engineering. Define an acceptable error rate (say, 5% of responses are unhelpful). As long as you are within budget, the system is healthy. When you exceed it, halt new feature work and focus on quality. This prevents the common pattern of ignoring gradual quality degradation until it becomes a crisis.

**Escalation paths.** Every AI interaction should have a path to a human. Not because users will always need it, but because knowing it exists changes how they interact with the system. Users are more willing to try the AI path when they know they can escalate. The escalation path should preserve context — the user should never have to repeat themselves.

**"I don't know" as a feature.** A model that says "I'm not confident enough to answer this" is more trustworthy than one that fabricates an answer. Build abstention into your system. Set confidence thresholds below which the model defers rather than responds. This is harder than it sounds — you need to balance abstention rate against user frustration — but it is critical for high-stakes applications.

## The Organizational Challenge

AI products fail more often from organizational dysfunction than from technical failure. The technology works. The organization cannot operate it.

**Middle management incentive misalignment.** The executives who fund AI initiatives measure success in strategic impact. The middle managers who execute them measure success in hitting quarterly targets with minimal disruption. AI projects are disruptive by nature. This creates a pattern where AI is funded at the top, resisted in the middle, and confused at the bottom.

**The handoff problem.** AI projects typically start in a data science team or an innovation lab. At some point, they need to transfer to a product team or an operations team that will run them in production. This handoff fails more often than it succeeds. The receiving team did not build it, does not understand it, and was not consulted about whether they could support it. Build the production team into the project from the start, not the end.

**Who owns AI after launch.** A deployed AI model is not like deployed software. Software can run unattended for months. Models degrade. Data distributions shift. User behavior changes. Someone needs to monitor quality, retrain when necessary, and make the judgment call about when the model needs intervention. That someone needs a name, a budget, and authority — not a vague mention in someone's job description.

> Nobody gets fired for buying AI. Someone eventually answers for why the investment didn't deliver. The answer is almost always: nobody owned what happened after the purchase.

## Legal and Compliance

The legal landscape for AI is evolving rapidly, but several constraints are already clear enough to design around.

**Copyright.** AI-generated content exists in a legal gray area. In the US, purely AI-generated works cannot be copyrighted (per current Copyright Office guidance). Content with meaningful human involvement in the creative process likely can be. For products that generate content for users, this matters. Be transparent about what is AI-generated and ensure your users understand the intellectual property implications.

**Data privacy.** If your AI processes personal data — and most useful AI does — GDPR and CCPA apply. Key obligations: users must be able to request deletion of their data, including data used for training. Automated decisions that significantly affect individuals (credit scoring, hiring screening) require human oversight under GDPR. Data sent to third-party model providers (OpenAI, etc.) may constitute a data transfer that requires contractual safeguards. Review your model provider's data processing terms carefully. Do not assume that "they probably don't train on our data" — verify it contractually.

**Regulated industries.** Healthcare (HIPAA), financial services (SOC 2, various banking regulations), and legal services have specific requirements around data handling, auditability, and human oversight. AI does not get an exemption. If anything, regulators scrutinize AI decisions more closely. In these industries, every AI output that influences a decision about a person needs an audit trail — what data went in, what model produced the output, what version it was, and who reviewed it.

**Liability for AI outputs.** If your AI gives advice and someone acts on it and is harmed, who is liable? This is largely untested in courts, but the safe assumption is that the company deploying the AI bears responsibility. Design accordingly: include appropriate disclaimers, maintain human oversight for high-stakes outputs, and carry adequate insurance.

## Defensibility and Moats

AI capabilities commoditize rapidly. The model you fine-tuned last quarter is outperformed by this quarter's base model. Your RAG pipeline uses the same open-source tools as everyone else's. Where, then, is the competitive advantage?

**Data moats.** Proprietary data that no competitor has is the strongest moat in AI. Every user interaction, every correction, every piece of domain-specific feedback becomes training data that improves your system. The more users you have, the more data you collect, the better your system gets, the more users you attract. This flywheel is real, but it takes time to spin up, and the data must be genuinely unique.

**Feedback loops.** A system that improves from usage builds a compounding advantage. But this requires infrastructure: logging, evaluation, retraining pipelines, and the ability to deploy improvements quickly. The feedback loop only works if you close it.

**Integration depth.** The deeper your product integrates into a customer's workflow, the harder it is to replace. An AI that reads their data, connects to their systems, learns their terminology, and fits into their existing processes creates switching costs that have nothing to do with the model itself.

**Workflow embedding.** The most defensible AI products do not feel like AI products. They feel like the normal way work gets done. When the AI is so embedded in the workflow that removing it would require changing how people work, you have a moat. This is why "AI as a feature" inside an existing workflow tool often beats "AI as a product" that requires people to change their behavior.

## The Pilot-to-Production Gap

> Your AI pilot worked. That's the dangerous part. The pilot proved the model works — it proved nothing about whether the organization can operate it. A curated dataset, a dedicated team, and executive attention created conditions that scaling will destroy.

The pilot-to-production gap is where most AI projects die. Understanding why requires understanding what a pilot actually proves — and what it does not.

**What the pilot proved:** The model can perform the task on the pilot dataset. The users in the pilot group found it useful. The infrastructure can handle pilot-scale traffic.

**What the pilot did not prove:** The model performs well on the full distribution of production data. The team can monitor and maintain the system without dedicated pilot resources. The edge cases that did not appear in pilot volume will not cause problems at scale. The integration with production systems works reliably. Users outside the pilot group (who did not volunteer and are not being watched) will adopt it.

**What to measure differently in production:**

- **Latency at the 99th percentile**, not the median. Pilot volumes hide tail latency problems.
- **Accuracy on data segments**, not overall accuracy. A model that is 90% accurate overall but 40% accurate on a specific customer segment is failing that segment.
- **Operational cost per query** including all infrastructure, not just inference cost. Retrieval, logging, monitoring, human review — it all adds up.
- **Time to detect and recover from failures.** In the pilot, someone was probably watching closely. In production, how long until a quality drop is noticed?
- **User behavior without executive sponsorship.** Pilot users often have a manager encouraging them to use the tool. Production users do not. Measure adoption without that pressure.

The gap is closable, but only if you plan for it from the beginning. Staff the production operations team during the pilot. Run the pilot on production infrastructure, not a separate environment. Include hostile data in the pilot dataset. Remove the special attention gradually and see what happens before declaring success.

> I've watched the meeting where AI projects actually die. Someone says "let's work with what we have and iterate," everyone nods, and the project is dead — it just doesn't know it yet. The right answer is moving the date, reducing scope, or funding the gap. Not proceeding as planned with caveats nobody will track.

## Putting It All Together

The technical skills from the preceding chapters are necessary. They are not sufficient. Building an AI system that works in a demo takes days. Building one that works in production takes months. Building one that delivers sustained business value takes a team that understands the technology, the users, the organization, and the constraints — and makes deliberate decisions about all four.

The pattern across every section of this chapter is the same: the technical question is easier than the surrounding questions. Choosing the right model is easier than choosing the right use case. Training the model is easier than training the organization. Achieving accuracy is easier than maintaining accuracy. Launching is easier than operating.

Every decision in this chapter comes back to a single discipline: being honest about what you know, what you do not know, and what you are assuming. The AI will not do that for you. That is your job.
