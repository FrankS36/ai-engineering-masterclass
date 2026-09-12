# LLM Foundations

> **From the field — Frank Sellhausen, Sellhausen AI Systems.** You do not need the training run. You need to know why the same prompt can be right twice and wrong once — and which knobs actually change that.

## What you will be able to do

Leave this chapter able to:

1. **Explain tokenize → generate** in one breath — what the model sees, what it does next, and why it can fail politely.
2. **Name the knobs that change behavior** — context, cache, cutoff, sampling, reasoning effort, tools — without memorizing this quarter’s IDs.
3. **Decide what belongs in the window** versus what the model should fetch.
4. **Know when to escalate** — a better prompt, a tool, a different tier, fine-tune, or self-host.

This is Skill 1 / LLM foundations on [Ng’s map](https://www.deeplearning.ai/the-batch/he-ai-engineering-skills-map-in-detail-building-and-deploying-ai-applications). Prompting is Chapter 3. APIs are Chapter 4. This chapter is the mental model underneath both.

## Tokenize, then generate

The model does not see words. It sees **tokens** — chunks of text (sometimes a word, sometimes a piece of one, sometimes a space plus a syllable). Your string is mapped to a list of IDs. Those IDs are the only input.

Then it **generates**: one token at a time, each conditioned on everything before it. That is the whole trick. There is no separate “understanding” pass and “answering” pass. The answer *is* the continuation.

Three consequences you can use on Monday:

- **Cost and limits are in tokens, not characters.** Code, JSON keys, and non-English text often spend more tokens for the same meaning. A “128k window” is smaller than it looks once you put a repo and a ticket in it.
- **Order is a feature.** The model attends across the window, but the start and the end of the prompt are the seats that get heard. Do not bury the instruction in the middle of a dump.
- **A wrong first token steers the rest.** Autoregression doubles down. If the model starts “Yes, we can refund that,” it will write a confident refund that does not exist.

[INTERACTIVE: TOKENIZER_DEMO]

> You can count on next-token prediction to continue a pattern. You cannot count on it to know when the pattern is false.

## When they fail

Language models fail **politely**. Traditional software throws, times out, or returns empty. An LLM completes the sentence.

The failure modes that matter for shipping:

| Failure | What it is | What you do |
|---|---|---|
| **Confident continuation** | A fluent answer to a question it cannot know | Ground it (Chapter 5) or give it a tool. Do not “ask it to be honest.” |
| **Cutoff** | Weights freeze. Last week’s policy is not in the model | Put the current fact in context, or fetch it. Look up the cutoff when it matters. |
| **Window overflow** | You stuffed more than fits; the middle drops out | Summarize, retrieve, or move history to memory. Do not paste the company. |
| **Cache miss** | You changed a prefix you thought was stable | Keep the static prefix byte-identical if you want the cache hit. |
| **Sampling noise** | Temperature made a classification wander | Lower temperature for extract / route / classify. Measure, don’t vibe. |
| **No hands** | It invents a price, a SKU, a status | Give it a tool. Generation is not a database. |

Ng’s point: once you see tokenize and generate, you can predict *when* to trust the model and when the architecture has to do the work.

## The knobs that change behavior

These survive a model generation. Names on the API change. The job of each knob does not.

| Knob | What it actually does | Default judgment |
|---|---|---|
| **Context** | What tokens the model can see *this call* | Only what moves the answer. The rest is noise and bill. |
| **Cache** | Reuse compute on a repeated prefix (system prompt, tools, policy) | Put the stable text first. Do not interpolate the date into the prefix. |
| **Cutoff** | The last day the weights saw the world | Current facts go in context or behind a tool. Never in the prompt as folklore. |
| **Sampling** | How wildly it picks the next token (temperature, top-p) | Near-zero for structured work. Higher only when variety is the product. |
| **Reasoning effort** | Extra test-time compute before it answers | Spend it when the eval says mid-tier fails. Latency is a product decision. |
| **Tools** | The model may call code, search, or your API instead of guessing | If a fact can be fetched, fetch it. Do not prompt it into existence. |

[INTERACTIVE: DECODING_STRATEGIES]

A **mix of models** is normal. Cheap/fast for route and extract. Mid for the everyday path. Flagship or reasoning when the eval fails. Chapter 1’s tier table still holds.

> **Look up now.** Prompt: "For OpenAI, Anthropic, and Google: what is the current context-window size, knowledge-cutoff language, and the name of the reasoning-effort or thinking control on the flagship and cheap tiers? Cite official docs only."

## Worked example — the ticket reply

You are shipping: *summarize this support ticket and draft a reply.*

**What the model sees.** Ticket body, the three sentences of policy that apply, maybe the last two messages. Not the help-center dump. Not last quarter’s pricing PDF. Tokens are scarce; the policy is the seat that must be heard.

**What you freeze.** System prompt + tool schemas + the policy paragraph. That prefix should be identical across thousands of tickets so the cache hits. The ticket text is the suffix that changes.

**What you do not ask it to know.** Order status, refund eligibility, the SKU that shipped. Those are tools: `get_order`, `get_policy(id)`. If the tool returns “ineligible,” the draft cannot invent a yes.

**How it should sample.** This is not a poem. Temperature near zero. You want the same ticket to get the same shape of reply.

**How it fails if you skip this.** The model writes a kind, specific refund. The customer screenshots it. Your agent of record is now a paragraph you cannot defend.

**When you escalate.** If mid-tier mangles messy threads, try flagship on *those* tickets only. If the same error class repeats after you have traces, that is Chapter 7 — not a fine-tune. Fine-tune or self-host when the eval is stuck *and* you have a constraint (data cannot leave, style must lock, cost at volume). Chapter 10 and Chapter 12.

That is the mental model applied: tokenize the right things, generate under a tight sample, fetch what you cannot know, measure before you change the model.

## Multimodal, fine-tune, self-host

Three escalations people reach for too early.

**Multimodal** when the evidence *is* pixels, audio, or a page image — a screenshot of the error, a scanned form, a diagram. If you can get the text out first and the text is enough, you do not need the bigger multimodal call. Chapter 11.

**Fine-tune** when prompting + tools + a better tier still fail a *stable* error class, and you have labeled examples of the behavior you want. It is not how you add last week’s facts. Facts go in context or a store. Chapter 10.

**Self-host** when data cannot leave, when the unit economics at volume beat the API, or when you need a model you can air-gap. You also take the ops. Chapter 12.

The common mistake is treating these as identity. They are responses to a measured constraint.

## Start here

You now have the picture Ng asked for: how the model tokenizes, how it generates, when to count on it, and the knobs that change the run.

Chapter 3 is how you write what goes in the window. Chapter 4 is how you call it. If a ticket-reply already failed in production, skip to Chapter 5 (grounding) or Chapter 7 (evals) with this chapter in your pocket.

The SKUs will move. Tokenize → generate will not.
