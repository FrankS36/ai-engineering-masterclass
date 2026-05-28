# Chapter 2: How Foundation Models Work

You do not need to understand every detail of backpropagation to build production systems with foundation models. But you do need a working mental model of what is happening inside these systems — what they are good at, where they break, and why the API parameters you set actually matter. This chapter gives you that mental model.

---

## What Makes a Foundation Model

The term "foundation model" was coined by Stanford's Center for Research on Foundation Models in 2021 to describe a specific pattern: a single model, trained on broad data at scale, that can be adapted to a wide range of downstream tasks.

Three properties define them:

**Scale.** Foundation models are trained on hundreds of billions to trillions of tokens of text (and increasingly, images, audio, and video). Training runs cost tens to hundreds of millions of dollars in compute. GPT-4's training cost is estimated at over $100M. Meta's Llama 3 405B used 15.6 trillion tokens across 30.8 million GPU-hours.

**Generality.** Unlike task-specific models (a spam classifier, a named-entity recognizer), foundation models develop broad capabilities during pre-training. A single model can summarize, translate, write code, reason about math, and answer questions — without being explicitly trained on labeled examples for each task.

**Adaptability.** Through prompting, fine-tuning, or retrieval-augmented generation, you can steer a foundation model toward your specific use case without retraining from scratch. This is the property that makes them useful to engineers: you get a capable base and specialize from there.

---

## Scaling Laws and Emergent Capabilities

In 2020, Kaplan et al. at OpenAI showed that model performance (measured as loss on next-token prediction) follows predictable power laws with respect to three variables: the number of parameters, the amount of training data, and the compute budget. Bigger models trained on more data with more compute get predictably better.

DeepMind's Chinchilla paper (2022) refined this, finding that for compute-optimal training, models should be trained on roughly **20 tokens per parameter**. A 70B parameter model should see ~1.4 trillion tokens.

> **Important caveat:** The Chinchilla ratio is compute-optimal for a *fixed training budget*. In practice, modern models are often trained on far more data per parameter than Chinchilla suggests — Llama 3 8B was trained on 15 trillion tokens, nearly 1,900 tokens per parameter. Why? Because inference cost dominates in production. A smaller model trained longer is cheaper to serve than a larger model trained to Chinchilla-optimal. The "right" ratio depends on your deployment economics, not just training efficiency.

**Emergent capabilities** — abilities that appear suddenly as models scale — have been a subject of both excitement and debate. Chain-of-thought reasoning, for instance, works poorly in small models but becomes effective around the 60B+ parameter range. However, Schaeffer et al. (2023) argued that many "emergent" abilities are artifacts of the metrics used: switch from nonlinear metrics (exact match) to linear ones (token-level accuracy), and the improvement looks smooth, not sudden.

For engineers, the practical takeaway: do not assume a smaller model simply cannot do something because a paper showed emergence at scale. Test it. But also do not assume capabilities transfer uniformly — some tasks genuinely require larger models.

**Cost trends.** The cost of equivalent intelligence has dropped dramatically. What cost $100 in API fees for a given task in 2021 might cost $3-10 today — roughly a 10-30x reduction, depending on the task and provider. This is driven by smaller and more efficient models, quantization, better inference infrastructure, and competition. The trend is real and continuing, but claims of 100x cost collapse overstate the case for most workloads.

---

## The Transformer Architecture: Why It Matters for Engineers

Nearly every foundation model today is built on the Transformer architecture (Vaswani et al., 2017). You need to understand two things about it: **self-attention** and **parallelizability**.

### Self-Attention as Relevance Scoring

At each layer, every token in the sequence computes attention scores against every other token. Think of it as: for each word, the model asks "how relevant is every other word to predicting what comes next here?" These scores are computed from learned Query, Key, and Value projections:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

This is why transformers handle long-range dependencies well. A pronoun at position 500 can attend directly to the noun it refers to at position 12 — no information needs to pass through a chain of recurrent steps.

For engineers, the consequence is: **token order matters, but so does token distance within the context window.** Models can in principle attend to anything in context, but in practice, attention to very distant tokens can degrade, which is why placement of key information in your prompts matters.

### Parallelization

Unlike RNNs, which process tokens sequentially, transformers process all tokens in a sequence simultaneously during training. This is what made modern scale possible — training can be distributed across thousands of GPUs efficiently. Without this property, training a trillion-token dataset would be computationally infeasible.

### Decoder-Only vs. Encoder-Decoder

Most current LLMs (GPT-4, Claude, Llama, Gemini) use a **decoder-only** architecture: they process tokens left-to-right and generate one token at a time. Encoder-decoder models (like T5 or the original BART) use a bidirectional encoder to process the input and a decoder to generate output. Decoder-only models won out in practice because they are simpler to scale and the single architecture handles both understanding and generation.

---

## The Training Pipeline

Building a foundation model is a multi-stage process. Each stage has different goals, costs, and data requirements.

### Stage 1: Pre-training

The model learns to predict the next token on a massive corpus. This is where the bulk of compute goes.

| Aspect | Typical Range |
|---|---|
| Data | 1-15+ trillion tokens |
| Cost | $2M (7B model) to $100M+ (frontier models) |
| Duration | Weeks to months on thousands of GPUs |
| Objective | Next-token prediction (causal language modeling) |

Pre-training produces a base model that is good at text completion but not at following instructions.

### Stage 2: Supervised Fine-Tuning (SFT)

Human annotators write high-quality (prompt, response) pairs. The model is trained on these to learn the format and style of helpful responses. Typical SFT datasets range from tens of thousands to low millions of examples. This stage is comparatively cheap — often under $1M.

### Stage 3: Preference Tuning (RLHF / DPO / Constitutional AI)

The model is further refined using human preferences. In RLHF (Reinforcement Learning from Human Feedback), annotators rank model outputs, a reward model is trained on these rankings, and the LLM is optimized against the reward model using PPO or similar algorithms.

DPO (Direct Preference Optimization) simplifies this by skipping the reward model and optimizing preferences directly.

> **On Constitutional AI:** Constitutional AI (Anthropic, 2022) is sometimes described as a replacement for RLHF. It is not. Constitutional AI provides a framework where the model critiques its own outputs against a set of principles (a "constitution"), generating preference data that supplements human annotations. Current Claude models use a combination of SFT, Constitutional AI, and RLHF — these techniques are complementary, not competing.

### Stage 4: Reasoning Training

A more recent addition to the pipeline, models like OpenAI's o1/o3 and Anthropic's Claude undergo additional training specifically to improve step-by-step reasoning. This typically involves reinforcement learning on reasoning tasks (math, code, logic), where the model is rewarded for producing correct chains of thought. This is an active and rapidly evolving area.

---

## Training Data: Sources, Biases, and Contamination

Foundation models are shaped by their training data. Understanding what goes in helps you predict what comes out.

**Common sources:** Common Crawl (web text), Wikipedia, books, academic papers (Semantic Scholar, arXiv), code repositories (GitHub, Stack Overflow), and curated datasets. Increasingly, providers also use synthetic data — model-generated text that is filtered for quality.

**Language bias:** The web is dominated by English. Common Crawl is roughly 45-50% English. Models trained on web-scale data inevitably perform better on English than on low-resource languages. If you are building for non-English users, test carefully — and consider that some providers train with more balanced multilingual corpora than others.

**Domain models:** Models like Bloomberg's BloombergGPT (trained on financial data), Google's Med-PaLM (medical), and Code Llama (code) demonstrate that mixing domain-specific data into pre-training or fine-tuning can significantly improve performance in specialized areas.

**Data contamination:** If your benchmark or evaluation dataset appeared in the training corpus, your metrics are inflated. This is a real and widespread problem. Always assume some contamination exists and supplement benchmarks with held-out, custom evaluations for your specific use case.

---

## Tokenization: Why It Matters More Than You Think

Models do not see text as characters or words. They see **tokens** — subword units produced by algorithms like Byte-Pair Encoding (BPE). Understanding tokenization matters for three practical reasons:

**Cost.** API pricing is per-token. The string "unhappiness" might be 1-3 tokens depending on the tokenizer. Code and structured data are often more token-dense than prose. JSON keys like `"customer_id"` consume tokens that carry little semantic value — this is one reason function calling and structured output formats can be more token-efficient than asking the model to produce raw JSON.

**Context limits.** Your context window is measured in tokens, not characters or words. A rough English approximation is ~0.75 words per token (or ~4 characters per token), but this varies by content type. Code often tokenizes less efficiently than natural English.

**Non-English languages.** BPE tokenizers trained on English-heavy corpora produce more tokens for the same meaning in other languages. A sentence in Japanese or Arabic can easily require 2-4x as many tokens as its English equivalent. This means non-English users effectively get a smaller context window and pay more per query.

```python
# Example: comparing token counts across languages (using tiktoken for GPT-4)
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4")

english = "The weather is nice today."      # 6 tokens
japanese = "今日はいい天気ですね。"              # 11 tokens
arabic = "الطقس جميل اليوم."                 # 10 tokens
```

---

## Decoding and Sampling Strategies

When the model generates output, it produces a probability distribution over the vocabulary for the next token. How you sample from that distribution controls the output's creativity, coherence, and determinism.

| Parameter | What It Does | When to Use |
|---|---|---|
| **Temperature** | Scales logits before softmax. Lower = more deterministic, higher = more random. | Set to 0-0.2 for factual/code tasks. 0.7-1.0 for creative writing. |
| **Top-p (nucleus)** | Samples from the smallest set of tokens whose cumulative probability exceeds p. | Default 0.9-0.95 is a solid starting point. Reduce for more focused output. |
| **Top-k** | Samples from the k most probable tokens only. | Less commonly used in production. k=40-100 is typical when used. |
| **Min-p** | Filters out tokens below a minimum probability threshold relative to the top token. | Newer alternative to top-k. 0.05-0.1 works well. More adaptive than top-k. |

**Practical guidance:** For most production applications, set temperature between 0 and 0.3 and leave top-p at ~0.95. For deterministic outputs (structured data extraction, classification), use temperature 0. Avoid stacking too many sampling parameters — temperature + one of top-p/min-p is usually sufficient.

---

## Context Windows: What Is Possible and What It Costs

Context windows have grown dramatically: from GPT-3's 2K tokens (2020) to Gemini 1.5 Pro's 1M tokens and Claude's 200K tokens (2024-2025).

Longer context windows enable new architectures — you can fit entire codebases, long documents, or extended conversation histories into a single prompt. But there are engineering tradeoffs:

**Attention is quadratic.** Standard self-attention scales as O(n^2) with sequence length. A 1M token context requires computing attention scores between every pair of tokens. Optimizations like FlashAttention, ring attention, and sparse attention patterns mitigate this, but long contexts still cost more in latency and compute.

**The "lost in the middle" problem.** Research from Liu et al. (2023) showed that models retrieve information less reliably from the middle of long contexts than from the beginning or end. This has improved with newer models, but it is still worth placing critical information at the start or end of your prompts.

**KV cache memory.** During generation, the model caches Key and Value tensors for all previous tokens. For large models, this gets expensive. A 70B parameter model at bf16 precision caching 1M tokens of context requires roughly 500GB of KV cache memory. For a 7B model, the figure is closer to 50GB. The range matters — do not quote a single number without specifying the model size.

---

## Mixture of Experts (MoE)

Mixture of Experts is an architecture that allows models to have a very large total parameter count while only activating a fraction of those parameters for each token.

**How it works:** Instead of one large feed-forward network (FFN) per transformer layer, you have N "expert" FFNs and a routing network that selects the top-k experts for each token. Mixtral 8x7B, for example, has 8 experts per layer but routes each token through only 2, giving it 46.7B total parameters but only ~13B active per token.

**Why it matters:**
- **Inference efficiency.** Active parameter count determines inference cost. An MoE model can match a dense model's quality at a fraction of the per-token compute.
- **Training efficiency.** More total parameters mean more model capacity for learning, without proportional increases in training compute.

GPT-4 is widely reported to be an MoE model (rumored ~1.8T total parameters, ~280B active). DeepSeek-V2 and DBRX also use MoE architectures.

**The tradeoff:** MoE models require more memory (all experts must be loaded even if only a few are active per token) and can be harder to fine-tune effectively, since not all experts see all training examples.

---

## Multimodal Models

Foundation models increasingly handle more than text. Understanding the two main approaches helps you choose the right model for your use case.

**Native multimodal models** are trained from the ground up on multiple modalities. Google's Gemini processes text, images, audio, and video through a single model with shared representations. This tends to produce better cross-modal understanding — the model can reason about the relationship between an image and text, not just describe each independently.

**Composite / pipeline models** bolt vision encoders or audio encoders onto a language model backbone. LLaVA, for instance, connects a CLIP vision encoder to a Llama language model. These are easier to build and iterate on, but can struggle with deep cross-modal reasoning.

**What works well in 2025:**
- Image understanding (describing, analyzing, extracting data from images and charts)
- Code generation from screenshots or mockups
- Document parsing (PDFs, receipts, forms) with vision models
- Audio transcription and understanding (Gemini, GPT-4o)

**What is still limited:**
- Fine-grained spatial reasoning ("what is 3cm to the left of the red box")
- Consistent image generation that follows complex multi-constraint prompts
- Real-time video understanding at scale

---

## Inference Optimization

Running foundation models in production requires serious engineering to manage cost and latency. Here are the key techniques:

### Quantization

Reducing the precision of model weights from fp16/bf16 (16-bit) to int8 or int4. A 70B model at bf16 requires ~140GB of memory; at int4, it fits in ~35GB — runnable on a single high-end GPU.

| Precision | Memory (70B model) | Quality Impact |
|---|---|---|
| bf16 | ~140 GB | Baseline |
| int8 (GPTQ/AWQ) | ~70 GB | Minimal for most tasks |
| int4 (GPTQ/AWQ) | ~35 GB | Small degradation, noticeable on reasoning-heavy tasks |

### KV Caching

Stores the Key and Value tensors from previous tokens so they do not need to be recomputed during generation. This is not optional — without it, generation time would scale quadratically with sequence length. Every production system uses KV caching. Techniques like PagedAttention (used in vLLM) manage KV cache memory more efficiently using virtual memory concepts.

### Speculative Decoding

Uses a small, fast "draft" model to generate candidate tokens, then verifies them in a single pass through the large model. Because the large model can check multiple tokens in parallel (whereas generation is sequential), this can yield 2-3x speedups with no quality loss. Works best when the draft model has high acceptance rates — i.e., for tasks where the output is relatively predictable.

### Continuous Batching

Traditional batching waits until a batch of requests is ready, processes them together, then returns all results. Continuous batching (used by vLLM, TensorRT-LLM, and others) dynamically adds and removes requests from the batch as they arrive and complete. This dramatically improves GPU utilization and throughput, reducing per-request latency under load.

---

## Putting It Together

As an engineer building with foundation models, here is what matters from this chapter:

1. **Model choice is an engineering decision.** Bigger is not always better. A well-quantized 8B model with good fine-tuning can outperform a 70B model on your specific task at a fraction of the cost.
2. **Tokenization affects your budget and your users.** Count tokens, not words. Test with your actual data, especially for non-English use cases.
3. **Sampling parameters are not magic.** Temperature controls randomness. Top-p controls diversity. Set them deliberately based on your use case, not by copying defaults.
4. **Context windows have real costs.** Just because you *can* send 200K tokens does not mean you *should*. Retrieve what is relevant, not everything.
5. **The training pipeline explains model behavior.** When a model refuses a harmless request or responds in an oddly formal style, it is usually traceable to the SFT or RLHF stages. Understanding the pipeline helps you debug unexpected behavior.

The models are impressive, but they are also engineering artifacts with knowable properties and predictable failure modes. The more you understand about how they work, the better systems you will build with them.
