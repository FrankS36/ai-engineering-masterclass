# Chapter 10: Fine-Tuning and Customization

You can get remarkably far with prompt engineering and retrieval-augmented generation. Most teams should exhaust both before they consider fine-tuning. But when you need a model that behaves differently -- adopts a specific tone, follows a rigid output format without constant reminders, or handles a domain where general-purpose models stumble -- fine-tuning is the lever that changes the model itself rather than the instructions around it.

This chapter covers the decision framework, the techniques, and the practical engineering of fine-tuning. The goal is to leave you equipped to ship a fine-tuned model to production without wasting weeks on avoidable mistakes.

---

## When to Fine-Tune (And When Not To)

The first question is always whether fine-tuning is the right tool. Three strategies sit on a spectrum of effort and capability:

| Problem | Best approach | Why |
|---|---|---|
| Model lacks domain knowledge (medical codes, internal APIs, product catalog) | RAG | Inject knowledge at inference time. No training needed. |
| Model knows enough but needs a different behavior (tone, format, reasoning style) | Fine-tuning | Change how the model responds, not what it knows. |
| Task is straightforward and well-defined | Prompting | A good system prompt with examples often suffices. |
| Model needs both new knowledge and new behavior | RAG + fine-tuning | Fine-tune for behavior, retrieve for knowledge. |

A useful heuristic: if you find yourself writing the same correction in your prompt over and over ("always respond in JSON," "never apologize," "use this exact citation format"), that repetition is a signal that fine-tuning could bake the behavior in and simplify your prompt.

Fine-tuning is not a fix for models that hallucinate facts. The model does not reliably memorize training examples -- it adjusts behavioral patterns. If you need factual grounding, use retrieval.

---

## Types of Fine-Tuning

### Full Fine-Tuning

Every parameter in the model is updated during training. This gives maximum flexibility but requires enormous GPU memory (roughly 4x the model size in FP16 for optimizer states) and risks catastrophic forgetting -- the model loses general capabilities while learning your task. In practice, full fine-tuning is rare outside of organizations training foundation models.

### LoRA (Low-Rank Adaptation)

LoRA is the dominant fine-tuning method for practitioners. The core idea: instead of updating the full weight matrix W, you freeze W and train two small matrices A and B such that the effective weight becomes W + BA. If W is a 4096x4096 matrix, and you choose rank r=16, then B is 4096x16 and A is 16x4096. You train 131,072 parameters instead of 16,777,216 -- a 128x reduction.

At inference time, the LoRA weights can be merged into the base model with zero latency overhead, or kept separate so you can hot-swap adapters for different tasks on the same base model.

### QLoRA

QLoRA combines LoRA with quantization. The base model is loaded in 4-bit precision (cutting memory by 4x), while the LoRA adapters train in higher precision. This lets you fine-tune a 70B-parameter model on a single 48GB GPU -- something that would otherwise require a multi-GPU cluster. Quality is surprisingly close to full-precision LoRA for most tasks.

### Instruction Tuning

Instruction tuning is not a separate technique but a data format. You train the model on instruction-response pairs so it learns to follow instructions in a specific way. All the chat models you use daily (GPT-4o, Claude, Llama-3-Instruct) were instruction-tuned on top of their base pre-trained versions. When you fine-tune a chat model, you are essentially doing a second round of instruction tuning for your domain.

---

## Data Preparation

Data quality is the single biggest determinant of fine-tuning success. A model fine-tuned on 500 high-quality examples will outperform one trained on 10,000 sloppy ones.

### Format

Most fine-tuning pipelines expect JSONL files where each line is a conversation. OpenAI's format has become a de facto standard:

```jsonl
{"messages": [{"role": "system", "content": "You are a legal contract summarizer. Output JSON with keys: parties, term_months, governing_law, key_obligations."}, {"role": "user", "content": "Summarize this contract: [contract text here]"}, {"role": "assistant", "content": "{\"parties\": [\"Acme Corp\", \"Widget Inc\"], \"term_months\": 24, \"governing_law\": \"Delaware\", \"key_obligations\": [\"Monthly SaaS fee of $5,000\", \"99.9% uptime SLA\", \"30-day termination notice\"]}"}]}
{"messages": [{"role": "system", "content": "You are a legal contract summarizer. Output JSON with keys: parties, term_months, governing_law, key_obligations."}, {"role": "user", "content": "Summarize this contract: [different contract text]"}, {"role": "assistant", "content": "{\"parties\": [\"TechStart LLC\", \"CloudServe Inc\"], \"term_months\": 12, \"governing_law\": \"California\", \"key_obligations\": [\"Annual license fee of $120,000\", \"Quarterly business reviews\", \"Data deletion within 30 days of termination\"]}"}]}
```

For HuggingFace-based training, the Alpaca format (instruction, input, output fields) and ShareGPT format (conversations array) are also common. The specific format matters less than consistency.

### Data Sources

- **Production logs**: Your best source. Real user queries with expert-corrected responses.
- **Expert annotation**: Subject-matter experts write gold-standard responses to representative queries.
- **Synthetic generation**: Use a stronger model (GPT-4o, Claude) to generate training data, then have humans verify. This is common and effective when done carefully.
- **Existing datasets**: HuggingFace Hub hosts thousands of instruction-tuning datasets for common tasks.

### Data Cleaning Checklist

1. Remove duplicates and near-duplicates.
2. Validate that every example follows your schema exactly.
3. Check for and remove personally identifiable information (PII).
4. Ensure the assistant responses are actually correct -- spot-check at least 10% manually.
5. Filter out excessively long or short examples that could skew the distribution.
6. Split into train (80-90%), validation (5-10%), and test (5-10%) sets. Split by semantic category, not randomly, to avoid data leakage.

### How Much Data?

There is no universal minimum, but rough guidelines from practical experience:

| Goal | Typical dataset size |
|---|---|
| Style/format adaptation | 50-200 examples |
| Domain-specific behavior | 500-2,000 examples |
| Complex multi-step reasoning | 2,000-10,000 examples |
| General instruction following | 10,000+ examples |

Start with the smallest viable dataset, evaluate, then add more data where the model fails.

---

## The Training Process

### Key Hyperparameters

| Parameter | Typical range | Notes |
|---|---|---|
| Learning rate | 1e-5 to 5e-5 | Lower for larger models. Start at 2e-5. |
| Epochs | 1-5 | More epochs risk overfitting. 2-3 is a good default. |
| Batch size | 4-32 | Limited by GPU memory. Use gradient accumulation to simulate larger batches. |
| LoRA rank (r) | 8-64 | Higher rank = more capacity but more parameters. 16 is a solid default. |
| LoRA alpha | 16-64 | Usually set to 2x the rank. Controls the scaling of LoRA updates. |
| LoRA target modules | q_proj, v_proj (minimum) | Adding k_proj, o_proj, gate_proj, up_proj, down_proj improves quality at modest cost. |
| Warmup ratio | 0.03-0.1 | Ramp up learning rate gradually to stabilize early training. |

### A Practical Training Script

Using the `trl` library from HuggingFace, a LoRA fine-tuning run looks like this:

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# Load base model in 4-bit (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16",
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # typically < 1% of total

# Load your dataset
dataset = load_dataset("json", data_files={"train": "train.jsonl", "validation": "val.jsonl"})

# Train
training_config = SFTConfig(
    output_dir="./lora-output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # effective batch size = 16
    learning_rate=2e-5,
    warmup_ratio=0.05,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    bf16=True,
)

trainer = SFTTrainer(
    model=model,
    args=training_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer,
)

trainer.train()
trainer.save_model("./final-adapter")
```

### Monitoring Training

Watch two curves: training loss and validation loss. Training loss should decrease steadily. If validation loss starts increasing while training loss continues dropping, you are overfitting -- stop training and use the checkpoint from the lowest validation loss.

A training loss that plateaus immediately suggests the learning rate is too low or the data is too noisy. A loss that spikes or oscillates means the learning rate is too high.

---

## Common Pitfalls

### Catastrophic Forgetting

The model gets good at your specific task but loses general capabilities. A customer-support model that can no longer handle basic math or follow simple instructions has forgotten too aggressively.

Prevention: use LoRA (inherently limits the scope of changes), keep epochs low, and mix in a small percentage (5-10%) of general-purpose instruction data alongside your domain data.

### Overfitting

The model memorizes training examples instead of learning patterns. Signs: perfect performance on training data, poor performance on held-out data, and the model regurgitating training examples verbatim.

Prevention: use a validation set and stop early. Smaller datasets are especially prone -- with 200 examples and 5 epochs, the model sees each example 5 times, which is enough to memorize.

### Distribution Shift

Your training data does not reflect real production queries. You trained on well-formed questions, but users send typo-filled, ambiguous, or adversarial inputs.

Prevention: include messy, real-world examples in your training data. If you generated synthetic data, add noise and variation. Test with actual production traffic before deploying.

### Evaluation Contamination

Your test set overlaps with your training set -- or worse, with the data used to generate your synthetic training examples. This gives falsely optimistic results.

Prevention: create your test set first, quarantine it, and never let it touch the training pipeline.

---

## Hosted vs Self-Hosted Fine-Tuning

### OpenAI Fine-Tuning API

The simplest path. Upload a JSONL file, start a job, wait, get a model ID you can use in the same API.

```python
from openai import OpenAI

client = OpenAI()

# Upload training file
file = client.files.create(file=open("train.jsonl", "rb"), purpose="fine-tune")

# Start fine-tuning job
job = client.fine_tuning.jobs.create(
    training_file=file.id,
    model="gpt-4o-mini-2024-07-18",
    hyperparameters={"n_epochs": 3},
)

# Check status
status = client.fine_tuning.jobs.retrieve(job.id)
print(status.status)  # "running", "succeeded", etc.

# Use the fine-tuned model (after training completes)
response = client.chat.completions.create(
    model=status.fine_tuned_model,  # e.g., "ft:gpt-4o-mini-2024-07-18:org:suffix:id"
    messages=[{"role": "user", "content": "Summarize this contract..."}],
)
```

Advantages: no infrastructure to manage, fast iteration, models are served for you. Disadvantages: limited model selection (GPT-4o-mini, GPT-4o), no control over hyperparameters beyond epochs and learning rate multiplier, data leaves your environment, ongoing per-token cost for inference.

### Self-Hosted (HuggingFace + Your GPUs)

Full control. You choose the base model, hyperparameters, and training strategy. You own the resulting weights and can serve them however you want.

Advantages: any open model (Llama, Mistral, Qwen, Phi), full hyperparameter control, data stays in your environment, no per-token inference cost after deployment. Disadvantages: you manage GPUs (or rent them), you handle serving infrastructure, and debugging training issues is on you.

### Cost Comparison

| Dimension | OpenAI fine-tuning | Self-hosted (cloud GPU) |
|---|---|---|
| Training cost | ~$0.008/1K tokens (4o-mini) | $2-4/hr per A100 GPU |
| Inference cost | ~1.5-2x base model pricing | Fixed GPU cost, unlimited tokens |
| Time to first result | Hours (mostly queue time) | Hours (mostly training time) |
| Infra effort | None | Significant |
| Data privacy | Data sent to OpenAI | Data stays on your machines |
| Model selection | GPT-4o-mini, GPT-4o | Any open model |

For low-volume use cases (under 1M tokens/day inference), hosted fine-tuning is almost always cheaper when you account for engineering time. For high-volume or privacy-sensitive use cases, self-hosted wins.

---

## Evaluation

### During Training

Track validation loss at regular intervals. A decreasing validation loss means the model is learning generalizable patterns. But loss alone is not enough -- a model can have low loss while still producing subtly wrong outputs.

### After Training

Run your fine-tuned model and the base model (with your best prompt) on the same held-out test set. Compare across the dimensions that matter for your use case:

```python
import json

def evaluate_model(model_fn, test_data):
    results = {"correct_format": 0, "correct_content": 0, "total": 0}
    for example in test_data:
        response = model_fn(example["input"])
        results["total"] += 1

        # Check structural correctness (does it output valid JSON?)
        try:
            parsed = json.loads(response)
            results["correct_format"] += 1
        except json.JSONDecodeError:
            continue

        # Check content correctness (domain-specific checks)
        if validates_against_ground_truth(parsed, example["expected"]):
            results["correct_content"] += 1

    return {
        "format_accuracy": results["correct_format"] / results["total"],
        "content_accuracy": results["correct_content"] / results["total"],
    }

# Compare base model (with prompt engineering) vs fine-tuned model
base_results = evaluate_model(base_model_with_prompt, test_set)
ft_results = evaluate_model(fine_tuned_model, test_set)

print(f"Base model:       format={base_results['format_accuracy']:.1%}, content={base_results['content_accuracy']:.1%}")
print(f"Fine-tuned model: format={ft_results['format_accuracy']:.1%}, content={ft_results['content_accuracy']:.1%}")
```

### Regression Testing

Fine-tuning can improve your target task while degrading performance on other tasks the model previously handled. Maintain a regression test suite that covers general capabilities you care about, and run it after every training run.

---

## Cost and ROI

Fine-tuning pays for itself in two scenarios:

1. **Prompt reduction**: If your current prompt is 2,000 tokens of instructions and examples, and fine-tuning lets you drop to 200 tokens, you save 1,800 tokens per request. At $0.01/1K input tokens and 100K requests/day, that is $1,800/day in savings.

2. **Model downgrade**: Fine-tuning a smaller model to match a larger model's performance on your specific task. If fine-tuned GPT-4o-mini matches GPT-4o on your use case, you save roughly 10-20x on per-token costs.

### Break-Even Analysis

Suppose fine-tuning costs $500 (training compute + engineering time) and saves $0.002 per request through prompt reduction. Break-even is at 250,000 requests. If you process 10,000 requests per day, the investment pays for itself in 25 days.

The less obvious cost is maintenance. Models evolve, data distributions shift, and you will need to retrain periodically. Budget for quarterly retraining cycles and treat your training data pipeline as production infrastructure -- because it is.

---

## Key Takeaways

Fine-tuning is a precision tool, not a first resort. Exhaust prompting and RAG before reaching for it. When you do fine-tune, invest heavily in data quality, start with LoRA or QLoRA, monitor validation loss religiously, and always compare against your best prompt-engineered baseline. The goal is not a fine-tuned model -- it is a measurably better system at lower operating cost.