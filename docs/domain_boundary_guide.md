# Domain Boundary Guide

By default a fine-tuned SLM answers every question because it has only seen
in-scope examples during training. This guide explains how to restrict the
model to a specific domain so it refuses out-of-scope questions.

## How It Works

Two mechanisms are combined, each reinforcing the other.

### 1. System Prompt Slot

A `### System:` block is added to the training format:

```text
### System:
{system_prompt}

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

Every training example — both in-scope and boundary — is shown with the same
system prompt. The model learns that the system prompt is a constraint, not
just background text. At inference the identical block must be present so the
learned behaviour activates.

The system prompt is set once in `configs/default_training.yaml`:

```yaml
system_prompt: >
  You are a UK mortgage assistant. Only answer questions about UK mortgage
  products, interest rates, LTV, fees, eligibility, and related topics.
  For any other question, respond that the question is outside your knowledge scope.
```

**Limitations**

- A system prompt only enforces behaviour reliably on models trained through
  RLHF (Reinforcement Learning from Human Feedback) — such as Claude, GPT-4,
  or Llama-3-Instruct — where the model was explicitly rewarded for following
  system-level constraints.
- For small fine-tuned SLMs like TinyLlama-1.1B, the system prompt is just
  more text in the input window. The model has no learned obligation to obey
  it and will ignore it when pre-trained knowledge is stronger.
- In this pipeline the system prompt acts as a **weak context signal**, not a
  hard constraint. It is only useful alongside boundary training examples, not
  as a standalone mechanism.

### 2. Boundary Training Examples

A set of out-of-scope examples with a consistent refusal output is mixed into
the training data:

```json
{
  "instruction": "What is the weather forecast for London this weekend?",
  "input": "",
  "output": "This question is outside my area of expertise. I am a UK mortgage assistant and can only help with questions about mortgage products, interest rates, LTV, fees, eligibility, and related topics."
}
```

These examples are stored in `data/boundary_examples.jsonl` and cover a wide
range of off-topic categories (weather, cooking, sport, history, technology,
health, and more). The mix ratio is controlled in the config:

```yaml
boundary_data:
  path: data/boundary_examples.jsonl
  ratio: 0.2   # 20% of the domain sample count
```

At the default ratio of 0.2, 200 domain examples produce 40 boundary examples
per training run.

**Limitations**

- Repetition (`repeat`) amplifies gradient signal but does not add new
  patterns. Repeating 40 examples 3× gives 120 instances but only 40 unique
  phrasings — the model may still answer novel out-of-scope questions it has
  not seen during training.
- The boundary ratio competes with domain signal. Too high a ratio degrades
  mortgage answer quality; too low and refusal is not learned reliably.
- Pre-trained knowledge is trained on billions of tokens and is very hard to
  override with a small fine-tuning dataset. The model may still answer
  off-topic questions whose categories were not covered in boundary examples.

**Rough Estimates of Boundary Examples Needed by Model Size**

| Model size | Examples needed | Generalises to unseen topics? |
|---|---|---|
| **1B (e.g. TinyLlama-1.1B)** | 500–1000+ | Weakly — high leakage expected |
| **7B (e.g. Llama-3, Mistral-7B)** | 200–400 | Reasonably — some leakage |
| **13B+** | 100–200 | Well — low leakage |

The current `data/boundary_examples.jsonl` contains 40 examples, which is
sufficient to demonstrate the mechanism but too few for reliable generalisation
at the 1B scale. Diversity across categories matters more than raw count —
new unique examples are always more valuable than repetition beyond a point.
Expanding to 400–500 diverse examples is recommended before production use.

## Why Both Are Needed

The system prompt alone gives the model a context anchor but does not
demonstrate what refusal looks like. The boundary examples alone teach refusal
but without a stable context signal. Together:

- The system prompt is the **trigger** — the model associates refusal with its presence.
- The boundary examples are the **demonstration** — the model learns what to say when the trigger fires on an off-topic question.

## Future Control 

### Recommended Request Pipeline

Before a user message reaches the SLM, two lightweight checks should sit in
front of it:

```
User input
    ↓
[Scope classifier → mortgage?]    ← cheapest, fastest, reject early
    ↓ in scope
[Moderation API]                  ← catch harmful text (abuse, threats etc.)
    ↓ clean
Your SLM
```

**Why this order matters:**

- The scope classifier is the cheapest check — most off-topic requests are
  rejected here before incurring any moderation API cost or latency.
- The moderation API only runs on confirmed in-scope requests, keeping cost
  and latency proportionate.
- The SLM only receives input that is both on-topic and clean.

The SLM's own trained boundary behaviour (system prompt + boundary examples)
acts as a final safety net for anything that slips through the classifier.

For moderation API options see `docs/sensitive_data_filtering_guide.md`.
