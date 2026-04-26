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
in each training run.

## Why Both Are Needed

The system prompt alone gives the model a context anchor but does not
demonstrate what refusal looks like. The boundary examples alone teach refusal
but without a stable context signal. Together:

- The system prompt is the **trigger** — the model associates refusal with its presence.
- The boundary examples are the **demonstration** — the model learns what to say when the trigger fires on an off-topic question.

## Inference

The system prompt must be present at inference time with the same wording and
format used during training.

**Route 1 — Hugging Face / `run_model.py`**

The system block is included in `DEFAULT_PROMPT` and prepended automatically
when using `--prompt` or `--prompt-file`. Override with `--system-prompt` or
disable with `--system-prompt ""`.

**Route 2 — Ollama**

The `Modelfile` `TEMPLATE` and `SYSTEM` directives handle injection
automatically. See `docs/model_export_workflow.md` Step 5 for the full
Modelfile content.

## Updating the Domain

To change the domain (e.g. from mortgages to insurance):

1. Replace `system_prompt` in `configs/default_training.yaml`.
2. Replace or extend `data/boundary_examples.jsonl` with examples relevant to
   the new out-of-scope categories.
3. Retrain.
4. Update the `SYSTEM` value in the Ollama `Modelfile` to match.
