# Evaluation and Benchmarking Guide

This guide covers all evaluation methods available in this repo. Two complementary approaches are provided: a fast training-signal check based on eval loss and perplexity, and a semantic quality evaluation powered by RAGAS.

## Why two methods?

Eval loss tells you whether the model learned the training distribution. RAGAS tells you whether the model's answers are faithful, relevant, and accurate in the way a user would care about. Neither method alone is sufficient. Eval loss can improve while answer quality degrades (overfitting to format), and RAGAS scores can look reasonable even when training was inefficient. Running both gives a more complete picture.

---

## Method 1: Eval Loss and Perplexity

**Script:** `src/compare_eval_loss.py`  
**Makefile target:** `make compare-eval-loss`

### What it measures

Eval loss is the cross-entropy loss computed on evaluation examples that were held out from training. It measures how confidently the model predicts the expected output token by token. Lower eval loss means the model more closely matches the reference responses on this split.

Perplexity is derived directly from eval loss:

```
perplexity = exp(eval_loss)
```

It is easier to read at a glance but carries the same information. Lower perplexity is better.

### What it does not measure

Eval loss does not measure whether the model's generated answers are helpful, accurate, or faithful to a knowledge source. A model with lower eval loss still produces outputs that need to be judged for quality. This is what RAGAS addresses.

### How to run

```bash
make compare-eval-loss
```

Equivalent command:

```bash
python src/compare_eval_loss.py --config configs/default_training.yaml
```

The script loads both the base model and the LoRA adapter, evaluates each on the same deterministic 90/10 eval split used during training, and prints:

```
Base model eval_loss:       2.4000 | perplexity: 11.02
LoRA adapter eval_loss:     1.6700 | perplexity: 5.31
Delta adapter - base:       -0.7300
Result: the LoRA adapter improved eval loss on this eval split.
```

### Interpreting the delta

| Delta | Interpretation |
|---|---|
| Negative | Adapter improved eval loss — the fine-tuning had a measurable effect |
| Zero | No change — the adapter did not affect prediction confidence |
| Positive | Adapter made eval loss worse — possible overfitting or harmful fine-tuning |

A negative delta is necessary but not sufficient. Even if eval loss dropped, run RAGAS to verify the improvement translates into better answer quality.

### Command options

```bash
# Use a different config
python src/compare_eval_loss.py --config configs/mps_training.yaml

# Point at a specific adapter directory
python src/compare_eval_loss.py --adapter-dir artifacts/experiments/my-run

# Override the base model
python src/compare_eval_loss.py --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Change eval batch size
python src/compare_eval_loss.py --batch-size 2
```

### Reading checkpoint eval loss from training logs

The Hugging Face Trainer logs eval loss for the fine-tuned model during training. You can find it inside the checkpoint:

```
artifacts/experiments/base-run/checkpoint-3/trainer_state.json
```

Look for `eval_loss` in the `log_history` array. This value represents the fine-tuned model only. To compare it against the base model, run `compare_eval_loss.py`.

---

## Method 2: RAGAS Semantic Evaluation

**Script:** `src/evaluate_ragas.py`  
**Makefile target:** `make eval-ragas`

### Overview

RAGAS (Retrieval-Augmented Generation Assessment) is a framework for evaluating language model outputs on dimensions that go beyond prediction confidence. It judges whether generated answers are faithful to a source, whether they address the question, and how accurate and similar they are to a reference answer.

This repo uses a local Ollama model as the judge, so no API key or cloud dependency is needed.

### Prerequisites

Ollama must be installed and running locally. Pull the judge model before the first run:

```bash
ollama pull llama3.1
```

The evaluated model must be exported first:

```bash
make export-merged
```

RAGAS evaluates the merged model at `artifacts/exports/base-run-merged/`.

### How to run

```bash
make eval-ragas
```

Equivalent command:

```bash
python src/evaluate_ragas.py --max-samples 50
```

With options:

```bash
# Evaluate only 20 examples (faster)
python src/evaluate_ragas.py --max-samples 20

# Use a different Ollama judge model
python src/evaluate_ragas.py --judge-model mistral

# Save results to a JSON file
python src/evaluate_ragas.py --output artifacts/ragas_results.json

# Evaluate a different exported model
python src/evaluate_ragas.py --model-dir artifacts/exports/my-run-merged

# Evaluate on a different dataset
python src/evaluate_ragas.py --data data/uk_mortgage_dataset.jsonl
```

### Data flow

The script maps the JSONL evaluation dataset to RAGAS's expected schema:

| JSONL field | RAGAS field | Role |
|---|---|---|
| `instruction` | `user_input` | The question asked |
| `input` | `retrieved_contexts` | Product details used as the knowledge source |
| `output` | `reference` | The expected correct answer |
| model output | `response` | The answer generated by the evaluated SLM |

### Metrics

#### Faithfulness

Measures whether every claim in the generated answer can be traced back to the retrieved context (the `input` field in the JSONL data). A faithful answer does not introduce facts that are not present in the product details.

- Score range: 0.0 to 1.0
- 1.0 means every claim in the answer is supported by the context
- Low scores indicate hallucination — the model invented information not present in the source

This is the most important metric for a domain-specific assistant where factual grounding matters.

#### Answer Relevancy

Measures whether the generated answer actually addresses the question that was asked. A relevant answer stays on topic and does not drift into tangential or unrelated content.

- Score range: 0.0 to 1.0
- High scores mean the answer is tightly focused on the question
- Low scores indicate the model responded with something that does not match the question

Uses both the LLM judge and embeddings to compute semantic alignment between question and answer.

#### Factual Correctness

Measures how accurately the generated answer matches the reference answer. The judge checks whether the facts in the generated answer are correct relative to the expected output.

- Score range: 0.0 to 1.0
- High scores mean the answer agrees with the reference on the key facts
- Low scores indicate the model produced incorrect or conflicting information

This is distinct from Faithfulness: an answer can be faithful (grounded in the context) but still factually incorrect relative to the reference answer.

#### Semantic Similarity

Measures how closely the generated answer resembles the reference answer in meaning, using embeddings rather than exact text matching.

- Score range: typically 0.0 to 1.0
- High scores mean the generated answer is semantically close to the reference
- Low scores indicate the model used a very different framing or structure

This metric is sensitive to phrasing differences even when the facts are correct.

### Example output

```
=== RAGAS Results ===
  faithfulness                 0.8423
  answer_relevancy             0.7891
  factual_correctness          0.7124
  semantic_similarity          0.8012
```

### Interpreting scores

| Score | Rough signal |
|---|---|
| Above 0.80 | Strong — the model handles this dimension well |
| 0.60–0.80 | Moderate — some room for improvement |
| Below 0.60 | Weak — likely needs more data, better prompting, or longer training |

Scores should be interpreted together rather than in isolation. A model with high Semantic Similarity but low Faithfulness is producing plausible-sounding but hallucinated answers. A model with high Faithfulness but low Answer Relevancy is staying grounded but not answering the question well.

### How the judge works

The script passes each (question, context, reference, response) tuple to the Ollama judge model. The judge evaluates each metric using RAGAS's built-in prompting framework, which asks structured questions such as "Can this claim be inferred from the context?" and returns structured scores. The judge model is used for all metrics that require reasoning (Faithfulness, Answer Relevancy, Factual Correctness). Semantic Similarity is computed using the judge model's embedding layer via Ollama's OpenAI-compatible embedding endpoint.

---

## Choosing which method to run

| Situation | Recommended method |
|---|---|
| After every training run, quick sanity check | Eval loss (`make compare-eval-loss`) |
| Before declaring a model ready for testing | Both methods |
| Investigating hallucination behaviour | RAGAS Faithfulness |
| Checking whether answers address the right question | RAGAS Answer Relevancy |
| Comparing two fine-tuned checkpoints for answer quality | RAGAS all metrics |
| Running on CPU without Ollama available | Eval loss only |
| Tracking improvement over multiple training experiments | Both, log results to `artifacts/` |

---

## Suggested workflow

```
1. Train with make train
2. Run make compare-eval-loss
   → If delta is positive (adapter made things worse), revisit training config
   → If delta is negative or neutral, continue
3. Run make export-merged
4. Run make eval-ragas
   → Review Faithfulness first (hallucination signal)
   → Review Answer Relevancy (focus signal)
   → Review Factual Correctness and Semantic Similarity for overall quality
5. Adjust data, config, or training length based on weakest metric
6. Repeat from step 1
```

---

## Limitations

**Eval loss** is measured on the same dataset used for training (different split, but same distribution). It does not generalise to unseen question types or adversarial inputs.

**RAGAS scores** depend on the quality of the judge model. A weaker judge (e.g., a small Ollama model) may score inconsistently on ambiguous answers. Larger judge models produce more reliable scores but take longer to run.

