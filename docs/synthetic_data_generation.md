# Synthetic Data Generation for Domain Fine-Tuning

A practical guide to the data generation pipeline used in this repository to produce high-quality, diverse instruction-tuning data for UK mortgage domain fine-tuning.

## Why synthetic data is needed here

Fine-tuning a small language model on domain-specific data requires instruction-response pairs that reflect the kinds of questions the model will face at inference time. The initial dataset in this repository (`data/uk_mortgage_dataset.jsonl`) contains 64 manually written examples, but these have two structural problems that limit training quality.

First, the dataset is too small. Supervised fine-tuning generally requires hundreds to thousands of examples to produce a model that generalises reliably rather than overfit to a narrow pattern.

Second, and more importantly, the examples are all the same type. Every instruction asks the model to summarise a mortgage product, and every output follows the same template. A model trained only on this data learns to produce summaries — nothing more. It will struggle with questions that require eligibility reasoning, rate analysis, fee trade-off calculations, risk assessment, or factual verification, because it has never seen those tasks during training.

The synthetic data pipeline in this repository solves both problems. It generates a large number of diverse, domain-grounded examples that cover a wide range of question types, all anchored to realistic UK mortgage product data.

## Pipeline overview

The pipeline has three sequential phases. Each phase has a clear input, a clear output, and a specific rationale for why it is designed the way it is.

```text
┌─────────────────────────────────────────────────────────────┐
│  Phase 1 — Product Sampling (no AI)                         │
│                                                             │
│  Parameterised template sampler draws from:                 │
│  · 18 real UK lenders                                       │
│  · 6 term types (2yr/5yr/10yr Fixed, Tracker, Discount)     │
│  · 7 LTV bands (60% → 95%)                                  │
│  · Rate spreads calibrated to base rate by LTV tier         │
│  · Pool of 30 realistic product notes                       │
│                                                             │
│  Output: 250 unique MortgageProduct records                 │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 2 — Q&A Generation (Claude Sonnet)                   │
│                                                             │
│  For each product, a capable model generates 7 Q&A pairs    │
│  sampled from 9 distinct question types:                    │
│                                                             │
│  · summarisation      · eligibility                         │
│  · rate_analysis      · fee_tradeoff                        │
│  · risk_suitability   · comparison                          │
│  · factual_correctness · terminology                        │
│  · borrower_advice                                          │
│                                                             │
│  The product record is enforced as the exact "input" field. │
│  Output: ~1,750 raw instruction-response pairs              │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 3 — Quality Filtering (Claude Haiku as judge)        │
│                                                             │
│  Each pair scored 1–5 on four domain-specific criteria:     │
│  · factual_accuracy    — figures match the product record   │
│  · financial_reasoning — advice is sound for UK market      │
│  · completeness        — answer fully addresses the task    │
│  · no_hallucination    — no invented rates, fees, or terms  │
│                                                             │
│  Pairs scoring < 4 on any criterion are discarded.          │
│  Output: ~1,200–1,500 high-quality training examples        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
              data/synthetic_mortgage_dataset.jsonl
```

## Phase 1: Product sampling

**File:** `src/data_gen/product_sampler.py`

The product sampler generates realistic UK mortgage product records without making any API calls. This is a deliberate design choice: language models tend to produce implausible rate and fee combinations when asked to invent financial data freely. A deterministic sampler with calibrated parameters is cheaper, faster, and more domain-faithful.

The sampler draws from the following pools with configurable weights that reflect real UK market proportions:

| Parameter | Options |
|---|---|
| Providers | 18 real UK lenders including Nationwide, HSBC UK, Barclays, Halifax, NatWest, Santander UK, First Direct, Yorkshire BS |
| Term types | 2 Year Fixed (weighted highest), 5 Year Fixed, 10 Year Fixed, 2 Year Tracker, 5 Year Tracker, 2 Year Discount |
| LTV bands | 60%, 70%, 75% (weighted highest), 80%, 85%, 90%, 95% |
| Booking fees | £0, £499, £995, £999 (most common), £1,499 |
| Notes | Pool of 30 realistic product conditions |

Interest rates are calculated as `base_rate + spread`, where the spread range is calibrated to each LTV band and adjusted for term length. Fixed products carry a modest term premium for longer terms, trackers price tightly to the base rate, and discount products price slightly below equivalent fixed products — matching real UK market structure.

All sampling uses a seeded `random.Random` instance, which means the product set is fully reproducible across runs.

> **Implementation note:** The sampler deduplicates on `(provider, product_name, ltv)` to avoid training on trivially repeated inputs. With a pool of 18 providers, 6 term types, 15 product suffixes, and 7 LTV tiers, this leaves enough headroom to draw 250 unique records comfortably.

## Phase 2: Question-answer generation

**File:** `src/data_gen/qa_generator.py`

For each product record, a Claude model is asked to generate a batch of question-answer pairs covering different question types. Batching all questions for a single product into one API call is more efficient than individual calls and allows the model to produce meaningfully varied questions rather than repetitive phrasings.

Each call selects 7 types at random from the 9 available, which ensures variety without exact repetition across products:

| Question type | What it tests |
|---|---|
| `summarisation` | Extracting and describing the key product terms |
| `eligibility` | Reasoning whether the product fits a specific borrower profile |
| `rate_analysis` | Evaluating the rate against the current base rate |
| `fee_tradeoff` | Calculating or reasoning about the break-even for a booking fee |
| `risk_suitability` | Identifying risks for a given borrower type or economic scenario |
| `comparison` | Contrasting this product against an alternative type |
| `factual_correctness` | Verifying or correcting a specific claim about the product |
| `terminology` | Explaining a mortgage term (APRC, LTV, ERC) in context |
| `borrower_advice` | Advising what to check before committing |

The `input` field of every generated pair is overwritten with the exact product text from the sampler, regardless of what the model produced. This ensures the structured product data is always correctly formatted and prevents the model from silently paraphrasing or omitting fields.

> **Model choice:** The generator uses `claude-sonnet-4-6` by default. This is configurable via `configs/data_generation.yaml`. A larger model improves output quality; a smaller one reduces cost. The generator is the main cost driver in the pipeline.

## Phase 3: Quality filtering

**File:** `src/data_gen/quality_filter.py`

Generated pairs are passed through an LLM-as-judge using a smaller, cheaper model. This is more appropriate than a general-purpose reward model for this domain because the scoring rubric is explicitly domain-specific: it checks financial accuracy and UK market validity, not just general helpfulness or fluency.

Each example is scored on four criteria:

| Criterion | What is checked |
|---|---|
| `factual_accuracy` | Does the output correctly cite rates, fees, and LTV from the product record? |
| `financial_reasoning` | Is the financial advice or analysis plausible within the UK mortgage market? |
| `completeness` | Does the output fully address what the instruction asks, not just partially? |
| `no_hallucination` | Does the output avoid inventing figures not present in the product record? |

A pair must score at least 4 out of 5 on every criterion to be retained. The filter uses `claude-haiku-4-5-20251001` by default for cost efficiency.

## Step-by-step instructions

### Choosing a provider

The pipeline supports two providers, switched by a single line in `configs/data_generation.yaml`. The default is Groq because it has a free tier.

| Provider | Cost | Model | Rate limit |
|---|---|---|---|
| **Groq** (default) | Free | Llama 3.1 70B | ~30 requests/min |
| Anthropic | ~$8–10 per full run | Claude Sonnet + Haiku | Higher limits |

The full run with Groq takes longer due to rate limiting (Phase 2 ~8 min, Phase 3 ~1 hour), but costs nothing. Anthropic completes the same run in minutes.

### Prerequisites

Ensure the repository environment is set up. See `docs/environment_setup_tutorial.md` for full setup instructions.

Install all dependencies including the provider SDKs:

```bash
pip install -r requirements.txt
```

#### Using Groq (free)

Sign up at [console.groq.com](https://console.groq.com), create an API key, and export it:

```bash
export GROQ_API_KEY=gsk_...
```

#### Using Anthropic (paid)

Sign up at [console.anthropic.com](https://console.anthropic.com), create an API key, and export it:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Then change the provider in `configs/data_generation.yaml`:

```yaml
provider:
  name: anthropic   # was: groq

qa_generator:
  model: claude-sonnet-4-6          # Anthropic generation model

quality_filter:
  model: claude-haiku-4-5-20251001  # Anthropic filter model
```

### Step 1: Verify the sampler (no API cost)

Run a dry-run to confirm the product sampler works and preview the product record format before making any API calls:

```bash
python src/data_gen/generate_dataset.py --dry-run
```

This samples all 250 products locally and prints the first two records. No API calls are made regardless of which provider is configured.

### Step 2: Inspect and adjust the config

Review `configs/data_generation.yaml` before running the full pipeline:

```yaml
provider:
  name: groq          # switch to: anthropic

product_sampler:
  num_products: 250               # increase for more data volume
  base_rate: 3.75                 # update if the Bank of England rate changes
  report_date: "2026-04-23"

qa_generator:
  model: llama-3.1-70b-versatile  # groq model; anthropic: claude-sonnet-4-6
  questions_per_product: 7        # max is 9 (number of available question types)
  temperature: 0.8                # higher = more varied phrasing

quality_filter:
  model: llama-3.1-70b-versatile  # groq model; anthropic: claude-haiku-4-5-20251001
  min_score: 4                    # raise to 5 for stricter filtering
```

Key decisions:
- **`num_products`**: 250 products × 7 questions = ~1,750 raw pairs, yielding ~1,200–1,500 after filtering. Reduce to 50–100 for a quick first run.
- **`base_rate`**: Keep this aligned with the actual Bank of England base rate so that rate analysis questions generate accurate answers.
- **`min_score`**: A threshold of 4 is recommended. Lowering to 3 increases volume at the cost of quality.

### Step 3: Run the full pipeline

```bash
# Groq (free, slower)
GROQ_API_KEY=gsk_... python src/data_gen/generate_dataset.py

# Anthropic (paid, faster)
ANTHROPIC_API_KEY=sk-ant-... python src/data_gen/generate_dataset.py
```

The script prints a progress bar for each phase and reports how many pairs passed the filter. Typical output:

```text
Phase 1: Sampling mortgage product records...
  Sampled 250 products.

Provider: groq  |  Gen model: llama-3.1-70b-versatile  |  Filter model: llama-3.1-70b-versatile

Phase 2: Generating Q&A pairs...
[##################################################] 100.0%  product 250/250
  Generated 1,742 raw pairs.

Phase 3: Filtering with LLM-as-judge...
[##################################################] 100.0%  pair 1742/1742
  Kept 1,381 / 1,742 pairs (rejected 361).

Saved 1,381 examples to data/synthetic_mortgage_dataset.jsonl
```

### Step 4: Skip filtering for fast iteration

During development it can be useful to skip the quality filter and inspect raw output before committing to a full run:

```bash
GROQ_API_KEY=gsk_... python src/data_gen/generate_dataset.py --skip-filter
```

This is especially useful with Groq because it avoids the ~1 hour filtering phase entirely.

### Step 5: Use the generated data for training

Point `configs/default_training.yaml` at the new dataset:

```yaml
dataset:
  path: data/synthetic_mortgage_dataset.jsonl
  field_names:
    instruction: instruction
    input: input
    output: output
```

Then run training as normal:

```bash
python src/train_slm.py --config configs/default_training.yaml
```

## Output format

The output file uses the same JSONL format as the existing dataset, with one JSON object per line:

```jsonl
{"instruction": "Would this mortgage be suitable for a first-time buyer with a 10% deposit?", "input": "Report date: 2026-04-23\nBase rate: 3.75%\nProvider: Halifax\n...", "output": "This product has a maximum LTV of 90%, which is compatible with a 10% deposit..."}
{"instruction": "Is the interest rate competitive given the current base rate of 3.75%?", "input": "Report date: 2026-04-23\nBase rate: 3.75%\nProvider: NatWest\n...", "output": "The rate of 4.92% sits approximately 1.17 percentage points above the base rate..."}
```

Every example has the same three fields required by `src/train_slm.py`:

| Field | Content |
|---|---|
| `instruction` | A natural question or task directive, varied in phrasing and type |
| `input` | The exact mortgage product record (provider, rate, LTV, fee, notes) |
| `output` | A thorough, domain-accurate response grounded in the product data |

## Cost and timing reference

| Provider | Phase 2 time | Phase 3 time | Total cost |
|---|---|---|---|
| Groq (free tier) | ~8 min | ~60 min | $0 |
| Anthropic | ~2 min | ~5 min | ~$8–10 USD |

Groq's free tier enforces approximately 30 requests per minute. The pipeline automatically sleeps 2 seconds between requests to stay within this limit. If you exceed it, the client retries with exponential back-off before raising an error.

To reduce Groq run time, lower `num_products` in the config. 100 products yields ~550 filtered examples and completes Phase 3 in roughly 25 minutes.
