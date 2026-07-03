# SLM Pipeline — GCP Migration and Improvement Plan

This document maps every component of the current local SLM training pipeline to its GCP replacement, explains which service is used at each layer, and provides a phased implementation roadmap.

---

## Current Architecture Overview

The existing pipeline is a single-machine, file-system-based workflow:

```
[Local Machine]
│
├── data/                        ← JSONL files (training data, boundary examples, test split)
├── artifacts/                   ← LoRA adapter weights, TensorBoard logs, RAGAS results
├── configs/                     ← YAML training and data-gen configs
│
├── src/data_gen/
│   ├── product_sampler.py       ← Template-based UK mortgage product generation
│   ├── qa_generator.py          ← Anthropic / Groq API calls for QA pairs
│   ├── quality_filter.py        ← LLM judge scoring (4 criteria, threshold 4/5)
│   └── generate_dataset.py      ← Orchestrates all three phases above
│
├── src/train_slm.py             ← LoRA fine-tuning (HF Trainer + PEFT)
├── src/export_merged_model.py   ← Merges LoRA adapter into base TinyLlama weights
├── src/evaluate_ragas.py        ← Semantic evaluation via local Ollama judge
├── src/compare_eval_loss.py     ← Perplexity comparison (base vs fine-tuned)
├── src/scope_classifier.py      ← DistilBERT-MNLI domain boundary gate
└── src/run_model.py             ← Local inference with classifier gate
```

**Key pain points addressed by this migration:**
- Training is blocked by local hardware (CPU/MPS only on laptop)
- Data and model artifacts are lost if the machine is wiped
- No experiment history or metric trending across runs
- RAGAS evaluation requires Ollama running locally
- Inference requires the developer's machine to be running
- API keys stored in plaintext `.env` files

---

## GCP Service Map by Pipeline Layer

| Pipeline Layer | Current (Local) | GCP Replacement | GCP Service |
|---|---|---|---|
| Secret / API key storage | `.env` file | Secret Manager | **Secret Manager** |
| Training data storage | `data/*.jsonl` | Object store | **Cloud Storage (GCS)** |
| Model artifact storage | `artifacts/` | Object store | **Cloud Storage (GCS)** |
| Config storage | `configs/*.yaml` | Object store | **Cloud Storage (GCS)** |
| Synthetic data generation | Local async script | Containerised job | **Cloud Run Jobs** |
| Training compute | Laptop GPU / MPS | Managed GPU VM | **Vertex AI Custom Training** |
| Container image registry | Local Docker | Managed registry | **Artifact Registry** |
| Experiment / metric tracking | Local TensorBoard | Managed TensorBoard | **Vertex AI TensorBoard** |
| RAGAS LLM judge | Ollama localhost | Managed LLM endpoint | **Vertex AI Model Garden** |
| Metric history & trending | `ragas_results.json` | Analytical database | **BigQuery** |
| Pipeline orchestration | Manual Makefile | Managed ML pipeline | **Vertex AI Pipelines** |
| Model versioning | Local directories | Managed model store | **Vertex AI Model Registry** |
| Model serving / inference | `python src/run_model.py` | Serverless container | **Cloud Run** |
| CI/CD for containers | None | Managed build | **Cloud Build** |
| Event-driven triggers | None | Event routing | **Eventarc** |

---

## Phase 1 — Storage and Secrets Baseline

**Effort:** Low (2–4 hours)
**Goal:** Move all data, artifacts, and credentials off the local filesystem. No code architecture changes required.

### Layer: Secret Management

**Service: Secret Manager**

Replace every plaintext `.env` reference with Secret Manager:

```
ANTHROPIC_API_KEY   →   projects/YOUR_PROJECT/secrets/anthropic-api-key/versions/latest
GROQ_API_KEY        →   projects/YOUR_PROJECT/secrets/groq-api-key/versions/latest
```

**Files to update:**
- `src/data_gen/llm_client.py` — swap `os.getenv("ANTHROPIC_API_KEY")` for `google.cloud.secretmanager.SecretManagerServiceClient`
- Remove `.env` from repository; add to `.gitignore`

**Why Secret Manager here:** API keys used by `llm_client.py` are the highest-risk plaintext secrets in the repo. Secret Manager provides IAM-scoped access, automatic audit logging, and rotation without code changes.

### Layer: Object Storage

**Service: Cloud Storage (GCS)**

Create three buckets with distinct access policies:

| Bucket | Contents | Retention |
|---|---|---|
| `slm-training-data` | `data/*.jsonl`, `configs/*.yaml` | Versioned; soft-delete 30 days |
| `slm-artifacts` | `artifacts/experiments/`, `artifacts/exports/` | Versioned; lifecycle archive after 90 days |
| `slm-logs` | TensorBoard event files | Delete after 180 days |

**Files to update:**
- `configs/default_training.yaml` — change `output_dir: artifacts/experiments/base-run` → `output_dir: gs://slm-artifacts/experiments/base-run`
- `configs/mps_training.yaml` — same `output_dir` swap
- `src/train_slm.py` — add `gcsfs` to `requirements.txt`; HF Trainer writes to GCS natively when the path starts with `gs://`
- `src/data_gen/generate_dataset.py` — change output path to `gs://slm-training-data/generated/`
- `src/evaluate_ragas.py` — write `ragas_results.json` to `gs://slm-artifacts/eval/`
- `src/compare_eval_loss.py` — load checkpoint from `gs://slm-artifacts/experiments/` instead of local path

**Why GCS here:** HuggingFace `datasets`, `transformers.Trainer`, and standard Python file I/O all support `gs://` URIs via `gcsfs` with no architectural change. This is the lowest-friction first step.

---

## Phase 2 — Training on Vertex AI Custom Training

**Effort:** Medium (1–2 days)
**Goal:** Replace `make train` on a local machine with a managed, GPU-accelerated training job. Eliminates hardware dependency; enables spot GPU pricing.

### Layer: Container Image Registry

**Service: Artifact Registry**

All training code is packaged into a Docker image and stored here before submission to Vertex AI.

```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY configs/ ./configs/

ENTRYPOINT ["python", "src/train_slm.py"]
```

Build and push:
```bash
docker build -t us-central1-docker.pkg.dev/YOUR_PROJECT/slm/trainer:latest .
docker push us-central1-docker.pkg.dev/YOUR_PROJECT/slm/trainer:latest
```

**Why Artifact Registry here:** Vertex AI Custom Training requires a container URI in Artifact Registry (or Docker Hub). Storing images here keeps everything within GCP IAM boundaries and enables vulnerability scanning.

### Layer: Training Compute

**Service: Vertex AI Custom Training**

Submit a training job referencing the image above:

```bash
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=slm-lora-finetune \
  --worker-pool-spec=\
machine-type=n1-standard-8,\
accelerator-type=NVIDIA_TESLA_T4,\
accelerator-count=1,\
container-image-uri=us-central1-docker.pkg.dev/YOUR_PROJECT/slm/trainer:latest \
  --env-vars=TRAINING_CONFIG_URI=gs://slm-training-data/configs/default_training.yaml
```

**Config loading change in `src/train_slm.py`:**
```python
# Before
with open("configs/default_training.yaml") as f:
    config = yaml.safe_load(f)

# After
import gcsfs
config_uri = os.getenv("TRAINING_CONFIG_URI", "configs/default_training.yaml")
if config_uri.startswith("gs://"):
    fs = gcsfs.GCSFileSystem()
    with fs.open(config_uri) as f:
        config = yaml.safe_load(f)
else:
    with open(config_uri) as f:
        config = yaml.safe_load(f)
```

**GPU options and spot pricing:**

| GPU | vCPU | RAM | On-demand/hr | Spot/hr | TinyLlama run cost (est.) |
|---|---|---|---|---|---|
| T4 | 8 | 30 GB | $0.35 | $0.11 | ~$0.05–$0.10 |
| L4 | 8 | 30 GB | $0.70 | $0.22 | ~$0.10–$0.20 |
| A100 40GB | 12 | 85 GB | $2.93 | $0.88 | ~$0.40–$0.80 |

Spot instances are safe here because `save_steps=100` in `default_training.yaml` means checkpoints land in GCS every 100 steps. If the job is preempted, resume from the latest checkpoint.

**Why Vertex AI Custom Training here:** It is the managed equivalent of `python src/train_slm.py` — same code, same HF Trainer, but running on a cloud GPU with no local setup. Unlike SageMaker or AzureML, Vertex AI Custom Training accepts any container image and does not require framework-specific wrappers.

### Layer: Experiment and Metric Tracking

**Service: Vertex AI TensorBoard**

TensorBoard logs currently land at `artifacts/experiments/base-run/logs/`. Redirect them to a managed instance:

1. Create a Vertex AI TensorBoard instance in the console
2. In `configs/default_training.yaml`, set `logging_dir: gs://slm-logs/experiments/base-run/`
3. Run the Vertex AI TensorBoard uploader alongside training, or point the Vertex AI Training job at the TensorBoard instance via `--tensorboard` flag

Result: persistent, shareable TensorBoard URLs accessible without running a local server.

**Why Vertex AI TensorBoard here:** Local TensorBoard is ephemeral and only visible on the developer's machine. Vertex AI TensorBoard stores runs permanently and links directly to the Vertex AI Training job that produced them.

---

## Phase 3 — Synthetic Data Generation at Scale

**Effort:** Medium (1–2 days)
**Goal:** Replace the local `python src/data_gen/generate_dataset.py` script with a horizontally scalable, resumable cloud job.

### Layer: Data Generation Compute

**Service: Cloud Run Jobs**

The three phases of data generation map to three Cloud Run Job steps:

**Step 1: Product Sampling (`product_sampler.py`)**
- No external API calls, CPU-only
- Accepts `--shard-index N --total-shards M` to split products across parallel invocations
- Writes `gs://slm-training-data/raw/products/shard-{N}.jsonl`

**Step 2: QA Generation (`qa_generator.py`)**
- Reads product shard from GCS
- Calls Anthropic / Groq API asynchronously (rate limit: 5 concurrent for Anthropic, 2 for Groq)
- Reads API keys from Secret Manager instead of `.env`
- Writes `gs://slm-training-data/raw/qa/shard-{N}.jsonl`

**Step 3: Quality Filtering (`quality_filter.py`)**
- Reads QA shard from GCS
- Calls LLM judge to score each example (threshold: 4/5)
- Writes `gs://slm-training-data/filtered/shard-{N}.jsonl`

**Changes to `generate_dataset.py`:**
```python
# Add CLI args for sharding
parser.add_argument("--shard-index", type=int, default=0)
parser.add_argument("--total-shards", type=int, default=1)

# Replace local output path
output_path = f"gs://slm-training-data/filtered/shard-{args.shard_index}.jsonl"
```

**Job submission (run 10 shards in parallel):**
```bash
for i in $(seq 0 9); do
  gcloud run jobs execute slm-data-gen \
    --args="--shard-index,$i,--total-shards,10" \
    --region=us-central1 &
done
```

**Why Cloud Run Jobs here:** `generate_dataset.py` is already structured as a batch script with async rate limiting. Cloud Run Jobs adds parallel sharding, managed retries, and removes the "laptop must stay awake" constraint without requiring Kubernetes.

### Layer: Dataset Storage and Versioning

**Service: BigQuery**

After all shards are written to GCS, merge and load into BigQuery for queryable, versioned dataset management:

```sql
-- Table: slm_datasets.training_examples
-- Schema:
--   run_id STRING
--   created_at TIMESTAMP
--   instruction STRING
--   input STRING
--   output STRING
--   quality_score FLOAT64
--   source_product_id STRING
--   split STRING  -- 'train' | 'test'

LOAD DATA INTO slm_datasets.training_examples
FROM FILES (
  format = 'NEWLINE_DELIMITED_JSON',
  uris = ['gs://slm-training-data/filtered/*.jsonl']
)
WITH PARTITION COLUMNS (run_id);
```

**Benefits over local JSONL:**
- SQL-based deduplication: `SELECT DISTINCT instruction, output FROM training_examples`
- Version tagging: filter by `run_id` to reproduce any historical training set
- Quality analysis: `SELECT AVG(quality_score), COUNT(*) GROUP BY source_product_id`
- Export to GCS for training: `EXPORT DATA OPTIONS(uri='gs://...', format='NEWLINE_DELIMITED_JSON')`

**Why BigQuery here:** The existing JSONL files have no deduplication, no version history, and no way to query quality distributions. BigQuery is the natural fit for a structured dataset store that a training pipeline can query before each run.

---

## Phase 4 — Evaluation Pipeline

**Effort:** Low–Medium (1 day)
**Goal:** Replace the local Ollama judge and local `ragas_results.json` file with managed evaluation that runs automatically after every training job.

### Layer: RAGAS LLM Judge

**Service: Vertex AI Model Garden**

`src/evaluate_ragas.py` currently hits `http://localhost:11434` (Ollama). Replace with a Vertex AI endpoint:

1. Deploy Llama 3 8B (or Gemma 3 9B) from Model Garden as an online endpoint
2. The endpoint exposes an OpenAI-compatible API — the same interface Ollama uses
3. Update `src/evaluate_ragas.py`:

```python
# Before
from openai import OpenAI
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# After
VERTEX_ENDPOINT = os.getenv("RAGAS_JUDGE_ENDPOINT")  # e.g. https://us-central1-aiplatform.googleapis.com/...
client = OpenAI(base_url=VERTEX_ENDPOINT, api_key=google_auth_token)
```

Alternatively, use **Claude claude-sonnet-4-6 via the Anthropic API** (already integrated in `llm_client.py`) as the judge — eliminates the need for a separate endpoint entirely.

**Why Vertex AI Model Garden here:** Removes the hard dependency on Ollama running locally. The Model Garden endpoint is always available, scales automatically, and does not require the developer's machine.

### Layer: Metric Storage and Trending

**Service: BigQuery**

Replace `artifacts/ragas_results.json` with a BigQuery table:

```sql
-- Table: slm_datasets.experiment_metrics
CREATE TABLE slm_datasets.experiment_metrics (
  run_id            STRING NOT NULL,
  training_job_id   STRING,
  evaluated_at      TIMESTAMP,
  eval_set          STRING,   -- 'dev' | 'test'
  faithfulness      FLOAT64,
  answer_relevancy  FLOAT64,
  factual_correctness FLOAT64,
  semantic_similarity FLOAT64,
  eval_loss         FLOAT64,
  base_model_loss   FLOAT64
);
```

Update `src/evaluate_ragas.py` to write to BigQuery after scoring:

```python
from google.cloud import bigquery
bq = bigquery.Client()
bq.insert_rows_json("YOUR_PROJECT.slm_datasets.experiment_metrics", [row])
```

**Trend query example:**
```sql
SELECT
  run_id,
  evaluated_at,
  faithfulness,
  factual_correctness,
  semantic_similarity,
  (base_model_loss - eval_loss) AS loss_improvement
FROM slm_datasets.experiment_metrics
ORDER BY evaluated_at DESC
LIMIT 20;
```

**Why BigQuery here:** A JSON file cannot be queried across runs. BigQuery enables cross-run regression detection, quality gates (`WHERE faithfulness < 0.75` blocks promotion), and Looker Studio dashboards with zero extra infrastructure.

### Layer: Automated Evaluation Trigger

**Service: Eventarc**

When a training job writes its final adapter weights to GCS, automatically trigger evaluation:

1. Training job writes `gs://slm-artifacts/experiments/{run_id}/adapter_model.bin`
2. Eventarc routes the GCS `finalize` event to a Cloud Run service
3. Cloud Run service runs `src/evaluate_ragas.py` against the new checkpoint
4. Results land in BigQuery `experiment_metrics` table

**Why Eventarc here:** Replaces the manual `make eval-ragas` step. Every training run gets evaluated automatically — no human needs to remember to run it.

---

## Phase 5 — Model Serving

**Effort:** Medium–High (2–3 days)
**Goal:** Replace `python src/run_model.py` (developer's laptop required) with a managed, always-on inference endpoint.

### Layer: Model Versioning

**Service: Vertex AI Model Registry**

After `src/export_merged_model.py` merges the LoRA adapter into the base TinyLlama weights, register the result:

```bash
gcloud ai models upload \
  --region=us-central1 \
  --display-name=tinyllama-mortgage-v1 \
  --artifact-uri=gs://slm-artifacts/exports/base-run-merged/ \
  --container-image-uri=us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.2-1:latest
```

Each registered version is immutable, tagged, and linked back to the training job that produced it. Supports A/B traffic splitting between versions.

**Why Vertex AI Model Registry here:** The current `artifacts/exports/` directory has no version history, no metadata linking a model to its training run, and no promotion workflow. Model Registry provides all three.

### Layer: Inference Serving

**Service: Cloud Run**

For TinyLlama-1.1B (1.1B parameters, ~2.2 GB in fp16), Cloud Run with CPU is sufficient for moderate traffic. No GPU required at this model size.

**Serving container structure:**

```
inference/
├── Dockerfile
├── server.py        ← FastAPI app wrapping run_model.py logic
└── requirements.txt
```

```python
# server.py
from fastapi import FastAPI
from src.scope_classifier import ScopeClassifier
from src.run_model import load_model, generate

app = FastAPI()
classifier = ScopeClassifier()
model, tokenizer = load_model("gs://slm-artifacts/exports/base-run-merged/")

@app.post("/predict")
async def predict(request: dict):
    question = request["question"]
    if not classifier.is_in_scope(question):
        return {"answer": "I can only answer UK mortgage questions.", "in_scope": False}
    answer = generate(model, tokenizer, question)
    return {"answer": answer, "in_scope": True}
```

**Deploy:**
```bash
gcloud run deploy slm-mortgage-api \
  --image=us-central1-docker.pkg.dev/YOUR_PROJECT/slm/inference:latest \
  --region=us-central1 \
  --memory=16Gi \
  --cpu=8 \
  --concurrency=4 \
  --min-instances=0 \
  --max-instances=10
```

**The scope classifier (`src/scope_classifier.py`) co-deploys** inside the same container — no separate service needed. The DistilBERT-MNLI model (~256 MB) loads at container startup and is reused across requests.

**Serving options comparison:**

| Option | Best for | Cold start | Cost at 0 traffic |
|---|---|---|---|
| Cloud Run (CPU) | Demo / low traffic | ~8–15 sec | $0 (scales to zero) |
| Cloud Run (GPU) | Moderate traffic, latency matters | ~20–30 sec | $0 (scales to zero) |
| Vertex AI Online Prediction | Production, monitoring, A/B testing | ~30–60 sec | $0 (scales to zero) |
| GKE + vLLM | High throughput, multi-model | None (always on) | ~$200+/month |

**Recommendation for this project:** Start with **Cloud Run (CPU)**. TinyLlama-1.1B fits in 16 GB RAM, inference takes ~2–5 seconds on 8 vCPUs, and cost is near-zero at demo traffic levels.

**Why Cloud Run here:** It is the direct cloud equivalent of `python src/run_model.py` — the same Python code runs inside a container, with the addition of an HTTP interface, autoscaling, and HTTPS termination.

---

## Phase 6 — End-to-End MLOps Pipeline

**Effort:** High (3–5 days, after Phases 1–5 are complete)
**Goal:** Replace the manual Makefile workflow with a fully automated, version-tracked ML pipeline triggered by new data.

### Layer: Pipeline Orchestration

**Service: Vertex AI Pipelines (Kubeflow Pipelines)**

The complete automated pipeline:

```
[Trigger]
New JSONL files land in gs://slm-training-data/raw/
         │
         ▼ Eventarc → Cloud Run → Vertex AI Pipeline
         │
[Step 1] Data Generation
│  Service: Cloud Run Jobs
│  Input:   gs://slm-training-data/raw/products/
│  Output:  gs://slm-training-data/filtered/*.jsonl
│
[Step 2] Dataset Preparation
│  Service: Vertex AI Pipelines (Python component)
│  Action:  Load filtered JSONL → BigQuery, create train/test split, export to GCS
│  Output:  gs://slm-training-data/splits/{run_id}/train.jsonl
│           gs://slm-training-data/splits/{run_id}/test.jsonl
│
[Step 3] LoRA Fine-Tuning
│  Service: Vertex AI Custom Training
│  Input:   gs://slm-training-data/splits/{run_id}/train.jsonl
│           gs://slm-training-data/configs/default_training.yaml
│  Output:  gs://slm-artifacts/experiments/{run_id}/
│
[Step 4] Evaluation
│  Service: Vertex AI Custom Training (eval job) + Vertex AI Model Garden (judge)
│  Action:  Run evaluate_ragas.py + compare_eval_loss.py
│  Output:  BigQuery slm_datasets.experiment_metrics (row for this run_id)
│
[Step 5] Quality Gate
│  Service: Vertex AI Pipelines (Python component)
│  Condition: faithfulness >= 0.75 AND semantic_similarity >= 0.80
│  On pass → continue to Step 6
│  On fail → notify via email/Slack, stop pipeline
│
[Step 6] Model Export and Registration
│  Service: Vertex AI Custom Training + Vertex AI Model Registry
│  Action:  Run export_merged_model.py, upload to GCS, register in Model Registry
│  Output:  Vertex AI Model version linked to run_id and BigQuery metrics row
│
[Step 7] Deployment
   Service: Cloud Run (or Vertex AI Online Prediction)
   Action:  Deploy new model version; route 10% traffic, monitor, ramp to 100%
   Output:  Live inference endpoint serving the new model
```

### Layer: CI/CD for Container Images

**Service: Cloud Build**

Automatically rebuild and push Docker images when code changes:

```yaml
# cloudbuild.yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'us-central1-docker.pkg.dev/$PROJECT_ID/slm/trainer:$COMMIT_SHA', '-f', 'Dockerfile.train', '.']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'us-central1-docker.pkg.dev/$PROJECT_ID/slm/trainer:$COMMIT_SHA']

triggers:
  - github:
      owner: Xiaoxuan3x
      name: SLM-Training-Local-Environment
      push:
        branch: main
```

**Why Vertex AI Pipelines here:** The Makefile is a linear, manual sequence. Pipelines adds conditional branching (quality gate), automatic retries, input/output lineage tracking, and a visual DAG that shows which data produced which model version.

---

## Complete GCP Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DEVELOPER / CI                                      │
│  git push → Cloud Build → Artifact Registry (trainer:sha, inference:sha)    │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                            │
│                                                                              │
│  ┌─────────────────────────┐    ┌──────────────────────────────────────┐   │
│  │  Cloud Storage (GCS)    │    │  BigQuery                            │   │
│  │                         │    │                                      │   │
│  │  slm-training-data/     │    │  slm_datasets.training_examples      │   │
│  │  ├── raw/               │    │  slm_datasets.experiment_metrics     │   │
│  │  ├── filtered/          │◄──►│                                      │   │
│  │  ├── splits/{run_id}/   │    │  (versioned, queryable, deduplicated)│   │
│  │  └── configs/           │    └──────────────────────────────────────┘   │
│  │                         │                                                │
│  │  slm-artifacts/         │    ┌──────────────────────────────────────┐   │
│  │  ├── experiments/       │    │  Secret Manager                      │   │
│  │  ├── exports/           │    │  ├── anthropic-api-key               │   │
│  │  └── eval/              │    │  └── groq-api-key                    │   │
│  └─────────────────────────┘    └──────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       TRAINING LAYER                                         │
│                                                                              │
│  ┌──────────────────────┐    ┌─────────────────────┐    ┌────────────────┐ │
│  │  Cloud Run Jobs      │    │  Vertex AI Custom   │    │  Vertex AI     │ │
│  │                      │    │  Training           │    │  TensorBoard   │ │
│  │  Data generation     │    │                     │    │                │ │
│  │  (sharded, parallel) │    │  LoRA fine-tuning   │───►│  Loss curves   │ │
│  │                      │    │  T4/L4 GPU          │    │  Metric logs   │ │
│  │  product_sampler.py  │    │  train_slm.py       │    │                │ │
│  │  qa_generator.py     │    │  (spot pricing)     │    └────────────────┘ │
│  │  quality_filter.py   │    └─────────────────────┘                       │
│  └──────────────────────┘                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      EVALUATION LAYER                                        │
│                                                                              │
│  ┌──────────────────────┐    ┌─────────────────────┐    ┌────────────────┐ │
│  │  Vertex AI           │    │  Vertex AI          │    │  BigQuery      │ │
│  │  Custom Training     │    │  Model Garden       │    │                │ │
│  │  (eval job)          │    │                     │    │  experiment_   │ │
│  │                      │    │  Llama 3 / Gemma    │    │  metrics table │ │
│  │  compare_eval_loss   │    │  as RAGAS judge     │───►│                │ │
│  │  evaluate_ragas      │    │  (replaces Ollama)  │    │  (trend query, │ │
│  │                      │    │                     │    │   quality gate)│ │
│  └──────────────────────┘    └─────────────────────┘    └────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    REGISTRY AND SERVING LAYER                                │
│                                                                              │
│  ┌──────────────────────┐    ┌─────────────────────┐    ┌────────────────┐ │
│  │  Vertex AI           │    │  Cloud Run          │    │  Eventarc      │ │
│  │  Model Registry      │    │                     │    │                │ │
│  │                      │    │  inference server   │    │  GCS finalize  │ │
│  │  tinyllama-mortgage  │───►│  (FastAPI)          │    │  → trigger     │ │
│  │  v1, v2, v3...       │    │  scope_classifier   │    │    eval job    │ │
│  │  (linked to run_id)  │    │  + run_model logic  │    │                │ │
│  └──────────────────────┘    └─────────────────────┘    └────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     ORCHESTRATION LAYER                                      │
│                                                                              │
│  Vertex AI Pipelines                                                         │
│                                                                              │
│  [Data Gen] → [Dataset Prep] → [Training] → [Eval] → [Gate] → [Deploy]     │
│       │              │              │           │        │          │        │
│  Cloud Run      BigQuery       Vertex AI    BigQuery  Pass/    Cloud Run     │
│    Jobs          export        Custom         write    Fail     deploy       │
│                               Training                          new ver.     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Roadmap

| Phase | Focus | Key Services | Est. Effort | Priority |
|---|---|---|---|---|
| 1 | Storage + secrets baseline | GCS, Secret Manager | 2–4 hours | **Start here** |
| 2 | GPU training on Vertex AI | Vertex AI Custom Training, Artifact Registry, TensorBoard | 1–2 days | High |
| 3 | Scalable data generation | Cloud Run Jobs, BigQuery | 1–2 days | High |
| 4 | Automated evaluation | Vertex AI Model Garden, BigQuery, Eventarc | 1 day | Medium |
| 5 | Model serving endpoint | Cloud Run, Vertex AI Model Registry | 2–3 days | Medium |
| 6 | Full MLOps pipeline | Vertex AI Pipelines, Cloud Build | 3–5 days | Low (future) |

---

## Files Changed Per Phase

### Phase 1
- `src/data_gen/llm_client.py` — Secret Manager for API keys
- `configs/default_training.yaml` — `output_dir` → `gs://`
- `configs/mps_training.yaml` — `output_dir` → `gs://`
- `src/data_gen/generate_dataset.py` — output path → GCS
- `src/evaluate_ragas.py` — results path → GCS
- `requirements.txt` — add `gcsfs`, `google-cloud-secret-manager`

### Phase 2
- `Dockerfile` (new) — training container
- `src/train_slm.py` — GCS config loading from `TRAINING_CONFIG_URI` env var
- `requirements.txt` — add `google-cloud-aiplatform`

### Phase 3
- `src/data_gen/generate_dataset.py` — add `--shard-index`, `--total-shards` args; GCS output
- `src/data_gen/llm_client.py` — Secret Manager (already done in Phase 1)
- `Dockerfile.datagen` (new) — data generation container

### Phase 4
- `src/evaluate_ragas.py` — replace Ollama base URL with Vertex AI endpoint; write results to BigQuery
- `src/compare_eval_loss.py` — write eval loss to BigQuery `experiment_metrics`
- `requirements.txt` — add `google-cloud-bigquery`

### Phase 5
- `inference/server.py` (new) — FastAPI wrapper around `run_model.py`
- `inference/Dockerfile` (new) — inference container
- `src/run_model.py` — minor: accept model path from `MODEL_URI` env var

### Phase 6
- `pipeline/pipeline.py` (new) — Kubeflow Pipelines DSL
- `cloudbuild.yaml` (new) — Cloud Build trigger config
