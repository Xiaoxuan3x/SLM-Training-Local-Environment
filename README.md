# SLM-Training-Local-Environment

A beginner-friendly starter kit for experimenting with small language model (SLM) training on your laptop or workstation. It packages a LoRA-based fine-tuning script, a configurable YAML file, sample data, and documentation so you can iterate quickly without building the plumbing from scratch.

## Key Components

### 1. Core Training Pipeline
- `src/train_slm.py` – LoRA fine-tuning entry point using Hugging Face + PEFT on TinyLlama-1.1B.
- `configs/default_training.yaml` / `configs/mps_training.yaml` – Full config control over LoRA rank, learning rate, quantisation, dataset path, and system prompt.
- Apple Silicon MPS support: disables CUDA-only 4-bit path, auto-detects device.

### 2. Synthetic Data Generation
- `src/data_gen/` – Full pipeline: `generate_dataset.py`, `qa_generator.py`, `quality_filter.py`, `product_sampler.py`, `llm_client.py`.
- `data/synthetic_mortgage_dataset.jsonl` – Teacher-LLM-generated QA pairs (knowledge distillation approach).
- `data/uk_mortgage_dataset.jsonl` – Curated domain dataset.

### 3. Domain Boundary Control (3-layer stack)

| Layer | Component | Status |
|---|---|---|
| 1st gate | DistilBERT NLI zero-shot scope classifier (`src/scope_classifier.py`) | Done |
| 2nd gate | System prompt in training format | Done |
| 3rd gate | Boundary training examples (`data/boundary_examples.jsonl`, 40 examples) | Done, limited coverage |
| Moderation API | Harmful content filter before SLM | TODO |

### 4. Evaluation
- `src/compare_eval_loss.py` – Compares base model vs LoRA adapter on the same eval split, prints eval loss delta and perplexity.
- `src/evaluate_ragas.py` – Runs RAGAS semantic evaluation (Faithfulness, Answer Relevancy, Factual Correctness, Semantic Similarity) using a local Ollama model as the LLM judge. No API key required.

### 5. Export & Deployment
- `src/export_merged_model.py` – Merges LoRA adapter into base model, saves standalone HF folder.
- `src/run_model.py` – Local inference with optional scope classifier gate.
- GGUF/Ollama conversion path documented in `docs/model_export_workflow.md`.

## Quick start
1. **Create a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # Windows: .venv\\Scripts\\Activate.ps1
   ```
2. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. **Kick off the sample run**
   ```bash
   make train
   # or
   python src/train_slm.py --config configs/default_training.yaml
   ```
Artifacts (adapter weights + tokenizer) are written to `artifacts/experiments/base-run/`. For a slower but CPU-friendly pass, set `quantization.load_in_4bit: false` in the config before running.

For Apple Silicon training, use the MPS-safe config:
```bash
python src/train_slm.py --config configs/mps_training.yaml
```
That config disables CUDA-only 4-bit loading and mixed precision defaults that do not apply to MPS.

4. **Compare the adapter with the base model**
   ```bash
   make compare-eval-loss
   ```
5. **Run RAGAS semantic evaluation** (requires Ollama running with a judge model)
   ```bash
   ollama pull llama3.1        # first time only
   make eval-ragas
   # or with options:
   python src/evaluate_ragas.py --max-samples 20 --judge-model mistral
   ```
   See `docs/evaluation_benchmark_guide.md` for a full description of all evaluation methods.

6. **Export a standalone Hugging Face model**
   ```bash
   make export-merged
   ```
The merged model folder is written to `artifacts/exports/base-run-merged/`. Move or deploy the whole folder, not only `model.safetensors`.

7. **Run the exported model locally**
   ```bash
   make run-model
   # or
   python src/run_model.py --prompt "What is LTV?" --use-classifier --allow-downloads
   ```
This loads `artifacts/exports/base-run-merged/` and gates every question through the scope classifier before the SLM is called. `make run-model` enables the classifier automatically. Pass `--allow-downloads` on first run to fetch the classifier model (~256 MB); omit it afterwards.

Need more detailed screenshots or OS-specific notes? See `docs/environment_setup_tutorial.md` for the full guide.

For a clean local deployment guide, including a brand-new environment and optional GGUF/Ollama route, see `docs/model_export_workflow.md`.

## Configuring the environment
All knobs live inside `configs/default_training.yaml`. The major sections are:

- **Top-level**
  - `project_name`: Used when naming artifact folders.
  - `model_name`: Any causal Hugging Face model (e.g., `TinyLlama/TinyLlama-1.1B-Chat-v1.0`).
  - `output_dir`: Where checkpoints and adapters are saved.
  - `max_seq_length`: Tokenization length cap.
  - `seed`: Controls shuffling and splits.

- **`dataset`**
  - `path`: Local path to a JSONL file with `instruction`, `input`, and `output` fields.
  - `field_names`: Map your column names to what the script expects.
  - `max_samples`: Optional quick cap for smoke tests.

- **`training`**
  - `epochs`, `batch_size`, `micro_batch_size`: Standard Hugging Face Trainer knobs. Gradient accumulation steps are computed from these values.
  - `learning_rate`, `warmup_ratio`, `logging_steps`, `eval_steps`, `save_steps`: Control optimisation and reporting cadence.
  - `fp16`, `gradient_checkpointing`: Enable if your hardware supports mixed precision or checkpointing to fit longer sequences.

- **`lora`**
  - `r`, `alpha`, `dropout`, `bias`: LoRA hyperparameters.
  - `target_modules`: Attention modules to adapt. Adjust to match the architecture of the chosen base model.

- **`quantization`**
  - `load_in_4bit`: Enable to load the base model with bitsandbytes 4-bit weights. Set to `false` for CPU-only environments.
  - `bnb_4bit_compute_dtype`, `bnb_4bit_use_double_quant`: Fine-tune 4-bit behaviour.

- **Apple Silicon / MPS**
  - The training script now detects `mps` automatically when available.
  - Use `configs/mps_training.yaml` as the starting point for Mac training.
  - Keep `quantization.load_in_4bit: false` on MPS because the 4-bit path in this repo is CUDA-only.

Create additional YAML files under `configs/` for different experiments and pass them through `--config`.

## Usage patterns
- **Train on your own data**: Copy your JSONL file into `data/`, update `dataset.path`, and re-run `make train`.
- **Compare eval loss**: Run `make compare-eval-loss` after training to compare the base model against the saved LoRA adapter.
- **Run RAGAS evaluation**: Run `make eval-ragas` (Ollama must be running) to measure Faithfulness, Answer Relevancy, Factual Correctness, and Semantic Similarity on the exported model. See `docs/evaluation_benchmark_guide.md`.
- **Export a runnable model**: Run `make export-merged` to merge the LoRA adapter into the base model and save a standalone Hugging Face model folder.
- **Test the exported model**: Run `make run-model`, or call `python src/run_model.py --prompt "..." --use-classifier --allow-downloads` for a custom prompt.
- **Gate off-topic questions**: Pass `--use-classifier` to reject questions outside the UK mortgage domain before the SLM loads. See `docs/domain_boundary_guide.md`.
- **Deploy outside this repo**: Archive and move the full `artifacts/exports/base-run-merged/` folder. See `docs/model_export_workflow.md`.
- **Create a GGUF/Ollama model**: Convert the merged Hugging Face export to GGUF only when you need Ollama, LM Studio, or `llama.cpp`. See `docs/model_export_workflow.md`.
- **Change the base model**: Swap `model_name` for another instruction-tuned checkpoint that fits your VRAM budget. Adjust `lora.target_modules` accordingly.
- **Disable quantisation**: Set `quantization.load_in_4bit` to `false` if bitsandbytes or GPU drivers are unavailable.
- **Avoid Hugging Face network checks when cached**: Run `python src/train_slm.py --config configs/default_training.yaml --local-files-only`.
- **Automate installs**: `make install` provisions the `.venv` and installs requirements end-to-end.

## Documentation
- **Environment tutorial**: `docs/environment_setup_tutorial.md` – extended instructions, troubleshooting notes, and cleanup steps.
- **Technique reference**: `docs/small_language_model_techniques.md` – overview of the LoRA → domain tuning → distillation → quantisation pipeline selected for this project.
- **Evaluation benchmark guide**: `docs/evaluation_benchmark_guide.md` – all evaluation methods in this repo: eval loss comparison, perplexity, and RAGAS semantic metrics (Faithfulness, Answer Relevancy, Factual Correctness, Semantic Similarity).
- **Model comparison**: `docs/model_comparison_techniques.md` – first steps for comparing fine-tuned LoRA results against the original base model.
- **Model export workflow**: `docs/model_export_workflow.md` – how to run the Hugging Face export in a fresh local environment and optionally convert it to GGUF for Ollama.
- **Domain boundary guide**: `docs/domain_boundary_guide.md` – how the scope classifier, system prompt, and boundary training examples combine to restrict the model to the UK mortgage domain.

Feel free to open issues or extend the repo with evaluation notebooks, dataset builders, or deployment scripts as you deepen your SLM experiments.
