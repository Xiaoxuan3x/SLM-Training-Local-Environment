# SLM-Training-Local-Environment

A beginner-friendly starter kit for experimenting with small language model (SLM) training on your laptop or workstation. It packages a LoRA-based fine-tuning script, a configurable YAML file, sample data, and documentation so you can iterate quickly without building the plumbing from scratch.

## Repository layout
- **Core training and inference**
  - `src/train_slm.py` – Hugging Face/PEFT training entry point with LoRA and optional 4-bit loading.
  - `src/compare_eval_loss.py` – Compares evaluation loss for the base model versus a saved LoRA adapter.
  - `src/export_merged_model.py` – Merges a trained LoRA adapter into the base model and saves a standalone Hugging Face model folder.
  - `src/run_model.py` – Loads the merged local model and runs a text-generation smoke test.

- **Synthetic data generation**
  - `src/data_gen/generate_dataset.py` – Main synthetic dataset generation pipeline.
  - `src/data_gen/qa_generator.py` – Builds instruction/response style QA examples.
  - `src/data_gen/llm_client.py` – Wraps the LLM provider calls used during generation.
  - `src/data_gen/product_sampler.py` – Samples source product or domain inputs for generation.
  - `src/data_gen/quality_filter.py` – Applies quality filtering to generated examples.
  - `src/data_gen/__init__.py` – Package marker for the data generation module.

- **Configuration**
  - `configs/default_training.yaml` – Central configuration for model, dataset, LoRA, and quantisation settings.
  - `configs/data_generation.yaml` – Configuration for synthetic data generation and filtering.

- **Sample and generated data**
  - `data/sample_dataset.jsonl` – Minimal toy instruction dataset showing the expected JSONL schema.
  - `data/uk_mortgage_dataset.jsonl` – Domain dataset used for mortgage-focused fine-tuning experiments.
  - `data/synthetic_mortgage_dataset.jsonl` – Synthetic mortgage QA dataset produced by the generation pipeline.

- **Documentation**
  - `docs/environment_setup_tutorial.md` – Step-by-step walkthrough for bringing the environment online.
  - `docs/small_language_model_techniques.md` – Reference guide describing the SLM methodology used in this project.
  - `docs/model_comparison_techniques.md` – Notes on comparing the fine-tuned adapter against the base model.
  - `docs/model_export_workflow.md` – How to export a merged Hugging Face model and optionally convert it to GGUF for Ollama.
  - `docs/synthetic_data_generation.md` – Workflow and configuration guide for synthetic dataset creation.

- **Project tooling**
  - `Makefile` – Convenience targets for install, training, evaluation, export, generation, and local smoke tests.
  - `requirements.txt` – Python dependencies tested on Python 3.10+.
  - `README.md` – Project overview, quick start, and usage notes.

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
5. **Export a standalone Hugging Face model**
   ```bash
   make export-merged
   ```
The merged model folder is written to `artifacts/exports/base-run-merged/`. Move or deploy the whole folder, not only `model.safetensors`.

6. **Run the exported model locally**
   ```bash
   make run-model
   # or
   python src/run_model.py --prompt-file prompt.txt
   ```
This loads `artifacts/exports/base-run-merged/` and runs a generation smoke test using the default prompt embedded in `src/run_model.py`.

To test a different prompt without editing code, update the `DEFAULT_PROMPT` constant in `src/run_model.py` and rerun the same command.

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
- **Export a runnable model**: Run `make export-merged` to merge the LoRA adapter into the base model and save a standalone Hugging Face model folder.
- **Test the exported model**: Run `make run-model`, or call `python src/run_model.py --prompt-file prompt.txt` for a custom prompt.
- **Deploy outside this repo**: Archive and move the full `artifacts/exports/base-run-merged/` folder. See `docs/model_export_workflow.md`.
- **Create a GGUF/Ollama model**: Convert the merged Hugging Face export to GGUF only when you need Ollama, LM Studio, or `llama.cpp`. See `docs/model_export_workflow.md`.
- **Change the base model**: Swap `model_name` for another instruction-tuned checkpoint that fits your VRAM budget. Adjust `lora.target_modules` accordingly.
- **Disable quantisation**: Set `quantization.load_in_4bit` to `false` if bitsandbytes or GPU drivers are unavailable.
- **Avoid Hugging Face network checks when cached**: Run `python src/train_slm.py --config configs/default_training.yaml --local-files-only`.
- **Automate installs**: `make install` provisions the `.venv` and installs requirements end-to-end.

## Documentation
- **Environment tutorial**: `docs/environment_setup_tutorial.md` – extended instructions, troubleshooting notes, and cleanup steps.
- **Technique reference**: `docs/small_language_model_techniques.md` – overview of the LoRA → domain tuning → distillation → quantisation pipeline selected for this project.
- **Model comparison**: `docs/model_comparison_techniques.md` – first steps for comparing fine-tuned LoRA results against the original base model.
- **Model export workflow**: `docs/model_export_workflow.md` – how to run the Hugging Face export in a fresh local environment and optionally convert it to GGUF for Ollama.

Feel free to open issues or extend the repo with evaluation notebooks, dataset builders, or deployment scripts as you deepen your SLM experiments.
