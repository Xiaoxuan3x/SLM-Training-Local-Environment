# SLM-Training-Local-Environment

A beginner-friendly starter kit for experimenting with small language model (SLM) training on your laptop or workstation. It packages a LoRA-based fine-tuning script, a configurable YAML file, sample data, and documentation so you can iterate quickly without building the plumbing from scratch.

## Repository layout
- `src/train_slm.py` – Hugging Face/PEFT training entry point with LoRA + optional 4-bit loading.
- `src/compare_eval_loss.py` – Eval-loss comparison for the original base model versus a saved LoRA adapter.
- `configs/default_training.yaml` – Central configuration for model, data, LoRA, and quantisation settings.
- `data/sample_dataset.jsonl` – Two toy instruction/response pairs that demonstrate the expected JSONL schema.
- `requirements.txt` – Python dependencies tested on Python 3.10+.
- `Makefile` – Convenience targets for installing dependencies and launching training.
- `docs/environment_setup_tutorial.md` – Step-by-step walkthrough for bringing the environment online.
- `docs/small_language_model_techniques.md` – Reference guide (provided by you) describing the SLM methodology being followed.

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

Need more detailed screenshots or OS-specific notes? See `docs/environment_setup_tutorial.md` for the full guide.

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

Create additional YAML files under `configs/` for different experiments and pass them through `--config`.

## Usage patterns
- **Train on your own data**: Copy your JSONL file into `data/`, update `dataset.path`, and re-run `make train`.
- **Compare eval loss**: Run `make compare-eval-loss` after training to compare the base model against the saved LoRA adapter.
- **Change the base model**: Swap `model_name` for another instruction-tuned checkpoint that fits your VRAM budget. Adjust `lora.target_modules` accordingly.
- **Disable quantisation**: Set `quantization.load_in_4bit` to `false` if bitsandbytes or GPU drivers are unavailable.
- **Automate installs**: `make install` provisions the `.venv` and installs requirements end-to-end.

## Documentation
- **Environment tutorial**: `docs/environment_setup_tutorial.md` – extended instructions, troubleshooting notes, and cleanup steps.
- **Technique reference**: `docs/small_language_model_techniques.md` – overview of the LoRA → domain tuning → distillation → quantisation pipeline selected for this project.
- **Model comparison**: `docs/model_comparison_techniques.md` – first steps for comparing fine-tuned LoRA results against the original base model.

Feel free to open issues or extend the repo with evaluation notebooks, dataset builders, or deployment scripts as you deepen your SLM experiments.
