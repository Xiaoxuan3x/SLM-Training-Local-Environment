# Beginner Environment Tutorial

This walkthrough shows how to bring up the small-language-model (SLM) training environment on macOS, Linux, or WSL. It covers prerequisites, Python environment creation, dependency installation, and running the starter training script.

## 1. Prerequisites
- **Git** for cloning this repository.
- **Python 3.10+** with `venv` module available.
- **Hardware**: CPU-only works for the sample run. A CUDA-capable GPU with at least 12 GB VRAM is recommended for real fine-tuning.
- **Optional**: NVIDIA drivers + CUDA 12.x toolkit if you plan to train on a GPU. Verify with `nvidia-smi` on Linux/WSL.
- **Apple Silicon**: MPS training is supported by the starter script, but use the dedicated MPS config instead of the CUDA-oriented default config.

## 2. Clone or update the repository
```bash
git clone <your-fork-url> slm-training
cd slm-training
```
If the repo already exists locally, pull the latest changes instead of cloning.

## 3. Create an isolated Python environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows (PowerShell): .venv\\Scripts\\Activate.ps1
```
Activating the virtual environment ensures this project’s dependencies do not interfere with global packages.

## 4. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
CPU-only users can keep `bitsandbytes` installed; it will gracefully fall back. If you hit GPU driver issues, uninstall it with `pip uninstall bitsandbytes` and set `load_in_4bit: false` inside `configs/default_training.yaml`.

On Apple Silicon, use `configs/mps_training.yaml`. It disables 4-bit loading and CUDA-only mixed precision defaults.

## 5. (Optional) GPU acceleration checks
1. Confirm the OS sees your GPU: `nvidia-smi`.
2. Ensure CUDA libraries match the PyTorch build. If you need to reinstall PyTorch, follow the selector at https://pytorch.org/get-started/locally/ and then rerun `pip install -r requirements.txt`.
3. Export `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64` to reduce fragmentation on smaller GPUs.

## 6. Understand the sample data format
The repository ships with `data/sample_dataset.jsonl` containing two toy instruction/response pairs:
```jsonl
{"instruction": "...", "input": "", "output": "..."}
```
To train on your own data, keep the same keys (`instruction`, `input`, `output`). Place the file anywhere and update `dataset.path` in your YAML config. For quick experiments you can duplicate the sample file and add more lines.

## 7. Configure your first run
Review `configs/default_training.yaml` and adjust:
- `model_name` for a different Hugging Face base model.
- `dataset.path` for your dataset location.
- `training.batch_size`, `micro_batch_size`, `epochs`, and `learning_rate`.
- `lora.target_modules` if the architecture uses different attention module names.
- Set `quantization.load_in_4bit` to `false` on CPU-only hardware.

If you are training on Apple Silicon, start from `configs/mps_training.yaml` instead of `configs/default_training.yaml`.

## 8. Launch a smoke test
With the virtual environment active:
```bash
make train
# or
python src/train_slm.py --config configs/default_training.yaml
```
For Apple Silicon:
```bash
python src/train_slm.py --config configs/mps_training.yaml
```
The script performs the following:
1. Loads the tokenizer/model.
2. Applies a LoRA adapter (optionally with 4-bit weights).
3. Tokenizes the dataset and splits 90/10 for eval.
4. Runs a short training loop with Hugging Face `Trainer`.
Artifacts land in `artifacts/experiments/base-run/`.

## 9. Monitor and iterate
- Training logs appear on stdout; redirect to a file if needed.
- Intermediate checkpoints are saved every `save_steps` steps. Delete the folder between runs to start clean.
- Use `tensorboard --logdir artifacts/experiments` if you enable TensorBoard logging inside the config.

## 10. Export a standalone model
Training saves a LoRA adapter in:

```text
artifacts/experiments/base-run/
```

That adapter is small, but it is not a standalone model. To create a runnable
Hugging Face model folder, merge the adapter into the base model:

```bash
make export-merged
```

The merged export is written to:

```text
artifacts/exports/base-run-merged/
```

Move or archive the whole folder when sharing the model. The file
`model.safetensors` contains the merged weights, but the tokenizer and config
files in the same folder are also required.

See `docs/model_export_workflow.md` for the difference between LoRA adapters,
merged Hugging Face exports, and optional GGUF runtime exports.

## 11. Run a local inference test
After exporting, run a smoke test from the merged model folder:

```bash
make run-model
```

To test your own prompt:

```bash
python src/run_model.py --prompt-file prompt.txt
```

The runner loads `artifacts/exports/base-run-merged/` by default and does not
attempt Hugging Face downloads unless you pass `--allow-downloads`.

## 12. Clean up
Deactivate the environment with `deactivate`. To remove everything generated by the starter run:
```bash
make clean
```

With this workflow you can repeatedly tweak configs, swap datasets, and retrain without rebuilding the entire environment.
