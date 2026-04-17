# Comparing the Trained Model With the Base Model

This guide explains how to measure whether a fine-tuning run improved the model compared with the original base model.

Start with one metric only: **eval loss**. Other checks, such as generated-answer review, domain-specific scoring, and safety review, can be added later.

## 1. Compare Eval Loss

**Eval loss** is a common standard benchmark for training performance. It measures how well a model predicts the expected output text on evaluation examples that were not used for training. Lower eval loss is better.

For a fair comparison, evaluate both of these on the same examples:

- **Base model only**: the original model from `model_name`.
- **Fine-tuned model**: the same base model with the saved LoRA adapter applied.

The comparison should use the same:

- dataset split or evaluation file
- tokenizer
- prompt format
- `max_seq_length`
- batch size
- precision and quantisation settings, where possible

### Run the Comparison

After training, run:

```bash
make compare-eval-loss
```

This uses:

- `configs/default_training.yaml` as the config
- `output_dir` from that config as the saved LoRA adapter directory
- the same deterministic 90/10 train/eval split used by `src/train_slm.py`

The Makefile command is equivalent to:

```bash
python src/compare_eval_loss.py --config configs/default_training.yaml
```

To compare a specific adapter directory, pass it explicitly:

```bash
python src/compare_eval_loss.py \
  --config configs/default_training.yaml \
  --adapter-dir artifacts/experiments/base-run
```

The script prints:

```text
Base model eval_loss:       2.4000 | perplexity: 11.02
LoRA adapter eval_loss:     1.6700 | perplexity: 5.31
Delta adapter - base:       -0.7300
Result: the LoRA adapter improved eval loss on this eval split.
```

Interpret the result using `Delta adapter - base`:

- negative delta means the LoRA adapter improved eval loss
- positive delta means the LoRA adapter made eval loss worse
- zero means both models performed the same on this eval split

### Command Options

Use a different config:

```bash
python src/compare_eval_loss.py --config configs/my_experiment.yaml
```

Use a different saved adapter:

```bash
python src/compare_eval_loss.py --adapter-dir artifacts/experiments/my-run
```

Override the base model:

```bash
python src/compare_eval_loss.py --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

Change evaluation batch size:

```bash
python src/compare_eval_loss.py --batch-size 2
```

By default, the script uses the base model recorded in the adapter's `adapter_config.json`. If that file is not available, it falls back to `model_name` from the config.

### Adapter Eval Loss From the Current Training Run

The training script already evaluates the fine-tuned LoRA model during training. The result is saved in the checkpoint `trainer_state.json`.

Example path:

```text
artifacts/experiments/base-run/checkpoint-3/trainer_state.json
```

Look for `eval_loss` inside `log_history`:

```json
{
  "eval_loss": 1.6673108339309692
}
```

This is the eval loss for the fine-tuned model at that checkpoint.

You can convert eval loss into perplexity:

```text
perplexity = exp(eval_loss)
```

For example:

```text
exp(1.6673) ~= 5.3
```

Perplexity is easier to read, but the rule is the same: lower is better.

### Why the Comparison Matters

The fine-tuned model's eval loss is only useful when compared with the base model's eval loss on the same evaluation examples.

For example:

```text
Base model eval loss:       2.40
Fine-tuned model eval loss: 1.67
Result: fine-tuning improved prediction on the eval set
```

Or:

```text
Base model eval loss:       1.70
Fine-tuned model eval loss: 1.67
Result: fine-tuning only made a very small difference
```

Or:

```text
Base model eval loss:       1.55
Fine-tuned model eval loss: 1.67
Result: fine-tuning made the model worse on the eval set
```

### Notes and Limitations

The comparison script evaluates both models on the same split from the configured dataset. With the current small dataset, the eval split may contain only a few examples, so treat the result as an early signal rather than final proof of quality.

If the base model is gated or private, authenticate with Hugging Face before running the command.

If 4-bit loading causes hardware or driver errors, set this in the config and run the command again:

```yaml
quantization:
  load_in_4bit: false
```

## To Be Updated

Future sections should cover:

- generated-answer comparison
- manual scoring rubrics
- domain-specific UK mortgage evaluation examples
- hallucination and safety checks
- overfitting checks
- repeatable evaluation scripts
