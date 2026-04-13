# Small Language Model Techniques

A practical technical guide to the main methods used to build small language models (SLMs) from open-source pretrained language models.

## Overview

In most real-world projects, a small language model is not trained from scratch. A more practical approach is to begin with an open-source pretrained base model and then apply a small set of targeted techniques to adapt it to a business domain, improve task performance, and reduce deployment cost.

These techniques do not all belong to the same stage of development. Some are primarily used during training and adaptation, while others are mainly used later to improve serving efficiency. Understanding that distinction is important, because an effective SLM workflow is not just a list of methods, but a sequence of decisions made for different engineering purposes.

In practice, the most common goals are to make the model more useful for a specific task, strengthen its performance within a limited parameter budget, and make it feasible to run on available infrastructure.

## Recommended Practical Pipeline

A common implementation sequence looks like this:

```text
1. Choose an open-source base model
2. Apply efficient fine-tuning
3. Fine-tune on domain-specific data
4. Use distillation if stronger capability is still needed (optional)
5. Apply quantisation for deployment
6. Use pruning only if further compression is required (optional)
```

This order is common because each step addresses a different problem. Efficient fine-tuning keeps adaptation affordable. Domain-specific tuning makes the model useful for the target business environment. Distillation can improve capability when a small model still underperforms. Quantisation reduces memory and serving cost. Pruning is usually left until later because it is more selective and often less predictable in practice.

## 1. LoRA and Efficient Fine-Tuning

**Category:** Training / Adaptation

**LoRA and related parameter-efficient fine-tuning methods** are used to adapt a pretrained model without updating all of its weights. Instead of retraining the full model, only a much smaller set of trainable parameters is introduced and optimized. This greatly reduces memory requirements, training cost, and experiment time.

This is one of the most common starting points in SLM development because many use cases do not require a full model rewrite. If a base model already has decent general language ability, efficient fine-tuning is often enough to adapt it to a new task, response style, or workflow. It is especially valuable when compute resources are limited or when a team wants to maintain several task-specific variants on top of the same base model.

In practical engineering terms, LoRA is often the first method tried when adapting an open-source model for tasks such as customer support, instruction following, response formatting, or internal assistant behavior. It offers one of the best cost-to-impact ratios among SLM techniques.

> **Implementation example – LoRA adapters in this repo:** `src/train_slm.py` builds a `LoraConfig` and wraps the base model with PEFT before training. Minimal version:

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora_cfg)
```

## 2. Domain-Specific Fine-Tuning

**Category:** Training / Adaptation

**Domain-specific fine-tuning** means continuing training on data from a particular field such as banking, healthcare, legal services, insurance, telecom, or customer support. The purpose is to make the model better at handling specialist terminology, document patterns, workflow conventions, and expected response structures within that domain.

A general-purpose model may be fluent, but still weak in a specialist environment because it does not reliably understand the language, categories, or priorities of the business context. Domain tuning addresses this problem by aligning the model more closely with the target use case.

This step is often what gives a small model its real business value. A smaller model that has been carefully aligned to a narrow domain can perform very well on that domain, sometimes more effectively than a larger general-purpose model that has not been specialized. Typical training data may include support conversations, policy documents, internal FAQs, ticket histories, or labelled intent-response pairs.

> **Implementation example – JSONL + YAML mapping:** Store curated dialogs and point `configs/default_training.yaml -> dataset.path` at the file.

```jsonl
{"instruction": "Handle a lost-card report", "input": "Card #8742 blocked?", "output": "I've locked the card and issued a replacement."}
{"instruction": "Escalate outage ticket", "input": "Incident-192", "output": "Routing to Tier-2 and posting ETA."}
```

```yaml
dataset:
  path: data/support_tickets.jsonl
  field_names:
    instruction: instruction
    input: input
    output: output
```

## 3. Knowledge Distillation

**Category:** Training / Capability Improvement

**Knowledge distillation** is the process of training a smaller student model using supervision from a stronger teacher model. The teacher may be a larger open-source model, a stronger internal model, or a hosted system used to generate better answers, labels, or synthetic training examples.

The purpose of distillation is to improve the capability of the smaller model without increasing its size. This is useful when a model has already been fine-tuned but still lacks answer quality, consistency, instruction-following ability, or robustness. Instead of making the model larger, distillation tries to transfer part of the stronger model's behavior into the smaller one.

In practice, this may involve training the student on teacher-generated outputs, synthetic instruction-response pairs, or other forms of guided supervision. Distillation is not always required, which is why it is often treated as an optional stage. It becomes especially valuable when deployment constraints make larger models impractical, but stronger capability is still needed.

**Optional step:** Use distillation when the tuned small model is still not strong enough and a stronger teacher model is available.

> **Implementation example – teacher-guided supervision:** Generate synthetic pairs with a larger teacher, then fine-tune the student.

```python
synthetic = []
for prompt in domain_prompts:
    teacher_answer = teacher_model.generate(prompt)
    synthetic.append({"instruction": prompt, "input": "", "output": teacher_answer})

Dataset.from_list(synthetic).to_json("data/distilled_pairs.jsonl")
```

## 4. Quantisation

**Category:** Compression / Efficiency

**Quantisation** reduces the numerical precision used to store model weights, typically moving from higher-precision formats to lower-bit representations such as 8-bit or 4-bit. The main goal is to reduce memory usage and make inference cheaper and easier to run.

In most SLM projects, quantisation is a deployment technique rather than a training technique. Once the model reaches an acceptable quality level, quantisation helps make it feasible to serve under real infrastructure constraints such as GPU memory limits, lower-cost hardware, or high-volume inference requirements.

This is one of the most common deployment optimizations in practice because it directly improves operational feasibility. It is widely used when a model must fit into tighter VRAM or RAM budgets, or when inference cost needs to be reduced without retraining the entire system. Although quantisation can introduce some quality trade-offs, it is often the standard next step after training and adaptation.

> **Implementation example – bitsandbytes 4-bit loading:** Controlled via `quantization` in `configs/default_training.yaml`.

```python
bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(base_model_name, quantization_config=bnb_cfg)
model = prepare_model_for_kbit_training(model)
```

## 5. Model Pruning

**Category:** Compression / Optimization

**Model pruning** removes less important parts of a model in order to reduce size or computation. Depending on the pruning strategy, this may involve removing weights, neurons, attention heads, or other structural components.

Pruning can be useful when additional compression is still needed after quantisation, but it is usually not the first optimization method teams apply. Compared with quantisation, pruning often requires more careful validation because the relationship between compression and performance can be less predictable. Some pruning methods may reduce size without delivering a meaningful real-world speedup, while others may affect task quality more sharply than expected.

For that reason, pruning is generally treated as a later-stage optimization rather than a default baseline technique.

**Optional step:** Use pruning only when further compression or compute reduction is still required after simpler techniques have already been applied.

> **Implementation example – structured pruning before export:** Run against the saved checkpoint in `artifacts/`.

```python
from optimum.intel.neural_compressor import INCQuantizer

pruned = INCQuantizer.prune(
    model,
    pruning_config={"target_sparsity": 0.3, "pattern": "m4n2"}
)
pruned.save_pretrained("artifacts/pruned-run")
```

## How These Techniques Fit Together

These techniques should not be treated as a flat checklist, because they solve different engineering problems at different stages of the pipeline.

LoRA and other efficient fine-tuning methods are mainly used to make adaptation affordable. Domain-specific fine-tuning is used to align the model with the target business environment. Knowledge distillation is used to strengthen a smaller model when direct tuning alone is not enough. Quantisation is mainly used to reduce memory footprint and serving cost. Pruning is a more selective optimization step for cases where even more compression is needed.

In practice, the most widely used techniques are efficient fine-tuning, domain-specific tuning, distillation, and quantisation. Pruning remains useful, but is often more optional and later-stage.

## Example Application Flow

A typical SLM project might begin by selecting an open-source base model that matches the required size, license, language coverage, and hardware constraints. The team then applies LoRA or another efficient fine-tuning method to adapt the model cheaply. After that, the model is trained on domain-specific data such as support conversations, policy documents, internal FAQs, or labelled task examples so that it becomes more useful in the target environment.

If the resulting model is still too weak, a stronger teacher model can be used to generate supervision for distillation. That step is optional, but often valuable when the quality gap remains too large. Once the model reaches acceptable performance, quantisation is typically applied to make deployment more feasible. Only if infrastructure limits are still too tight does pruning become worth considering.

## Practical Pipeline Diagram

```text
Open-source pretrained model
        ↓
LoRA / efficient fine-tuning
        ↓
Domain-specific fine-tuning
        ↓
Knowledge distillation (optional, if capability is still insufficient)
        ↓
Quantisation
        ↓
Model pruning (optional, if further compression is needed)
```

## Conclusion

The usual path from an open-source large language model to a practical small language model is not to start from zero, but to apply a small set of techniques in the right order. In most cases, that means choosing a capable base model, adapting it efficiently, specializing it to the domain, improving it through distillation only if necessary, and then compressing it for deployment.

The main value of these techniques is not only what they do individually, but how they work together across the development lifecycle. Efficient fine-tuning and domain tuning improve usefulness. Distillation improves capability when needed. Quantisation and pruning make the final system more realistic to deploy.
