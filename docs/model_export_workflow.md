# Model Export and Local Deployment Workflow

This guide shows two clear ways to run the exported model locally.

- **Route 1: Hugging Face / Transformers** runs the existing folder at
  `artifacts/exports/base-run-merged/`.
- **Route 2: GGUF / Ollama** converts that same exported folder into a GGUF file
  and registers it with Ollama.

Use Route 1 as the source-of-truth workflow for Python, evaluation, and further
ML work. Use Route 2 when you want a compact local runtime model for Ollama,
LM Studio, or `llama.cpp`.

## Starting Point

After training and merging, this folder should exist:

```text
artifacts/exports/base-run-merged/
  model.safetensors
  config.json
  tokenizer.json
  tokenizer_config.json
  generation_config.json
  chat_template.jinja
```

The whole folder is the runnable Hugging Face model. The file
`model.safetensors` contains the merged weights, but it is not enough on its own
because the model also needs `config.json` and tokenizer files.

If the folder does not exist yet, create it:

```bash
make export-merged
```

## Route 1: Run the Hugging Face Export Locally

Use this route when you want to run the model with Python and `transformers`.

### 1. Run From This Repo

If you are inside this repository and have already run `make export-merged`, use
the project runner:

```bash
make run-model
```

That command runs:

```bash
python src/run_model.py
```

The runner loads this path by default:

```text
artifacts/exports/base-run-merged/
```

For a custom prompt:

```bash
python src/run_model.py --prompt-file prompt.txt
```

The runner uses local files only unless you pass `--allow-downloads`.

### 2. Create a Portable Model Archive

From this repo:

```bash
tar -czf artifacts/exports/base-run-merged.tar.gz \
  -C artifacts/exports \
  base-run-merged
```

Move this archive to the target machine:

```text
artifacts/exports/base-run-merged.tar.gz
```

### 3. Create a Fresh Local Environment

On the target machine:

```bash
mkdir slm-local-run
cd slm-local-run
tar -xzf /path/to/base-run-merged.tar.gz

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install torch transformers sentencepiece safetensors accelerate
```

After unpacking, the model folder should be:

```text
slm-local-run/base-run-merged/
```

### 4. Run a Smoke Test Without This Repo

Use this fallback when the new machine only has `base-run-merged.tar.gz` and does
not have this repository or `src/run_model.py`.

Run this from the fresh environment:

```bash
python - <<'PY'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = "base-run-merged"
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"

prompt = """### Instruction:
Summarise the mortgage product and highlight the main lending terms.

### Input:
Provider: Halifax
Mortgage name: 5 Year Fixed Remortgage
Interest rate: 4.65%
Maximum LTV: 75%
Term type: Fixed
Length: 5 years
Booking fee: £999
APRC: 5.80%
Notes: Free valuation and standard legal work included.

### Response:
"""

tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    local_files_only=True,
    low_cpu_mem_usage=True,
)
model.to(device)
model.eval()

inputs = tokenizer(prompt, return_tensors="pt").to(device)
input_length = inputs["input_ids"].shape[-1]
with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_length=input_length + 160,
        do_sample=False,
        repetition_penalty=1.08,
        pad_token_id=tokenizer.eos_token_id,
    )

text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text[len(prompt):].strip() if text.startswith(prompt) else text)
PY
```

Expected result: the model should produce a short mortgage summary using the
fields in the prompt.

## Route 2: Convert to GGUF and Run With Ollama

Use this route when you want a local Ollama model. The flow is:

```text
Hugging Face merged folder -> GGUF F16 -> GGUF Q4_K_M -> Ollama model
```

The quantized `Q4_K_M` file is much smaller than the full `float32`
`model.safetensors` export and is usually better for local CPU inference.

### 1. Install Ollama

Install Ollama from:

```text
https://ollama.com/download
```

Check that the CLI is available:

```bash
ollama --version
```

On macOS, the Ollama app usually starts the local service. On Linux or a headless
machine, start it manually in a separate terminal if needed:

```bash
ollama serve
```

### 2. Build llama.cpp Tools

The conversion and quantization tools come from `llama.cpp`.

From this repo root:

```bash
mkdir -p external artifacts/exports/gguf
git clone https://github.com/ggml-org/llama.cpp external/llama.cpp

cd external/llama.cpp
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

cmake -B build
cmake --build build --config Release -j
```

The build should create:

```text
external/llama.cpp/build/bin/llama-quantize
```

### 3. Convert the Hugging Face Export to GGUF

Still inside `external/llama.cpp`:

```bash
python convert_hf_to_gguf.py \
  ../../artifacts/exports/base-run-merged \
  --outfile ../../artifacts/exports/gguf/base-run-f16.gguf \
  --outtype f16
```

This creates an unquantized or lightly quantized GGUF file:

```text
artifacts/exports/gguf/base-run-f16.gguf
```

### 4. Quantize to Q4_K_M

Still inside `external/llama.cpp`:

```bash
./build/bin/llama-quantize \
  ../../artifacts/exports/gguf/base-run-f16.gguf \
  ../../artifacts/exports/gguf/base-run-Q4_K_M.gguf \
  Q4_K_M
```

This creates the Ollama-friendly runtime file:

```text
artifacts/exports/gguf/base-run-Q4_K_M.gguf
```

### 5. Create an Ollama Modelfile

Return to the repo root:

```bash
cd ../..
mkdir -p artifacts/exports/ollama
```

Create this file:

```text
artifacts/exports/ollama/Modelfile
```

Content:

```text
FROM ../gguf/base-run-Q4_K_M.gguf

PARAMETER temperature 0
PARAMETER repeat_penalty 1.08
PARAMETER num_ctx 2048

TEMPLATE """{{ .Prompt }}{{ .Response }}"""
```

Why this template is simple: this model was trained on prompts that already
contain sections such as `### Instruction:`, `### Input:`, and `### Response:`.
The template passes your prompt through directly instead of wrapping it in a
different chat format.

### 6. Register the Model With Ollama

From the repo root:

```bash
ollama create uk-mortgage-slm -f artifacts/exports/ollama/Modelfile
```

Check that Ollama knows about it:

```bash
ollama list
```

### 7. Run the Ollama Model

Interactive:

```bash
ollama run uk-mortgage-slm
```

Then paste:

```text
### Instruction:
Summarise the mortgage product and highlight the main lending terms.

### Input:
Provider: Halifax
Mortgage name: 5 Year Fixed Remortgage
Interest rate: 4.65%
Maximum LTV: 75%
Term type: Fixed
Length: 5 years
Booking fee: £999
APRC: 5.80%
Notes: Free valuation and standard legal work included.

### Response:
```

One-shot command:

```bash
ollama run uk-mortgage-slm "### Instruction:
Summarise the mortgage product and highlight the main lending terms.

### Input:
Provider: Halifax
Mortgage name: 5 Year Fixed Remortgage
Interest rate: 4.65%
Maximum LTV: 75%
Term type: Fixed
Length: 5 years
Booking fee: £999
APRC: 5.80%
Notes: Free valuation and standard legal work included.

### Response:"
```

Expected result: Ollama should generate a concise mortgage summary.

## Which Route Should You Use?

| Goal | Route |
| --- | --- |
| Python inference | Route 1: Hugging Face |
| Evaluation scripts | Route 1: Hugging Face |
| Further ML work or future fine-tuning | Route 1: Hugging Face |
| FastAPI or custom Python service | Route 1: Hugging Face |
| Local desktop runtime | Route 2: GGUF / Ollama |
| Smaller single-file local model | Route 2: GGUF / Ollama |
| CPU-friendly local deployment | Route 2: GGUF / Ollama |

Recommended project workflow:

```text
1. Keep artifacts/exports/base-run-merged/ as the source-of-truth export.
2. Use Route 1 to test model quality.
3. Create GGUF only when you want Ollama or another GGUF runtime.
```

## Troubleshooting

### `base-run-merged` is missing

Run:

```bash
make export-merged
```

### Transformers tries to download files

Use a local folder path and `local_files_only=True`. In this repo:

```bash
python src/run_model.py --model-dir artifacts/exports/base-run-merged
```

### `llama-quantize` is missing

Rebuild `llama.cpp`:

```bash
cd external/llama.cpp
cmake -B build
cmake --build build --config Release -j
```

Then check:

```bash
ls build/bin/llama-quantize
```

### Ollama cannot find the GGUF file

In the `Modelfile`, the path after `FROM` is relative to the `Modelfile`
location. This project uses:

```text
artifacts/exports/ollama/Modelfile
artifacts/exports/gguf/base-run-Q4_K_M.gguf
```

So the `FROM` line is:

```text
FROM ../gguf/base-run-Q4_K_M.gguf
```

### Output quality is weak

The export route is separate from model quality. If the model answers awkwardly,
improve the dataset, train for more steps, and compare eval loss before
exporting again.

## References

- Ollama Modelfile reference: https://docs.ollama.com/modelfile
- llama.cpp repository and GGUF tooling: https://github.com/ggml-org/llama.cpp
- Hugging Face `llama.cpp` integration notes: https://huggingface.co/docs/transformers/community_integrations/llama_cpp
