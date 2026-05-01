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
There are two common scenarios:

- **Scenario A: Run from this repo** when you are still working in the training
  project and want a quick smoke test.
- **Scenario B: Run from a fresh environment** when you want to prove the
  exported model folder works outside the repo.

### Scenario A: Run From This Repo

Use this when you are still inside this repo and the project `.venv` already
has the required dependencies.

Run the built-in smoke test:

```bash
make run-model
```

The runner loads this path by default:

```text
artifacts/exports/base-run-merged/
```

`make run-model` passes `--use-classifier --allow-downloads` automatically, so
the scope classifier gates every question before it reaches the SLM. For a
custom prompt:

```bash
python src/run_model.py --prompt "What is LTV?" --use-classifier --allow-downloads
```

The runner uses local files only unless you pass `--allow-downloads`.

Expected result: in-scope mortgage questions produce a model response;
out-of-scope questions are rejected by the classifier before the SLM loads.

### Scenario B: Run From a Fresh Environment

Use this when you want to move the merged Hugging Face export to another folder
or target machine and run it without relying on this repo.

#### 1. Create a Portable Model Archive

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

#### 2. Create a Fresh Local Environment

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

Also copy `src/scope_classifier.py` and `src/run_model.py` from this repo
into `slm-local-run/`. The `transformers` package above satisfies their
only dependency.

After unpacking, the folder should be:

```text
slm-local-run/
  base-run-merged/       ← unpacked model
  scope_classifier.py    ← copied from src/
  run_model.py           ← copied from src/
```

#### 3. Run a Smoke Test

Run from inside `slm-local-run/` with the venv active. The classifier gates
every question before the SLM is loaded.

```bash
# in-scope — classifier passes to SLM
python run_model.py \
  --model-dir base-run-merged \
  --prompt "What is LTV?" \
  --use-classifier \
  --allow-downloads

# out-of-scope — classifier rejects, SLM never loads
python run_model.py \
  --model-dir base-run-merged \
  --prompt "Where is the capital of France?" \
  --use-classifier \
  --allow-downloads
```

`--allow-downloads` is required on first run to download the classifier model
(~256 MB) into `~/.cache/huggingface/hub/`. Omit it on subsequent runs.

Expected result: mortgage questions produce a model response; out-of-scope
questions print the refusal message and exit without loading the SLM.

## Route 2: Convert to GGUF and Run With Ollama

Use this route when you want a local Ollama model. The flow is:

> **Classifier limitation:** Ollama is a standalone runtime with no Python
> hook. `scope_classifier.py` does not run when you call `ollama run` directly,
> so off-topic questions reach the model unfiltered.

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

On macOS, install the build prerequisites with Homebrew:

```bash
brew install cmake python@3.11
```

From this repo root:

```bash
mkdir -p external artifacts/exports/gguf
git clone https://github.com/ggml-org/llama.cpp external/llama.cpp

cd external/llama.cpp
python3.11 -m venv .venv
source .venv/bin/activate
python --version
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

cmake -B build
cmake --build build --config Release -j
```

If `external/llama.cpp` already exists, skip the `git clone` command and start
from:

```bash
cd external/llama.cpp
```

The `python --version` check should print Python 3.11.x or 3.12.x. Avoid
Python 3.13 for this environment: `llama.cpp` currently pulls
`numpy~=1.26.4`, and that NumPy release does not provide Python 3.13 wheels.
On Python 3.13, `pip` may fall back to building NumPy from source and fail
during installation.

If you already created `external/llama.cpp/.venv` with Python 3.13, recreate it
before installing requirements:

```bash
deactivate 2>/dev/null || true
mv .venv ".venv-py313-broken-$(date +%Y%m%d-%H%M%S)"
python3.11 -m venv .venv
source .venv/bin/activate
python --version
```

On macOS, if the build fails with missing standard C++ headers such as
`'array' file not found`, `'mutex' file not found`, or `'cstdio' file not
found`, re-run CMake with the SDK C++ include path:

```bash
cmake -B build \
  -DCMAKE_CXX_FLAGS="-isystem $(xcrun --show-sdk-path)/usr/include/c++/v1"
cmake --build build --config Release -j
```

The build should create:

```text
external/llama.cpp/build/bin/llama-quantize
```

Verify it before continuing:

```bash
test -x build/bin/llama-quantize
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

TEMPLATE """### System:
{{ .System }}

### Instruction:
{{ .Prompt }}

### Response:
"""

SYSTEM "You are a UK mortgage assistant. Only answer questions about UK mortgage products, interest rates, LTV, fees, eligibility, and related topics. For any other question, respond that the question is outside your knowledge scope."
```

The `TEMPLATE` reproduces exactly the `### System: / ### Instruction: / ### Response:`
format the model was trained on. Ollama injects the `SYSTEM` value automatically
for every request, so the domain boundary and refusal behaviour learned during
fine-tuning are always active. You do not need to include `### System:` or
`### Response:` manually when running the model.

### 6. Register the Model With Ollama

From the repo root:

```bash
ollama create uk-mortgage-slm -f artifacts/exports/ollama/Modelfile
```

If Ollama returns `invalid model name` while gathering model components, check
that the GGUF file referenced by the Modelfile exists:

```bash
test -f artifacts/exports/gguf/base-run-Q4_K_M.gguf
```

Create it with the quantization step above before running `ollama create`.

Check that Ollama knows about it:

```bash
ollama list
```

### 7. Run the Ollama Model

The Modelfile template handles `### System:`, `### Instruction:`, and
`### Response:` automatically. Supply only the instruction text and, when
needed, the `### Input:` block.

Interactive:

```bash
ollama run uk-mortgage-slm
```

Then paste:

```text
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
```

One-shot command:

```bash
ollama run uk-mortgage-slm "Summarise the mortgage product and highlight the main lending terms.

### Input:
Provider: Halifax
Mortgage name: 5 Year Fixed Remortgage
Interest rate: 4.65%
Maximum LTV: 75%
Term type: Fixed
Length: 5 years
Booking fee: £999
APRC: 5.80%
Notes: Free valuation and standard legal work included."
```

```bash
ollama run uk-mortgage-slm "List the key pros and cons of this mortgage product.

### Input:
Provider: Halifax
Mortgage name: 5 Year Fixed Remortgage
Interest rate: 4.65%
Maximum LTV: 75%
Term type: Fixed
Length: 5 years
Booking fee: £999
APRC: 5.80%
Notes: Free valuation and standard legal work included."
```

To verify the domain boundary, send an out-of-scope question:

```bash
ollama run uk-mortgage-slm "What is the weather forecast for London this weekend?"
```

Expected result: the model should respond that the question is outside its
knowledge scope.

Expected result for mortgage questions: Ollama generates a response to the
prompt you send.

## Which Route Should You Use?

| Goal | Route | Classifier works? |
| --- | --- | --- |
| Python inference | Route 1: Hugging Face | Yes — built in |
| Evaluation scripts | Route 1: Hugging Face | Yes — built in |
| Further ML work or future fine-tuning | Route 1: Hugging Face | Yes — built in |
| FastAPI or custom Python service | Route 1: Hugging Face | Yes — built in |
| Local desktop runtime | Route 2: GGUF / Ollama | No — not supported |
| Smaller single-file local model | Route 2: GGUF / Ollama | No — not supported |
| CPU-friendly local deployment | Route 2: GGUF / Ollama | No — not supported |

Recommended project workflow:

```text
1. Keep artifacts/exports/base-run-merged/ as the source-of-truth export.
2. Use Route 1 to test model quality.
3. Create GGUF only when you want Ollama or another GGUF runtime.
```
