# Human-in-the-Loop Qualitative Analysis

A local research application for producing researcher-reviewed preference data for qualitative coding. The application combines Streamlit, Ollama, and SQLite to present interview context, generate two blinded assessments of a researcher-supplied qualitative code, record a preference decision, and export eligible preference pairs for further DPO training.

This repository is extension work based on [KaiJie-5/Direct-Preference-Optimization-of-LLMs-for-Critical-Thinking](https://github.com/KaiJie-5/Direct-Preference-Optimization-of-LLMs-for-Critical-Thinking). Refer to that upstream repository for information about preparing data, training a DPO model, and the original DPO pipeline. This repository begins after training and focuses on local human review and preference-data collection.

The application runs locally and has three pages:

- **Setup:** configure a study, reviewer, research questions, transcript dataset, context window, and Ollama model.
- **Review:** assess transcript segments, generate two blinded model responses, and record a preference or audit decision.
- **Progress and export:** monitor completion and export eligible `adaptation` preference pairs as chat JSONL.

Candidate assessments are shown as structured cards. Each explicit regeneration uses two new
seeds and replaces the prior pair only after the replacement finishes successfully; saved
decisions remain immutable.

## Getting Started

Install [Git for Windows](https://git-scm.com/download/win) and [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) or Anaconda, then open **Anaconda Prompt**.

Clone this repository:

```bat
git clone https://github.com/KaiJie-5/Human-In-The-Loop-Qualitative-Analysis.git
```

Enter the repository:

```bat
cd /d "Human-In-The-Loop-Qualitative-Analysis"
```

If you cloned it into another directory, replace the path above with the location of your clone.

## Installation

### Tested system

The pipeline has been tested on the following local system. These are tested versions, not strict minimum requirements.

| Component | Tested configuration |
|---|---|
| Operating system | Windows 11 Home Single Language, build 26200 |
| Processor | Intel Core i7-12700H, 14 cores / 20 logical processors |
| Memory | 15.71 GiB RAM |
| GPU | NVIDIA GeForce RTX 3070 Ti Laptop GPU, 8 GiB VRAM |
| NVIDIA driver / CUDA reported by driver | 576.52 / CUDA 12.9 |
| Conda | 25.1.1 |
| Python | CPython 3.10.20, AMD64 |
| Git | 2.49.0.windows.1 |
| Streamlit / Starlette | 1.61.0 / 1.3.1 |
| HTTPX / Pydantic | 0.28.1 / 2.13.4 |
| Ollama | 0.32.5 |
| llama.cpp conversion checkout | `b10284-3-gb06aa774c` |
| Validated local model | SmolLM3 3B DPO model, BF16 source converted to Q8_0 GGUF |

The application requires Python 3.10 or newer. Python 3.10 is recommended for reproducing the tested environment.

### Create the Conda environment

Run each command from **Anaconda Prompt**.

```bat
conda create --name hitl-qda python=3.10 -y
```

```bat
conda activate hitl-qda
```

```bat
python -m pip install --upgrade pip setuptools wheel
```

From the repository root, install the application and its development dependencies:

```bat
python -m pip install -e ".[dev]"
```

Confirm that the environment has no broken Python dependencies:

```bat
python -m pip check
```

Create the local configuration file:

```bat
copy /Y local_config.toml.example local_config.toml
```

`local_config.toml` is ignored by Git. It controls the SQLite database path, export directory, Ollama address, timeouts, transcript context limits, and generation settings. The default runtime locations are relative to the repository:

```text
runtime/hitl.sqlite3
exports/
```

## Ollama Installation

Ollama runs the selected language model locally and exposes the API used by the application. The official Windows installer supports NVIDIA GPU acceleration and starts Ollama in the background.

Open the official download page:

```bat
start https://ollama.com/download/windows
```

Download and run `OllamaSetup.exe`. When installation finishes, close and reopen Anaconda Prompt, reactivate the environment, and verify Ollama:

```bat
conda activate hitl-qda
```

```bat
where ollama
```

```bat
ollama --version
```

```bat
curl.exe http://localhost:11434/api/tags
```

A successful fresh installation returns an empty model list until a model is pulled or imported. See the official [Ollama Windows documentation](https://docs.ollama.com/windows) for installation and GPU requirements.

## Add a Custom DPO Model to Ollama

The following workflow is for a **complete, merged Hugging Face model** containing files such as `config.json`, tokenizer files, and one or more `.safetensors` weight files. A directory containing only a PEFT/LoRA adapter must first be merged with the exact base model used for DPO training. Refer to the [upstream DPO repository](https://github.com/KaiJie-5/Direct-Preference-Optimization-of-LLMs-for-Critical-Thinking) for the training and model-saving workflow.

The commands below create three GGUF variants:

- **BF16:** highest-fidelity local GGUF and the source used for Q4_K_M quantization.
- **Q8_0:** smaller than BF16 with relatively little precision loss.
- **Q4_K_M:** recommended for machines with limited VRAM, including an 8 GiB GPU.

Quantization can reduce memory use and improve inference speed, but lower precision can affect model quality. Evaluate each quantization on research-relevant examples before collecting preference data.

### 1. Define the model location

Replace the example with the directory containing your merged DPO model:

```bat
set "DPO_MODEL_DIR=C:\path\to\your\final_model"
```

Confirm that the directory exists:

```bat
if exist "%DPO_MODEL_DIR%\" (echo MODEL DIRECTORY FOUND) else (echo MODEL DIRECTORY NOT FOUND)
```

Inspect its files:

```bat
dir /b /a-d "%DPO_MODEL_DIR%"
```

Check whether the directory contains only an adapter:

```bat
if exist "%DPO_MODEL_DIR%\adapter_config.json" (echo PEFT OR LORA ADAPTER DETECTED - MERGE IT FIRST) else (echo NO ADAPTER CONFIG - CONTINUE WITH CONVERSION)
```

### 2. Create a conversion environment and install llama.cpp

The conversion environment is separate from `hitl-qda` so model-conversion packages do not alter the application environment.

```bat
conda create --name gguf-convert python=3.10 -y
```

```bat
conda activate gguf-convert
```

Choose where llama.cpp will be stored:

```bat
set "LLAMA_CPP_DIR=%USERPROFILE%\Desktop\llama.cpp"
```

Clone the official llama.cpp repository if it is not already present:

```bat
git clone https://github.com/ggml-org/llama.cpp.git "%LLAMA_CPP_DIR%"
```

To reproduce the tested conversion-tool revision, check it out explicitly:

```bat
git -C "%LLAMA_CPP_DIR%" checkout b06aa774c03dbbb624e726664b714a57d1f49815
```

Enter the llama.cpp directory:

```bat
cd /d "%LLAMA_CPP_DIR%"
```

Install the official conversion requirements:

```bat
python -m pip install --upgrade pip setuptools wheel
```

```bat
python -m pip install -r requirements.txt
```

The conversion commands use llama.cpp's official [`convert_hf_to_gguf.py`](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py) script.

### 3. Convert the BF16 model to BF16 GGUF

Create an output directory and a reusable model name:

```bat
set "GGUF_DIR=%DPO_MODEL_DIR%\gguf"
```

```bat
set "MODEL_BASENAME=smollm3-3b-reflective-dpo"
```

```bat
if not exist "%GGUF_DIR%" mkdir "%GGUF_DIR%"
```

Convert the merged Hugging Face BF16 model to a high-precision BF16 GGUF:

```bat
python convert_hf_to_gguf.py "%DPO_MODEL_DIR%" --outfile "%GGUF_DIR%\%MODEL_BASENAME%-bf16.gguf" --outtype bf16
```

Confirm the output:

```bat
dir "%GGUF_DIR%\%MODEL_BASENAME%-bf16.gguf"
```

### 4. Create the Q8_0 GGUF

Convert directly from the original Hugging Face model rather than requantizing another GGUF:

```bat
python convert_hf_to_gguf.py "%DPO_MODEL_DIR%" --outfile "%GGUF_DIR%\%MODEL_BASENAME%-q8_0.gguf" --outtype q8_0
```

Confirm the output:

```bat
dir "%GGUF_DIR%\%MODEL_BASENAME%-q8_0.gguf"
```

### 5. Build llama-quantize for Q4_K_M

Q4_K_M is produced from the BF16 GGUF with `llama-quantize`; it must not be produced by requantizing the Q8_0 file.

On Windows, install [Visual Studio 2022 Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and select the **Desktop development with C++** workload. Include **C++ CMake tools for Windows**, as required by the official [llama.cpp Windows build guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md).

After installation, reopen Anaconda Prompt and activate the compiler environment:

```bat
call "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64
```

Reactivate the conversion environment:

```bat
conda activate gguf-convert
```

Restore the working variables if this is a new prompt:

```bat
set "DPO_MODEL_DIR=C:\path\to\your\final_model"
```

```bat
set "GGUF_DIR=%DPO_MODEL_DIR%\gguf"
```

```bat
set "MODEL_BASENAME=smollm3-3b-reflective-dpo"
```

```bat
set "LLAMA_CPP_DIR=%USERPROFILE%\Desktop\llama.cpp"
```

Configure a CPU-only release build; GPU support is not required for the quantization utility:

```bat
cd /d "%LLAMA_CPP_DIR%"
```

```bat
cmake -B build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=OFF
```

```bat
cmake --build build --config Release -j 8
```

Confirm that the quantizer was built:

```bat
dir "%LLAMA_CPP_DIR%\build\bin\Release\llama-quantize.exe"
```

### 6. Create the Q4_K_M GGUF

Quantize from the BF16 GGUF:

```bat
"%LLAMA_CPP_DIR%\build\bin\Release\llama-quantize.exe" "%GGUF_DIR%\%MODEL_BASENAME%-bf16.gguf" "%GGUF_DIR%\%MODEL_BASENAME%-q4_k_m.gguf" Q4_K_M
```

Confirm all three files:

```bat
dir "%GGUF_DIR%\*.gguf"
```

Optionally record their SHA-256 checksums:

```bat
certutil -hashfile "%GGUF_DIR%\%MODEL_BASENAME%-bf16.gguf" SHA256
```

```bat
certutil -hashfile "%GGUF_DIR%\%MODEL_BASENAME%-q8_0.gguf" SHA256
```

```bat
certutil -hashfile "%GGUF_DIR%\%MODEL_BASENAME%-q4_k_m.gguf" SHA256
```

The two-stage BF16-to-Q4_K_M process follows the official [llama.cpp quantization workflow](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md).

### 7. Register the GGUF models with Ollama

Change to the GGUF directory:

```bat
cd /d "%GGUF_DIR%"
```

Create a Modelfile for the BF16 variant:

```bat
(
echo FROM ./%MODEL_BASENAME%-bf16.gguf
echo PARAMETER num_ctx 65536
echo SYSTEM /no_think
) > Modelfile.hitl-bf16
```

Create a Modelfile for the Q8_0 variant:

```bat
(
echo FROM ./%MODEL_BASENAME%-q8_0.gguf
echo PARAMETER num_ctx 65536
echo SYSTEM /no_think
) > Modelfile.hitl-q8
```

Create a Modelfile for the Q4_K_M variant:

```bat
(
echo FROM ./%MODEL_BASENAME%-q4_k_m.gguf
echo PARAMETER num_ctx 65536
echo SYSTEM /no_think
) > Modelfile.hitl-q4-k-m
```

Register each model with a distinct Ollama tag:

```bat
ollama create hitl-smollm3-dpo:bf16 -f Modelfile.hitl-bf16
```

```bat
ollama create hitl-smollm3-dpo:q8 -f Modelfile.hitl-q8
```

```bat
ollama create hitl-smollm3-dpo:q4-k-m -f Modelfile.hitl-q4-k-m
```

Verify that the models are available:

```bat
ollama list
```

Inspect the selected model and its configured context length:

```bat
ollama show hitl-smollm3-dpo:q4-k-m
```

```bat
ollama show hitl-smollm3-dpo:q4-k-m --parameters
```

Run a short local check:

```bat
ollama run hitl-smollm3-dpo:q4-k-m "Reply with only the word ready."
```

The application sends its saved context length to Ollama as `num_ctx` for every generation. A 65,536-token context is the tested SmolLM3 model's architectural maximum, but it requires substantially more memory than a shorter context. If generation becomes slow or Ollama reports insufficient memory, select `32768` or `16384` under **Setup → Advanced generation settings**. See the official [Ollama model-import documentation](https://docs.ollama.com/import) and [Modelfile reference](https://docs.ollama.com/modelfile) for additional model formats and parameters.

## Verification (Optional)

The automated suite uses temporary SQLite databases and a fake Ollama client, so the normal
tests do not require a running model. From Anaconda Prompt, activate the application environment
and refresh the editable installation:

```bat
conda activate hitl-qda
```

```bat
cd /d "C:\path\to\Human-In-The-Loop-Qualitative-Analysis"
```

```bat
python -m pip install -e ".[dev]"
```

Run the candidate and transcript tests:

```bat
python -m pytest tests\test_transcripts_and_candidates.py -vv --tb=long
```

Run the workflow, regeneration, recovery, and export tests:

```bat
python -m pytest tests\test_workflow_and_export.py -vv --tb=long
```

Run the complete local suite while excluding optional external compatibility checks:

```bat
python -m pytest -vv --tb=long -m "not upstream_compat and not real_data_compat"
```

Save that output for diagnosis:

```bat
python -m pytest -vv --tb=long -m "not upstream_compat and not real_data_compat" > pytest-output.txt 2>&1
```

The optional compatibility checks use configurable environment variables so no external drive
or repository path is committed in application logic:

```bat
set "DPO_REPOSITORY_PATH=C:\path\to\Direct-Preference-Optimization-of-LLMs-for-Critical-Thinking"
```

```bat
python -m pytest tests\test_optional_compatibility.py::test_export_row_with_configured_upstream_loader -vv --tb=long -m upstream_compat
```

```bat
set "DPO_REFERENCE_JSONL=X:\path\to\preference_pairs_category_evidence.jsonl"
```

```bat
python -m pytest tests\test_optional_compatibility.py::test_first_twenty_real_rows_include_all_heading_contracts -vv --tb=long -m real_data_compat
```

Run and save both optional compatibility checks after configuring both paths:

```bat
python -m pytest tests\test_optional_compatibility.py -vv --tb=long -m "upstream_compat or real_data_compat" > pytest-compatibility-output.txt 2>&1
```

## Run the Pipeline

Open Anaconda Prompt and activate the application environment:

```bat
conda activate hitl-qda
```

Enter the cloned repository:

```bat
cd /d "C:\path\to\Human-In-The-Loop-Qualitative-Analysis"
```

Start the local application:

```bat
python -m streamlit run app.py
```
