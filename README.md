# Assignment 3 — Hybrid biomedical image-analysis pipeline

Local pipeline for the assigned **fluorescence-nuclei** mini-dataset:

`raw image → (VLM description) → Otsu / U-Net mask → regionprops → structured JSON → short narrative`

Outputs are intended to be **auditable**. None of the models are cleared for clinical use.

## Dataset

Images already live in `nuclei_dataset/` (synthetic DAPI-like 256×256 RGB, with binary masks). If you need to regenerate them:

```bash
python nuclei_dataset/make_dataset.py
```

## Setup

Python 3.9+ (3.11 recommended). Apple Silicon uses MPS automatically.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Ollama (Tasks 1, 2, 4) — llama3.2-vision

The assignment requires **llama3.2-vision**. Ollama **0.30+ dropped `mllama`**, so that model 500s on current releases. Use **Ollama 0.24.0** (already unpacked under `tools/ollama-0.24/` if you downloaded it):

```bash
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_MODELS="$PWD/.ollama/models"   # optional; keeps weights in the repo
./tools/ollama-0.24/ollama serve               # leave running

# other terminal
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_MODELS="$PWD/.ollama/models"
./tools/ollama-0.24/ollama pull llama3.2-vision
./tools/ollama-0.24/ollama pull llava          # extra-credit second VLM
```

Do not substitute llava/gemma3: the report and prompts target llama3.2-vision.

## Run everything

```bash
source .venv/bin/activate
python run_all.py                 # preprocess, EDA, VLM, Otsu, U-Net ablation, test CSV
python run_all.py --epochs 25
python run_all.py --skip-llm      # reuse cached LLM JSON, or skip if Ollama is down
python run_all.py --skip-train    # reuse outputs/models/*.pt
python run_all.py --extras-only          # second VLM + MedSAM only (needs checkpoints)
python run_all.py --extras-only --skip-medsam
```

On first run, U-Net training (three losses × 25 epochs) takes several minutes on Apple MPS.

## What gets written

| Path | Contents |
|------|----------|
| `outputs/processed/{train,val,test}/` | grayscale 256×256 PNGs |
| `outputs/figures/` | EDA, Otsu, U-Net panels, curves, robustness |
| `outputs/models/unet_{bce,dice,bce_dice}.pt` | best-by-val-Dice checkpoints |
| `outputs/tables/` | metrics JSON, regionprops CSV, test pipeline CSV |
| `outputs/llm/` | cached Ollama responses |

The test-set CSV required by Task 4 is `outputs/tables/test_pipeline_records.csv`.

## Layout of the code

| Module | Role |
|--------|------|
| `src/data.py` | load, grayscale, resize, histograms |
| `src/prompts.py` | **all LLM prompts** (also printed in the report) |
| `src/llm.py` | Ollama client, JSON parse, disk cache |
| `src/classical.py` | Otsu, morphology, `regionprops_table` |
| `src/unet_model.py` | Small U-Net + BCE / Dice / BCE+Dice |
| `src/train_eval.py` | training loop, Dice/IoU, Otsu comparison |
| `src/tasks.py` | Task 1 (VLM) and Task 2 (numbers-first) |
| `src/pipeline.py` | Task 4 hybrid test-set records |
| `src/robustness.py` | extra-credit blur trace |
| `src/figures.py` | EDA, Otsu, U-Net, robustness, and MedSAM figures |

## Design notes (for markers)

- Numeric JSON fields `n_objects` and `mean_area` are **overwritten** from regionprops after the LLM returns, so the CSV cannot silently adopt a hallucinated count.
- Prompts are descriptive-not-diagnostic and explicitly allow `"uncertain"`.
- Loss ablation (BCE vs Dice vs BCE+Dice), a blur-corruption trace, a second VLM (`llava` vs `llama3.2-vision`), and MedSAM vs U-Net on the validation split.

## Disclaimer

For educational use only. Hallucinations in a medical context can cause harm; do not use these outputs for diagnosis or treatment.
