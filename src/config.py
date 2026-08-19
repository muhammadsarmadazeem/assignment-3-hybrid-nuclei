"""Paths and hyperparameters for the nuclei pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "nuclei_dataset"
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"
LLM_DIR = OUTPUT_DIR / "llm"
TABLE_DIR = OUTPUT_DIR / "tables"
PROCESSED_DIR = OUTPUT_DIR / "processed"

IMAGE_SIZE = 256
SEED = 42
BATCH_SIZE = 8
EPOCHS = 25
BASE_CHANNELS = 16
LEARNING_RATE = 1e-3
NUM_WORKERS = 0
THRESHOLD = 0.5

# Ollama
OLLAMA_HOST = "http://127.0.0.1:11434"
VLM_MODEL = "llama3.2-vision"
SECOND_VLM_MODEL = "llava"  # extra-credit second vision model on the same prompt
TEXT_MODEL = "llama3.2-vision"  # text-only calls; no image is attached
OLLAMA_TIMEOUT = 600  # first llama3.2-vision load on Metal is slow

REPRESENTATIVE_ID = "train_001"  # normal density, used for Task 1/2 comparison
TASK2_IMAGE_ID = "train_001"

LOSSES = ("bce", "dice", "bce_dice")
PRIMARY_LOSS = "bce_dice"

DENSITY_BINS = {
    "sparse": (0, 14),
    "normal": (15, 44),
    "dense": (45, 10_000),
}


def ensure_dirs() -> None:
    for d in (FIGURE_DIR, MODEL_DIR, LLM_DIR, TABLE_DIR, PROCESSED_DIR):
        d.mkdir(parents=True, exist_ok=True)
