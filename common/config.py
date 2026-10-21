from pathlib import Path

ROLL_NUMBER = "M24MA2001"
SEED = int("".join(ch for ch in ROLL_NUMBER if ch.isdigit()))

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
SPLIT_PATH = RESULTS_DIR / "split.npz"
FIG_DIR = RESULTS_DIR / "figures"
LOG_DIR = RESULTS_DIR / "logs"

N_CLASSES = 40
N_TRAIN, N_VAL, N_TEST = 6, 2, 2

MASK_ROWS = slice(24, 40)
MASK_COLS = slice(24, 40)
