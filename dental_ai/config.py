import os
from pathlib import Path

BASE = Path(__file__).parent
MODEL_DIR = BASE / "models"
STORAGE = BASE / "storage" / "results"
# Copy best.pt of each YOLO run to these names (see README)
YOLO_WEIGHTS = {t: MODEL_DIR / f"{t}.pt" for t in ("quadrant", "enum", "disease")}
CLASSICAL = MODEL_DIR / "classical" / "classical_models.joblib"

_env = os.getenv("USE_MOCK")                       # optional override: 1 = demo, 0 = real
USE_MOCK = (_env == "1") if _env is not None else not all(p.exists() for p in YOLO_WEIGHTS.values())
_sz = MODEL_DIR / "imgsz.txt"                      # written by the training notebook
IMGSZ = int(_sz.read_text()) if _sz.exists() else 1024   # must equal the training image size
CONF = {"quadrant": 0.25, "enum": 0.25, "disease": 0.25}
MAX_MB = 20

QUAD_NAMES = {1: "Upper right", 2: "Upper left", 3: "Lower left", 4: "Lower right"}
QUAD_COLORS = {1: "#12a594", 2: "#3e63dd", 3: "#d6409f", 4: "#6e9b00"}
DISEASES = {"Caries": "#f5a623", "Deep Caries": "#e5484d",
            "Periapical Lesion": "#8e4ec6", "Impacted": "#0091ff"}
