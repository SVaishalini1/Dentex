"""Traditional models (trained by train_classical.py) applied to the abnormality boxes found by YOLO.
Features are IDENTICAL to training: 64x64 crop -> CLAHE -> HOG + 4 position features."""
import cv2, numpy as np, joblib
from skimage.feature import hog
from config import CLASSICAL

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
LABELS = {"logreg": "Logistic Regression", "ridge": "Ridge", "rf": "Random Forest",
          "svm": "SVM", "knn": "KNN", "dt": "Decision Tree"}


def features(gray, box):
    H, W = gray.shape
    x, y, w, h = int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])
    x, y = max(x, 0), max(y, 0)
    crop = gray[y:y + max(h, 1), x:x + max(w, 1)]
    if crop.size == 0:
        return None
    crop = _clahe.apply(cv2.resize(crop, (64, 64)))
    return np.concatenate([hog(crop, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)),
                           [(x + w / 2) / W, (y + h / 2) / H, w / W, h / H]])


class Classical:
    def __init__(self):
        self.ok = CLASSICAL.exists()
        if self.ok:
            d = joblib.load(CLASSICAL)
            self.models, self.classes, self.metrics = d["models"], d["classes"], d.get("metrics", [])

    def compare(self, gray, findings):
        if not self.ok:
            return {"available": False, "message": "Run train_classical.py and copy classical_models.joblib to models/classical/"}
        keep = [(i, features(gray, f["box"])) for i, f in enumerate(findings)]
        keep = [(i, x) for i, x in keep if x is not None]
        rows = []
        if keep:
            X = np.array([x for _, x in keep])
            preds = {n: [self.classes[int(p)] for p in m.predict(X)] for n, m in self.models.items()}
            for k, (i, _) in enumerate(keep):
                rows.append({"finding": i, "fdi": findings[i]["fdi"], "yolo": findings[i]["disease"],
                             "predictions": {n: preds[n][k] for n in preds}})
        agree = {n: (sum(r["predictions"][n] == r["yolo"] for r in rows) / len(rows) if rows else None)
                 for n in self.models}
        return {"available": True, "models": {n: LABELS.get(n, n) for n in self.models},
                "rows": rows, "agreement": agree, "test_metrics": self.metrics}
