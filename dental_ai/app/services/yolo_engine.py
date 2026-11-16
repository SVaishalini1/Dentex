"""Real detector: your 3 YOLOv8 models (quadrant, enum, disease), same imgsz as training."""
from config import YOLO_WEIGHTS, IMGSZ, CONF


class YoloEngine:
    def __init__(self):
        from ultralytics import YOLO
        missing = [str(p) for p in YOLO_WEIGHTS.values() if not p.exists()]
        if missing:
            raise FileNotFoundError("Missing YOLO weights: " + ", ".join(missing))
        self.m = {t: YOLO(str(p)) for t, p in YOLO_WEIGHTS.items()}

    def _run(self, task, img):
        r = self.m[task].predict(img, imgsz=IMGSZ, conf=CONF[task], verbose=False)[0]
        return [([float(v) for v in b.xyxy[0].tolist()], str(r.names[int(b.cls)]), float(b.conf))
                for b in r.boxes]

    def predict(self, img_bgr):
        return {
            "quadrants": [{"box": b, "quadrant": int(n), "conf": c} for b, n, c in self._run("quadrant", img_bgr)],
            "teeth":     [{"box": b, "tooth": int(n), "conf": c} for b, n, c in self._run("enum", img_bgr)],
            "findings":  [{"box": b, "disease": n, "conf": c} for b, n, c in self._run("disease", img_bgr)],
        }
