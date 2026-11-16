"""Demo detector (USE_MOCK=1) so the app can be tried without the YOLO weights."""
import random
from config import DISEASES


class MockEngine:
    def predict(self, img):
        h, w = img.shape[:2]
        rnd = random.Random(w * h)
        quads = [{"box": [w * (0.06 + 0.44 * (q in (2, 3))), h * (0.22 if q in (1, 2) else 0.5),
                          w * (0.50 + 0.44 * (q in (2, 3))), h * (0.52 if q in (1, 2) else 0.80)],
                  "quadrant": q, "conf": 0.99} for q in (1, 2, 3, 4)]
        teeth, findings = [], []
        for q in (1, 2, 3, 4):
            for t in range(1, 9):
                left = q in (1, 4)
                k = (8 - t) if left else (t - 1)
                x1 = w * (0.08 + 0.43 * (not left) + 0.05 * k)
                y1 = h * (0.30 if q in (1, 2) else 0.55)
                box = [x1, y1, x1 + w * 0.04, y1 + h * 0.17]
                teeth.append({"box": box, "tooth": t, "conf": round(rnd.uniform(.6, .98), 2)})
                if rnd.random() < 0.2:
                    findings.append({"box": box[:], "disease": rnd.choice(list(DISEASES)),
                                     "conf": round(rnd.uniform(.4, .9), 2)})
        return {"quadrants": quads, "teeth": teeth, "findings": findings}
