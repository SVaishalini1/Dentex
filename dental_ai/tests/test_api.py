import numpy as np, cv2, joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import config

# tiny dummy classical models so the compare page can be tested without DENTEX
config.CLASSICAL.parent.mkdir(parents=True, exist_ok=True)
X = np.random.rand(40, 1764 + 4); y = np.arange(40) % 4
joblib.dump({"models": {"logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=200)).fit(X, y)},
             "classes": {0: "Impacted", 1: "Caries", 2: "Periapical Lesion", 3: "Deep Caries"},
             "metrics": [{"model": "logreg", "accuracy": .5, "precision": .5, "recall": .5, "f1": .5}]}, config.CLASSICAL)

from fastapi.testclient import TestClient
from app.main import app
c = TestClient(app)

def _post(img):
    ok, buf = cv2.imencode(".png", img)
    return c.post("/api/analyze", files={"file": ("opg.png", buf.tobytes(), "image/png")})

def test_full_flow():
    r = _post(np.random.randint(0, 255, (600, 1200, 3), np.uint8)); assert r.status_code == 200
    j = r.json()
    assert j["summary"]["teeth_detected"] == 32 and j["summary"]["quadrants_found"] == 4
    assert {t["fdi"] for t in j["teeth"]} == {q * 10 + t for q in range(1, 5) for t in range(1, 9)}
    for k in ("quadrants", "fdi", "abnormalities"):
        assert c.get(j["images"][k]).status_code == 200
    assert j["compare"]["available"] and len(j["compare"]["rows"]) == len(j["findings"])
    assert c.get("/compare").status_code == 200 and c.get(f"/api/results/{j['id']}").status_code == 200

def test_bad_file():
    assert c.post("/api/analyze", files={"file": ("x.txt", b"hi", "text/plain")}).status_code == 400
