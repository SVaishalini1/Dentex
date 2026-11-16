import json, uuid
import cv2, numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config import BASE, STORAGE, USE_MOCK, MAX_MB, DISEASES, QUAD_COLORS
from app.services.postprocess import merge
from app.services import annotate
from app.services.classical import Classical

app = FastAPI(title="Dental OPG Analyzer")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
app.mount("/files", StaticFiles(directory=STORAGE), name="files")

if USE_MOCK:
    from app.services.mock_engine import MockEngine as Engine
else:
    from app.services.yolo_engine import YoloEngine as Engine
engine, classical = Engine(), Classical()


@app.get("/")
def index(): return FileResponse(BASE / "static" / "index.html")


@app.get("/compare")
def compare_page(): return FileResponse(BASE / "static" / "compare.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "mock": USE_MOCK, "classical_models": classical.ok,
            "diseases": DISEASES, "quadrant_colors": QUAD_COLORS}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"File larger than {MAX_MB} MB")
    buf = np.frombuffer(data, np.uint8)
    img, gray = cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(400, "Not a valid image (use PNG or JPG)")
    H, W = img.shape[:2]
    try:
        res = merge(engine.predict(img), W, H)
        compare = classical.compare(gray, res["findings"])
    except Exception as e:
        raise HTTPException(500, f"Model error: {e}")

    rid = uuid.uuid4().hex[:10]
    d = STORAGE / rid; d.mkdir(parents=True)
    cv2.imwrite(str(d / "original.png"), img)
    for name, fn in [("quadrants", annotate.quadrants_image), ("fdi", annotate.fdi_image),
                     ("abnormalities", annotate.abnormal_image)]:
        cv2.imwrite(str(d / f"{name}.png"), fn(img, res))
    for i, f in enumerate(res["findings"]):
        x1, y1, x2, y2 = [int(v) for v in f["box"]]
        c = img[max(y1, 0):max(y2, 1), max(x1, 0):max(x2, 1)]
        if c.size: cv2.imwrite(str(d / f"crop_{i}.png"), c)
        f["crop_url"] = f"/files/{rid}/crop_{i}.png"
    for r in compare.get("rows", []):
        r["crop_url"] = f"/files/{rid}/crop_{r['finding']}.png"
    res.update(id=rid, width=W, height=H, mock=USE_MOCK, compare=compare,
               images={k: f"/files/{rid}/{k}.png" for k in ("original", "quadrants", "fdi", "abnormalities")})
    (d / "result.json").write_text(json.dumps(res, indent=2))
    return res


@app.get("/api/results/{rid}")
def get_result(rid: str):
    p = STORAGE / rid / "result.json"
    if not p.exists():
        raise HTTPException(404, "Result not found")
    return json.loads(p.read_text())
