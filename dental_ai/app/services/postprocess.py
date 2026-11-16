import math
from config import DISEASES, QUAD_NAMES


def overlap(a, b):
    """Share of box a that lies inside box b."""
    iw = min(a[2], b[2]) - max(a[0], b[0]); ih = min(a[3], b[3]) - max(a[1], b[1])
    return max(iw, 0) * max(ih, 0) / max((a[2] - a[0]) * (a[3] - a[1]), 1e-6)


def _quadrant_of(box, quads, W, H):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    scored = [(overlap(box, q["box"]),
               -math.hypot(cx - (q["box"][0] + q["box"][2]) / 2, cy - (q["box"][1] + q["box"][3]) / 2),
               q["quadrant"]) for q in quads]
    if scored and max(scored)[0] > 0:
        return max(scored)[2]
    # fallback if the quadrant model found nothing: image halves (patient's right is on image left)
    right, upper = cx < W / 2, cy < H / 2
    return {(True, True): 1, (False, True): 2, (False, False): 3, (True, False): 4}[(right, upper)]


def merge(raw, W, H):
    """quadrant boxes + tooth-number boxes + disease boxes -> one record per tooth (FDI)."""
    best = {}
    for t in raw["teeth"]:
        q = _quadrant_of(t["box"], raw["quadrants"], W, H)
        fdi = q * 10 + t["tooth"]
        if fdi not in best or t["conf"] > best[fdi]["conf"]:     # one box per FDI number
            best[fdi] = {"fdi": fdi, "quadrant": q, "quadrant_name": QUAD_NAMES[q], "tooth": t["tooth"],
                         "box": t["box"], "conf": t["conf"], "abnormalities": []}
    teeth = sorted(best.values(), key=lambda t: t["fdi"])

    findings = []
    for f in raw["findings"]:
        if f["disease"] not in DISEASES:
            continue
        cand = max(teeth, key=lambda t: overlap(f["box"], t["box"]), default=None)
        ok = cand is not None and overlap(f["box"], cand["box"]) >= 0.3
        if ok:
            cand["abnormalities"].append({"disease": f["disease"], "conf": f["conf"]})
        findings.append({"box": f["box"], "disease": f["disease"], "conf": f["conf"],
                         "fdi": cand["fdi"] if ok else None})
    counts = {d: sum(1 for f in findings if f["disease"] == d) for d in DISEASES}
    return {"quadrants": raw["quadrants"], "teeth": teeth, "findings": findings,
            "summary": {"quadrants_found": len({q["quadrant"] for q in raw["quadrants"]}),
                        "teeth_detected": len(teeth), "findings": len(findings),
                        "teeth_with_abnormality": sum(1 for t in teeth if t["abnormalities"]),
                        "counts": counts}}
