import cv2
from config import DISEASES, QUAD_COLORS, QUAD_NAMES


def _bgr(h):
    h = h.lstrip("#"); return tuple(int(h[i:i + 2], 16) for i in (4, 2, 0))


def _label(out, x1, y1, text, col, s):
    fs, th = 0.6 * s, max(1, int(2 * s))
    (tw, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    y0 = max(int(y1), h + 8)
    cv2.rectangle(out, (int(x1), y0 - h - 8), (int(x1) + tw + 6, y0), col, -1)
    cv2.putText(out, text, (int(x1) + 3, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)


def _scale(img): return max(img.shape[1] / 1500, 0.6)


def quadrants_image(img, res):
    out, s = img.copy(), _scale(img)
    for q in res["quadrants"]:
        col = _bgr(QUAD_COLORS[q["quadrant"]]); x1, y1, x2, y2 = map(int, q["box"])
        tint = out.copy(); cv2.rectangle(tint, (x1, y1), (x2, y2), col, -1)
        out = cv2.addWeighted(tint, 0.14, out, 0.86, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, max(3, int(3 * s)))
        _label(out, x1, y1, f"Q{q['quadrant']} {QUAD_NAMES[q['quadrant']]}", col, s)
    return out


def fdi_image(img, res):
    out, s = img.copy(), _scale(img)
    for t in res["teeth"]:
        col = _bgr(QUAD_COLORS[t["quadrant"]]); x1, y1, x2, y2 = map(int, t["box"])
        cv2.rectangle(out, (x1, y1), (x2, y2), col, max(2, int(2 * s)))
        _label(out, x1, y1, str(t["fdi"]), col, s * 0.8)
    return out


def abnormal_image(img, res):
    out, s = img.copy(), _scale(img)
    for t in res["teeth"]:
        x1, y1, x2, y2 = map(int, t["box"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (200, 200, 200), 1)
    for f in res["findings"]:
        col = _bgr(DISEASES[f["disease"]]); x1, y1, x2, y2 = map(int, f["box"])
        cv2.rectangle(out, (x1, y1), (x2, y2), col, max(3, int(3 * s)))
        _label(out, x1, y1, f"{f['fdi'] or '?'} {f['disease']} {f['conf']:.2f}", col, s * 0.8)
    return out
