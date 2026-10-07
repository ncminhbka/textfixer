"""Dò dòng chữ (kể cả chữ giun) bằng rapidocr (ONNX nhỏ, CPU). Mỗi dòng: khung, đa giác 4 điểm, chữ đọc được, độ tin, góc."""

from __future__ import annotations

import math

import numpy as np

_OCR = None


def _engine():
    global _OCR
    if _OCR is None:
        import rapidocr_onnxruntime as ro
        _OCR = ro.RapidOCR()
    return _OCR


MULTI_SCALE = 0.5   # lượt dò thứ hai ở ảnh thu nhỏ: chữ RẤT TO / chữ trang trí bị sót ở cỡ gốc
MULTI_MIN = 600     # chỉ chạy lượt thu nhỏ khi cạnh ngắn ảnh >= 600 px


def _once(img, k: float = 1.0) -> list[dict]:
    res, _ = _engine()(img)
    out = []
    for poly, text, score in res or []:
        p = np.asarray(poly, dtype=float) / k
        # góc cạnh trên (điểm 0 -> 1): dòng nghiêng / phối cảnh là dấu hiệu chữ thuộc cảnh
        ang = math.degrees(math.atan2(p[1, 1] - p[0, 1], p[1, 0] - p[0, 0]))
        out.append({"box": [float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())],
                    "poly": p.tolist(), "text": text, "score": float(score), "angle": ang})
    return out


def _cover(a, b) -> float:
    """Phần diện tích khung a nằm trong khung b."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h) / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))


def read_lines(img: np.ndarray | str, multi: bool = True) -> list[dict]:
    """Dòng chữ ở cỡ gốc + (multi) dòng chỉ dò được ở ảnh thu nhỏ MULTI_SCALE, không chồng dòng đã có (khung chồng < 30% cả hai phía)."""
    out = _once(img)
    if not multi or isinstance(img, str) or min(img.shape[:2]) < MULTI_MIN:
        return out
    from PIL import Image
    H, W = img.shape[:2]
    small = np.asarray(Image.fromarray(img).resize((int(W * MULTI_SCALE), int(H * MULTI_SCALE)), Image.LANCZOS))
    for x in _once(small, MULTI_SCALE):
        if all(_cover(x["box"], o["box"]) < 0.3 and _cover(o["box"], x["box"]) < 0.3 for o in out):
            out.append(x)
    return out
