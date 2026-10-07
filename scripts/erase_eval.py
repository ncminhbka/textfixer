#!/usr/bin/env python3
"""CHẤM LỜI DẶN XOÁ (máy cá nhân): kết quả scripts/erase_probe.py (giải nén vào output/erase_probe/) so với nháp output/pairs.

  python scripts/erase_eval.py                    # mọi lời dặn có trong output/erase_probe (+ A = output/pairs)
  python scripts/erase_eval.py --only h06_s0,r02_s0 --no-sheet

Đo mỗi lời dặn:
  sót     dòng chữ OCR đọc được trên bản xoá, trùng vị trí một dòng trên nháp, KHÔNG phải chữ trên sản phẩm -> chữ poster chưa xoá
  SP giữ  (ca có ảnh tham chiếu) dòng chữ trên sản phẩm (khớp gần đúng chữ của ảnh tham chiếu, pairs.json scene_texts) còn trên
          bản xoá / có trên nháp
  đổi     % điểm ảnh đổi (ΔE > 20) NGOÀI khung các dòng chữ của nháp: vỏ / icon bị xoá + phần ảnh bị vẽ lại (xem ảnh để phân biệt)
Ảnh so sánh output/erase_eval/<key>.jpg: nháp | A | B | ... (cùng thứ tự bảng).
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", line_buffering=True)

PAIRS = ROOT / "output" / "pairs"
PROBE = ROOT / "output" / "erase_probe"
OUT = ROOT / "output" / "erase_eval"


def _norm(t: str) -> str:
    t = unicodedata.normalize("NFD", t.upper())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn").replace("Đ", "D")
    return re.sub(r"[^A-Z0-9]", "", t)


def is_scene(text: str, scene: list[str]) -> bool:
    """Dòng OCR là chữ trên sản phẩm: gần giống (>= 0.6) một chữ / cụm chữ của ảnh tham chiếu, hoặc chứa nó."""
    n = _norm(text)
    if len(n) < 2:
        return False
    toks = {_norm(w) for s in scene for w in [s] + s.split()} - {""}
    return any(t in n or (len(t) >= 3 and difflib.SequenceMatcher(None, n, t).ratio() >= 0.6) for t in toks)


def cov(a, b) -> float:
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    return max(0, w) * max(0, h) / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))


def main(argv: list[str] | None = None) -> int:
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw
    from textfix.ocr import read_lines
    from textfix.overlay import _lab
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--no-sheet", action="store_true")
    a = ap.parse_args(argv)
    meta = json.loads((PAIRS / "pairs.json").read_text(encoding="utf-8"))
    info = json.loads((PROBE / "variants.json").read_text(encoding="utf-8")) if (PROBE / "variants.json").exists() else {}
    vs = ["A"] + sorted(d.name for d in PROBE.iterdir() if d.is_dir()) if PROBE.exists() else ["A"]
    keys = [k for k in sorted(meta) if (PAIRS / f"{k}_draft.png").exists() and (not a.only or k in a.only.split(","))]
    OUT.mkdir(parents=True, exist_ok=True)
    plate_f = lambda v, k: PAIRS / f"{k}_plate.png" if v == "A" else PROBE / v / f"{k}_plate.png"   # noqa: E731
    tot = {v: {"n": 0, "lines": 0, "left": 0, "sp": 0, "sp_kept": 0, "chg": 0.0} for v in vs}
    rows = []
    for k in keys:
        d = np.asarray(Image.open(PAIRS / f"{k}_draft.png").convert("RGB"))
        H, W = d.shape[:2]
        scene = meta[k].get("scene_texts") or []
        Ld = [l for l in read_lines(d) if l["score"] > 0.5]
        sp_d = [l for l in Ld if scene and is_scene(l["text"], scene)]
        poster = [l for l in Ld if l not in sp_d]
        boxes = np.zeros((H, W), np.uint8)
        for l in Ld:
            x0, y0, x1, y1 = map(int, l["box"])
            p = int(0.3 * (y1 - y0))
            boxes[max(0, y0 - p):y1 + p, max(0, x0 - p):x1 + p] = 1
        A = _lab(d)
        line, ims = [f"{k:<8}"], [d]
        for v in vs:
            f = plate_f(v, k)
            if not f.exists():
                line.append(f"{v}: --")
                ims.append(None)
                continue
            p = np.asarray(Image.open(f).convert("RGB").resize((W, H)))
            Lp = [l for l in read_lines(p) if l["score"] > 0.5]
            left = [l for l in Lp if any(cov(l["box"], m["box"]) > 0.5 for m in poster) and not (scene and is_scene(l["text"], scene))]
            kept = [m for m in sp_d if any(cov(m["box"], l["box"]) > 0.3 and is_scene(l["text"], scene) for l in Lp)]
            e = cv2.GaussianBlur(np.linalg.norm(A - _lab(p), axis=2), (0, 0), 1.0) > 20
            chg = float((e & (boxes == 0)).mean() * 100)
            t = tot[v]
            t["n"] += 1; t["lines"] += len(poster); t["left"] += len(left); t["sp"] += len(sp_d); t["sp_kept"] += len(kept)
            t["chg"] += chg
            line.append(f"{v}: sót {len(left)}/{len(poster)}" + (f" SP {len(kept)}/{len(sp_d)}" if sp_d else "") + f" đổi {chg:.1f}%")
            rows.append({"key": k, "v": v, "left": [l["text"] for l in left], "sp_kept": len(kept), "sp": len(sp_d), "chg": chg})
            ims.append(p)
        print("  ".join(line))
        if not a.no_sheet:
            h = 420
            tiles = []
            for name, im in zip(["nháp"] + vs, ims):
                if im is None:
                    continue
                t = Image.fromarray(im).resize((int(W * h / H), h))
                ImageDraw.Draw(t).rectangle([0, 0, 44, 18], fill="black")
                ImageDraw.Draw(t).text((4, 3), name, fill="white")
                tiles.append(t)
            sheet = Image.new("RGB", (sum(t.width + 6 for t in tiles), h), "white")
            x = 0
            for t in tiles:
                sheet.paste(t, (x, 0))
                x += t.width + 6
            sheet.save(OUT / f"{k}.jpg", quality=80)
    print(f"\n{'lời dặn':<8}{'ảnh':>5}{'chữ sót':>14}{'chữ SP còn':>14}{'đổi ngoài chữ':>15}   lời dặn")
    for v in vs:
        t = tot[v]
        if not t["n"]:
            continue
        pr = (info.get(v) or {}).get("prompt") or ""
        print(f"{v:<8}{t['n']:>5}{t['left']:>7}/{t['lines']:<4}{100 * t['left'] / max(1, t['lines']):>3.0f}%"
              f"{t['sp_kept']:>7}/{t['sp']:<4}{100 * t['sp_kept'] / max(1, t['sp']):>3.0f}%{t['chg'] / t['n']:>12.1f}%   {pr[:70]}")
    (OUT / "result.json").write_text(json.dumps({"total": tot, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"ảnh so sánh: {OUT}/<key>.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
