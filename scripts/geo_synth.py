#!/usr/bin/env python3
"""MÁY CÁ NHÂN: bộ thử HÌNH HỌC CHỮ tổng hợp (đáp án đúng tuyệt đối) cho đợt nâng cấp chữ cong / chéo / một dòng nhiều cỡ.
Nền = bản xoá thật (output/dev2/*/v*/plate.png); vẽ chữ bằng Chromium với hình học biết trước -> "nháp" giả; bản xoá = nền.

  python scripts/geo_synth.py make [--n 60]      # -> output/geo_synth/<k>_draft.png, <k>_plate.png, <k>.json (đáp án)
  python scripts/geo_synth.py eval               # B2.5 đo lại -> so với đáp án (chữ cong / góc / các đoạn cỡ)

Đáp án mỗi mục: {kind: arc | line | mixed, text, font, size, ...}
  arc:   cx, cy, r (px), a0, a1 (độ, 0 = phải, 90 = dưới -- toạ độ ảnh), up (True: chữ trên cung lồi, False: cung lõm / dưới)
  line:  x, y (tâm), angle (độ, dương = quay chiều kim đồng hồ như OCR)
  mixed: x, y (đầu đường chân), parts [{text, size}]
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", line_buffering=True)

OUT = ROOT / "output" / "geo_synth"
TEXTS = ["SIÊU SALE CUỐI TUẦN", "KHAI TRƯƠNG", "NGON KHÓ CƯỠNG", "TIỆM BÁNH NHÀ GẠO", "BẢO HÀNH 24 THÁNG", "QUÀ TẶNG TRI ÂN",
         "MÙA HÈ SÔI ĐỘNG", "Ưu đãi mùa thu", "Trà Sữa Mây", "FLASH SALE", "GIẢM 30%", "HÀNG MỚI VỀ", "Chỉ hôm nay"]
MIXED = [[("GIẢM", 0.4), ("50%", 1.0)], [("Chỉ", 0.35), ("99K", 1.0), ("/hộp", 0.35)], [("Thứ Bảy", 0.4), ("21.12", 1.0)],
         [("Ưu đãi", 0.45), ("MÙA THU", 1.0)], [("Giảm đến", 0.5), ("40%", 1.0)]]
FONTS = ["bevietnam", "montserrat", "oswald", "baloo", "anton", "playfair", "paytone"]


def _page():
    from textfix.browser import Browser
    return Browser()


def _arc_path(cx, cy, r, a0, a1, up):
    """Cung tròn đi TRÁI -> PHẢI (chữ đọc xuôi). up: cung trên (lồi), góc a0 < a1 trong (180, 360); dưới: (0, 180) đi ngược."""
    p0 = (cx + r * math.cos(math.radians(a0)), cy + r * math.sin(math.radians(a0)))
    p1 = (cx + r * math.cos(math.radians(a1)), cy + r * math.sin(math.radians(a1)))
    large = 1 if abs(a1 - a0) > 180 else 0
    sweep = 1 if up else 0
    return f"M {p0[0]:.1f} {p0[1]:.1f} A {r:.1f} {r:.1f} 0 {large} {sweep} {p1[0]:.1f} {p1[1]:.1f}"


def _item(rng: random.Random, W: int, H: int) -> dict:
    kind = rng.choice(["arc", "arc", "line", "mixed", "wave", "grow"])
    font = rng.choice(FONTS)
    if kind == "arc":
        up = rng.random() < 0.6
        text = rng.choice(TEXTS)
        size = rng.uniform(0.045, 0.08) * W
        span = rng.uniform(50, 110)   # độ cung chữ chiếm
        r = max(size * len(text) * 0.62 / math.radians(span), 0.18 * W)
        span = math.degrees(size * len(text) * 0.62 / r)
        cx = rng.uniform(0.35, 0.65) * W
        mid = 270.0 if up else 90.0
        cy = (rng.uniform(0.3, 0.6) * H + r) if up else (rng.uniform(0.4, 0.7) * H - r)
        a0, a1 = (mid - span / 2, mid + span / 2) if up else (mid + span / 2, mid - span / 2)
        return {"kind": "arc", "text": text, "font": font, "size": round(size, 1), "cx": round(cx, 1), "cy": round(cy, 1),
                "r": round(r, 1), "a0": round(a0, 1), "a1": round(a1, 1), "up": up}
    if kind == "line":
        text = rng.choice(TEXTS)
        size = rng.uniform(0.04, 0.08) * W
        ang = rng.choice([-1, 1]) * rng.uniform(6, 35)
        return {"kind": "line", "text": text, "font": font, "size": round(size, 1), "x": round(rng.uniform(0.35, 0.65) * W, 1),
                "y": round(rng.uniform(0.25, 0.75) * H, 1), "angle": round(ang, 1)}
    if kind == "wave":
        text = rng.choice(TEXTS)
        size = rng.uniform(0.045, 0.075) * W
        length = size * len(text) * 0.62
        x0 = rng.uniform(0.05, 0.95) * W - length / 2
        x0 = min(max(0.04 * W, x0), 0.96 * W - length * 1.1)
        y0 = rng.uniform(0.3, 0.7) * H
        amp = rng.uniform(0.35, 0.8) * size
        lam = rng.uniform(0.6, 1.2) * length
        ph = rng.uniform(0, 2 * math.pi)
        xs = np.linspace(x0, x0 + length * 1.1, 80)
        base = [[round(float(x), 1), round(float(y0 + amp * math.sin(2 * math.pi * (x - x0) / lam + ph)), 1)] for x in xs]
        return {"kind": "wave", "text": text, "font": font, "size": round(size, 1), "baseline": base}
    if kind == "grow":
        text = rng.choice(TEXTS)
        s0 = rng.uniform(0.03, 0.05) * W
        s1 = s0 * rng.uniform(1.6, 2.4)
        if rng.random() < 0.5:
            s0, s1 = s1, s0
        x = rng.uniform(0.06, 0.2) * W
        y = rng.uniform(0.3, 0.8) * H
        ang = rng.choice([0, 0, rng.uniform(-20, 20)])
        return {"kind": "grow", "text": text, "font": font, "size": round((s0 + s1) / 2, 1), "s0": round(s0, 1),
                "s1": round(s1, 1), "x": round(x, 1), "y": round(y, 1), "angle": round(ang, 1)}
    parts = rng.choice(MIXED)
    big = rng.uniform(0.09, 0.15) * W
    return {"kind": "mixed", "text": " ".join(p for p, _ in parts), "font": font, "size": round(big, 1),
            "x": round(rng.uniform(0.08, 0.3) * W, 1), "y": round(rng.uniform(0.35, 0.8) * H, 1),
            "parts": [{"text": p, "size": round(big * f, 1)} for p, f in parts]}


def _svg(it: dict, color: str, stroke: str) -> str:
    from textfix.fonts import family
    fam = family(it["font"]).replace('"', "'")
    st = f"font-family:{fam};font-weight:800;fill:{color};paint-order:stroke;stroke:{stroke};stroke-width:{it['size'] * 0.06:.1f}px"
    if it["kind"] == "arc":
        pid = f"p{abs(hash(it['text'])) % 10 ** 6}"
        return (f'<path id="{pid}" d="{_arc_path(it["cx"], it["cy"], it["r"], it["a0"], it["a1"], it["up"])}" fill="none"/>'
                f'<text style="{st};font-size:{it["size"]}px"><textPath href="#{pid}" startOffset="50%" text-anchor="middle">'
                f'{it["text"]}</textPath></text>')
    if it["kind"] == "wave":
        pid = f"w{abs(hash(it['text'] + str(it['baseline'][0]))) % 10 ** 6}"
        d = "M " + " L ".join(f"{x} {y}" for x, y in it["baseline"])
        return (f'<path id="{pid}" d="{d}" fill="none"/><text style="{st};font-size:{it["size"]}px"><textPath href="#{pid}">'
                f'{it["text"]}</textPath></text>')
    if it["kind"] == "grow":
        n = max(1, len(it["text"]) - 1)
        sp = "".join(f'<tspan style="font-size:{it["s0"] + (it["s1"] - it["s0"]) * i / n:.1f}px">{ch}</tspan>'
                     for i, ch in enumerate(it["text"]))
        return (f'<text x="{it["x"]}" y="{it["y"]}" transform="rotate({it["angle"]} {it["x"]} {it["y"]})" '
                f'style="{st};white-space:pre">{sp}</text>')
    if it["kind"] == "line":
        return (f'<text x="{it["x"]}" y="{it["y"]}" text-anchor="middle" dominant-baseline="central" '
                f'transform="rotate({it["angle"]} {it["x"]} {it["y"]})" style="{st};font-size:{it["size"]}px">{it["text"]}</text>')
    sp = "".join(f'<tspan style="font-size:{p["size"]}px">{p["text"]} </tspan>' for p in it["parts"])
    return f'<text x="{it["x"]}" y="{it["y"]}" style="{st}">{sp}</text>'


def make(n: int, seed: int = 7) -> None:
    from PIL import Image
    from textfix.fonts import faces_for
    rng = random.Random(seed)
    plates = sorted((ROOT / "output" / "dev2").glob("run_*/v*/plate.png"))
    if not plates:
        raise SystemExit("cần output/dev2/*/v*/plate.png (giải nén zip dev2)")
    OUT.mkdir(parents=True, exist_ok=True)
    with _page() as B:
        for k in range(n):
            plate = Image.open(rng.choice(plates)).convert("RGB")
            W, H = plate.size
            items = [_item(rng, W, H) for _ in range(rng.choice([1, 1, 2]))]
            if len(items) == 2:   # hai mục không chồng nhau (poster thật không đè chữ lên chữ): mục 2 nửa dưới / mục 1 nửa trên
                for it, (lo, hi) in zip(items, ((0.12, 0.42), (0.6, 0.88))):
                    ref = ((it["cy"] - it["r"] if it["up"] else it["cy"] + it["r"]) if it["kind"] == "arc" else
                           float(np.mean([p[1] for p in it["baseline"]])) if it["kind"] == "wave" else it["y"])
                    dy = rng.uniform(lo, hi) * H - ref
                    for key in ("cy", "y"):
                        if key in it:
                            it[key] = round(it[key] + dy, 1)
                    if it["kind"] == "wave":
                        it["baseline"] = [[x, round(y + dy, 1)] for x, y in it["baseline"]]
            arr = np.asarray(plate).astype(float)
            lum = float((arr @ [0.2126, 0.7152, 0.0722]).mean())
            color, stroke = ("#1b1b1b", "#ffffff") if lum > 140 else ("#ffffff", "#1b1b1b")
            buf = io.BytesIO(); plate.save(buf, "PNG")
            import base64
            uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
            svg = "".join(_svg(it, color, stroke) for it in items)
            fonts = tuple(sorted({it["font"] for it in items}))
            B.page.set_viewport_size({"width": W, "height": H})
            B.page.set_content(f"<html><head><style>{faces_for(fonts)} body{{margin:0}}</style></head><body>"
                               f"<svg width='{W}' height='{H}' xmlns='http://www.w3.org/2000/svg'><image href='{uri}' width='{W}' "
                               f"height='{H}'/>{svg}</svg></body></html>")
            B.page.evaluate("async () => { await Promise.all([...document.fonts].map(x => x.load().catch(() => 0))); "
                            "await document.fonts.ready; }")
            img = Image.open(io.BytesIO(B.page.screenshot())).convert("RGB")
            # khung đáp án = khung NÉT của từng mục (dựng riêng từng mục trên nền, so với nền) -- getBBox của textPath sai
            base = np.asarray(img).astype(np.int16)
            for j, it in enumerate(items):
                B.page.evaluate(f"() => document.querySelectorAll('text').forEach((t, i) => t.style.visibility = i === {j} ? '' : 'hidden')")
                one = np.asarray(Image.open(io.BytesIO(B.page.screenshot())).convert("RGB")).astype(np.int16)
                d = np.abs(one - np.asarray(plate).astype(np.int16)).max(2) > 40
                ys, xs = np.nonzero(d)
                it["box"] = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)] if len(xs) else [0, 0, 0, 0]
                if it["kind"] == "arc":
                    aq = np.radians(np.linspace(it["a0"], it["a1"], 120))
                    it["baseline"] = [[round(float(it["cx"] + it["r"] * math.cos(t)), 1), round(float(it["cy"] + it["r"] * math.sin(t)), 1)]
                                      for t in aq]
                elif it["kind"] == "grow":
                    L = it["box"][2] - it["box"][0] + it["box"][3] - it["box"][1]
                    t = math.radians(it["angle"])
                    it["baseline"] = [[round(it["x"] + q * math.cos(t), 1), round(it["y"] + q * math.sin(t), 1)]
                                      for q in np.linspace(0, 1.2 * L, 120)]
            del base
            key = f"g{k:03d}"
            img.save(OUT / f"{key}_draft.png")
            plate.save(OUT / f"{key}_plate.png")
            (OUT / f"{key}.json").write_text(json.dumps({"items": items}, ensure_ascii=False, indent=1), encoding="utf-8")
            print(key, [(it["kind"], it["text"]) for it in items])


def _iou(a, b) -> float:
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    i = max(0, w) * max(0, h)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def evaluate(only: str = "") -> dict:
    """Mỗi mục đáp án: ô L nào phủ nó (IoU khung hợp các ô L chạm khung đáp án), số ô L tách ra (1 = đúng), loại hình học đo
    được (curve / angle / runs) và sai số so với đáp án."""
    from PIL import Image
    from textfix import geo, slots
    geo.ENABLED = True   # đo bước hình học chữ dù máy chủ đang tắt nó
    keys = sorted(p.stem for p in OUT.glob("g*.json"))
    if only:
        keys = [k for k in keys if k in set(only.split(","))]
    stat = {"arc": [], "wave": [], "grow": [], "line": [], "mixed": []}
    for k in keys:
        gt = json.loads((OUT / f"{k}.json").read_text(encoding="utf-8"))["items"]
        D = np.asarray(Image.open(OUT / f"{k}_draft.png").convert("RGB"))
        P = np.asarray(Image.open(OUT / f"{k}_plate.png").convert("RGB"))
        M = slots.build(D, P)
        for it in gt:
            gb = it["box"]
            hit = [l for l in M["L"] if _iou(l["box"], gb) > 0 and slots._cov(gb, l["box"]) >= 0.5]
            iu = 0.0
            if hit:
                u = [min(l["box"][0] for l in hit), min(l["box"][1] for l in hit), max(l["box"][2] for l in hit),
                     max(l["box"][3] for l in hit)]
                iu = _iou(u, gb)
            row = {"key": k, "n_L": len(hit), "iou": round(iu, 2), "text": it["text"]}
            if it["kind"] in ("arc", "wave", "grow"):
                g = next((l for l in hit if l.get("path")), None)
                row["geo"] = g.get("geo") if g else None
                if g:   # sai lệch đường chân: khoảng cách từng điểm đường chân dò được tới đường chân đáp án / cỡ chữ
                    gtb = np.array(it["baseline"], float)
                    d = [float(np.min(np.hypot(*(gtb - np.array(p)).T))) for p in g["path"][2:-2]]
                    row["base_err"] = round(float(np.median(d)) / it["size"], 2)
                    if it["kind"] == "grow":
                        sc = g.get("scale")
                        row["scale_ok"] = bool(sc) and (sc[-1] > sc[0]) == (it["s1"] > it["s0"])
            elif it["kind"] == "line":
                a = [l.get("angle") or 0 for l in hit]
                row["ang_err"] = round(min((abs(x - it["angle"]) for x in a), default=99), 1)
            else:
                rr = next((l.get("runs") for l in hit if l.get("runs")), None)
                row["runs"] = len(rr) if rr else 0
                row["parts"] = len(it["parts"])
            stat[it["kind"]].append(row)
    for kind, rows in stat.items():
        if not rows:
            continue
        one = sum(r["n_L"] == 1 for r in rows) / len(rows)
        print(f"{kind:6} {len(rows):3} mục | đúng 1 ô L {one:.0%} | IoU khung TB {np.mean([r['iou'] for r in rows]):.2f}", end="")
        if kind in ("arc", "wave", "grow"):
            cv = [r for r in rows if r.get("geo")]
            print(f" | có đường chân {len(cv) / len(rows):.0%} (đúng loại {np.mean([r['geo'] == kind for r in rows]):.0%})"
                  + (f", lệch đường chân trung vị {np.median([r['base_err'] for r in cv]):.2f} cỡ" if cv else "")
                  + (f", chiều cỡ đúng {np.mean([r.get('scale_ok', False) for r in cv]):.0%}" if kind == "grow" and cv else ""))
        elif kind == "line":
            print(f" | sai góc trung vị {np.median([r['ang_err'] for r in rows]):.1f} độ")
        else:
            print(f" | tách đúng số đoạn cỡ {np.mean([r['runs'] == r['parts'] for r in rows]):.0%}")
    return stat


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "eval"])
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--only", default="")
    a = ap.parse_args(argv)
    if a.cmd == "make":
        make(a.n)
    else:
        evaluate(a.only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
