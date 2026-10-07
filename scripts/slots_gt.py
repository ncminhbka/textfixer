#!/usr/bin/env python3
"""BỘ ĐÁP ÁN Ô (S# vỏ / I# chi tiết / L# dòng chữ / P chữ thuộc cảnh) và CHẤM bộ đo ô. Chạy ở máy cá nhân (CPU: OCR + lớp phủ).

  python scripts/slots_gt.py prefill            # output/pairs/*_draft.png + _plate.png -> bench/slots_gt/<key>.json (khung đo hiện
                                                # tại, trạng thái "prefill"); KHÔNG ghi đè tệp đã sửa tay (trạng thái khác)
  # sửa tay bằng tools/annotate.html (Chrome / Edge, mở thư mục repo), luật gán nhãn: docs/SLOTS_GT.md
  python scripts/slots_gt.py eval               # chấm bộ đo hiện tại trên các tệp "done" -> bảng + output/slots_eval/<key>.jpg
  python scripts/slots_gt.py eval --all         # chấm cả tệp chưa "done" (chỉ để thử)
  python scripts/slots_gt.py eval --save base   # lưu kết quả làm mốc (output/slots_eval/base.json) ...
  python scripts/slots_gt.py eval --vs base     # ... rồi so lần sau với mốc

Tệp đáp án: {"key", "size": [W, H], "status": "prefill" | "editing" | "done", "boxes": [{"cls": "L|S|I|P", "box": [x0,y0,x1,y1]}]}
Khớp: theo từng lớp, IoU lớn nhất trước, IoU >= 0.5. Dòng L dự đoán khớp một khung P (chữ in trên sản phẩm / vật trong ảnh) tính
riêng ("L trúng P"), không tính là L sai.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", line_buffering=True)

PAIRS = ROOT / "output" / "pairs"
GT = ROOT / "bench" / "slots_gt"
OUT = ROOT / "output" / "slots_eval"
CLASSES = ("L", "S", "I")
IOU = 0.5


def _load(key: str, pairs: Path):
    import numpy as np
    from PIL import Image
    d = np.asarray(Image.open(pairs / f"{key}_draft.png").convert("RGB"))
    p = np.asarray(Image.open(pairs / f"{key}_plate.png").convert("RGB").resize((d.shape[1], d.shape[0])))
    return d, p


def predict(draft, plate) -> list[dict]:
    """Ô của bộ đo hiện tại (slots.build) -> [{cls, box, id}]. Dòng nghiêng: khung thẳng ôm ngoài."""
    from textfix.slots import build
    M = build(draft, plate)
    return [{"cls": c, "box": [round(float(v), 1) for v in m["box"]], "id": m["id"]} for c in CLASSES for m in M[c]]


def keys(pairs: Path) -> list[str]:
    return sorted(p.name[:-len("_draft.png")] for p in pairs.glob("*_draft.png") if (pairs / p.name.replace("_draft", "_plate")).exists())


# ---------------------------------------------------------------------------------------------------- prefill
def prefill(a) -> int:
    GT.mkdir(parents=True, exist_ok=True)
    ks = [k for k in keys(a.pairs) if not a.only or k in a.only.split(",")]
    if not ks:
        print(f"không có cặp nào trong {a.pairs} (cần <key>_draft.png + <key>_plate.png, giải nén zip từ make_pairs.py vào đây)")
        return 1
    for k in ks:
        f = GT / f"{k}.json"
        if f.exists():
            st = json.loads(f.read_text(encoding="utf-8")).get("status")
            if st != "prefill" or not a.force:
                print(f"{k}: đã có ({st}), giữ nguyên" + ("" if st != "prefill" else " -- --force để đo lại"))
                continue
        t0 = time.time()
        d, p = _load(k, a.pairs)
        boxes = [{"cls": b["cls"], "box": b["box"]} for b in predict(d, p)]
        f.write_text(json.dumps({"key": k, "size": [d.shape[1], d.shape[0]], "status": "prefill", "boxes": boxes},
                                ensure_ascii=False, indent=1), encoding="utf-8")
        n = {c: sum(b["cls"] == c for b in boxes) for c in CLASSES}
        print(f"{k}: {n} ({time.time() - t0:.1f}s)")
    return 0


# ---------------------------------------------------------------------------------------------------- chấm
def iou(a, b) -> float:
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    i = w * h
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def match(pred: list, gt: list, thr: float = IOU) -> list[tuple[int, int, float]]:
    """Khớp tham lam IoU lớn nhất trước -> [(chỉ số pred, chỉ số gt, iou)]."""
    cand = sorted(((iou(p["box"], g["box"]), i, j) for i, p in enumerate(pred) for j, g in enumerate(gt)), reverse=True)
    used_p, used_g, out = set(), set(), []
    for v, i, j in cand:
        if v < thr:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        out.append((i, j, v))
    return out


def score(pred: list, gt: list) -> dict:
    """Điểm một ảnh: theo lớp {tp, fp, fn, iou_sum} + L trúng P; kèm nhãn từng khung để vẽ."""
    res, tags = {}, {"pred": {}, "gt": {}}
    P = [g for g in gt if g["cls"] == "P"]
    for c in CLASSES:
        pi = [i for i, p in enumerate(pred) if p["cls"] == c]
        gi = [j for j, g in enumerate(gt) if g["cls"] == c]
        m = match([pred[i] for i in pi], [gt[j] for j in gi])
        mp, mg = {pi[i] for i, _, _ in m}, {gi[j] for _, j, _ in m}
        on_p = set()
        if c == "L" and P:   # dòng L chưa khớp L nào mà trúng chữ cảnh P
            rest = [i for i in pi if i not in mp]
            on_p = {rest[i] for i, _, _ in match([pred[i] for i in rest], P, 0.3)}
        res[c] = {"tp": len(m), "fp": len(pi) - len(m) - len(on_p), "fn": len(gi) - len(m), "iou_sum": sum(v for _, _, v in m),
                  **({"on_p": len(on_p), "p": len(P)} if c == "L" else {})}
        for i in pi:
            tags["pred"][i] = "tp" if i in mp else "p" if i in on_p else "fp"
        for j in gi:
            tags["gt"][j] = "tp" if j in mg else "fn"
    return {"by_cls": res, "tags": tags}


def _prf(r: dict) -> tuple[float, float, float, float]:
    tp, fp, fn = r["tp"], r["fp"], r["fn"]
    p = tp / (tp + fp) if tp + fp else 1.0
    rc = tp / (tp + fn) if tp + fn else 1.0
    f = 2 * p * rc / (p + rc) if p + rc else 0.0
    return p, rc, f, (r["iou_sum"] / tp if tp else 0.0)


COL = {"tp": (20, 170, 60), "fp": (230, 20, 20), "p": (150, 60, 220), "fn": (20, 90, 240)}


def draw(draft, pred, gt, tags, path: Path) -> None:
    """Nháp mờ: GT chưa khớp (xanh dương, nét dày), dự đoán đúng (xanh lá), sai (đỏ), trúng chữ cảnh P (tím); nhãn lớp."""
    from PIL import Image, ImageDraw
    im = Image.fromarray((draft * 0.6 + 100).astype("uint8"))
    d = ImageDraw.Draw(im)
    for j, g in enumerate(gt):
        t = tags["gt"].get(j)
        if g["cls"] == "P":
            d.rectangle(g["box"], outline=(150, 60, 220), width=1)
        elif t == "fn":
            d.rectangle(g["box"], outline=COL["fn"], width=4)
            d.text((g["box"][0] + 2, g["box"][1] + 1), f"thiếu {g['cls']}", fill=COL["fn"])
    for i, p in enumerate(pred):
        t = tags["pred"].get(i, "fp")
        d.rectangle(p["box"], outline=COL[t], width=2)
        d.text((p["box"][0] + 2, p["box"][3] - 11), p.get("id", p["cls"]) + ("" if t == "tp" else f" {t}"), fill=COL[t])
    im.save(path, quality=85)


def evaluate(a) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(GT.glob("*.json"))
    docs = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    docs = [g for g in docs if (a.all or g.get("status") == "done") and (not a.only or g["key"] in a.only.split(","))]
    docs = [g for g in docs if (a.pairs / f"{g['key']}_draft.png").exists()]
    if not docs:
        print("chưa có tệp đáp án 'done' nào có ảnh (sửa bằng tools/annotate.html rồi đánh dấu xong; hoặc --all để thử)")
        return 1
    tot = {c: {"tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0, "on_p": 0, "p": 0} for c in CLASSES}
    per = {}
    t0 = time.time()
    for g in docs:
        d, p = _load(g["key"], a.pairs)
        pred = predict(d, p)
        s = score(pred, g["boxes"])
        per[g["key"]] = s["by_cls"]
        for c in CLASSES:
            for k, v in s["by_cls"][c].items():
                tot[c][k] += v
        draw(d, pred, g["boxes"], s["tags"], OUT / f"{g['key']}.jpg")
        line = "  ".join(f"{c} {r['tp']}/{r['tp'] + r['fn']} sai {r['fp']}" for c, r in s["by_cls"].items())
        print(f"{g['key']:<14} {line}" + (f"  L trúng P {s['by_cls']['L']['on_p']}" if s["by_cls"]["L"].get("on_p") else ""))
    print(f"\n{len(docs)} ảnh, {time.time() - t0:.0f}s. IoU >= {IOU}")
    base = json.loads((OUT / f"{a.vs}.json").read_text(encoding="utf-8"))["total"] if a.vs else None
    print(f"{'lớp':<4}{'precision':>10}{'recall':>9}{'F1':>8}{'IoU khớp':>10}{'thiếu':>7}{'sai':>6}")
    for c in CLASSES:
        pr, rc, f, mi = _prf(tot[c])
        row = f"{c:<4}{pr:>10.3f}{rc:>9.3f}{f:>8.3f}{mi:>10.3f}{tot[c]['fn']:>7}{tot[c]['fp']:>6}"
        if base:
            bp, br, bf, bi = _prf(base[c])
            row += f"   (F1 {f - bf:+.3f}, recall {rc - br:+.3f}, precision {pr - bp:+.3f}, IoU {mi - bi:+.3f})"
        print(row)
    if tot["L"]["p"]:
        print(f"chữ cảnh P: {tot['L']['p']} khung, bộ đo lấy nhầm thành L {tot['L']['on_p']}")
    print(f"ảnh so khớp: {OUT}/<key>.jpg (xanh lá đúng, đỏ sai, xanh dương thiếu, tím trúng chữ cảnh)")
    if a.save:
        (OUT / f"{a.save}.json").write_text(json.dumps({"total": tot, "per": per, "keys": [g["key"] for g in docs]},
                                                       ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"đã lưu mốc {OUT / (a.save + '.json')}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prefill", "eval"])
    ap.add_argument("--pairs", type=Path, default=PAIRS)
    ap.add_argument("--only", default="")
    ap.add_argument("--force", action="store_true", help="prefill: đo lại tệp còn ở trạng thái prefill")
    ap.add_argument("--all", action="store_true", help="eval: chấm cả tệp chưa 'done'")
    ap.add_argument("--save", default="", help="eval: lưu kết quả làm mốc với tên này")
    ap.add_argument("--vs", default="", help="eval: so với mốc đã lưu")
    a = ap.parse_args(argv)
    return prefill(a) if a.cmd == "prefill" else evaluate(a)


if __name__ == "__main__":
    raise SystemExit(main())
