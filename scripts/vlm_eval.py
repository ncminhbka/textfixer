#!/usr/bin/env python3
"""CHẤM kết quả scripts/vlm_probe.py (giải nén zip vào output/vlm_probe/) ở máy cá nhân.

  python scripts/vlm_eval.py                 # bảng theo cấu hình + output/vlm_eval/<key>.jpg (nháp | các cấu hình cạnh nhau)

Mỗi cấu hình:
  lỗi1 / lỗi2   lỗi khách quan (render.check) lượt đầu / sau vòng sửa
  thiếu         câu khách designer khai không vẽ
  sai DK        câu khách có DỮ KIỆN (cụm số: giá, điện thoại, ngày, %) mà poster thiếu một cụm số đó và designer không khai thiếu
  P keep        ô nằm >= 70% trong khung chữ cảnh P (bench/slots_gt) được "keep" / tổng ô như vậy (chữ trên sản phẩm phải giữ)
  token, giây   lời gọi lập lệnh (không tính vòng sửa)
"""

from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", line_buffering=True)

PROBE = ROOT / "output" / "vlm_probe"
OUT = ROOT / "output" / "vlm_eval"
GT = ROOT / "bench" / "slots_gt"
NUM = re.compile(r"\d[\d.,:/-]*\d|\d")


def plain(h: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", h or "")).split())


def inside(a, b) -> float:
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h) / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))


def main() -> int:
    from PIL import Image, ImageDraw
    meta = json.loads((ROOT / "output" / "pairs" / "pairs.json").read_text(encoding="utf-8"))
    views = sorted(d.name for d in PROBE.iterdir() if d.is_dir() and not d.name.startswith("_"))
    keys = sorted({f.stem for v in views for f in (PROBE / v).glob("*.json")})
    OUT.mkdir(parents=True, exist_ok=True)
    tot = {v: {"n": 0, "fail": 0, "e1": 0, "e2": 0, "miss": 0, "fact": 0, "facts": 0, "pk": 0, "pn": 0, "tok": [], "s": []} for v in views}
    bad_rows = []
    for k in keys:
        texts = [t for t in meta[k]["texts"] if not t.get("scene")]
        gtf = GT / f"{k}.json"
        P = [b["box"] for b in json.loads(gtf.read_text(encoding="utf-8"))["boxes"] if b["cls"] == "P"] if gtf.exists() else []
        tiles = [("nháp", Image.open(ROOT / "output" / "pairs" / f"{k}_draft.png").convert("RGB"))]
        for v in views:
            f = PROBE / v / f"{k}.json"
            if not f.exists():
                continue
            r = json.loads(f.read_text(encoding="utf-8"))
            t = tot[v]
            t["n"] += 1
            if r.get("error"):
                t["fail"] += 1
                continue
            t["e1"] += len(r.get("errs_first") or [])
            t["e2"] += len(r.get("errs_final") or [])
            miss = r.get("missing") or []
            t["miss"] += len(miss)
            ops = r.get("ops") or []
            shown = " ".join(plain(o.get("html")) for o in ops if o.get("kind") == "text")
            for tx in texts:   # dữ kiện: mọi cụm số của câu khách phải có nguyên văn, trừ khi designer khai thiếu câu đó
                nums = NUM.findall(tx["text"])
                if not nums or tx["text"] in miss:
                    continue
                t["facts"] += 1
                lost = [n for n in nums if n not in shown]
                if lost:
                    t["fact"] += 1
                    bad_rows.append((v, k, tx["text"], lost))
            by = {m["id"]: m for m in r["slots"]["L"] + r["slots"]["S"] + r["slots"]["I"]}
            kind = {m: o.get("kind") for o in ops for m in (o.get("marks") or o.get("slots") or [])}
            for sid, m in by.items():
                if any(inside(m["box"], q) >= 0.7 for q in P):
                    t["pn"] += 1
                    t["pk"] += kind.get(sid) == "keep"
            pm = r.get("plan_meta") or {}
            if pm.get("prompt_tokens"):
                t["tok"].append(pm["prompt_tokens"] + (pm.get("completion_tokens") or 0))
            if pm.get("s"):
                t["s"].append(pm["s"])
            pf = PROBE / v / f"{k}.jpg"
            if pf.exists():
                tiles.append((f"{v}  lỗi {len(r.get('errs_final') or [])}  thiếu {len(miss)}", Image.open(pf).convert("RGB")))
        h = 640
        ims = [(n, im.resize((int(im.width * h / im.height), h))) for n, im in tiles]
        sheet = Image.new("RGB", (sum(im.width + 8 for _, im in ims), h + 22), "white")
        x = 0
        for n, im in ims:
            sheet.paste(im, (x, 22))
            ImageDraw.Draw(sheet).text((x + 4, 4), n, fill="black")
            x += im.width + 8
        sheet.save(OUT / f"{k}.jpg", quality=82)
    print(f"{'cấu hình':<9}{'ảnh':>4}{'hỏng':>5}{'lỗi1':>6}{'lỗi2':>6}{'thiếu':>7}{'sai DK':>9}{'P keep':>9}{'token':>8}{'giây':>7}")
    for v in views:
        t = tot[v]
        mean = lambda xs: sum(xs) / len(xs) if xs else 0   # noqa: E731
        print(f"{v:<9}{t['n']:>4}{t['fail']:>5}{t['e1']:>6}{t['e2']:>6}{t['miss']:>7}{t['fact']:>5}/{t['facts']:<3}"
              f"{t['pk']:>5}/{t['pn']:<3}{mean(t['tok']):>8.0f}{mean(t['s']):>7.1f}")
    if bad_rows:
        print("\nsai dữ kiện (cụm số không có trên poster, không khai thiếu):")
        for v, k, tx, lost in bad_rows:
            print(f"  {v:<4} {k:<8} {lost}  <- {tx}")
    print(f"\nảnh so sánh: {OUT}/<key>.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
