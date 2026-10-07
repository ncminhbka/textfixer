#!/usr/bin/env python3
"""THỬ ENGINE trên bộ prompt cố định (bench/prompts_C.json: prompt FLUX + câu khách + cỡ), nhìn bằng mắt.

  %run scripts/probe.py                                 # máy chủ, trong ô notebook: FLUX distill vẽ nháp + xoá, VLM thật;
                                                        # cuối cùng gói output/probe thành zip < 27 MB và tự tải về
  python scripts/probe.py                               # terminal: như trên, zip ở output/probe_zip/ (tải tay)
  python scripts/probe.py --only p11_distill_s4,q21_distill_s0
  python scripts/probe.py --data <thư mục>              # dùng nháp / bản xoá có sẵn: <key>_draft.png (+ <key>_plate.png)
  python scripts/probe.py --data <thư mục> --dry        # không VLM (lệnh giả: vỏ tô màu đo, chi tiết = chấm, chữ = chữ OCR):
                                                        # kiểm phần đo / dựng ở máy không GPU, không mạng

Ra output/probe/<key>.jpg (nháp | ô | bản xoá | poster), <key>_plan.json (ô, lệnh, lỗi, nhật ký, thời gian), <key>_poster.png;
gói output/probe_zip/probe_NN.zip (--no-ship: không gói).
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


def dry_vlm(system: str, parts: list) -> dict:
    """VLM giả: đọc danh sách ô trong lời nhắn, mỗi ô một lệnh tối thiểu."""
    import re
    text = parts[0]
    if "Involved ops" in text:   # vòng sửa: không vá gì
        return {"patches": [], "why": "dry"}
    ops = []
    for m in re.finditer(r"^(S\d+): shell .*?fill (#[0-9a-f]{6}), radius ~(\d+)px", text, re.M):
        ops.append({"id": f"s{m.group(1)}", "slots": [m.group(1)], "kind": "shape", "html": "",
                    "box_style": f"background:{m.group(2)};border-radius:{m.group(3)}px"})
    for m in re.finditer(r"^(I\d+): detail .*?color (#[0-9a-f]{6})", text, re.M):
        ops.append({"id": f"i{m.group(1)}", "slots": [m.group(1)], "kind": "icon", "html": f'<i-icon name="circle" color="{m.group(2)}"></i-icon>'})
    for m in re.finditer(r'^(L\d+): "(.*)" \(text', text, re.M):
        ops.append({"id": f"t{m.group(1)}", "slots": [m.group(1)], "kind": "text", "html": m.group(2)})
    return {"style": {}, "ops": ops, "missing": []}


def main() -> int:
    import numpy as np
    from PIL import Image
    from textfix import config
    from textfix.engine import Engine
    from textfix.render import ops_image
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--set", type=Path, default=ROOT / "bench" / "prompts_C.json")
    ap.add_argument("--data", type=Path, default=None, help="thư mục nháp có sẵn (<key>_draft.png, tuỳ chọn <key>_plate.png)")
    ap.add_argument("--plate-suffix", default="plate", help="bản xoá có sẵn: <key>_<hậu tố>.png")
    ap.add_argument("--dry", action="store_true", help="VLM giả (kiểm đo / dựng)")
    ap.add_argument("-o", "--out", type=Path, default=ROOT / "output" / "probe")
    ap.add_argument("--no-ship", action="store_true", help="không gói / tải zip")
    a = ap.parse_args()
    config.load_env()
    jobs = [j for j in json.loads(a.set.read_text(encoding="utf-8")) if not a.only or j["key"] in a.only.split(",")]
    a.out.mkdir(parents=True, exist_ok=True)
    F = None
    if a.data is None or any(not (a.data / f"{j['key']}_{a.plate_suffix}.png").exists() for j in jobs):
        from textfix.flux import Flux
        t0 = time.time()
        F = Flux("distill").load()
        print(f"nạp FLUX distill {time.time() - t0:.0f}s")
    fake = dry_vlm if a.dry else None
    with Engine(F, progress=lambda m: print("   ..", m, flush=True), llm=fake, vlm=fake) as E:
        for j in jobs:
            k = j["key"]
            seed = int(k.rsplit("_s", 1)[1]) if "_s" in k else (j.get("seeds") or [0])[0]
            B = {"prompt_en": j["prompt"], "texts": [t for t in j["texts"] if not t.get("scene")]}
            product = None
            if j.get("ref"):   # ảnh sản phẩm người dùng tải lên (bench/refs), thu nhỏ như máy chủ
                im = Image.open(ROOT / j["ref"]).convert("RGB")
                im.thumbnail((1024, 1024))
                product = np.asarray(im)
            t0 = time.time()
            try:
                if a.data is not None and (a.data / f"{k}_draft.png").exists():
                    draft = np.asarray(Image.open(a.data / f"{k}_draft.png").convert("RGB"))
                    pf = a.data / f"{k}_{a.plate_suffix}.png"
                    plate = np.asarray(Image.open(pf).convert("RGB").resize((draft.shape[1], draft.shape[0]))) if pf.exists() else None
                    r = E.fix(draft, B, seed, plate)
                else:
                    r = E.make(B, j["w"], j["h"], seed, product)
            except Exception as e:
                print(f"{k}: LỖI {type(e).__name__}: {str(e)[:300]}", flush=True)
                continue
            Image.fromarray(r["poster"]).save(a.out / f"{k}_poster.png")
            row = Image.fromarray(r["steps"])
            ops = ops_image(r["draft"], {**r["plan"]["slots"], "size": [r["draft"].shape[1], r["draft"].shape[0]],
                                         "by": {m["id"]: m for m in r["plan"]["slots"]["L"] + r["plan"]["slots"]["S"] + r["plan"]["slots"]["I"]}},
                            r["plan"])
            ops = ops.resize((int(ops.width * row.height / ops.height), row.height))
            sheet = Image.new("RGB", (row.width + ops.width + 10, row.height), "white")
            sheet.paste(row, (0, 0))
            sheet.paste(ops, (row.width + 10, 0))
            sheet.save(a.out / f"{k}.jpg", quality=86)
            (a.out / f"{k}_plan.json").write_text(json.dumps({**r["plan"], "timing": r["timing"]}, ensure_ascii=False, indent=1),
                                                  encoding="utf-8")
            kinds: dict = {}
            for op in r["plan"]["ops"]:
                kinds[op.get("kind")] = kinds.get(op.get("kind"), 0) + 1
            print(f"{k}: lệnh {kinds}; lỗi {len(r['plan']['errors'])}; thiếu {len(r['plan']['missing'])}; {r['timing']} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    print(f"xong: {a.out}/<key>.jpg")
    if not a.no_ship:
        from textfix import ship
        ship.offer(ship.pack_dir(a.out, a.out.parent / f"{a.out.name}_zip", a.out.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
