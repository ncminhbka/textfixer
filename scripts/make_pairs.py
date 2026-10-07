#!/usr/bin/env python3
"""MÁY CHỦ: sinh cặp NHÁP + BẢN XOÁ cho bộ đáp án ô (S# / I# / L#). Chỉ phần cần GPU; đo ô, gán nhãn, chấm điểm chạy ở máy cá nhân.

  # trong ô notebook JupyterLab (để tự tải zip về):
  %run scripts/make_pairs.py                         # bộ gán nhãn bench/prompts_gt.json: 17 prompt (dễ / trung / khó), 30 cặp
  %run scripts/make_pairs.py --only h01,h04          # vài prompt
  %run scripts/make_pairs.py --set bench/prompts_C.json --seeds 0,1,2
  # terminal (không tự tải, tải tay các zip trong output/pairs_zip/):
  python scripts/make_pairs.py

Ra output/pairs/<key>_draft.png, <key>_plate.png (đặt tên như probe.py --data), pairs.json (prompt, câu khách, cỡ, seed, căn bản
xoá, thời gian), rồi gói output/pairs_zip/pairs_NN.zip (mỗi zip < 27 MB) và tự tải về.
key = <mã prompt>_s<seed>, mã prompt = phần trước dấu "_" đầu tiên của key trong bộ prompt (p11_distill_s4 -> p11). Seed: --seeds
nếu có, không thì "seeds" của từng prompt, không có nữa thì 0,1,2.
Prompt có "ref" (ảnh sản phẩm trong bench/refs/): ảnh đó làm ảnh tham chiếu cho FLUX, như khi người dùng tải ảnh sản phẩm lên
(thu nhỏ cạnh dài 1024 px như máy chủ). Cặp đã có thì bỏ qua (chạy lại được sau khi đứt giữa chừng); --force vẽ lại.
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


def main(argv: list[str] | None = None) -> int:
    import numpy as np
    from PIL import Image
    from textfix import config, ship
    from textfix.flux import ERASE_PROMPT, Flux, register
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", type=Path, default=ROOT / "bench" / "prompts_gt.json")
    ap.add_argument("--seeds", default="", help="ghi đè seed của từng prompt, vd 0,1,2")
    ap.add_argument("--only", default="", help="mã prompt, cách nhau dấu phẩy (p11,q05)")
    ap.add_argument("--model", default="distill", choices=["distill", "base"])
    ap.add_argument("-o", "--out", type=Path, default=ROOT / "output" / "pairs")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-ship", action="store_true", help="không gói / tải zip")
    a = ap.parse_args(argv)
    config.load_env()
    force_seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    only = {s.strip() for s in a.only.split(",") if s.strip()}
    jobs = [j for j in json.loads(a.set.read_text(encoding="utf-8")) if not only or j["key"].split("_")[0] in only]
    a.out.mkdir(parents=True, exist_ok=True)
    meta_f = a.out / "pairs.json"
    meta = json.loads(meta_f.read_text(encoding="utf-8")) if meta_f.exists() else {}
    t0 = time.time()
    F = Flux(a.model).load()
    seeds_of = lambda j: force_seeds or j.get("seeds") or [0, 1, 2]   # noqa: E731
    def ref_of(j):
        if not j.get("ref"):
            return None
        im = Image.open(ROOT / j["ref"]).convert("RGB")
        im.thumbnail((1024, 1024))   # như server/app.py decode_image
        return np.asarray(im)

    print(f"nạp FLUX {a.model} {time.time() - t0:.0f}s, {len(jobs)} prompt, {sum(len(seeds_of(j)) for j in jobs)} cặp")
    for j in jobs:
        for seed in seeds_of(j):
            key = f"{j['key'].split('_')[0]}_s{seed}"
            fd, fp = a.out / f"{key}_draft.png", a.out / f"{key}_plate.png"
            if fd.exists() and fp.exists() and key in meta and not a.force:
                print(f"{key}: đã có, bỏ qua")
                continue
            try:
                t1 = time.time()
                ref = ref_of(j)
                draft = F.generate(j["prompt"], j["w"], j["h"], seed, refs=[ref] if ref is not None else None)
                t2 = time.time()
                plate, reg = register(draft, F.edit(draft, ERASE_PROMPT, seed))
                t3 = time.time()
            except Exception as e:
                print(f"{key}: LỖI {type(e).__name__}: {str(e)[:300]}")
                continue
            Image.fromarray(draft).save(fd)
            Image.fromarray(plate).save(fp)
            meta[key] = {"prompt_key": j["key"], **{k: j[k] for k in ("level", "category", "focus", "ref", "scene_texts") if k in j},
                         "prompt": j["prompt"], "texts": j["texts"], "w": j["w"], "h": j["h"],
                         "seed": seed, "model": a.model, "erase_prompt": ERASE_PROMPT, "register": reg,
                         "timing": {"ve_nhap": round(t2 - t1, 1), "xoa": round(t3 - t2, 1)}}
            meta_f.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"{key}: nháp {t2 - t1:.1f}s, xoá {t3 - t2:.1f}s, căn {reg}")
    print(f"xong {len(meta)} cặp: {a.out}")
    if not a.no_ship:
        files = [(meta_f, "pairs.json")] + [(p, p.name) for p in sorted(a.out.glob("*_draft.png")) + sorted(a.out.glob("*_plate.png"))
                                            if p.name.rsplit("_", 1)[0] in meta]
        ship.offer(ship.pack(files, a.out.parent / "pairs_zip", "pairs"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
