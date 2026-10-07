#!/usr/bin/env python3
"""MÁY CHỦ: sinh cặp NHÁP + BẢN XOÁ cho bộ đáp án ô (S# / I# / L#). Chỉ phần cần GPU; đo ô, gán nhãn, chấm điểm chạy ở máy cá nhân.

  # trong ô notebook JupyterLab (để tự tải zip về):
  %run scripts/make_pairs.py                         # bộ gán nhãn bench/prompts_gt.json: 17 prompt (dễ / trung / khó), 30 cặp
  %run scripts/make_pairs.py --only h01,h04          # vài prompt
  %run scripts/make_pairs.py --set bench/prompts_C.json --seeds 0,1,2
  # terminal (không tự tải, tải tay các zip trong output/pairs_zip/):
  python scripts/make_pairs.py
  %run scripts/make_pairs.py --ship-only             # không vẽ, chỉ gói lại zip các cặp đã có (vd. đổi cỡ zip: --zip-mb 15)

Ra output/pairs/<key>_draft.png, <key>_plate.png (đặt tên như probe.py --data), pairs.json (prompt, câu khách, cỡ, seed, căn bản
xoá, thời gian), rồi gói output/pairs_zip/pairs_NN.zip (mỗi zip < 24 MB) và tự tải về.
key = <mã prompt>_s<seed>, mã prompt = phần trước dấu "_" đầu tiên của key trong bộ prompt (p11_distill_s4 -> p11). Seed: --seeds
nếu có, không thì "seeds" của từng prompt, không có nữa thì 0,1,2.
Prompt có "ref" (ảnh sản phẩm trong bench/refs/): ảnh đó làm ảnh tham chiếu cho FLUX, như khi người dùng tải ảnh sản phẩm lên
(thu nhỏ cạnh dài 1024 px như máy chủ). Cặp đã có thì bỏ qua (chạy lại được sau khi đứt giữa chừng); --force vẽ lại. Mặc định dùng MỌI GPU (mỗi card một bản FLUX,
chia đều các cặp; textfix/multigpu.py); --gpus 1: một tiến trình như cũ.
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
    from textfix import config, multigpu, ship
    from textfix.flux import ERASE_PROMPT, Flux, register
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", type=Path, default=ROOT / "bench" / "prompts_gt.json")
    ap.add_argument("--seeds", default="", help="ghi đè seed của từng prompt, vd 0,1,2")
    ap.add_argument("--only", default="", help="mã prompt, cách nhau dấu phẩy (p11,q05)")
    ap.add_argument("--model", default="distill", choices=["distill", "base"])
    ap.add_argument("-o", "--out", type=Path, default=ROOT / "output" / "pairs")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--ship-only", action="store_true", help="không nạp FLUX / vẽ, chỉ gói lại zip các cặp đã có")
    ap.add_argument("--zip-mb", type=float, default=None, help="cỡ tối đa mỗi zip (MiB), mặc định 22 (< 24 MB)")
    multigpu.add_args(ap)
    a = ap.parse_args(argv)
    config.load_env()
    force_seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    only = {s.strip() for s in a.only.split(",") if s.strip()}
    jobs = [j for j in json.loads(a.set.read_text(encoding="utf-8")) if not only or j["key"].split("_")[0] in only]
    a.out.mkdir(parents=True, exist_ok=True)
    meta_f = a.out / "pairs.json"
    meta = multigpu.merge(meta_f) if not a.shard else json.loads(meta_f.read_text(encoding="utf-8")) if meta_f.exists() else {}
    seeds_of = lambda j: force_seeds or j.get("seeds") or [0, 1, 2]   # noqa: E731
    key_of = lambda j, seed: f"{j['key'].split('_')[0]}_s{seed}"   # noqa: E731
    done = lambda k: (a.out / f"{k}_draft.png").exists() and (a.out / f"{k}_plate.png").exists() and k in meta   # noqa: E731
    todo = [] if a.ship_only else [(j, s) for j in jobs for s in seeds_of(j) if a.force or not done(key_of(j, s))]
    if todo and multigpu.fanout(a, __file__, argv):   # mỗi card một tiến trình con, chia đều các cặp
        meta = multigpu.merge(meta_f)
        todo = []
    todo = multigpu.shard(todo, a)
    mine = meta if not a.shard else {}   # con chỉ ghi phần của mình vào pairs.shard<k>.json, cha gộp
    out_f = multigpu.part(meta_f, a)

    def ref_of(j):
        if not j.get("ref"):
            return None
        im = Image.open(ROOT / j["ref"]).convert("RGB")
        im.thumbnail((1024, 1024))   # như server/app.py decode_image
        return np.asarray(im)

    if todo:
        t0 = time.time()
        F = Flux(a.model).load()
        print(f"nạp FLUX {a.model} {time.time() - t0:.0f}s, {len(todo)} cặp cần vẽ")
    for j, seed in todo:
        key = key_of(j, seed)
        fd, fp = a.out / f"{key}_draft.png", a.out / f"{key}_plate.png"
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
        mine[key] = {"prompt_key": j["key"], **{k: j[k] for k in ("level", "category", "focus", "ref", "scene_texts") if k in j},
                     "prompt": j["prompt"], "texts": j["texts"], "w": j["w"], "h": j["h"],
                     "seed": seed, "model": a.model, "erase_prompt": ERASE_PROMPT, "register": reg,
                     "timing": {"ve_nhap": round(t2 - t1, 1), "xoa": round(t3 - t2, 1)}}
        out_f.write_text(json.dumps(mine, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{key}: nháp {t2 - t1:.1f}s, xoá {t3 - t2:.1f}s, căn {reg}")
    if a.shard:
        return 0
    print(f"xong {len(meta)} cặp: {a.out}")
    if not a.no_ship:
        files = [(meta_f, "pairs.json")] + [(p, p.name) for p in sorted(a.out.glob("*_draft.png")) + sorted(a.out.glob("*_plate.png"))
                                            if p.name.rsplit("_", 1)[0] in meta]
        ship.offer(ship.pack(files, a.out.parent / "pairs_zip", "pairs", int(a.zip_mb * 2**20) if a.zip_mb else None,
                               group=lambda arc: arc.rsplit("_", 1)[0]))   # <key>_draft.png + <key>_plate.png chung một zip
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
