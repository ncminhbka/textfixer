#!/usr/bin/env python3
"""MÁY CHỦ: thử các LỜI DẶN XOÁ trên các nháp có sẵn (output/pairs, từ make_pairs.py). Không vẽ nháp mới, chỉ xoá lại.

  %run scripts/erase_probe.py                        # mọi lời dặn x mọi nháp -> zip < 24 MB, tự tải về
  %run scripts/erase_probe.py --variants B,F --only h06_s0,r02_s0
  %run scripts/erase_probe.py --ship-only            # chỉ gói lại kết quả đã có
Mặc định dùng MỌI GPU (mỗi card một bản FLUX, chia đều; textfix/multigpu.py); --gpus 1: một tiến trình.

Ra output/erase_probe/<lời dặn>/<key>_plate.png (đã căn theo nháp như erase()), variants.json (lời dặn, thời gian, căn).
Lời dặn A (hiện tại) = bản xoá trong output/pairs, không chạy lại. A2 = xoá lần hai trên bản xoá A (lời dặn A).
Chấm ở máy cá nhân: python scripts/erase_eval.py (chữ còn sót, chữ trên sản phẩm còn / mất, ảnh so sánh).
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

# lời dặn thử. A = lời dặn hiện tại (flux.ERASE_PROMPT). Lời dặn kể tên vật từng làm model distill vẽ thêm vật đó (07/10):
# mỗi biến thể chỉ thêm MỘT ý để biết ý nào có tác dụng / gây hại.
VARIANTS = {
    "B": "Remove every piece of text from this image: all letters, numbers and symbols, including text inside buttons, badges and "
         "labels. Keep everything else exactly the same.",
    "C": "Remove all text, icons and star ratings from this image. Keep everything else exactly the same.",
    "D": "Remove all overlaid text and graphic elements from this poster. Keep the photo exactly the same, including any text "
         "printed on products.",
    "E": "Remove all text from this image. Keep everything else exactly the same, including the products and their labels.",
    "F": "Remove all text, numbers, icons, logos and star ratings from this image, including inside buttons, badges and labels. "
         "Keep everything else exactly the same.",
    "A2": None,   # xoá lần hai bằng lời dặn A trên bản xoá A
}


def main(argv: list[str] | None = None) -> int:
    import numpy as np
    from PIL import Image
    from textfix import config, multigpu, ship
    from textfix.flux import ERASE_PROMPT, Flux, register
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, default=ROOT / "output" / "pairs")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--only", default="", help="key, cách nhau dấu phẩy (h06_s0,r02_s0)")
    ap.add_argument("-o", "--out", type=Path, default=ROOT / "output" / "erase_probe")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--ship-only", action="store_true")
    ap.add_argument("--zip-mb", type=float, default=None)
    multigpu.add_args(ap)
    a = ap.parse_args(argv)
    config.load_env()
    meta = json.loads((a.pairs / "pairs.json").read_text(encoding="utf-8"))
    keys = [k for k in sorted(meta) if (a.pairs / f"{k}_draft.png").exists() and (not a.only or k in a.only.split(","))]
    vs = [v for v in a.variants.split(",") if v in VARIANTS]
    a.out.mkdir(parents=True, exist_ok=True)
    info_f = a.out / "variants.json"
    todo = [] if a.ship_only else [(v, k) for v in vs for k in keys if a.force or not (a.out / v / f"{k}_plate.png").exists()]
    if todo and multigpu.fanout(a, __file__, argv):   # mỗi card một tiến trình con, chia đều các bản xoá
        todo = []
    todo = multigpu.shard(todo, a)
    info = {} if a.shard else multigpu.merge(info_f, deep=("runs",))
    info_out = multigpu.part(info_f, a)
    if not a.shard:
        info["A"] = {"prompt": ERASE_PROMPT, "note": "bản xoá trong output/pairs"}
    if todo:
        t0 = time.time()
        F = Flux("distill").load()
        print(f"nạp FLUX {time.time() - t0:.0f}s; {len(todo)} bản xoá cần chạy")
    for v, k in todo:
        prompt = VARIANTS[v] or ERASE_PROMPT
        info.setdefault(v, {"prompt": prompt, "note": "xoá lần hai trên bản xoá A" if v == "A2" else "", "runs": {}})
        (a.out / v).mkdir(exist_ok=True)
        seed = int(meta[k].get("seed", 0))
        draft = np.asarray(Image.open(a.pairs / f"{k}_draft.png").convert("RGB"))
        src = np.asarray(Image.open(a.pairs / f"{k}_plate.png").convert("RGB")) if v == "A2" else draft
        try:
            t1 = time.time()
            plate, reg = register(draft, F.edit(src, prompt, seed))
            dt = time.time() - t1
        except Exception as e:
            print(f"{v} {k}: LỖI {type(e).__name__}: {str(e)[:300]}")
            continue
        Image.fromarray(plate).save(a.out / v / f"{k}_plate.png")
        info[v]["runs"][k] = {"s": round(dt, 1), "register": reg}
        info_out.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{v} {k}: {dt:.1f}s")
    if a.shard:
        return 0
    info_out.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    info = multigpu.merge(info_f, deep=("runs",))
    files = [(info_f, "variants.json")] + [(p, f"{p.parent.name}/{p.name}") for v in vs if (a.out / v).is_dir()
                                           for p in sorted((a.out / v).glob("*_plate.png"))]
    print(f"xong: {len(files) - 1} bản xoá ở {a.out}")
    if not a.no_ship:
        ship.offer(ship.pack(files, a.out.parent / "erase_probe_zip", "erase", int(a.zip_mb * 2**20) if a.zip_mb else None))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
