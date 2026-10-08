#!/usr/bin/env python3
"""MÁY CHỦ: thử CẤU HÌNH ĐẦU VÀO của VLM designer (ảnh gửi kèm) trên cặp nháp / bản xoá có sẵn (output/pairs, make_pairs.py).
Không cần GPU: không FLUX; dọn chữ sót bằng inpaint để mọi cấu hình cùng một nền. Chỉ đổi ảnh gửi designer (slots.VIEWS);
tách ô, system prompt, dựng, kiểm, vòng sửa giữ nguyên như engine.

  %run scripts/vlm_probe.py                          # mọi cấu hình (A, AX, BX, CX) x ảnh dev -> zip < 24 MB, tự tải về
  %run scripts/vlm_probe.py --views A,BX --only h01_s0,r01_s0
  %run scripts/vlm_probe.py --split test             # CHỈ khi chốt (bench/slots_gt/split.json)
  %run scripts/vlm_probe.py --ship-only

Ra output/vlm_probe/<cấu hình>/<key>.jpg (poster), <key>.json (lệnh, lỗi lượt đầu / sau sửa, câu thiếu, nhật ký, token, giây),
summary.json. Lời gọi VLM chạy song song (--workers), dựng Chromium tuần tự. Kết quả VLM có cache (~/.cache/textfix/designer):
chạy lại không gọi lại; --fresh để gọi mới (đo độ dao động).
Chấm ở máy cá nhân: python scripts/vlm_eval.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", line_buffering=True)


def main(argv: list[str] | None = None) -> int:
    import numpy as np
    from PIL import Image
    from textfix import cleanup, config, render, ship, slots
    from textfix.browser import Browser
    from textfix.llm import json_call
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, default=ROOT / "output" / "pairs")
    ap.add_argument("--views", default=",".join(slots.VIEWS))
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--only", default="")
    ap.add_argument("--workers", type=int, default=4, help="số lời gọi VLM song song")
    ap.add_argument("--fresh", action="store_true", help="bỏ qua cache VLM (gọi mới)")
    ap.add_argument("-o", "--out", type=Path, default=ROOT / "output" / "vlm_probe")
    ap.add_argument("--ship-only", action="store_true")
    ap.add_argument("--no-ship", action="store_true")
    ap.add_argument("--zip-mb", type=float, default=None)
    a = ap.parse_args(argv)
    config.load_env()
    meta = json.loads((a.pairs / "pairs.json").read_text(encoding="utf-8"))
    keys = [k for k in sorted(meta) if (a.pairs / f"{k}_draft.png").exists() and (not a.only or k in a.only.split(","))]
    if a.split != "all":
        test = set(json.loads((ROOT / "bench" / "slots_gt" / "split.json").read_text(encoding="utf-8"))["test"])
        keys = [k for k in keys if (k.split("_")[0] in test) == (a.split == "test")]
    views = [v for v in a.views.split(",") if v in slots.VIEWS]
    a.out.mkdir(parents=True, exist_ok=True)
    if not a.ship_only:
        m = config.models()["vlm"]
        vlm = json_call(m["url"], m["model"], m["key_env"], tag="designer" + ("_fresh" + str(int(time.time())) if a.fresh else ""))
        # 1. nền + ô (một lần mỗi ảnh, dùng chung mọi cấu hình)
        base_dir = a.out / "_base"
        base_dir.mkdir(exist_ok=True)
        D = {}
        for k in keys:
            d = np.asarray(Image.open(a.pairs / f"{k}_draft.png").convert("RGB"))
            bf = base_dir / f"{k}.png"
            if bf.exists():
                cl = np.asarray(Image.open(bf).convert("RGB"))
            else:
                p = np.asarray(Image.open(a.pairs / f"{k}_plate.png").convert("RGB").resize((d.shape[1], d.shape[0])))
                cl = cleanup.clean(d, p)[0]
                Image.fromarray(cl).save(bf)
            D[k] = {"draft": d, "plate": cl, "M": slots.build(d, cl), "prompt": meta[k]["prompt"],
                    "texts": [t for t in meta[k]["texts"] if not t.get("scene")], "product": bool(meta[k].get("ref"))}
            print(f"ô {k}: L {len(D[k]['M']['L'])} S {len(D[k]['M']['S'])} I {len(D[k]['M']['I'])}")
        def done(v, k):   # đã có kết quả không lỗi (lượt lỗi VLM chạy lại)
            f = a.out / v / f"{k}.json"
            return f.exists() and not json.loads(f.read_text(encoding="utf-8")).get("error")

        jobs = [(v, k) for v in views for k in keys if a.fresh or not done(v, k)]
        print(f"{len(jobs)} lượt ({len(views)} cấu hình x {len(keys)} ảnh), {a.workers} lời gọi song song")

        def plan(job):
            v, k = job
            x, mt = D[k], {}
            t0 = time.time()
            try:
                P = slots.plan(x["draft"], x["M"], x["prompt"], x["texts"], vlm, product=x["product"], base=x["plate"], view=v, meta=mt)
                return job, P, mt, None, time.time() - t0
            except Exception as e:
                return job, None, mt, f"{type(e).__name__}: {str(e)[:300]}", time.time() - t0

        def repair(item):
            (v, k), P, poster, errs = item
            x, mt = D[k], {}
            try:
                P2 = slots.repair(x["draft"], poster, x["M"], P, errs, x["prompt"], x["texts"], vlm, product=x["product"], meta=mt)
                return (v, k), P2, mt, None
            except Exception as e:
                return (v, k), None, mt, f"{type(e).__name__}: {str(e)[:300]}"

        bad = lambda E: (len(E), sum(e.get("frac", 0) + e.get("px", 0) / 100 for e in E))   # noqa: E731  như engine
        with Browser() as B, ThreadPoolExecutor(a.workers) as ex:
            first, todo_fix = {}, []
            # 2. lập lệnh (song song) -> 3. dựng + kiểm (tuần tự)
            for (v, k), P, mt, err, dt in ex.map(plan, jobs):
                rec = {"view": v, "key": k, "plan_meta": {**mt, "wall_s": round(dt, 1)}, "error": err}
                if P is None:
                    print(f"{v} {k}: LỖI VLM {err}")
                    first[(v, k)] = (rec, None, None, None)
                    continue
                x = D[k]
                P, vlog = slots.validate(P, x["M"], x["texts"])
                poster, res, rlog = render.render(B.page, slots.base_image(x["draft"], x["plate"], x["M"], P), P, x["M"])
                errs = render.check(P, x["M"], res)
                rec.update(errs_first=errs, log=vlog + rlog)
                first[(v, k)] = (rec, P, poster, errs)
                if errs:
                    todo_fix.append(((v, k), P, poster, errs))
            # 4. vòng sửa (song song) -> dựng lại, giữ bản ít lỗi hơn
            fixed = {}
            for (v, k), P2, mt, err in ex.map(repair, todo_fix):
                fixed[(v, k)] = (P2, mt, err)
            for (v, k), (rec, P, poster, errs) in first.items():
                if P is None:
                    out_P, out_poster, out_errs = None, None, None
                else:
                    out_P, out_poster, out_errs = P, poster, errs
                    if (v, k) in fixed:
                        P2, mt, err = fixed[(v, k)]
                        rec["repair_meta"], rec["repair_error"] = mt, err
                        if P2 is not None:
                            x = D[k]
                            P2, vlog2 = slots.validate(P2, x["M"], x["texts"])
                            poster2, res2, rlog2 = render.render(B.page, slots.base_image(x["draft"], x["plate"], x["M"], P2), P2, x["M"])
                            errs2 = render.check(P2, x["M"], res2)
                            rec["errs_repair"] = errs2
                            rec["log"] += ["-- vòng sửa --"] + vlog2 + rlog2
                            if bad(errs2) < bad(errs):
                                out_P, out_poster, out_errs = P2, poster2, errs2
                                rec["kept"] = "repair"
                (a.out / v).mkdir(exist_ok=True)
                if out_poster is not None:
                    Image.fromarray(out_poster).save(a.out / v / f"{k}.jpg", quality=85)
                tid = lambda m: int(m[1:]) if isinstance(m, str) and m[1:].isdigit() else m   # noqa: E731
                texts = D[k]["texts"]
                rec.update(errs_final=out_errs, style=(out_P or {}).get("style"), ops=(out_P or {}).get("ops"),
                           missing=[texts[i]["text"] for i in map(tid, (out_P or {}).get("missing") or [])
                                    if isinstance(i, int) and 0 <= i < len(texts)],
                           slots=slots.public(D[k]["M"]))
                (a.out / v / f"{k}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
                print(f"{v} {k}: lỗi {len(rec.get('errs_first') or [])} -> {len(out_errs or [])}; "
                      f"token {rec['plan_meta'].get('prompt_tokens')}+{rec['plan_meta'].get('completion_tokens')}, "
                      f"{rec['plan_meta'].get('s')}s")
    # tổng hợp nhanh
    summ = {}
    for v in views:
        R = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((a.out / v).glob("*.json"))] if (a.out / v).is_dir() else []
        if not R:
            continue
        num = lambda f: [x for x in (f(r) for r in R) if isinstance(x, (int, float))]   # noqa: E731
        summ[v] = {"n": len(R), "vlm_fail": sum(1 for r in R if r.get("error")),
                   "errs_first": sum(len(r.get("errs_first") or []) for r in R),
                   "errs_final": sum(len(r.get("errs_final") or []) for r in R),
                   "missing": sum(len(r.get("missing") or []) for r in R),
                   "prompt_tokens": sum(num(lambda r: r["plan_meta"].get("prompt_tokens"))),
                   "plan_s": round(sum(num(lambda r: r["plan_meta"].get("s"))), 1)}
        print(v, summ[v])
    (a.out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    if not a.no_ship:
        files = [(a.out / "summary.json", "summary.json")] + [(f, f"{f.parent.name}/{f.name}") for v in views if (a.out / v).is_dir()
                                                               for f in sorted((a.out / v).iterdir())]
        ship.offer(ship.pack(files, a.out.parent / "vlm_probe_zip", "vlm", int(a.zip_mb * 2**20) if a.zip_mb else None,
                             group=lambda arc: arc.rsplit(".", 1)[0]))   # poster + json một ảnh chung một zip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
