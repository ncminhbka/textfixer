#!/usr/bin/env python3
"""MÁY CHỦ: tóm tắt các lượt chạy gần đây của máy chủ TextFix (output/runs) -- lỗi lượt đầu theo loại, vòng sửa có chạy / được giữ
không, thời gian từng bước -- rồi gói plan.json + ảnh (poster, lượt đầu, nháp) thành zip < 24 MB, tự tải về.

  %run scripts/runs_report.py              # 10 lượt chạy gần nhất (+ server.log)
  %run scripts/runs_report.py --last 3
  %run scripts/runs_report.py --all        # mọi lượt máy chủ còn giữ (TEXTFIX_MAX_RUNS, mặc định 300)
  %run scripts/runs_report.py --light      # chỉ plan.json + poster (nhẹ: tải nhiều lượt nhanh)
Zip tự chia phần < 24 MB, mỗi ảnh (v<i>/) nằm gọn trong một phần; giải nén vào output/runs/ ở máy cá nhân.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", line_buffering=True)

SEVERE = ("unassigned", "content_overlap", "boxes_overlap", "overflow", "low_contrast", "mark_twice")


def main(argv: list[str] | None = None) -> int:
    from textfix import ship
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=ROOT / "output" / "runs")
    ap.add_argument("--last", type=int, default=10)
    ap.add_argument("--all", action="store_true", help="mọi lượt còn giữ")
    ap.add_argument("--light", action="store_true", help="chỉ plan.json + poster.png")
    ap.add_argument("--no-ship", action="store_true")
    a = ap.parse_args(argv)
    runs = sorted([d for d in a.runs.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime)
    runs = runs if a.all else runs[-a.last:]
    first, final, trig, kept, n = collections.Counter(), collections.Counter(), 0, 0, 0
    T = collections.defaultdict(list)
    files = []
    for rd in runs:
        for f in ["request.json", "brief.json"]:
            if (rd / f).exists():
                files.append((rd / f, f"{rd.name}/{f}"))
        for vd in sorted(rd.glob("v*")):
            pf = vd / "plan.json"
            if not pf.exists():
                continue
            P = json.loads(pf.read_text(encoding="utf-8"))
            n += 1
            e1 = P.get("errors_first") or []
            sev = [e for e in e1 if e["type"] in SEVERE]
            for e in e1:
                first[e["type"]] += 1
            for e in P.get("errors") or []:
                final[e["type"]] += 1
            log = P.get("log") or []
            ran = any(l.startswith("-- vòng duyệt") for l in log)
            trig += ran
            kept += any("giữ bản đã duyệt" in l for l in log)
            for k, v in (P.get("timing") or {}).items():
                T[k].append(v)
            print(f"{rd.name}/{vd.name}: lỗi nặng lượt đầu {[(e['type'], e.get('ops'), e.get('frac')) for e in sev]}"
                  f" -> vòng sửa {'CÓ' if ran else 'không'}; lỗi cuối {len(P.get('errors') or [])}")
            for f in (["plan.json", "poster.png"] if a.light else
                      ["plan.json", "poster.png", "poster_first.png", "draft.png", "plate_raw.png", "plate.png"]):
                if (vd / f).exists():
                    files.append((vd / f, f"{rd.name}/{vd.name}/{f}"))
    print(f"\n{n} ảnh, vòng sửa chạy {trig}, được giữ {kept}")
    print("lỗi lượt đầu:", dict(first))
    print("lỗi cuối:", dict(final))
    print("thời gian trung vị:", {k: round(st.median(v), 1) for k, v in T.items()})
    if (ROOT / "server.log").exists():
        files.append((ROOT / "server.log", "server.log"))
    if not a.no_ship and files:
        ship.offer(ship.pack(files, ROOT / "output" / "runs_zip", "runs", group=lambda arc: arc.rsplit("/", 1)[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
