#!/usr/bin/env python3
"""MÁY CHỦ: tóm tắt các lượt chạy gần đây của máy chủ TextFix (output/runs) -- lỗi lượt đầu theo loại, sửa bằng code / vòng duyệt
VLM có chạy / được giữ không, thời gian từng bước -- rồi gói plan.json + ảnh (poster, lượt đầu, nháp) thành zip < 24 MB, tự tải về.

  %run scripts/runs_report.py              # 10 lượt chạy gần nhất (+ server.log)
  %run scripts/runs_report.py --last 3
  %run scripts/runs_report.py --all        # mọi lượt máy chủ còn giữ (TEXTFIX_MAX_RUNS, mặc định 300)
  %run scripts/runs_report.py --light      # chỉ plan.json + poster (nhẹ: tải nhiều lượt nhanh)
  %run scripts/runs_report.py --again      # gửi lại cả lượt đã gửi
KHÔNG GỬI TRÙNG: lượt đã gửi ghi ở output/runs_zip/shipped.json, lần sau bỏ qua (tóm tắt cũng chỉ tính lượt mới). Chỉ ghi
lượt đã XONG (mọi ảnh có nháp đều có plan.json): lượt đang chạy dở vẫn gửi, lần sau gửi lại bản đủ. server.log chỉ gửi phần
mới từ lần trước. Zip mang tên theo giờ gói (runs_<ngày>_<giờ phút giây>_01.zip) để trình duyệt không đặt "(1)".
Zip tự chia phần < 24 MB, mỗi ảnh (v<i>/) nằm gọn trong một phần; giải nén vào output/runs/ ở máy cá nhân.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", line_buffering=True)

SEVERE = ("unassigned", "content_overlap", "boxes_overlap", "overflow", "low_contrast", "mark_twice")


def _done(rd: Path) -> bool:
    """Lượt đã XONG: đủ số ảnh khách xin đều có plan.json; hoặc lượt đã im > 30 phút (hỏng giữa chừng, không chạy tiếp nữa)."""
    try:
        want = int(json.loads((rd / "request.json").read_text(encoding="utf-8")).get("num_images") or 1)
    except (OSError, ValueError, TypeError):
        want = 1
    if sum((vd / "plan.json").exists() for vd in rd.glob("v*")) >= want:
        return True
    last = max([p.stat().st_mtime for p in rd.rglob("*")] + [rd.stat().st_mtime])
    return time.time() - last > 1800


def main(argv: list[str] | None = None) -> int:
    from textfix import ship
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=ROOT / "output" / "runs")
    ap.add_argument("--last", type=int, default=10)
    ap.add_argument("--all", action="store_true", help="mọi lượt còn giữ")
    ap.add_argument("--light", action="store_true", help="chỉ plan.json + poster.png")
    ap.add_argument("--no-ship", action="store_true")
    ap.add_argument("--again", action="store_true", help="gửi lại cả lượt đã gửi")
    a = ap.parse_args(argv)
    zdir = ROOT / "output" / "runs_zip"
    reg_f = zdir / "shipped.json"
    try:
        reg = json.loads(reg_f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        reg = {}
    shipped = set(reg.get("runs") or [])
    runs = sorted([d for d in a.runs.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime)
    if not a.again:
        old = [d for d in runs if d.name in shipped]
        runs = [d for d in runs if d.name not in shipped]
        if old:
            print(f"bỏ qua {len(old)} lượt đã gửi (--again để gửi lại)")
    runs = runs if a.all else runs[-a.last:]
    if not runs:
        print("không có lượt mới")
    done = [rd.name for rd in runs if _done(rd)]
    first, final, trig, kept, n = collections.Counter(), collections.Counter(), 0, 0, 0
    cfix, ckept = 0, 0
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
            cf = any(l.startswith("-- sửa bằng code") for l in log)
            cfix += cf
            ckept += any(l == "-> giữ bản sửa" for l in log)
            for k, v in (P.get("timing") or {}).items():
                T[k].append(v)
            print(f"{rd.name}/{vd.name}: lỗi nặng lượt đầu {[(e['type'], e.get('ops'), e.get('frac')) for e in sev]}"
                  f" -> sửa code {'CÓ' if cf else 'không'}, vòng duyệt VLM {'CÓ' if ran else 'không'};"
                  f" lỗi nặng cuối {[(e['type'], e.get('ops')) for e in P.get('errors') or [] if e['type'] in SEVERE]}")
            for f in (["plan.json", "poster.png"] if a.light else
                      ["plan.json", "poster.png", "poster_first.png", "draft.png", "plate_raw.png", "plate.png"]):
                if (vd / f).exists():
                    files.append((vd / f, f"{rd.name}/{vd.name}/{f}"))
    print(f"\n{n} ảnh, sửa bằng code {cfix} (giữ {ckept}), vòng duyệt VLM {trig} (giữ {kept})")
    print("lỗi lượt đầu:", dict(first))
    print("lỗi cuối:", dict(final))
    print("thời gian trung vị:", {k: round(st.median(v), 1) for k, v in T.items()})
    if a.no_ship:
        return 0
    stamp = time.strftime("%m%d_%H%M%S")
    lf, off = ROOT / "server.log", 0 if a.again else int(reg.get("log_offset") or 0)
    size = lf.stat().st_size if lf.exists() else 0
    if size > off or (size and size < off):   # log mới (hoặc log bị xoay / ghi lại từ đầu -> gửi cả)
        part = zdir / f"server_{stamp}.log"
        zdir.mkdir(parents=True, exist_ok=True)
        with open(lf, "rb") as f:
            f.seek(off if size > off else 0)
            part.write_bytes(f.read())
        files.append((part, part.name))
    if not files:
        return 0
    for z in zdir.glob("runs_*.zip"):   # zip lần trước đã tải về -> dọn khỏi máy chủ
        z.unlink()
    for g in zdir.glob("server_*.log"):
        if not files or g != files[-1][0]:
            g.unlink()
    ship.offer(ship.pack(files, zdir, f"runs_{stamp}", group=lambda arc: arc.rsplit("/", 1)[0]))
    reg = {"runs": sorted(shipped | set(done)), "log_offset": size}
    reg_f.write_text(json.dumps(reg, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"đã ghi {len(done)} lượt xong vào {reg_f.name}" + (f"; {len(runs) - len(done)} lượt dở sẽ gửi lại lần sau"
                                                             if len(runs) > len(done) else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
