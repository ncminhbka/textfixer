"""CHẠY HÀNG LOẠT TRÊN MỌI GPU: mỗi card một tiến trình con nạp một bản FLUX riêng (≈ 17 GB, vừa một A30 24 GB), chia đều
danh sách việc; tiến trình cha chờ các con xong rồi gom kết quả (gói zip...). VLM host ngoài nên không cần chừa card nào.

  ap = argparse.ArgumentParser(); multigpu.add_args(ap); a = ap.parse_args(argv)
  if multigpu.fanout(a, __file__, argv):   # cha: đã chạy xong các con (mỗi con --shard k/n --no-ship) -> chỉ còn gói kết quả
      ...ship...; return 0
  items = multigpu.shard(items, a)         # con (hoặc chạy một card): phần việc của mình

Con thấy đúng MỘT card (CUDA_VISIBLE_DEVICES=k) và mọi phần FLUX (DiT / bộ mã hoá chữ / VAE) đặt lên cuda:0 của nó, bỏ qua
TEXTFIX_DEVICE_* của môi trường (đặt cho chế độ một tiến trình). Nhật ký các con in kèm tiền tố [gpu k].
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading


def add_args(ap) -> None:
    ap.add_argument("--gpus", type=int, default=0, help="số card dùng (0 = mọi card thấy được; 1 = một tiến trình như cũ)")
    ap.add_argument("--shard", default="", help=argparse.SUPPRESS)
    ap.add_argument("--no-ship", action="store_true", help="không gói / tải zip")


def part(path, a):
    """Tệp nhật ký JSON của tiến trình này: con ghi <tên>.shard<k>.json (không tranh ghi với con khác), cha / một card ghi
    thẳng <tên>.json. Cha gộp bằng merge()."""
    from pathlib import Path
    path = Path(path)
    return path.with_name(f"{path.stem}.shard{a.shard.split('/')[0]}{path.suffix}") if a.shard else path


def merge(path, deep: tuple = ()) -> dict:
    """Gộp <tên>.shard*.json vào <tên>.json (xoá tệp shard). deep: các khoá cấp 1 mà giá trị là dict cần gộp sâu một mức
    (vd. {"B": {"runs": {...}}})."""
    from pathlib import Path
    path = Path(path)
    out = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for f in sorted(path.parent.glob(f"{path.stem}.shard*{path.suffix}")):
        for k, v in json.loads(f.read_text(encoding="utf-8")).items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = {**out[k], **{kk: ({**out[k].get(kk, {}), **vv} if kk in deep and isinstance(vv, dict) else vv)
                                       for kk, vv in v.items()}}
            else:
                out[k] = v
        f.unlink()
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def count() -> int:
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def shard(items: list, a) -> list:
    """Phần việc của tiến trình này: items[k::n] với --shard k/n; không có --shard: tất cả."""
    if not a.shard:
        return items
    k, n = (int(x) for x in a.shard.split("/"))
    return items[k::n]


def fanout(a, script: str, argv: list[str] | None) -> bool:
    """Cha: nếu dùng >= 2 card và chưa là con -> chạy mỗi card một tiến trình con với cùng tham số + --shard k/n --no-ship,
    chờ hết, trả True. Ngược lại trả False (chạy tại chỗ). Con lỗi -> báo, các con khác vẫn chạy xong."""
    if a.shard:
        return False
    n = a.gpus or count()
    if n < 2:
        return False
    args = list(sys.argv[1:] if argv is None else argv)
    procs = []
    for k in range(n):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(k), "TEXTFIX_DEVICE_DIT": "cuda:0", "TEXTFIX_DEVICE_TE": "cuda:0",
               "TEXTFIX_DEVICE_AE": "cuda:0", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
        p = subprocess.Popen([sys.executable, os.path.abspath(script), *args, "--shard", f"{k}/{n}", "--no-ship"], env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        procs.append(p)
    print(f"chạy song song trên {n} card (mỗi card một bản FLUX)", flush=True)
    lock = threading.Lock()

    def pump(k, p):
        for line in p.stdout:
            with lock:
                print(f"[gpu {k}] {line.rstrip()}", flush=True)

    ts = [threading.Thread(target=pump, args=(k, p), daemon=True) for k, p in enumerate(procs)]
    for t in ts:
        t.start()
    codes = [p.wait() for p in procs]
    for t in ts:
        t.join()
    bad = [k for k, c in enumerate(codes) if c != 0]
    if bad:
        print(f"CẢNH BÁO: tiến trình card {bad} kết thúc lỗi (mã {[codes[k] for k in bad]}); gói phần đã xong", flush=True)
    return True
