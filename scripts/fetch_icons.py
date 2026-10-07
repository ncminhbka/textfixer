#!/usr/bin/env python3
"""Tải bộ icon Lucide (giấy phép ISC -- mã mở, dùng thương mại được) cho textfix.icons -> assets/icons/lucide/<tên>.svg + LICENSE.

  python scripts/fetch_icons.py          # danh sách tên hay gặp
  python scripts/fetch_icons.py --all    # toàn bộ Lucide (designer chọn icon theo tên)
Nguồn: gói npm lucide-static (unpkg). Chỉ tải danh sách tên hay gặp trên poster quảng cáo (tính năng, lợi ích, liên hệ, thời gian...).
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "icons" / "lucide"
BASE = "https://unpkg.com/lucide-static@0.469.0"
NAMES = """
check circle-check badge-check shield-check shield star sparkles heart thumbs-up award trophy crown gem medal
clock timer calendar calendar-check hourglass alarm-clock
phone phone-call mail message-circle map-pin map globe link at-sign house building store
truck package gift shopping-cart shopping-bag tag percent ticket credit-card wallet banknote coins piggy-bank
leaf sprout flower salad apple coffee cup-soda utensils chef-hat cooking-pot wheat milk egg fish beef cake ice-cream-cone
droplet droplets flame snowflake sun moon wind zap battery-charging plug thermometer
dumbbell activity heart-pulse stethoscope pill syringe smile baby dog cat paw-print bath scissors sparkle
user users user-check graduation-cap book-open book pen-tool lightbulb brain target rocket trending-up chart-line chart-column
wifi smartphone laptop monitor headphones camera music video tv
car plane bike bus ship key lock wrench hammer settings sofa bed
recycle refresh-cw rotate-ccw infinity hand-heart handshake circle-dollar-sign
""".split()


def fetch_all() -> int:
    """--all: TOÀN BỘ bộ Lucide (~1.500 icon) từ gói npm (một tệp .tgz) -- designer VLM (textfix/slots.py) gọi icon theo tên tự do."""
    import io
    import tarfile
    ver = BASE.rsplit("@", 1)[1]
    with urllib.request.urlopen(f"https://registry.npmjs.org/lucide-static/-/lucide-static-{ver}.tgz", timeout=60) as r:
        tar = tarfile.open(fileobj=io.BytesIO(r.read()), mode="r:gz")
    n = 0
    for m in tar.getmembers():
        if m.name.startswith("package/icons/") and m.name.endswith(".svg"):
            (OUT / m.name.rsplit("/", 1)[1]).write_bytes(tar.extractfile(m).read())
            n += 1
        elif m.name == "package/LICENSE":
            (OUT / "LICENSE").write_bytes(tar.extractfile(m).read())
    print(f"{n} icon -> {OUT}")
    return 0


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if "--all" in sys.argv:
        return fetch_all()
    ok, bad = 0, []
    for n in NAMES:
        f = OUT / f"{n}.svg"
        if f.exists():
            ok += 1
            continue
        try:
            with urllib.request.urlopen(f"{BASE}/icons/{n}.svg", timeout=20) as r:
                f.write_bytes(r.read())
            ok += 1
        except Exception as e:
            bad.append((n, str(e)[:60]))
    try:
        with urllib.request.urlopen(f"{BASE}/LICENSE", timeout=20) as r:
            (OUT / "LICENSE").write_bytes(r.read())
    except Exception as e:
        print("không tải được LICENSE:", e)
    print(f"{ok} icon -> {OUT}; lỗi {len(bad)}: {bad}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
