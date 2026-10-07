"""Bộ icon Lucide (giấy phép ISC) ở assets/icons/lucide/<tên>.svg (scripts/fetch_icons.py --all): designer chọn theo tên."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ICON_DIR = ROOT / "assets" / "icons" / "lucide"


def available() -> dict[str, str]:
    """{tên: phần trong <svg> của icon (path, circle...)}."""
    out = {}
    if not ICON_DIR.exists():
        return out
    for f in sorted(ICON_DIR.glob("*.svg")):
        m = re.search(r"<svg[^>]*>(.*)</svg>", f.read_text(encoding="utf-8"), re.S)
        if m:
            out[f.stem] = " ".join(m.group(1).split())
    return out


def nearest(name: str, names) -> str | None:
    """Tên icon gần nhất trong bộ có sẵn (VLM gọi tên tự do: 'verified' -> 'badge-check' không bắt được, 'heart-filled' -> 'heart')."""
    import difflib
    name = (name or "").strip().lower().replace("_", "-").replace(" ", "-")
    if name in names:
        return name
    for tok in name.split("-"):   # 'heart-filled' -> 'heart'
        if tok in names:
            return tok
    hit = difflib.get_close_matches(name, list(names), n=1, cutoff=0.6)
    return hit[0] if hit else None
