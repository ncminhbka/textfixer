#!/usr/bin/env python3
"""ĐO ĐỘ RỘNG CHỮ của từng font trong fonts.CATALOG (Chromium, đúng renderer): em / ký tự, chữ thường lẫn hoa và chữ IN HOA,
ở độ đậm đậm nhất <= 800 font có -> in bảng WIDTH để dán vào textfix/fonts.py. Chạy lại khi thêm font.

  python scripts/font_widths.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MIXED = "Giảm giá đặc biệt cho khách hàng mới, Thứ Bảy 15/10"
CAPS = "KHAI TRƯƠNG GIẢM GIÁ ĐẶC BIỆT CUỐI TUẦN"

JS = """([fam, wt, a, b]) => { const c = document.createElement('canvas').getContext('2d');
  c.font = `${wt} 100px ${fam}`; return [c.measureText(a).width / 100 / a.length, c.measureText(b).width / 100 / b.length]; }"""


def main() -> int:
    from textfix.browser import Browser
    from textfix.fonts import CATALOG, faces_for, family
    rows = {}
    with Browser() as B:
        for key, (_, _, _, files) in CATALOG.items():
            ws = []   # khoá: độ đậm (700) hoặc dải font biến thiên ("100 900")
            for w in files:
                lo, hi = (int(x) for x in str(w).split()) if " " in str(w) else (int(w), int(w))
                ws += [min(hi, 800)] if lo <= 800 else [lo]
            wt = max(ws)
            B.page.set_content(f"<html><head><style>{faces_for((key,))}</style></head><body>"
                               f"<span style=\"font-family:{family(key)};font-weight:{wt}\">{MIXED}{CAPS}</span></body></html>")
            B.page.evaluate("async () => { await Promise.all([...document.fonts].map(f => f.load().catch(() => null))); }")
            m, c = B.page.evaluate(JS, [family(key), wt, MIXED, CAPS])
            rows[key] = (round(m, 2), round(c, 2))
    print("WIDTH = {   # em / ký tự (chữ thường lẫn hoa, IN HOA) ở độ đậm đậm nhất <= 800; scripts/font_widths.py")
    for k, v in rows.items():
        print(f"    {k!r}: {v},")
    print("}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
