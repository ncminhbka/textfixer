"""DESIGNER "ĐIỀN Ô": nháp - bản xoá = lớp phủ model đã vẽ -> các Ô đúng toạ độ; VLM chỉ quyết mỗi ô là gì và điền cho đẹp; code
dựng đúng khung từng ô (render.py), theo lớp vỏ -> chi tiết -> chữ.

  M = build(draft, plate)                         # S# vỏ, I# chi tiết nhỏ, L# dòng chữ (OCR, khung nắn theo lớp phủ, số đo nét)
  P = plan(draft, M, prompt, texts, call)         # VLM: {style, ops: [{slots, kind, html, box_style, client}], missing}
  P, log = validate(P, M, texts)                 # + chặn chữ: dấu theo câu khách, chữ bịa, dòng lặp
  base = base_image(draft, plate, M, P)           # bản xoá + dán lại pixel nháp ở dòng "keep" (chữ in trên sản phẩm / màn hình)
  P2 = repair(draft, poster, M, P, errs, prompt, texts, call)   # có lỗi (render.check): một vòng, chỉ các lệnh / ô lỗi

VLM không đưa toạ độ: chỉ trỏ id ô. Mỗi ô thuộc đúng một lệnh (lệnh chữ được gộp nhiều dòng liền nhau thành một khối).
"""

from __future__ import annotations

import io
import json
import os
import re

import numpy as np

CAP_EM = 0.72        # cao chữ hoa / em
X_EM = 0.53          # cao chữ thường (x-height) / em
SNAP_SIDE = 0.35     # nắn khung dòng: xét lớp phủ quanh đa giác OCR, nới ngang / xuống 0.35 cao dòng ...
SNAP_UP = 0.9        # ... và lên 0.9 cao dòng (dấu thanh / mũ tiếng Việt)
INK_IN_SHELL = 25.0  # trong vỏ: nét chữ = khác màu lòng vỏ >= 25 ΔE (cả lòng vỏ đều thuộc lớp phủ)
GAP_SPLIT = 1.0      # trong một dòng OCR, khoảng KHÔNG nét rộng >= 1 cao dòng = hai mục riêng (khoảng cách từ ~0.3 cao dòng;
                     # OCR gộp "địa chỉ [icon] website" ở chân trang thành một dòng)
GAP_MIN = 0.4        # khoảng không nét quanh chỗ OCR chèn dấu cách kép: >= 0.4 cao dòng (khoảng cách từ thường ~0.3)
ICON_W = 1.6         # khối kẹp giữa hai khoảng không ở chỗ cắt, rộng <= 1.6 cao dòng = icon giữa hai mục
INK_DE = 25.0        # điểm nét = khác màu nền khung dòng (trung vị viền khung) >= 25 ΔE
ROW_GAP = 0.5        # chi tiết giống nhau (cao lệch <= 25%) cùng hàng (tâm lệch <= 25% cao), cách nhau <= 0.5 cao = MỘT chi tiết
                     # (hàng sao, hàng chấm: designer dựng cả hàng bằng một thẻ)
DETAIL_MIN = 0.5     # chi tiết I: cạnh dài >= 0.5 cao dòng chữ trung vị của poster (icon designer đặt không nhỏ hơn nửa chữ thân;
                     # nhỏ hơn = vụn: mép vật FLUX vẽ lại, mẩu hoạ tiết)


def _lab(x):
    import cv2
    a = cv2.cvtColor(x, cv2.COLOR_RGB2LAB).astype(np.float32)
    a[..., 0] *= 100 / 255
    a[..., 1:] -= 128
    return a


def _png(im) -> bytes:
    from PIL import Image
    if isinstance(im, np.ndarray):
        im = Image.fromarray(im)
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def _cov(a, b) -> float:
    """phần diện tích b nằm trong a"""
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    return max(0, w) * max(0, h) / max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))


# ---------------------------------------------------------------------------------------------------- đo
def lines(img: np.ndarray) -> list[dict]:
    """Dòng OCR + số đo nét: cao thân chữ -> cỡ chữ (px), màu, góc nghiêng."""
    import cv2
    from .glyph import measure
    from .ocr import read_lines
    H, W = img.shape[:2]
    # "dòng" không có chữ cái / chữ số (hàng sao "★★★★★", gạch, chấm) không phải chữ: để lớp phủ tách thành chi tiết I
    raw = [l for l in read_lines(img) if any(ch.isalnum() for ch in l["text"])]
    raw = [q for l in raw for q in _split_gaps(img, l)]
    L = []
    for i, l in enumerate(raw, 1):
        ang = float(l.get("angle") or 0.0)
        if abs(ang) >= 2:   # dòng NGHIÊNG: xoay thẳng ảnh quanh dòng rồi đo (khung thẳng của dòng nghiêng làm cao thân chữ phồng)
            P_ = np.asarray(l["poly"], np.float32)
            c = P_.mean(0)
            R = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), ang, 1.0)
            img_r = cv2.warpAffine(img, R, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            q = np.hstack([P_, np.ones((4, 1), np.float32)]) @ R.T
            m = measure(img_r, [float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max())], l["text"], [])
        else:
            m = measure(img, l["box"], l["text"], [o["box"] for o in raw if o is not l])
        lt = [c for c in l["text"] if c.isalnum()]
        caps = bool(lt) and sum(c.isupper() or c.isdigit() for c in lt) >= 0.8 * len(lt)
        h = m["h"] if m else 0.7 * (l["box"][3] - l["box"][1])
        L.append({"id": f"L{i}", "box": [float(v) for v in l["box"]], "poly": l["poly"], "ocr": l["text"],
                  "size": float(h / (CAP_EM if caps else X_EM)), "color": m["color"] if m else None, "angle": ang,
                  "caps": caps, "stroke": round(float(m["stroke"]), 1) if m else None,
                  "stroke_contrast": round(float(m["contrast"]), 2) if m else None,
                  "effect": {k: v for k, v in (m.get("effect") or {}).items() if k in ("dx", "dy", "color")} if m and m.get("effect") else None})
    return L


def _rows(I: list[dict]) -> list[dict]:
    """Gộp chi tiết giống nhau xếp thành hàng (ROW_GAP) thành một chi tiết."""
    I = sorted(I, key=lambda c: c["box"][0])
    out = []
    for c in I:
        x0, y0, x1, y1 = c["box"]
        h = y1 - y0
        prev = next((o for o in reversed(out) if o.get("shell") == c.get("shell")
                     and abs((o["_h"]) - h) <= 0.25 * max(o["_h"], h)
                     and abs((o["box"][1] + o["box"][3]) / 2 - (y0 + y1) / 2) <= 0.25 * max(o["_h"], h)
                     and 0 <= x0 - o["box"][2] <= ROW_GAP * max(o["_h"], h)), None)
        if prev is None:
            out.append({**c, "box": list(c["box"]), "_h": h})
        else:
            b = prev["box"]
            prev["box"] = [min(b[0], x0), min(b[1], y0), max(b[2], x1), max(b[3], y1)]
    for o in out:
        o.pop("_h")
    return out


def _split_gaps(img: np.ndarray, l: dict) -> list[dict]:
    """Tách dòng OCR (thẳng) thành các MỤC riêng:
    (a) khoảng không nét rộng >= GAP_SPLIT cao dòng;
    (b) chỗ OCR chèn >= 2 dấu cách (nó đánh dấu khoảng cách lớn): cắt tại khoảng không nét >= GAP_MIN cao dòng gần vị trí đó
        nhất; nếu giữa hai khoảng không quanh đó là một khối hẹp (<= ICON_W cao dòng: icon giữa hai mục, "địa chỉ [điện thoại]
        số") thì khối đó tách riêng, không thuộc dòng nào (lớp phủ tách nó thành chi tiết I).
    Chữ chia theo dấu cách gần chỗ cắt nhất (OCR không cho toạ độ từng chữ: chỉ là gợi ý cho designer)."""
    import re
    if abs(l.get("angle") or 0) >= 2:
        return [l]
    x0, y0, x1, y1 = (int(round(v)) for v in l["box"])
    h = y1 - y0
    if h < 4 or x1 - x0 < 3 * h:
        return [l]
    lab = _lab(np.ascontiguousarray(img[max(0, y0):y1, max(0, x0):x1]))
    border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    ink = (np.linalg.norm(lab - np.median(border, 0), axis=2) >= INK_DE).any(0)
    cols = np.flatnonzero(ink)
    if len(cols) < 2:
        return [l]
    W = x1 - x0
    gaps = [(int(a) + 1, int(b)) for a, b in zip(cols[:-1], cols[1:]) if b - a - 1 >= GAP_MIN * h]   # [đầu, cuối) khoảng không nét
    text = l["text"]
    cuts = []   # (cột kết thúc mục trái, cột bắt đầu mục phải, vị trí chữ để chia)
    for a, b in gaps:
        if b - a >= GAP_SPLIT * h:
            cuts.append((a, b, None))
    for m in re.finditer(r" {2,}", text):
        want = (m.start() + m.end()) / 2 / max(1, len(text)) * W
        near = sorted(gaps, key=lambda g: abs((g[0] + g[1]) / 2 - want))
        if not near or abs((near[0][0] + near[0][1]) / 2 - want) > 2 * h:
            continue
        g = near[0]
        # khối hẹp kẹp giữa hai khoảng không liền nhau quanh chỗ cắt = icon tách riêng
        side = [q for q in gaps if q != g and abs((q[0] + q[1]) / 2 - want) <= 2 * h]
        pair = None
        for q in side:
            lo, hi = (g, q) if q[0] > g[0] else (q, g)
            if 0 < hi[0] - lo[1] <= ICON_W * h:
                pair = (lo[0], hi[1])
        cuts.append((pair[0], pair[1], m) if pair else (g[0], g[1], m))
    if not cuts:
        return [l]
    cuts = sorted({(a, b): (a, b, m) for a, b, m in cuts}.values(), key=lambda c: c[0])
    spans, start = [], int(cols[0])
    for a, b, _ in cuts:
        if a > start:
            spans.append((start, a))
        start = b
    spans.append((start, int(cols[-1]) + 1))
    # chia chữ: chỗ cắt có dấu cách kép dùng đúng vị trí đó, chỗ khác dùng dấu cách gần tỉ lệ vị trí cắt nhất
    parts, pos = [], 0
    for k, (a, b, m) in enumerate(cuts):
        if m is not None:
            cut, nxt = m.start(), m.end()
        else:
            want = int(round(len(text) * (a + b) / 2 / W))
            sp = [i for i, ch in enumerate(text) if ch == " " and i > pos]
            cut = min(sp, key=lambda i: abs(i - want)) if sp else max(pos + 1, want)
            nxt = cut
        parts.append(text[pos:cut])
        pos = max(pos, nxt)
    parts.append(text[pos:])
    if len(parts) != len(spans):
        return [l]
    out = []
    for (a, b), part in zip(spans, parts):
        part = part.strip()
        if not part or b - a < 2:
            continue
        bx = [float(x0 + a), float(y0), float(x0 + b), float(y1)]
        out.append({**l, "text": part, "box": bx, "poly": [[bx[0], bx[1]], [bx[2], bx[1]], [bx[2], bx[3]], [bx[0], bx[3]]]})
    return out or [l]


def _snap(L: dict, ink: np.ndarray, others: np.ndarray) -> list[float]:
    """Khung dòng = khung các mảng lớp phủ CHẠM đa giác OCR + mẩu cỡ dấu trong vùng nới (nét bay, dấu thanh, gạch đầu dòng);
    mảng chạm dòng OCR khác thuộc dòng đó. Dòng nghiêng giữ khung OCR (khung thẳng không ôm được dòng nghiêng)."""
    import cv2
    x0, y0, x1, y1 = L["box"]
    if abs(L.get("angle") or 0) >= 2:
        return list(L["box"])
    H, W = ink.shape
    h = y1 - y0
    X0, Y0 = int(max(0, x0 - SNAP_SIDE * h)), int(max(0, y0 - SNAP_UP * h))
    X1, Y1 = int(min(W, x1 + SNAP_SIDE * h)), int(min(H, y1 + SNAP_SIDE * h))
    sub = ink[Y0:Y1, X0:X1].astype(np.uint8)
    if not sub.any():
        return list(L["box"])
    oth = others[Y0:Y1, X0:X1]
    n, cc, st, _ = cv2.connectedComponentsWithStats(sub, 8)
    core = np.zeros_like(sub)
    cv2.fillPoly(core, [np.round(np.asarray(L["poly"]) - [X0, Y0]).astype(np.int32)], 1)
    hit = set(np.unique(cc[(core > 0) & (cc > 0)]).tolist())
    taken = set(np.unique(cc[oth & (cc > 0)]).tolist())
    keep = []
    for i in range(1, n):
        x, y, w, hh, a = st[i]
        on_edge = x == 0 or y == 0 or x + w >= sub.shape[1] or y + hh >= sub.shape[0]
        if i in taken and i not in hit:
            continue
        if i in hit or (not on_edge and max(w, hh) <= 0.4 * h and a >= 3):
            keep.append((x, y, x + w, y + hh))
    if not keep:
        return list(L["box"])
    bx = [min(k[0] for k in keep) + X0, min(k[1] for k in keep) + Y0, max(k[2] for k in keep) + X0, max(k[3] for k in keep) + Y0]
    # chặn nắn quá đà: không nở quá 1 cao dòng mỗi phía so với khung OCR
    return [float(max(bx[0], x0 - h)), float(max(bx[1], y0 - h)), float(min(bx[2], x1 + h)), float(min(bx[3], y1 + h))]


def _unstack(L: list[dict]) -> None:
    """Hai dòng xếp chồng có nét dính nhau (dấu "Ữ" chạm chân "MENU"): mảng dính chạm cả hai đa giác OCR nên khung nắn của dòng
    này nuốt cả dòng kia -> hai ô chồng nhau, designer viết hai dòng đè lên nhau (08/10 h02). Khung nắn không được vượt sang
    ranh giới với dòng bên cạnh (trên / dưới, trái / phải) = giữa khe (hoặc giữa phần giao) hai khung OCR. Khung OCR giao nhau
    >= nửa cao (rộng) theo cả hai chiều: không phân xử được, giữ nguyên."""
    new = []
    for a in L:
        x0, y0, x1, y1 = a["box"]
        ox0, oy0, ox1, oy1 = a["ocr_box"]
        for b in L if abs(a.get("angle") or 0) < 2 else []:
            if b is a:
                continue
            bx0, by0, bx1, by1 = b["ocr_box"]
            if min(x1, bx1) - max(x0, bx0) <= 0 or min(y1, by1) - max(y0, by0) <= 0:
                continue   # khung nắn của a không chạm khung OCR của b
            vy = (min(oy1, by1) - max(oy0, by0)) / max(1.0, min(oy1 - oy0, by1 - by0))
            vx = (min(ox1, bx1) - max(ox0, bx0)) / max(1.0, min(ox1 - ox0, bx1 - bx0))
            if vy < 0.5 and vy <= vx:   # trên / dưới
                if by0 + by1 < oy0 + oy1:
                    y0 = max(y0, (by1 + oy0) / 2)
                else:
                    y1 = min(y1, (by0 + oy1) / 2)
            elif vx < 0.5:              # trái / phải
                if bx0 + bx1 < ox0 + ox1:
                    x0 = max(x0, (bx1 + ox0) / 2)
                else:
                    x1 = min(x1, (bx0 + ox1) / 2)
        new.append([float(x0), float(y0), float(x1), float(y1)])
    for a, b in zip(L, new):
        a["box"] = b


def build(draft: np.ndarray, plate: np.ndarray) -> dict:
    """Ô của nháp: L# (dòng OCR + số đo nét, khung nắn theo lớp phủ, vỏ chứa nó), S# vỏ, I# chi tiết nhỏ (overlay.overlay)."""
    import cv2
    from .overlay import overlay
    H, W = draft.shape[:2]
    L = lines(draft)
    O = overlay(draft, plate, [{"box": l["box"], "poly": l["poly"]} for l in L])
    A = _lab(draft)
    polys = []
    for l in L:
        pm = np.zeros((H, W), np.uint8)
        cv2.fillPoly(pm, [np.asarray(l["poly"], np.int32)], 1)
        polys.append(pm.astype(bool))
    for j, l in enumerate(L):
        sh = next((s for s in O["S"] if j in s["lines"]), None)
        l["shell"] = sh["id"] if sh else None
        ink = (sh["_mask"] & (np.linalg.norm(A - sh["_fill_lab"], axis=2) > INK_IN_SHELL)) if sh else O["mask"]
        others = np.zeros((H, W), bool)
        for i, pm in enumerate(polys):
            if i != j:
                others |= pm
        l["ocr_box"] = list(l["box"])
        l["box"] = _snap(l, ink, others)
    _unstack(L)
    S =[{k: v for k, v in s.items() if not k.startswith("_")} for s in O["S"]]
    for s in S:
        s["lines"] = [L[j]["id"] for j in s["lines"]]
    # chi tiết không bao giờ ĐÈ lên chữ; bỏ chi tiết nằm trong khung dòng đã nắn (gạch đầu dòng thuộc dòng) và lồng trong chi tiết khác
    I = [c for c in O["I"] if all(_cov(c["box"], l["box"]) < 0.3 and _cov(l["box"], c["box"]) < 0.5 for l in L)]
    area = lambda b: (b[2] - b[0]) * (b[3] - b[1])   # noqa: E731
    I = [c for c in I if not any(o is not c and _cov(o["box"], c["box"]) >= 0.8 and area(o["box"]) > area(c["box"]) for o in I)]
    I = _rows(I)
    if L:   # vụn nhỏ hơn nửa chữ thân không phải chi tiết
        lh = float(np.median([l["box"][3] - l["box"][1] for l in L]))
        I = [c for c in I if max(c["box"][2] - c["box"][0], c["box"][3] - c["box"][1]) >= DETAIL_MIN * lh]
    for k, c in enumerate(I, 1):
        c["id"] = f"I{k}"
    for s in S:
        s["icons"] = [c["id"] for c in I if c["shell"] == s["id"]]
    for c in I:   # quan hệ với dòng gần nhất (gợi ý cho VLM: avatar cạnh tên, sao dưới tên, icon đầu dòng)
        cx, cy = (c["box"][0] + c["box"][2]) / 2, (c["box"][1] + c["box"][3]) / 2
        best = None
        for l in L:
            x0, y0, x1, y1 = l["box"]
            dx, dy = max(x0 - cx, 0, cx - x1), max(y0 - cy, 0, cy - y1)
            d = float(np.hypot(dx, dy))
            if best is None or d < best[0]:
                side = "left of" if cx < x0 and dx >= dy else "right of" if cx > x1 and dx >= dy else "above" if cy < y0 else \
                    "below" if cy > y1 else "on"
                best = (d, l["id"], side)
        c["near"] = f"{best[2]} {best[1]}" if best and best[0] <= 3 * max(8.0, c["box"][3] - c["box"][1]) else None
    # mask: lớp phủ (nháp - bản xoá đã dọn), cho "keep" dán lại trọn mảng bị xoá (không ghi JSON)
    return {"L": L, "S": S, "I": I, "size": [W, H], "by": {m["id"]: m for m in L + S + I}, "mask": O["mask"]}


def public(M: dict) -> dict:
    """Ô để ghi JSON."""
    return {"L": [{k: m[k] for k in ("id", "box", "ocr_box", "ocr", "size", "color", "angle", "shell") if k in m} for m in M["L"]],
            "S": M["S"], "I": M["I"]}


def marked_image(img: np.ndarray, M: dict):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(img).convert("RGB")
    d = ImageDraw.Draw(im)
    s = max(12, img.shape[1] // 75)
    try:
        f = ImageFont.truetype("arial.ttf", s)
    except OSError:
        try:
            f = ImageFont.truetype("DejaVuSans.ttf", s)
        except OSError:
            f = ImageFont.load_default()
    items = [(m, (0, 140, 255), 4, "tl") for m in M["S"]] + [(m, (230, 20, 20), 2, "tl") for m in M["L"]] + \
            [(m, (255, 150, 0), 2, "br") for m in M["I"]]
    for m, col, wd, corner in items:
        x0, y0, x1, y1 = m["box"]
        d.rectangle([x0, y0, x1, y1], outline=col, width=wd)
        tw, th = d.textbbox((0, 0), m["id"], font=f)[2:]
        tx, ty = (x0, max(0, y0 - th - 2)) if corner == "tl" else (max(0, x1 - tw - 2), min(img.shape[0] - th - 2, y1 + 1))
        d.rectangle([tx, ty, tx + tw + 2, ty + th + 2], fill=col)
        d.text((tx + 1, ty), m["id"], fill=(255, 255, 255), font=f)
    return im


# ---------------------------------------------------------------------------------------------------- VLM
SYSTEM = """You are a senior graphic designer finishing a poster. An image model was given the PROMPT below and drew the DRAFT. \
Its layout, composition and pictures are good; its overlay layer is broken: text is garbled, icons / avatars / stars are \
malformed, cards / pills / bands are rough. An eraser removed the WHOLE overlay layer: every text, card, pill, band, button, \
icon, avatar and star -- only the picture remains (the BASE). Code measured exactly what was removed and split it into SLOTS. \
You decide what each slot is and how it should look; the renderer draws your decision exactly inside that slot. The final \
poster = the BASE + your slots. Keep the model's layout: every slot stays where the model drew it, at its size.

AESTHETICS COME FIRST. For layout and look, follow the draft. A clean, consistent, well-fitted poster beats a complete one. \
But the WORDS come from the client texts, never from the image.

IMAGES: the message names each image. DRAFT = what the model drew. SLOTS = the draft with every slot marked (blue S#, red \
L#, orange I#); ZOOM images are enlarged parts of it. BASE (when given) = the erased poster the renderer draws your slots on: \
what will really be behind each slot (shells the eraser kept are still there; skipped slots leave this background).

SLOTS:
- S# SHELL: a card / pill / badge / band / button / panel the model drew; measured box, fill color, corner radius, and which \
L# / I# lie inside it.
- I# DETAIL: a small non-text element (icon, avatar, star row, check mark, bullet, logo, emoji...); measured box and color, \
the shell it is in, the nearest line.
- L# TEXT LINE: what OCR read (garbled), the drawn text size in px, how many characters the model wrote there (its CAPACITY), \
the measured color, the shell it is in.

FOR EACH SLOT, ONE op (a text op may take several consecutive lines of one text block):
- S#: kind "shape", slots ["S1"], box_style = CSS for the shell matching the draft (start from the measured fill and radius; \
add border / shadow / gradient / transparency if the draft shows it), html "". Or kind "skip" if the shell is junk, or kind \
"keep" if it is part of the picture (a graphic printed on a product, package or object: its original pixels are restored).
- I#: kind "icon", slots ["I3"], html = ONE helper tag without a size (it fills the slot): <i-icon name="<lucide name>" \
color="#hex"></i-icon>; a person avatar -> <i-icon name="circle-user" color="#hex"></i-icon>; a star row -> <i-stars n="5" \
fill="#f5b301" empty="#d9d9d9"></i-stars>. Pick the icon from what the model drew and what the nearby text means. Or kind \
"text" (html = the text) if the detail is really a big character or number OCR missed (a step number "1" in a circle), or kind "skip" for junk or duplicates, or kind "keep" for a detail that belongs to the picture (a logo or graphic printed on a product, \
package or object, a sign in the scene: its original pixels are restored).
- L#: kind "text", slots ["L5"] or ["L5","L6"], html = the text. Or kind "keep" for text printed on a product, package, screen \
or object in the photo (its original pixels are restored, never redraw such text), or kind "skip" for junk / duplicates.
- "keep" is only for things printed in the photographed scene. POSTER lettering is never kept, however decorated it is \
(a headline on a ribbon or banner, curved, 3D or metallic letters, badge and sticker text): the draft's letters are often \
misspelt, so it is always rewritten as a text op with the client's spelling.
- Every slot id appears in exactly ONE op -- never leave a slot out (use "skip" to drop it).

TEXT
- Decide what the model meant to write on each line and write it at about the size it drew, keeping its line breaks (<br> \
between the lines of a multi-line op). Write about as many characters per line as the model wrote there -- never much more: \
crammed text gets shrunk and looks bad.
- A client text longer than the lines the model gave it: shorten or rephrase it so it fits naturally; facts (names, prices, \
numbers, dates, phones, addresses, emails, links) are never changed or invented -- keep them exact or leave them out; the \
unit belongs to the number (15-25 triệu, 35.000đ, 30 phút, 50%): never drop it. Set \
"client": "T<i>" on ops writing (part of) a client text; list client texts the model did not draw, or that you dropped, in \
"missing".
- SPELLING: copy client words character for character, with every Vietnamese diacritic. The draft's letters are often \
wrong (TRƯỞNG or TRƯỜNG drawn for TRƯƠNG, Thứ Bẩy for Thứ Bảy): never take spelling from the image or the OCR.
- ONCE: each client text is written once. The image model often repeats a line (the same date, price or badge label twice): \
write it in the slot that fits it best and skip the copies. Never write a word that is in no client text inside an op that \
claims a client text.
- One client text drawn over several stacked lines (MENU / TRÀ SỮA): one op with all those slots, <br> between the lines. \
Read the OCR letters to find those lines even when garbled: a short slot below a line that ends its sentence ("chainn vi" \
under "Coffee rang moc") is that text's second line, never the start of another text.
- List items, steps and item / price rows keep the client's order and pairing (each name with its own price, steps \
1, 2, 3 in order), even where the draft drew them in another order.
- Text the model invented: write what it was meant to say from its role, position and the PROMPT, a natural line of about the \
same length in the poster's language (usually Vietnamese). Never copy garbled OCR. Commit; do not hedge. But a FACT the \
model drew that is in no client text (an address, phone, price, date, percentage, website) is never written: skip that slot.
- Star ratings are always <i-stars n="..."></i-stars>, never star characters (the fonts have no star glyph).
- html: inline spans only (color / weight / italic / size for emphasis), inline CSS only, no <img>, no URLs, no scripts. Sizes \
in px of the poster (as listed per line). The slot is a flex box centered both ways; for left-aligned text use box_style \
"justify-content:flex-start; text-align:left".
- One consistent type system: style.display_font for headlines, style.text_font for the rest (font keys from the catalog); use \
<i-font key="..." weight="700">...</i-font> only when a line needs another font. Colors: follow the draft; text must stay \
clearly readable on what is behind it (on the BASE the shell under a line may be gone if you skip it).

LOOK: REPRODUCE THE DRAFT'S STYLE -- the final poster should look like the draft with clean text, not like a different design.
- Typography per line: match what the model drew (look at the images; the measured weight / caps / strokes / shadow / outline \
are hints, the image wins when they disagree). Pick catalog fonts whose family matches the draft's letterforms (condensed, \
rounded, geometric, serif, script, display), then reproduce the rest with \
inline CSS: font-weight, font-style:italic, text-transform, letter-spacing for tracked headlines, -webkit-text-stroke or \
text-shadow for outlines, text-shadow for drop shadows and glows, background-clip:text with a linear-gradient for gradient or \
metallic (gold / silver) lettering. Keep effects as strong as in the draft -- no weaker, no stronger.
- Color harmony: put the draft's 3-5 main colors in style.palette (dominant background tone, 1-2 accents, dark and light text \
tones). Take every text, shell and icon color from that palette or from the measured colors; no new hues. Lines with the same \
role share the same color and font.
- Hierarchy: headline > price / offer / call-to-action > body > fine print, by size, weight and color contrast, as the draft \
shows it. Never make body text compete with the headline.
- Contrast: every text needs strong contrast with what is really behind it (check the BASE): light text on dark, dark on light; \
when the background is busy or mid-tone, add a subtle text-shadow or a shell, in the poster's palette.
- Lost shells: if the DRAFT shows a button, pill, badge or band behind a line but the BASE does not (the eraser removed it and \
no S# slot covers it), recreate it in that text op's box_style (background, border-radius, a little padding) in the draft's \
colors -- a call-to-action must never end up as bare text.
- Size: write every line at the size measured for its slot; fine print, labels and footers too -- never smaller "to be safe". \
If a text does not fit, shorten it (facts stay exact) rather than shrinking it.
- Lists: keep the draft's list structure -- one item per row, each row keeps its own text next to its bullet / check / icon; \
never merge list rows into one block and never leave a bullet or icon without its text.
- Slanted (italic / oblique) lettering in the draft stays italic; tracked (widely spaced) headlines keep their letter-spacing.
- Weight and family: a heavy / black headline stays heavy (font-weight 800-900 in a font that has it), a script stays a \
script font, a wide headline is never set in a condensed font (or the reverse). Unset weight = the weight measured on the draft.
- Letter width: each line lists how wide the drawn letters are (em per character) and the catalog lists each font's width \
(mixed case / caps). Pick a font within about 0.05em of the line (caps width for all-caps lines): a wider font does not fit \
the slot and gets shrunk, a narrower one looks lost in it.
- Same role, same look: rows of one list / menu / table, the items of one card set, the labels of one badge set share ONE \
font, size, weight and color -- write the same font-size on all of them (the smallest that fits), never one row big and the \
next small. Left-aligned columns: "justify-content:flex-start; text-align:left" on every row so the left edges line up. \
Give every op of such a set the same "group" name (e.g. "benefits", "menu-names", "menu-prices", "badges"): the renderer then \
draws the whole group at one size, the size that fits its longest member.
- More rows than items: the image model sometimes drew more list rows or bullets than there are client items. A long item \
may continue on the next row (that row's bullet skipped); otherwise skip the extra rows AND their bullets / icons.
- Badges / coins / stickers with a big middle line: put the key word or number of the client text big in the big slot and \
the rest small in the small slots (Giảm / 50%; Tặng / quà); skip slots left over rather than filling them with made-up words.
- Never a shell inside a shell: when the BASE still shows a pill / card / band behind a line, or its S# slot is drawn as a \
shape, the text op gets NO background, border or box-shadow of its own.

OUTPUT: one JSON object
{"style": {"display_font": "<key>", "text_font": "<key>", "palette": ["#hex"], "notes": "..."},
 "ops": [{"id": "o1", "slots": ["L1"], "kind": "text|keep|skip|shape|icon", "client": "T0" | null, "html": "...",
          "box_style": "...", "group": "name" | null, "why": "a few words"}],
 "missing": ["T5"], "notes": "..."}

FONT CATALOG (key: family, class): {fonts}

LUCIDE ICON NAMES: {icons}"""

REVIEW = """You are the same designer, now the ART DIRECTOR reviewing the rendered poster before it goes to the client. Your first pass \
was a plan made without seeing the result; now you SEE it. Use that: judge the PROOF with your eyes, not the plan.

You get: the DRAFT (the look to reach), the PROOF (exactly what the client will see), PROOF-IDS (the same with every op's \
box and id), COMPARE crops (draft left, proof right, enlarged, for the busy areas), the rendered ops (what each wrote; \
drafted size -> rendered font size, weight, font), the CODE CHANGES (what the engine changed in your plan on its own, see \
below) and the ERRORS a checker measured.

CODE CHANGES are deliberate and must not be undone: diacritics corrected to the client's spelling, duplicate lines skipped, \
invented words removed, cut-short client texts restored, font keys mapped, left edges of a column aligned, icons of one row \
/ column set to one size, lines of one role set to the size of the smallest one. When a change made something look worse (a \
whole list became small because one row was long), fix the cause (shorten or rebreak that row, use a narrower font for the \
role) -- do not reverse the change.

Look at the PROOF as a whole and then area by area against the DRAFT, and write down every flaw a careful designer would not \
ship. Typical flaws:
- an orphan: a price with no item on its row, a bullet / icon / number with no text, a lone word;
- an empty or under-filled shell: a badge, coin, pill, button or card the draft filled with big text that now holds tiny \
text or nothing;
- text not centred in its shell, or touching its edge;
- one role in different sizes, weights, colors or fonts (list rows, prices, card titles, badge labels);
- icons of one set in different sizes or styles;
- text that is hard to read on what is behind it;
- style that drifted from the draft: font family and width, weight, case, italic, effects, colors;
- awkward line breaks, crowding, text much smaller than the draft drew it.
Then fix each flaw with patches, and also every ERROR.

Your tools (you may use any of them):
- rewrite html / box_style of an op: font, weight, size, color, effects, line breaks, letter-spacing, padding, alignment;
- MERGE slots: one op over several slots (e.g. all lines of a coin or badge in one op, the key word or number big on its own \
line) -- the op is drawn over the union of its slots; stacked slots get one line each with <br> between them, never all \
on one line;
- move a client text to a better slot, split one over two slots, or give a slot to another op;
- turn a slot into a shape (recreate a lost pill / band behind text), an icon, keep or skip;
- delete an orphan or junk op that carries no client text.
Change only what is wrong -- ops that look right stay untouched. Facts stay exact; step / list numbers stay in order (1, 2, 3 -- \
never renumber a step to make room for another text). Never drop a client text: a poster \
missing its words is worse than any flaw. Slots never move.

OUTPUT one JSON object: {"findings": [{"ops": ["o7"], "flaw": "...", "fix": "..."}], "patches": [{"id": "o6", "slots": \
[...], "kind": "...", "html": "...", "box_style": "...", "client": ..., "delete": false}], "why": "..."} -- findings first \
(what you see), then one patch per fix; a patch lists only the fields it changes; "delete": true removes that op; a patch \
with a NEW id (e.g. "n1") and its slots adds a new op. Empty "patches" only when the PROOF is already as good as the draft's \
look allows."""


def _rewritten(op: dict, ops: list[dict], patches: dict) -> bool:
    """Chữ của lệnh op (bỏ dấu) có nằm trong một lệnh khác (sau vá) không: vòng duyệt gộp / chuyển chữ sang lệnh khác rồi mới xoá."""
    mine = _flat(_plain(op.get("html")))
    for o in ops + [p for p in patches.values() if p.get("id") not in {x["id"] for x in ops}]:
        if o.get("id") == op["id"]:
            continue
        q = patches.get(o.get("id")) or {}
        if q.get("delete"):
            continue
        if mine and mine in _flat(_plain(q.get("html", o.get("html")))):
            return True
    return False


def _px(b) -> str:
    return f"{b[2] - b[0]:.0f}x{b[3] - b[1]:.0f}px"


def _fits(w: float, m: dict, n: int = 4) -> list[str]:
    """Font trong danh mục có độ rộng chữ gần nét nháp nhất (cùng nhóm: nét dày-mảnh -> serif / viết tay, còn lại sans) --
    tên cụ thể từng dòng: con số độ rộng chung chung bị VLM bỏ qua (08/10: 70 lệnh bị co, font rộng hơn nét nháp 23%,
    gần như toàn Montserrat -- font rộng nhất bộ)."""
    from .fonts import CATALOG, WIDTH
    cls = ("serif", "script") if (m.get("stroke_contrast") or 0) >= 2.2 else ("sans",)
    col = 1 if m.get("caps") else 0
    keys = [k for k in WIDTH if CATALOG[k][1] in cls and "Italic" not in CATALOG[k][0]]   # bản nghiêng: VLM tự chọn khi nháp nghiêng
    return sorted(keys, key=lambda k: abs(WIDTH[k][col] - w))[:n]


def _look(m: dict) -> str:
    """Kiểu chữ đo trên nháp: độ đậm (độ dày nét / cỡ chữ), nét dày-mảnh (chữ có chân / viết tay), viết hoa, bóng / viền."""
    out = []
    if m.get("stroke") and m.get("size"):
        r = m["stroke"] / m["size"]   # độ dày nét / em
        out.append("weight " + ("light" if r < 0.07 else "regular" if r < 0.11 else "bold" if r < 0.16 else "black")
                   + f" (stroke ~{m['stroke']:.0f}px)")
    if (m.get("stroke_contrast") or 0) >= 2.2:
        out.append("thick-thin strokes (serif / script)")
    if m.get("caps"):
        out.append("all caps")
    n = len(m.get("ocr") or "")
    if n >= 4 and m.get("size") and abs(m.get("angle") or 0) < 2:
        # ĐỘ RỘNG CHỮ: rộng khung / (số ký tự x cỡ) -- so với độ rộng font ở danh mục (08/10: designer chọn font rộng hơn chữ nháp
        # -> phải co ~0.66, phần lớn lỗi too_small); OCR sai chữ nhưng số ký tự gần đúng
        w = (m["box"][2] - m["box"][0]) / (n * m["size"])
        # (08/10 lượt 4: ghi kèm tên font cùng độ rộng thì VLM đổi sang chữ condensed cả chỗ nháp là chữ thường -- bỏ)
        out.append(f"letters ~{w:.2f}em wide" + (" (condensed)" if w < 0.45 else " (wide)" if w > 0.62 else ""))
    e = m.get("effect")
    if e:
        if abs(e.get("dx", 0)) + abs(e.get("dy", 0)) >= 1:
            out.append(f"drop shadow {e.get('color')} offset ({e['dx']:.0f},{e['dy']:.0f})px")
        else:
            out.append(f"outline / glow {e.get('color')}")
    return (", " + ", ".join(out)) if out else ""


def describe(M: dict) -> str:
    sh = "\n".join(f"{s['id']}: shell {_px(s['box'])}, fill {s['fill']}, radius ~{s['radius']:.0f}px, contains "
                   f"{', '.join(s['lines'] + s['icons']) or 'nothing'}" for s in M["S"]) or "(none)"
    de = "\n".join(f"{c['id']}: detail {_px(c['box'])}, color {c['color']}" + (f", inside {c['shell']}" if c.get("shell") else "")
                   + (f", {c['near']}" if c.get("near") else "") for c in M["I"]) or "(none)"
    tilt = lambda m: f", drawn at {m['angle']:.0f} degrees" if abs(m.get("angle") or 0) >= 2 else ""   # noqa: E731
    li = "\n".join(f"{m['id']}: \"{m['ocr']}\" (text ~{m['size']:.0f}px, ~{len(m['ocr'])} characters, color {m['color']}"
                   + (f", inside {m['shell']}" if m.get("shell") else "") + f"{tilt(m)}{_look(m)})" for m in M["L"]) or "(none)"
    return f"Shells:\n{sh}\n\nDetails:\n{de}\n\nText lines:\n{li}"


def _system(base: str) -> str:
    from .fonts import CATALOG, WIDTH
    from .icons import available
    fonts = "; ".join(f"{k}: {v[0]} ({v[1]}" + (f", {WIDTH[k][0]:.2f} / caps {WIDTH[k][1]:.2f}em" if k in WIDTH else "") + ")"
                      for k, v in CATALOG.items())
    return base.replace("{fonts}", fonts).replace("{icons}", ", ".join(sorted(available())))


def _obj(x) -> dict:
    """Đầu ra VLM về dict: đôi khi trả JSON lồng trong chuỗi, hoặc mảng."""
    if isinstance(x, str):
        try:
            x = json.loads(x)
        except ValueError:
            return {}
    if isinstance(x, list):
        x = {"ops": x}
    return x if isinstance(x, dict) else {}


def _to_marks(P) -> dict:
    """slots -> marks (render dùng marks). Lệnh không phải object (VLM trả sai khuôn) bị bỏ, ghi vào "dropped"."""
    P = _obj(P)
    ops, bad = [], []
    for raw in P.get("ops") or []:
        op = _obj(raw) if isinstance(raw, str) else raw
        if not isinstance(op, dict) or not op:
            bad.append(str(raw)[:80])
            continue
        sl = op.get("slots") or op.get("marks") or []
        ops.append({**op, "marks": [m for m in (sl if isinstance(sl, list) else [sl]) if isinstance(m, str)]})
    return {**P, "ops": ops, **({"dropped": bad} if bad else {})}


def _texts(texts: list[dict]) -> str:
    return "\n".join(f"T{i} ({t.get('role', '')}): {t['text']}" for i, t in enumerate(texts))


PRODUCT_NOTE = ("The client uploaded a photo of their product and the image model copied it into the draft. Text, logos and "
                "graphics printed on that product (label, package, screen) are part of the photo: give every slot on them kind "
                "\"keep\", never redraw or skip them.")


VIEWS = {   # ảnh gửi designer (thử: scripts/vlm_probe.py). D nháp, S nháp đánh dấu ô, Q 4 góc phóng 2x, X bản xoá (nền dựng),
    "A": "DSQ",    # Z vùng cắt phóng to quanh cụm ô
    "AX": "DSQX",
    "BX": "DSX",
    "CX": "SXZ",
}
VIEW = "BX"  # chốt 07/10 (vlm_probe 23 ảnh dev): ít lỗi nhất, giữ chữ sản phẩm tốt nhất (29/30), rẻ nhất (~12k token)


def _quarters(mk) -> list[bytes]:
    """4 góc chồng 10%, phóng 2x: đọc được ô nhỏ."""
    from PIL import Image
    W, H = mk.size
    out = []
    for (a, b) in ((0, 0), (1, 0), (0, 1), (1, 1)):
        x0, y0 = int(a * W * 0.45), int(b * H * 0.45)
        q = mk.crop((x0, y0, x0 + int(W * 0.55), y0 + int(H * 0.55)))
        out.append(_png(q.resize((q.width * 2, q.height * 2), Image.LANCZOS)))
    return out


def _clusters(mk, M: dict, n_max: int = 4, side: int = 1024) -> list[bytes]:
    """Vùng cắt quanh CỤM ô, phóng cạnh dài = side: chỗ có ô được phóng, chỗ chỉ có ảnh thì không."""
    from PIL import Image
    out = []
    for b in _cluster_boxes(M, *mk.size, n_max=n_max):
        c = mk.crop(b)
        z = side / max(c.size)
        out.append(_png(c.resize((max(1, int(c.width * z)), max(1, int(c.height * z))), Image.LANCZOS)))
    return out


def _cluster_boxes(M: dict, W: int, H: int, n_max: int = 4) -> list[list[int]]:
    """CỤM ô: khung ô nới 4% cạnh ngắn chạm nhau = một cụm; gộp cặp cụm làm tăng diện tích ít nhất tới <= n_max."""
    pad = 0.04 * min(W, H)
    boxes = [[b["box"][0] - pad, b["box"][1] - pad, b["box"][2] + pad, b["box"][3] + pad] for b in M["L"] + M["S"] + M["I"]]
    if not boxes:
        return []
    ov = lambda a, b: a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]   # noqa: E731
    un = lambda a, b: [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]   # noqa: E731
    cl = [list(b) for b in boxes]
    merged = True
    while merged:
        merged = False
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                if ov(cl[i], cl[j]):
                    cl[i] = un(cl[i], cl.pop(j))
                    merged = True
                    break
            if merged:
                break
    while len(cl) > n_max:   # gộp cặp làm tăng diện tích ít nhất
        area = lambda b: (b[2] - b[0]) * (b[3] - b[1])   # noqa: E731
        i, j = min(((i, j) for i in range(len(cl)) for j in range(i + 1, len(cl))),
                   key=lambda ij: area(un(cl[ij[0]], cl[ij[1]])) - area(cl[ij[0]]) - area(cl[ij[1]]))
        cl[i] = un(cl[i], cl.pop(j))
    return [[max(0, int(b[0])), max(0, int(b[1])), min(W, int(b[2])), min(H, int(b[3]))] for b in sorted(cl, key=lambda b: (b[1], b[0]))]


def plan(draft: np.ndarray, M: dict, prompt: str, texts: list[dict], call, product: bool = False, base: np.ndarray | None = None,
         view: str | None = None, meta: dict | None = None) -> dict:
    """base: bản xoá đã dọn (nền dựng) cho các cấu hình có X. meta: nhận số token / thời gian (llm.json_call)."""
    mk = marked_image(draft, M)
    W, H = mk.size
    v = VIEWS[view or VIEW]
    imgs = []
    for c in v:
        if c == "D":
            imgs += ["DRAFT:", _png(draft)]
        elif c == "S":
            imgs += ["SLOTS (marked):", _png(mk)]
        elif c == "Q":
            imgs += ["ZOOM (4 quarters of SLOTS, 2x):", *_quarters(mk)]
        elif c == "X" and base is not None:
            imgs += ["BASE (erased poster, your slots are drawn on it):", _png(base)]
        elif c == "Z":
            imgs += ["ZOOM (slot clusters of SLOTS, enlarged):", *_clusters(mk, M)]
    user = (f"PROMPT given to the image model:\n{prompt}\n\nClient texts (exact words):\n{_texts(texts)}\n\nPoster size: {W}x{H}px.\n\n"
            + (PRODUCT_NOTE + "\n\n" if product else "") + f"{describe(M)}")
    args = (_system(SYSTEM), [user, *imgs]) + ((meta,) if meta is not None else ())
    return _to_marks(call(*args))


def validate(P: dict, M: dict, texts: list[dict] | None = None) -> tuple[dict, list[str]]:
    """Phòng ngừa nhẹ, có ghi log: id ô không có -> bỏ id; lệnh không còn ô nào -> bỏ lệnh. texts (câu khách): chặn chữ (_guard)."""
    log, ops = [f"bỏ lệnh sai khuôn: {d}" for d in P.get("dropped") or []], []
    by = M["by"]
    for op in P.get("ops", []):
        mk = [m for m in op.get("marks") or [] if m in by]
        if len(mk) < len(op.get("marks") or []):
            log.append(f"{op.get('id')}: bỏ id không có {sorted(set(op.get('marks') or []) - set(mk))}")
        if not mk:
            continue
        ops.append({**op, "marks": mk, "slots": mk})
    if texts:
        ops, lg = _guard(ops, M, texts)
        log += lg
    return {**P, "ops": ops}, log


def score(P: dict, errs: list[dict]) -> tuple:
    """So bản lượt đầu / sau vòng sửa (nhỏ hơn = tốt hơn): trước hết số câu khách được viết (08/10 m03: vòng sửa bỏ tiêu đề
    "KHAI TRƯƠNG" để hết lỗi too_small và được giữ vì ít lỗi hơn), rồi số lỗi, rồi mức chồng / tràn."""
    told = {_tid(op.get("client")) for op in P.get("ops", []) if op.get("kind") == "text" and op.get("html")} - {None}
    return -len(told), len(errs), sum(e.get("frac", 0) + e.get("px", 0) / 100 for e in errs)


def _fold(s: str) -> str:
    """Bỏ dấu tiếng Việt, chữ thường: "TRƯỞNG" == "trương" == "TRUONG"."""
    import unicodedata
    s = unicodedata.normalize("NFD", s.replace("đ", "d").replace("Đ", "D"))
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def _plain(h: str) -> str:
    import html as H
    # <br> / <BR> / thẻ khối = dấu cách; thẻ inline (span, i-font) không tách chữ (08/10 m01: "FLASH<BR>SALE" thành "FLASHSALE")
    h = re.sub(r"<\s*(br|/?div|/?p)\b[^>]*>", " ", h or "", flags=re.I)
    return " ".join(H.unescape(re.sub(r"<[^>]+>", "", h)).split())


def _flat(s: str) -> str:
    return re.sub(r"\W", "", _fold(s))


def _tid(c) -> int | None:
    c = str(c or "")
    return int(c[1:]) if c[:1] in "Tt" and c[1:].isdigit() else int(c) if c.isdigit() else None


WORD = re.compile(r"\w+", re.UNICODE)


def _guard(ops: list[dict], M: dict, texts: list[dict]) -> tuple[list[dict], list[str]]:
    """CHẶN CHỮ SAI bằng mã (08/10, BX 23 ảnh dev: VLM chép chữ từ ảnh nháp thay vì câu khách):
    1. DẤU: chữ khớp một chữ của câu khách khi bỏ dấu nhưng khác dấu ("TRƯỞNG", "TRƯỜNG" cho "TRƯƠNG") -> thay bằng chữ của
       khách (giữ hoa / thường). Chỉ xét lệnh đang viết câu khách: có "client", hoặc cả dòng (bỏ dấu) nằm trong một câu khách.
    2. CHỮ BỊA: lệnh khai viết câu khách T# mà không chữ nào có trong bất kỳ câu khách nào ("SƯ PRING" chép từ mẩu OCR vỡ) -> skip.
    2b. SỐ BỊA: lệnh không khai câu khách mà có số không nằm trong câu khách nào -> skip.
    3. LẶP: nháp hay vẽ một dòng hai lần (ngày, giá, nhãn huy hiệu); hai lệnh viết cùng một chữ (bỏ dấu, >= 3 ký tự) mà câu
       khách không lặp -> giữ lệnh có chữ OCR của ô giống nhất, lệnh kia skip (nền xoá sạch ở đó)."""
    from difflib import SequenceMatcher
    log, txt = [], [t["text"] for t in texts]
    ftxt = [" ".join(WORD.findall(_fold(t))) for t in txt]
    vocab_all = {w for t in txt for w in WORD.findall(t)}
    out = []
    for op in ops:
        if op.get("kind") != "text" or not op.get("html"):
            out.append(op)
            continue
        plain = _plain(op["html"])
        fp = " ".join(WORD.findall(_fold(plain)))
        k = _tid(op.get("client"))
        own = txt[k] if k is not None and 0 <= k < len(txt) else None
        words = WORD.findall(plain)
        if own is not None and words and not any(c.isdigit() for c in plain) and \
                not {_fold(w) for w in words} & {_fold(w) for w in vocab_all} and \
                not any(_flat(plain) in _flat(t) for t in txt):   # chữ dính / tách khác câu khách vẫn là chữ khách
            log.append(f"{op.get('id')}: chữ không có trong câu khách ({plain!r} khai {op.get('client')}) -> skip")
            out.append({**op, "kind": "skip", "html": "", "why": "guard: invented text"})
            continue
        nums = re.findall(r"\d+", plain)
        if own is None and nums and not all(any(n in t for t in txt) for n in nums):
            # SỐ BỊA: lệnh không khai câu khách mà viết số không có trong câu khách nào (địa chỉ / giá / điện thoại / ngày nháp tự
            # vẽ, 08/10 "123 Đường Số 1, Quận 1, TP. HCM") -> skip: dữ kiện không bao giờ bịa
            log.append(f"{op.get('id')}: số không có trong câu khách ({plain!r}) -> skip")
            out.append({**op, "kind": "skip", "html": "", "why": "guard: invented fact"})
            continue
        src = [own] if own is not None else [t for t, f in zip(txt, ftxt) if fp and f" {fp} " in f" {f} "]
        if src:
            cand = {}   # chữ bỏ dấu -> các dạng có dấu trong câu của lệnh
            for t in src:
                for w in WORD.findall(t):
                    cand.setdefault(_fold(w), set()).add(w.lower())

            def fix(m):
                w = m.group(0)
                opts = cand.get(_fold(w)) or set()
                if len(opts) != 1 or w.lower() in opts:
                    return w
                c = next(iter(opts))
                return c.upper() if w.isupper() else (c[:1].upper() + c[1:]) if w[:1].isupper() else c
            parts = re.split(r"(<[^>]+>|&\w+;|&#\d+;)", op["html"])
            new = "".join(p if p.startswith("<") or p.startswith("&") else WORD.sub(fix, p) for p in parts)
            if new != op["html"]:
                log.append(f"{op.get('id')}: sửa dấu theo câu khách {plain!r} -> {_plain(new)!r}")
                op = {**op, "html": new}
        out.append(op)
    # 3. lặp
    groups = {}
    for i, op in enumerate(out):
        if op.get("kind") == "text" and op.get("html"):
            f = " ".join(WORD.findall(_fold(_plain(op["html"]))))
            if len(f.replace(" ", "")) >= 3:
                groups.setdefault(f, []).append(i)
    for f, idx in groups.items():
        allowed = max(1, sum(1 for t in ftxt if t == f))
        if len(idx) <= allowed:
            continue
        ocr = lambda i: _fold(" ".join(M["by"][m].get("ocr", "") for m in out[i]["marks"] if m.startswith("L")))   # noqa: E731
        idx = sorted(idx, key=lambda i: -SequenceMatcher(None, ocr(i), f).ratio())
        for i in idx[allowed:]:
            log.append(f"{out[i].get('id')}: lặp chữ {_plain(out[i]['html'])!r} (giữ {out[idx[0]].get('id')}) -> skip")
            out[i] = {**out[i], "kind": "skip", "html": "", "why": "guard: duplicate"}
    # 4. CÂU NGẮN (<= 3 chữ: lương "15-25 triệu", giá, nhãn) không được cắt bớt -- câu dài mới được rút gọn (08/10 h06: vòng sửa
    #    bỏ "triệu" cho vừa ô). Một lệnh viết câu đó mà thiếu chữ -> trả nguyên câu khách vào đúng chỗ chữ trong html.
    for k, t in enumerate(txt):
        tw = WORD.findall(t)
        mine = [i for i, op in enumerate(out) if op.get("kind") == "text" and op.get("html") and _tid(op.get("client")) == k]
        if not tw or len(tw) > 3 or len(mine) != 1:
            continue
        op = out[mine[0]]
        plain = _plain(op["html"])
        lost = {_fold(w) for w in tw} - {_fold(w) for w in WORD.findall(plain)}
        if not lost or not plain or plain not in op["html"]:
            continue
        near = {_fold(w) for o in out if o is not op and o.get("kind") == "text" for w in WORD.findall(_plain(o.get("html") or ""))}
        if lost & near:   # chữ thiếu đã nằm ở lệnh khác (huy hiệu tách "Giảm" / "50%" thành hai ô)
            continue
        log.append(f"{op.get('id')}: câu ngắn bị cắt {plain!r} -> {t!r}")
        out[mine[0]] = {**op, "html": op["html"].replace(plain, t, 1)}
    return out, log


def _ids_image(proof: np.ndarray, M: dict, P: dict):
    """Bản dựng + khung và id từng lệnh vẽ (màu theo loại) -- để VLM trỏ đúng lệnh khi duyệt."""
    from PIL import Image, ImageDraw, ImageFont
    from .render import KIND_COL, op_box
    im = Image.fromarray(proof).convert("RGB")
    d = ImageDraw.Draw(im)
    f = ImageFont.load_default(size=max(12, im.width // 60))
    for op in P["ops"]:
        b = op_box(op, M) if op.get("kind") not in ("skip", "keep") else None
        if not b:
            continue
        col = KIND_COL.get(op.get("kind"), (230, 30, 30))
        d.rectangle(b, outline=col, width=2)
        tw, th = d.textbbox((0, 0), op["id"], font=f)[2:]
        d.rectangle([b[0], max(0, b[1] - th - 2), b[0] + tw + 2, max(0, b[1] - th - 2) + th + 2], fill=col)
        d.text((b[0] + 1, max(0, b[1] - th - 2)), op["id"], fill=(255, 255, 255), font=f)
    return im


def _compare(draft: np.ndarray, proof: np.ndarray, M: dict, n_max: int = 3, side: int = 1280) -> list[bytes]:
    """Ảnh SO SÁNH vùng đông ô: nháp (trái) | bản dựng (phải), cùng vùng cắt, phóng -- toàn cảnh 1024px quá nhỏ để thấy chữ
    trong huy hiệu / chân trang (08/10)."""
    from PIL import Image
    H, W = draft.shape[:2]
    out = []
    for b in _cluster_boxes(M, W, H, n_max=n_max):
        a, c = Image.fromarray(draft).crop(b), Image.fromarray(proof).crop(b)
        im = Image.new("RGB", (a.width * 2 + 12, a.height), "white")
        im.paste(a, (0, 0))
        im.paste(c, (a.width + 12, 0))
        z = min(side / im.width, side / im.height, 3.0)
        out.append(_png(im.resize((max(1, int(im.width * z)), max(1, int(im.height * z))), Image.LANCZOS)))
    return out


def review(draft: np.ndarray, proof: np.ndarray, M: dict, P: dict, errs: list[dict], res: list[dict], prompt: str,
           texts: list[dict], call, product: bool = False, meta: dict | None = None, changes: list[str] | None = None) -> dict:
    """VÒNG DUYỆT (luôn chạy, 08/10 thay vòng sửa chỉ-khi-có-lỗi): VLM xem bản dựng cả tấm như giám đốc nghệ thuật -- sửa lỗi
    khách quan (render.check) VÀ lỗi thẩm mỹ tự thấy (mồ côi, chữ nhỏ / mảnh trong huy hiệu, lệch tâm vỏ, cùng vai khác cỡ /
    độ đậm, icon không đều...). Gửi: nháp, bản dựng, bản dựng có id lệnh, danh sách lệnh kèm cỡ / độ đậm / font DỰNG THẬT (không
    cần OCR bản dựng: code biết chính xác đã viết gì ở đâu), lỗi. -> P mới (vá / xoá / thêm lệnh); rào chắn ở slots.accept."""
    by_res = {r["id"]: r for r in res or []}
    rows = []
    for op in P["ops"]:
        r = by_res.get(op["id"]) or {}
        row = {k: op.get(k) for k in ("id", "marks", "kind", "client", "html", "box_style") if op.get(k) not in (None, "")}
        Ls = [M["by"][m] for m in op.get("marks") or [] if m.startswith("L") and m in M["by"]]
        if r.get("fs"):
            row["rendered"] = {"drafted_px": round(float(np.median([m["size"] for m in Ls]))) if Ls else None,
                               "font_px": round(r["fs"]), "weight": r.get("fw"), "font": (r.get("ff") or "").split(",")[0].strip("'\" ")}
            if r.get("shrink", 1) < 0.9 and len(Ls) == 1 and len(Ls[0].get("ocr") or "") >= 4:
                # VÌ SAO nhỏ: font rộng hơn nét nháp (gợi ý font đúng độ rộng) hay chữ dài hơn chỗ nháp vẽ
                from .fonts import WIDTH
                from .render import font_key
                L0 = Ls[0]
                fk = font_key(row["rendered"]["font"])
                dw = (L0["box"][2] - L0["box"][0]) / (len(L0["ocr"]) * L0["size"])
                if fk in WIDTH:
                    row["rendered"]["why_small"] = (f"font {WIDTH[fk][1 if L0.get('caps') else 0] / dw:.2f}x wider than the drafted "
                                                    f"letters; same-width fonts (only if their letter shape matches the draft): "
                                                    f"{', '.join(_fits(dw, L0))}; text "
                                                    f"{len(_plain(op.get('html') or '')) / len(L0['ocr']):.2f}x the drafted length")
        rows.append(row)
    ch = [c for c in changes or [] if not c.startswith("--")]
    user = (f"PROMPT given to the image model:\n{prompt}\n\nClient texts (exact words):\n{_texts(texts)}\n\n"
            + (PRODUCT_NOTE + "\n\n" if product else "") + f"{describe(M)}\n\nStyle: {json.dumps(P.get('style'), ensure_ascii=False)}"
            f"\n\nRendered ops:\n{json.dumps(rows, ensure_ascii=False)}\n\nCODE CHANGES (engine log, op ids):\n"
            + ("\n".join(ch) or "(none)") + "\n\nERRORS:\n"
            + ("\n".join(json.dumps(e, ensure_ascii=False) for e in errs if REVIEW_MODE == "full" or e["type"] in SEVERE) or "(none)"))
    full = REVIEW_MODE == "full"
    imgs = ["DRAFT:", _png(draft), "PROOF:", _png(proof), "PROOF-IDS:", _png(_ids_image(proof, M, P))] + \
        (["COMPARE (draft | proof):", *_compare(draft, proof, M)] if full else [])
    sysmsg = _system(SYSTEM) + "\n\n" + REVIEW + ("" if full else "\n\n" + ONLY_ERRORS)
    out = _obj(call(*((sysmsg, [user, *imgs]) + ((meta,) if meta is not None else ()))))
    new, patches = [], {p.get("id"): p for p in out.get("patches") or [] if isinstance(p, dict) and p.get("id")}
    for op in P["ops"]:
        p = patches.pop(op["id"], None)
        if p is None:
            new.append(op)
        elif p.get("delete") and op.get("kind") == "text" and _tid(op.get("client")) is not None and op.get("html") and                 not _rewritten(op, P["ops"], patches):
            # xoá lệnh đang viết (một phần) câu khách mà không lệnh nào khác viết lại phần đó -> không xoá (08/10 server bánh mì
            # v1: vòng duyệt xoá dòng "Bánh Mì", poster chỉ còn "Cô Ba")
            new.append(op)
        elif p.get("delete"):   # xoá = không vẽ gì; ô vẫn đã quyết (skip), không thành "ô bỏ sót" (08/10 h08: cả lượt bị loại)
            new.append({"id": op["id"], "kind": "skip", "marks": op.get("marks") or [], "why": "review: delete"})
        else:
            q = {**op, **{k: v for k, v in p.items() if k not in ("id", "delete")}}
            if "slots" in p:
                q["marks"] = list(p["slots"] or [])
            new.append({k: v for k, v in q.items() if v is not None or k == "client"})
    for pid, p in patches.items():   # lệnh MỚI (ô chưa quyết)
        if p.get("slots") and not p.get("delete"):
            new.append({**{k: v for k, v in p.items() if k != "delete"}, "marks": list(p["slots"])})
    # ô bị lệnh khác lấy đi (gộp ô) thì bỏ khỏi lệnh cũ; lệnh cũ hết ô thì bỏ
    regrab = {p.get("id") for p in out.get("patches") or [] if isinstance(p, dict) and p.get("slots")}   # lệnh vá có ô mới thắng
    owner = {}
    for op in sorted(new, key=lambda o: o["id"] not in regrab):
        for m in op.get("marks") or []:
            owner.setdefault(m, op["id"])
    new = [{**op, "marks": [m for m in op.get("marks") or [] if owner.get(m) == op["id"]]} for op in new]
    new = [op for op in new if op["marks"]]
    return _to_marks({**P, "ops": new, "review_why": out.get("why"), "review_findings": out.get("findings"),
                      "review_patches": len(out.get("patches") or [])})


REVIEW_ROUNDS = 2   # chế độ "full": dựng -> duyệt -> dựng -> duyệt; dừng sớm khi VLM không sửa gì / bản duyệt bị loại
# CHẾ ĐỘ (08/10, sau lượt 3-4: duyệt thẩm mỹ tốn ~20 s / poster, ~2 / 15 poster đẹp hơn, ~5 xấu đi -> không đáng):
#   "errors" (mặc định) -- MỘT lượt, CHỈ khi lượt đầu có lỗi nặng (SEVERE), chỉ sửa đúng các lỗi đó (~1 / 3 poster)
#   "full"   -- duyệt thẩm mỹ cả tấm, REVIEW_ROUNDS lượt, có giám khảo (slots.judge)
#   "off"    -- không bao giờ gọi VLM lần hai (độ trễ cố định; lỗi nặng còn nguyên trên poster)
REVIEW_MODE = os.environ.get("TEXTFIX_REVIEW", "errors")
ONLY_ERRORS = """MODE: FIX ONLY THE ERRORS. Fix exactly the ERRORS listed and nothing else -- every op not involved in an error \
stays untouched; no aesthetic changes. Output "findings" for the errors only."""


def review_rounds() -> int:
    return REVIEW_ROUNDS if REVIEW_MODE == "full" else 1


def review_needed(errs: list[dict]) -> bool:
    """Có chạy vòng duyệt cho bản này không (theo REVIEW_MODE)."""
    if REVIEW_MODE == "off":
        return False
    return REVIEW_MODE == "full" or any(e["type"] in SEVERE for e in errs)
SEVERE = ("unassigned", "content_overlap", "boxes_overlap", "overflow", "low_contrast", "mark_twice")


JUDGE = """You are an art director choosing which of two finished renders of the same poster to send to the client. The DRAFT \
shows the intended look. Pick the better poster: closer to the draft's look (type style, weight, size, colors, layout), \
cleaner, every text easy to read, nothing cut off, overlapping, crammed, tiny or orphaned, badges / pills / buttons properly \
filled. Judge only what you see. Reply with one JSON object: {"better": 1 or 2, "why": "a few words"}."""


def judge(draft: np.ndarray, before: np.ndarray, after: np.ndarray, call, meta: dict | None = None) -> tuple[bool, str]:
    """GIÁM KHẢO thẩm mỹ cho vòng duyệt (08/10 lượt 4: bản duyệt qua được rào đo lường nhưng xấu đi -- vòng tròn -50% phình
    cắt chữ, huy hiệu lộn xộn hơn): VLM so trước / sau cạnh nhau (thứ tự xáo theo nội dung ảnh, chống thiên vị vị trí) -> True
    khi bản SAU đẹp hơn. Lỗi / trả sai khuôn -> giữ bản sau (rào đo lường đã qua)."""
    import hashlib
    swap = hashlib.md5(after[::16, ::16].tobytes()).digest()[0] % 2 == 1
    one, two = (after, before) if swap else (before, after)
    try:
        out = _obj(call(*((JUDGE, ["DRAFT:", _png(draft), "POSTER 1:", _png(one), "POSTER 2:", _png(two)])
                         + ((meta,) if meta is not None else ()))))
        b = int(str(out.get("better")).strip()[:1])
        assert b in (1, 2)
    except (ValueError, TypeError, AttributeError, AssertionError):
        return True, "judge: không đọc được -> giữ bản duyệt"
    return (b == 1) == swap, str(out.get("why") or "")[:200]


def salvage(P: dict, errs: list[dict], P2: dict, errs2: list[dict]) -> dict | None:
    """Bản duyệt bị loại (accept = False) thường chỉ hỏng ở một hai lệnh, phần còn lại là sửa tốt (08/10 h08: xoá một huy hiệu
    làm ô bỏ sót -> mất cả ba huy hiệu đã dựng lại đẹp). Hoàn lại ĐÚNG các lệnh gây lỗi nặng mới / làm mất câu khách về bản
    trước, giữ các vá khác. None khi không có gì để cứu."""
    key = lambda e: (e["type"], tuple(sorted(e.get("ops") or [])), tuple(sorted(e.get("slots") or [])))   # noqa: E731
    seen = {key(e) for e in errs if e["type"] in SEVERE}
    old = {o["id"]: o for o in P["ops"]}
    bad = set()
    for e in errs2:
        if e["type"] in SEVERE and key(e) not in seen:
            bad |= set(e.get("ops") or [])
            bad |= {o["id"] for o in P["ops"] for s in e.get("slots") or [] if s in (o.get("marks") or [])}
    told = lambda Q: {_tid(o.get("client")): o["id"] for o in Q["ops"] if o.get("kind") == "text" and o.get("html")}   # noqa: E731
    lost = set(told(P)) - set(told(P2)) - {None}
    bad |= {i for c, i in told(P).items() if c in lost}
    if not bad:
        return None
    new = [old[o["id"]] if o["id"] in old and o["id"] in bad else o for o in P2["ops"] if o["id"] in old or o["id"] not in bad]
    new += [old[i] for i in bad if i in old and i not in {o["id"] for o in new}]
    owner = {}
    for op in sorted(new, key=lambda o: o["id"] not in bad):   # lệnh hoàn lại lấy lại ô của nó
        for m in op.get("marks") or []:
            owner.setdefault(m, op["id"])
    new = [{**op, "marks": [m for m in op.get("marks") or [] if owner.get(m) == op["id"]]} for op in new]
    return {**P2, "ops": [op for op in new if op["marks"]], "salvaged": sorted(bad)}


def accept(P: dict, errs: list[dict], P2: dict, errs2: list[dict]) -> bool:
    """Giữ bản sau vòng duyệt? Sửa thẩm mỹ không đo được, nên tin VLM -- trừ khi nó làm hỏng điều đo được: viết ít câu khách hơn,
    hoặc thêm lỗi NẶNG (ô bỏ sót, chồng nhau, tràn, chìm nền). too_small / uneven_size được phép tăng (chữ to lên trong huy hiệu
    có thể chạm sàn co)."""
    sev = lambda E: sum(1 for e in E if e["type"] in SEVERE)   # noqa: E731
    return score(P2, [])[0] <= score(P, [])[0] and sev(errs2) <= sev(errs)


# ---------------------------------------------------------------------------------------------------- nền
def base_image(draft: np.ndarray, plate: np.ndarray, M: dict, P: dict) -> np.ndarray:
    """Bản xoá + DÁN LẠI pixel nháp ở các ô "keep" (chữ / logo / hình in trên sản phẩm, màn hình: bản xoá đã xoá cả chúng).
    Vùng dán = khung ô nới 0.25 cao + TRỌN các mảng lớp phủ chạm khung (vòng con dấu, phần logo ngoài khung dòng), trong giới
    hạn khung nới 1 cao ô mỗi phía (mảng lớp phủ to chạy xa -- dải, thẻ -- không bị kéo theo)."""
    import cv2
    H, W = draft.shape[:2]
    m = np.zeros((H, W), np.uint8)
    ov = M.get("mask")
    cc = None
    if ov is not None:
        _, cc = cv2.connectedComponents(cv2.morphologyEx(ov.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)), connectivity=8)
    for op in P.get("ops", []):
        if op.get("kind") != "keep":
            continue
        for i in op.get("marks") or []:
            if i not in M["by"]:
                continue
            x0, y0, x1, y1 = M["by"][i]["box"]
            h = y1 - y0
            p = 0.25 * h
            m[int(max(0, y0 - p)):int(min(H, y1 + p)), int(max(0, x0 - p)):int(min(W, x1 + p))] = 1
            if cc is not None:
                X0, Y0, X1, Y1 = int(max(0, x0 - h)), int(max(0, y0 - h)), int(min(W, x1 + h)), int(min(H, y1 + h))
                ids = np.unique(cc[int(max(0, y0)):int(min(H, y1)), int(max(0, x0)):int(min(W, x1))])
                ids = ids[ids > 0]
                if len(ids):
                    sel = np.isin(cc[Y0:Y1, X0:X1], ids)
                    m[Y0:Y1, X0:X1] |= cv2.dilate(sel.astype(np.uint8), np.ones((5, 5), np.uint8))
    if not m.any():
        return plate.copy()
    a = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 2.0)[..., None]
    return (draft * a + plate * (1 - a)).round().clip(0, 255).astype(np.uint8)
