"""DESIGNER "ĐIỀN Ô": nháp - bản xoá = lớp phủ model đã vẽ -> các Ô đúng toạ độ; VLM chỉ quyết mỗi ô là gì và điền cho đẹp; code
dựng đúng khung từng ô (render.py), theo lớp vỏ -> chi tiết -> chữ.

  M = build(draft, plate)                         # S# vỏ, I# chi tiết nhỏ, L# dòng chữ (OCR, khung nắn theo lớp phủ, số đo nét)
  P = plan(draft, M, prompt, texts, call)         # VLM: {style, ops: [{slots, kind, html, box_style, client}], missing}
  P, log = validate(P, M)
  base = base_image(draft, plate, M, P)           # bản xoá + dán lại pixel nháp ở dòng "keep" (chữ in trên sản phẩm / màn hình)
  P2 = repair(draft, poster, M, P, errs, prompt, texts, call)   # có lỗi (render.check): một vòng, chỉ các lệnh / ô lỗi

VLM không đưa toạ độ: chỉ trỏ id ô. Mỗi ô thuộc đúng một lệnh (lệnh chữ được gộp nhiều dòng liền nhau thành một khối).
"""

from __future__ import annotations

import io
import json

import numpy as np

CAP_EM = 0.72        # cao chữ hoa / em
X_EM = 0.53          # cao chữ thường (x-height) / em
SNAP_SIDE = 0.35     # nắn khung dòng: xét lớp phủ quanh đa giác OCR, nới ngang / xuống 0.35 cao dòng ...
SNAP_UP = 0.9        # ... và lên 0.9 cao dòng (dấu thanh / mũ tiếng Việt)
INK_IN_SHELL = 25.0  # trong vỏ: nét chữ = khác màu lòng vỏ >= 25 ΔE (cả lòng vỏ đều thuộc lớp phủ)
REPAIR_MAX = 6       # vòng sửa: tối đa 6 lỗi có ảnh (2 ảnh / lỗi) -- máy chủ VLM giới hạn số ảnh mỗi lời gọi
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
    S = [{k: v for k, v in s.items() if not k.startswith("_")} for s in O["S"]]
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

AESTHETICS COME FIRST. When the draft and the prompt / client texts disagree, follow the draft. A clean, consistent, \
well-fitted poster beats a complete one.

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
- Every slot id appears in exactly ONE op -- never leave a slot out (use "skip" to drop it).

TEXT
- Decide what the model meant to write on each line and write it at about the size it drew, keeping its line breaks (<br> \
between the lines of a multi-line op). Write about as many characters per line as the model wrote there -- never much more: \
crammed text gets shrunk and looks bad.
- A client text longer than the lines the model gave it: shorten or rephrase it so it fits naturally; facts (names, prices, \
numbers, dates, phones, addresses, emails, links) are never changed or invented -- keep them exact or leave them out. Set \
"client": "T<i>" on ops writing (part of) a client text; list client texts the model did not draw, or that you dropped, in \
"missing".
- Text the model invented: write what it was meant to say from its role, position and the PROMPT, a natural line of about the \
same length in the poster's language (usually Vietnamese). Never copy garbled OCR. Commit; do not hedge.
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

OUTPUT: one JSON object
{"style": {"display_font": "<key>", "text_font": "<key>", "palette": ["#hex"], "notes": "..."},
 "ops": [{"id": "o1", "slots": ["L1"], "kind": "text|keep|skip|shape|icon", "client": "T0" | null, "html": "...",
          "box_style": "...", "why": "a few words"}],
 "missing": ["T5"], "notes": "..."}

FONT CATALOG (key: family, class): {fonts}

LUCIDE ICON NAMES: {icons}"""

REPAIR = """You are the same designer. The renderer drew your ops and its checker found mechanical ERRORS (listed below): slots \
no op covers (unassigned -- decide them: text / icon / shape / keep / skip), content of two ops overlaps, content overflows its \
slot, or content had to be shrunk a lot to fit (too_small: you wrote more text than the slot holds -- shorten it, keep facts \
exact, or merge it with the next line of the same block). For each error you get the involved ops / slots (JSON) and two zoomed \
crops: the PROOF (rendered; each involved op outlined with its id) and the MARKED draft. Fix ONLY these errors. Slots never \
move: you change text, merging of consecutive lines, html, box_style.

OUTPUT one JSON object: {"patches": [{"id": "o6", "slots": [...], "kind": "...", "html": "...", "box_style": "...", "client": ...,
"delete": false}], "why": "..."} -- a patch lists only the fields it changes; "delete": true removes that op; a patch with a NEW id
(e.g. "n1") and its slots adds a new op (use it for unassigned slots)."""


def _px(b) -> str:
    return f"{b[2] - b[0]:.0f}x{b[3] - b[1]:.0f}px"


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
    from .fonts import CATALOG
    from .icons import available
    fonts = "; ".join(f"{k}: {v[0]} ({v[1]})" for k, v in CATALOG.items())
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
VIEW = "A"   # cấu hình dùng thật (chốt theo kết quả thử)


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
    """Vùng cắt quanh CỤM ô (khung ô nới 4% cạnh ngắn chạm nhau = một cụm; gộp cụm gần nhất tới <= n_max), phóng cạnh dài
    = side: chỗ có ô được phóng, chỗ chỉ có ảnh thì không."""
    from PIL import Image
    W, H = mk.size
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
    out = []
    for b in sorted(cl, key=lambda b: (b[1], b[0])):
        b = [max(0, int(b[0])), max(0, int(b[1])), min(W, int(b[2])), min(H, int(b[3]))]
        c = mk.crop(b)
        z = side / max(c.size)
        out.append(_png(c.resize((max(1, int(c.width * z)), max(1, int(c.height * z))), Image.LANCZOS)))
    return out


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


def validate(P: dict, M: dict) -> tuple[dict, list[str]]:
    """Phòng ngừa nhẹ, có ghi log: id ô không có -> bỏ id; lệnh không còn ô nào -> bỏ lệnh."""
    log, ops = [f"bỏ lệnh sai khuôn: {d}" for d in P.get("dropped") or []], []
    by = M["by"]
    for op in P.get("ops", []):
        mk = [m for m in op.get("marks") or [] if m in by]
        if len(mk) < len(op.get("marks") or []):
            log.append(f"{op.get('id')}: bỏ id không có {sorted(set(op.get('marks') or []) - set(mk))}")
        if not mk:
            continue
        ops.append({**op, "marks": mk, "slots": mk})
    return {**P, "ops": ops}, log


def repair(draft: np.ndarray, proof: np.ndarray, M: dict, P: dict, errs: list[dict], prompt: str, texts: list[dict], call,
           product: bool = False, meta: dict | None = None) -> dict:
    """Vòng sửa (chỉ khi render.check báo lỗi): một lượt VLM, gửi các lệnh / ô lỗi + ảnh phóng vùng lỗi (bản dựng có khung lệnh,
    nháp đánh dấu) -> P mới: vá đúng các lệnh đó, thêm lệnh cho ô chưa quyết; các lệnh khác giữ nguyên."""
    from PIL import Image, ImageDraw
    from .render import op_box
    by_id = {op["id"]: op for op in P["ops"]}
    mk = marked_image(draft, M)
    H, W = draft.shape[:2]
    cols = [(230, 30, 30), (30, 110, 230), (20, 160, 60), (200, 120, 0)]
    order = sorted(errs, key=lambda e: e["type"] != "unassigned")   # ô chưa quyết trước: mất chữ là lỗi nặng nhất
    parts, ids, n_img = [], [], 0
    for k, e in enumerate(order):
        boxes = [(i, op_box(by_id[i], M)) for i in e.get("ops") or [] if i in by_id]
        boxes += [(s, M["by"][s]["box"]) for s in e.get("slots") or [] if s in M["by"]]
        boxes = [(i, b) for i, b in boxes if b]
        parts.append(f"ERROR {k + 1}: {json.dumps(e, ensure_ascii=False)}")
        ids += [i for i in e.get("ops") or [] if i in by_id]
        if not boxes or n_img >= REPAIR_MAX:
            continue
        x0, y0 = min(b[0] for _, b in boxes), min(b[1] for _, b in boxes)
        x1, y1 = max(b[2] for _, b in boxes), max(b[3] for _, b in boxes)
        pad = 0.5 * max(y1 - y0, 0.05 * min(W, H))
        cb = (int(max(0, x0 - pad)), int(max(0, y0 - pad)), int(min(W, x1 + pad)), int(min(H, y1 + pad)))
        pr = Image.fromarray(proof).convert("RGB")
        d = ImageDraw.Draw(pr)
        for j, (i, b) in enumerate(boxes):
            d.rectangle(b, outline=cols[j % len(cols)], width=3)
            d.text((b[0] + 3, b[1] + 2), i, fill=cols[j % len(cols)])
        z = 1024 / max(cb[2] - cb[0], cb[3] - cb[1])
        sz = (max(1, int((cb[2] - cb[0]) * z)), max(1, int((cb[3] - cb[1]) * z)))
        parts += ["PROOF crop:", _png(pr.crop(cb).resize(sz, Image.LANCZOS)), "MARKED draft crop:", _png(mk.crop(cb).resize(sz, Image.LANCZOS))]
        n_img += 1
    ops = [by_id[i] for i in dict.fromkeys(ids)]
    user = (f"PROMPT given to the image model:\n{prompt}\n\nClient texts (exact words):\n{_texts(texts)}\n\n"
            + (PRODUCT_NOTE + "\n\n" if product else "") + f"{describe(M)}\n\nInvolved ops:\n{json.dumps(ops, ensure_ascii=False)}")
    out = call(*((_system(SYSTEM) + "\n\n" + REPAIR, [user, *parts]) + ((meta,) if meta is not None else ())))
    out = _obj(out)
    new, patches = [], {p.get("id"): p for p in out.get("patches") or [] if isinstance(p, dict) and p.get("id")}
    for op in P["ops"]:
        p = patches.pop(op["id"], None)
        if p is None:
            new.append(op)
        elif not p.get("delete"):
            q = {**op, **{k: v for k, v in p.items() if k not in ("id", "delete")}}
            if "slots" in p:
                q["marks"] = list(p["slots"] or [])
            new.append({k: v for k, v in q.items() if v is not None or k == "client"})
    for pid, p in patches.items():   # lệnh MỚI (ô chưa quyết)
        if p.get("slots") and not p.get("delete"):
            new.append({**{k: v for k, v in p.items() if k != "delete"}, "marks": list(p["slots"])})
    return _to_marks({**P, "ops": new, "repair_why": out.get("why")})


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
