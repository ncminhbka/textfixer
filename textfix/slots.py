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
    raw = read_lines(img)
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
                  "size": float(h / (CAP_EM if caps else X_EM)), "color": m["color"] if m else None, "angle": ang})
    return L


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

SLOTS (marked on the second image: blue S#, red L#, orange I#; then 4 zoomed quarters of it):
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
"skip" for junk or duplicates, or kind "keep" for a detail that belongs to the picture (a logo or graphic printed on a product, \
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


def describe(M: dict) -> str:
    sh = "\n".join(f"{s['id']}: shell {_px(s['box'])}, fill {s['fill']}, radius ~{s['radius']:.0f}px, contains "
                   f"{', '.join(s['lines'] + s['icons']) or 'nothing'}" for s in M["S"]) or "(none)"
    de = "\n".join(f"{c['id']}: detail {_px(c['box'])}, color {c['color']}" + (f", inside {c['shell']}" if c.get("shell") else "")
                   + (f", {c['near']}" if c.get("near") else "") for c in M["I"]) or "(none)"
    tilt = lambda m: f", drawn at {m['angle']:.0f} degrees" if abs(m.get("angle") or 0) >= 2 else ""   # noqa: E731
    li = "\n".join(f"{m['id']}: \"{m['ocr']}\" (text ~{m['size']:.0f}px, ~{len(m['ocr'])} characters, color {m['color']}"
                   + (f", inside {m['shell']}" if m.get("shell") else "") + f"{tilt(m)})" for m in M["L"]) or "(none)"
    return f"Shells:\n{sh}\n\nDetails:\n{de}\n\nText lines:\n{li}"


def _system(base: str) -> str:
    from .fonts import CATALOG
    from .icons import available
    fonts = "; ".join(f"{k}: {v[0]} ({v[1]})" for k, v in CATALOG.items())
    return base.replace("{fonts}", fonts).replace("{icons}", ", ".join(sorted(available())))


def _to_marks(P: dict) -> dict:
    """slots -> marks (render dùng marks)."""
    return {**P, "ops": [{**op, "marks": list(op.get("slots") or op.get("marks") or [])} for op in P.get("ops", [])]}


def _texts(texts: list[dict]) -> str:
    return "\n".join(f"T{i} ({t.get('role', '')}): {t['text']}" for i, t in enumerate(texts))


PRODUCT_NOTE = ("The client uploaded a photo of their product and the image model copied it into the draft. Text, logos and "
                "graphics printed on that product (label, package, screen) are part of the photo: give every slot on them kind "
                "\"keep\", never redraw or skip them.")


def plan(draft: np.ndarray, M: dict, prompt: str, texts: list[dict], call, product: bool = False) -> dict:
    from PIL import Image
    mk = marked_image(draft, M)
    W, H = mk.size
    quarters = []
    for (a, b) in ((0, 0), (1, 0), (0, 1), (1, 1)):   # 4 góc chồng 10%, phóng 2x: đọc được ô nhỏ
        x0, y0 = int(a * W * 0.45), int(b * H * 0.45)
        q = mk.crop((x0, y0, x0 + int(W * 0.55), y0 + int(H * 0.55)))
        quarters.append(_png(q.resize((q.width * 2, q.height * 2), Image.LANCZOS)))
    user = (f"PROMPT given to the image model:\n{prompt}\n\nClient texts (exact words):\n{_texts(texts)}\n\nPoster size: {W}x{H}px.\n\n"
            + (PRODUCT_NOTE + "\n\n" if product else "") + f"{describe(M)}")
    return _to_marks(call(_system(SYSTEM), [user, "DRAFT:", _png(draft), "SLOTS (marked):", _png(mk), "ZOOM (4 quarters, marked):",
                                            *quarters]))


def validate(P: dict, M: dict) -> tuple[dict, list[str]]:
    """Phòng ngừa nhẹ, có ghi log: id ô không có -> bỏ id; lệnh không còn ô nào -> bỏ lệnh."""
    log, ops = [], []
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
           product: bool = False) -> dict:
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
    out = call(_system(SYSTEM) + "\n\n" + REPAIR, [user, *parts])
    new, patches = [], {p.get("id"): p for p in out.get("patches", []) if p.get("id")}
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
