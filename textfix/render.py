"""DỰNG lệnh của designer lên nền (bản xoá) trong Chromium, đúng khung từng ô; báo lỗi khách quan cho vòng sửa.

  poster, res, log = render(page, base, P, M)   # mỗi lệnh = một div tại khung các ô của nó, HTML của VLM + thẻ tiện ích, co vừa khung
  errs = check(P, M, res, shrink_min=0.7)       # ô chưa quyết, khung / nội dung chồng nhau, tràn, phải co quá nhỏ

Khung lệnh = hợp khung các ô nó trỏ (lệnh chữ: chỉ các dòng L). Dòng model vẽ nghiêng: div dựng thẳng rồi xoay đúng góc.
Lớp: vỏ (shape) trước, rồi chi tiết / chữ. Code không quyết thiết kế; phòng ngừa duy nhất: HTML an toàn (bỏ script / on* / URL / img).
"""

from __future__ import annotations

import base64
import io
import re

import numpy as np

DRAW = ("text", "icon", "stars", "shape", "group")
OVERLAP = 0.1   # chồng nhau > 10% diện tích phần tử nhỏ hơn = lỗi báo cho vòng sửa
ROT_MIN = 2.0   # độ: dòng model vẽ nghiêng >= 2 độ thì dựng nghiêng đúng góc đó (dưới 2 độ là nhiễu đo của OCR)
STAR_PATH = "M50 3 L61.8 37.6 L98.1 38.2 L69.1 60.1 L79.4 95 L50 74.2 L20.6 95 L30.9 60.1 L1.9 38.2 L38.2 37.6 Z"


def to_data_uri(arr: np.ndarray) -> str:
    from PIL import Image
    b = io.BytesIO()
    Image.fromarray(arr).save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def plain(h: str) -> str:
    """Chữ trần của HTML (bỏ thẻ, gộp khoảng trắng)."""
    import html as _html
    return " ".join(_html.unescape(re.sub(r"<[^>]+>", " ", h or "")).split())


# ---------------------------------------------------------------------------------------------------- khung
def _ids(op: dict) -> list[str]:
    ids = op.get("marks") or []
    if op.get("kind") == "text" and any(m.startswith("L") for m in ids):
        ids = [m for m in ids if m.startswith("L")]
    return ids


def op_geom(op: dict, M: dict) -> dict | None:
    """Khung DỰNG của lệnh = hợp khung các ô. Dòng NGHIÊNG (trung vị góc OCR các dòng L >= ROT_MIN): khung trong hệ trục xoay
    theo góc đó (ôm sát dòng nghiêng), div dựng thẳng rồi xoay. -> {rect (chưa xoay), angle, poly (4 góc thật), aabb}."""
    by, (W, H) = M["by"], M["size"]
    ids = [m for m in _ids(op) if m in by]
    if not ids:
        return None
    Ls = [by[m] for m in ids if m.startswith("L")]
    ang = float(np.median([m.get("angle") or 0.0 for m in Ls])) if Ls else 0.0
    if abs(ang) < ROT_MIN:
        bs = [by[m]["box"] for m in ids]
        r = [max(0.0, min(b[0] for b in bs)), max(0.0, min(b[1] for b in bs)), min(float(W), max(b[2] for b in bs)),
             min(float(H), max(b[3] for b in bs))]
        if r[2] - r[0] < 2 or r[3] - r[1] < 2:
            return None
        return {"rect": r, "angle": 0.0, "poly": [[r[0], r[1]], [r[2], r[1]], [r[2], r[3]], [r[0], r[3]]], "aabb": r}
    t = np.radians(ang)
    c, sn = np.cos(t), np.sin(t)
    pts = []
    for m in ids:
        mk = by[m]
        pts += mk["poly"] if "poly" in mk else [[mk["box"][0], mk["box"][1]], [mk["box"][2], mk["box"][1]],
                                               [mk["box"][2], mk["box"][3]], [mk["box"][0], mk["box"][3]]]
    P_ = np.asarray(pts, float)
    u, v = P_[:, 0] * c + P_[:, 1] * sn, -P_[:, 0] * sn + P_[:, 1] * c   # sang hệ trục của dòng
    u0, u1, v0, v1 = u.min(), u.max(), v.min(), v.max()
    uc, vc, w, h = (u0 + u1) / 2, (v0 + v1) / 2, u1 - u0, v1 - v0
    cx, cy = uc * c - vc * sn, uc * sn + vc * c
    corners = [[uu * c - vv * sn, uu * sn + vv * c] for uu, vv in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))]
    xs, ys = [p[0] for p in corners], [p[1] for p in corners]
    return {"rect": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], "angle": ang, "poly": corners,
            "aabb": [max(0.0, min(xs)), max(0.0, min(ys)), min(float(W), max(xs)), min(float(H), max(ys))]}


def op_box(op: dict, M: dict) -> list[float] | None:
    """Khung thẳng ôm khung dựng của lệnh (để kiểm, cắt ảnh lỗi, vẽ debug)."""
    g = op_geom(op, M)
    return g["aabb"] if g else None


# ---------------------------------------------------------------------------------------------------- dựng
def _attr(tag: str, name: str, default=None):
    m = re.search(rf"""{name}\s*=\s*(["'])(.*?)\1""", tag)
    return m.group(2) if m else default


def font_key(k) -> str | None:
    """Khoá font VLM ghi -> khoá CATALOG: đúng khoá, tên họ font ("Dancing Script"), hoặc gần nhất ("bevietsans" -> bevietnam)."""
    from difflib import get_close_matches
    from .fonts import CATALOG
    if not isinstance(k, str) or not k.strip():
        return None
    if k in CATALOG:
        return k
    norm = lambda x: re.sub(r"[^a-z]", "", x.lower())   # noqa: E731
    names = {norm(v[0]): key for key, v in CATALOG.items()} | {norm(key): key for key in CATALOG}
    m = get_close_matches(norm(k), list(names), n=1, cutoff=0.6)
    return names[m[0]] if m else None


WEIGHT = lambda m: (300 if m < 0.07 else 400 if m < 0.11 else 700 if m < 0.16 else 900)   # noqa: E731  nét / cỡ chữ -> độ đậm (như slots._look)
HEAD = 1.4   # dòng cao >= 1.4 cỡ chữ trung vị của poster = tiêu đề -> mặc định style.display_font


def expand(h: str, icons: dict, fonts_used: set) -> tuple[str, list[str]]:
    """HTML của VLM -> HTML dựng được: an toàn (bỏ script / on* / URL / img) + thay thẻ tiện ích i-icon / i-stars / i-font."""
    from .fonts import family
    from .icons import nearest
    log = []
    h = re.sub(r"<script.*?</script>", "", h or "", flags=re.S | re.I)
    h = re.sub(r"\son\w+\s*=\s*([\"']).*?\1", "", h, flags=re.I)
    h = re.sub(r"url\([^)]*\)", "none", h, flags=re.I)
    h = re.sub(r"<img[^>]*>", "", h, flags=re.I)

    def icon(m):
        t = m.group(0)
        name = _attr(t, "name", "")
        n = nearest(name, icons)
        if n is None:
            log.append(f"icon '{name}' không có -> bỏ")
            return ""
        if n != name:
            log.append(f"icon '{name}' -> '{n}'")
        size, col = _attr(t, "size", "1em"), _attr(t, "color", "currentColor")
        return (f'<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" stroke="{col}" stroke-width="2" '
                f'stroke-linecap="round" stroke-linejoin="round" style="display:inline-block;vertical-align:middle;flex:none">{icons[n]}</svg>')

    def stars(m):
        t = m.group(0)
        try:
            n = float(_attr(t, "n", "5"))
        except ValueError:
            n = 5.0
        total = max(1, int(np.ceil(_attr(t, "total", None) and float(_attr(t, "total")) or max(5, np.ceil(n)))))
        size, fill, empty = _attr(t, "size", "1em"), _attr(t, "fill", "#f5b301"), _attr(t, "empty", "#d9d9d9")
        parts = []
        for i in range(total):
            frac = float(np.clip(n - i, 0, 1))
            gid = f"g{abs(hash((t, i))) % 10**8}"
            if 0 < frac < 1:   # nửa sao
                parts.append(f'<defs><linearGradient id="{gid}"><stop offset="{frac}" stop-color="{fill}"/><stop offset="{frac}" '
                             f'stop-color="{empty}"/></linearGradient></defs><path transform="translate({i * 112},0)" '
                             f'fill="url(#{gid})" d="{STAR_PATH}"/>')
            else:
                parts.append(f'<path transform="translate({i * 112},0)" fill="{fill if frac >= 1 else empty}" d="{STAR_PATH}"/>')
        return (f'<svg viewBox="0 0 {total * 112 - 12} 100" height="{size}" style="display:inline-block;vertical-align:middle;'
                f'height:{size};width:calc({size} * {(total * 112 - 12) / 100:.3f});flex:none">{"".join(parts)}</svg>')

    def font(m):
        t = m.group(0)
        raw, wt = _attr(t, "key", ""), _attr(t, "weight", "")
        key = font_key(raw)
        if key is None:
            log.append(f"font '{raw}' không có -> font mặc định")
            return f"<span style=\"{'font-weight:' + wt + ';' if wt else ''}\">"
        if key != raw:
            log.append(f"font '{raw}' -> '{key}'")
        fonts_used.add(key)
        return f"<span style=\"font-family:{family(key)};{'font-weight:' + wt + ';' if wt else ''}\">"

    h = re.sub(r"<i-icon\b[^>]*?(/>|>\s*</i-icon>|>)", icon, h, flags=re.I)
    h = re.sub(r"<i-stars\b[^>]*?(/>|>\s*</i-stars>|>)", stars, h, flags=re.I)
    h = re.sub(r"<i-font\b[^>]*>", font, h, flags=re.I)
    h = re.sub(r"</i-font>", "</span>", h, flags=re.I)
    # font ghi thẳng bằng CSS (font-family:'Dancing Script') cũng phải được nạp (07/10 h08: rơi về font dự phòng)
    for fam in re.findall(r"font-family\s*:\s*([^;\"]+)", h, flags=re.I):
        k = font_key(fam.split(",")[0].strip(" '\""))
        if k:
            fonts_used.add(k)
    return h, log


RENDER_JS = r"""
async (o) => {
  const c = document.getElementById('c');
  // nạp HẾT font khai báo trước khi đo: fonts.ready xong ngay khi chưa có chữ nào dùng font (chưa bắt đầu tải) -> lần dựng đầu
  // của trang đo bằng font dự phòng (cỡ / tràn sai), các lần sau đúng nhờ bộ nhớ đệm
  await Promise.all([...document.fonts].map(f => f.load().catch(() => null)));
  await document.fonts.ready;
  const out = [];
  for (const e of o.els) {
    const d = document.createElement('div');
    d.style.cssText = `position:absolute;left:${e.box[0]}px;top:${e.box[1]}px;box-sizing:border-box;display:flex;align-items:center;` +
      `justify-content:center;text-align:center;line-height:1.15;font-synthesis:style;font-family:${e.font};color:${e.color};font-size:${e.size}px;` +
      (e.fit === 'grow' ? `min-width:${e.box[2] - e.box[0]}px;min-height:${e.box[3] - e.box[1]}px;width:max-content;` :
                          `width:${e.box[2] - e.box[0]}px;height:${e.box[3] - e.box[1]}px;`) + (e.style || '');
    d.innerHTML = e.html;
    c.appendChild(d);
    const R0 = c.getBoundingClientRect();
    // TRÀN = nội dung thật (mọi con, kể cả SVG) vượt khung ở BẤT KỲ phía nào (nội dung căn giữa tràn cả trái -- scrollWidth chỉ
    // đếm phía phải: p20 hàng sao thò ra ngoài huy hiệu bên trái)
    // KHUNG Ô ôm NÉT MỰC (cả dấu thanh), còn hộp chữ của trình duyệt = ascent + descent của font (đệm trên / dưới nét): so
    // thẳng thì mọi dòng "tràn dọc" và bị co ~0.6-0.8 (07/10: 80% lỗi too_small, chữ nhỏ / nhạt hơn nháp). Trừ phần đệm đó --
    // đo bằng measureText trên chính chữ của khối, theo font của khối (tỉ lệ theo cỡ, đo một lần ở 100px).
    const pad = (() => {
      const t = d.textContent.trim();
      if (!t) return {top: 0, bot: 0};
      const cs = getComputedStyle(d), cx = document.createElement('canvas').getContext('2d');
      cx.font = `${cs.fontStyle} ${cs.fontWeight} 100px ${cs.fontFamily}`;
      const mt = cx.measureText(t);
      return {top: Math.max(0, mt.fontBoundingBoxAscent - mt.actualBoundingBoxAscent) / 100,
              bot: Math.max(0, mt.fontBoundingBoxDescent - mt.actualBoundingBoxDescent) / 100};
    })();
    const over = () => {
      const rr = document.createRange(); rr.selectNodeContents(d);
      const q = rr.getBoundingClientRect(), b = d.getBoundingClientRect();
      const fs = parseFloat(d.style.fontSize) || 0;
      return {q, v: Math.max(0, b.left - q.left, q.right - b.right, b.top - q.top - pad.top * fs, q.bottom - b.bottom - pad.bot * fs)};
    };
    // ô KHÔNG có nội dung (vỏ chỉ có CSS): không đo tràn -- khung nội dung rỗng nằm ở (0,0) cho số tràn ảo (07/10 p11: 429 / 916 px)
    if (!d.textContent.trim() && !d.querySelector('svg')) {
      const b0 = d.getBoundingClientRect();
      out.push({id: e.id, size: e.size, shrink: 1, over: 0, empty: true,
                ink: [b0.left - R0.left, b0.top - R0.top, b0.right - R0.left, b0.bottom - R0.top],
                box: [b0.left - R0.left, b0.top - R0.top, b0.right - R0.left, b0.bottom - R0.top]});
      continue;
    }
    let s = e.size, k = 0, m = over();
    if (e.fit !== 'grow' && e.fit !== 'none') {   // co nội dung cho vừa khung (em theo font-size của khung)
      while (m.v > 1 && s > o.floor && k < 80) { s *= 0.95; d.style.fontSize = s + 'px'; k++; m = over(); }
    }
    const m0 = m.v;   // tràn đo khi CHƯA xoay (sau khi xoay, khung thẳng ôm ngoài lệch so với nội dung)
    if (e.angle) { d.style.transformOrigin = '50% 50%'; d.style.transform = `rotate(${e.angle}deg)`; m = over(); m.v = m0; }
    const q = m.q, b = d.getBoundingClientRect();
    // MÀU CHỮ thật (màu tính được của phần tử chứa nhiều chữ nhất) + có hiệu ứng tách nền không (bóng / viền / chữ tô gradient)
    let fg = null, fx = false, best = 0, fs = null, ff = null;
    const tw = document.createTreeWalker(d, NodeFilter.SHOW_TEXT);
    for (let t = tw.nextNode(); t; t = tw.nextNode()) {
      const n = t.textContent.trim().length, cs = getComputedStyle(t.parentElement);
      if (cs.textShadow !== 'none' || parseFloat(cs.webkitTextStrokeWidth) > 0 || cs.backgroundClip === 'text' ||
          cs.webkitBackgroundClip === 'text') fx = true;
      if (n > best) { best = n; fg = cs.webkitTextFillColor || cs.color; fs = parseFloat(cs.fontSize); ff = cs.fontFamily; }
    }
    out.push({id: e.id, size: s, shrink: e.size ? s / e.size : 1, over: m.v, fg, fx, fs, ff,
              ink: [q.left - R0.left, q.top - R0.top, q.right - R0.left, q.bottom - R0.top],
              box: [b.left - R0.left, b.top - R0.top, b.right - R0.left, b.bottom - R0.top]});
  }
  return out;
}
"""


def render(page, plate: np.ndarray, P: dict, M: dict, floor: float | None = None) -> tuple[np.ndarray, list, list]:
    """Dựng lệnh vẽ lên NỀN = bản xoá trong Chromium -> (ảnh, kết quả từng lệnh {id, size, shrink, over, ink, box}, log).
    Mặc định khi VLM không nói: font = style.text_font, cỡ (1em) = cỡ chữ đo được của các dòng L trong lệnh (không có dòng: nửa cao
    khung), màu = màu đo của dòng."""
    from PIL import Image
    from .fonts import faces_for, family
    from .icons import available
    H, W = plate.shape[:2]
    icons = available()
    st = P.get("style") or {}
    tfont = font_key(st.get("text_font")) or "bevietnam"
    dfont = font_key(st.get("display_font")) or tfont
    used = {tfont, dfont}
    med = float(np.median([m["size"] for m in M["L"]])) if M.get("L") else 0.0
    els, log = [], []
    # LỚP: vỏ (shape / group) vẽ trước, chi tiết / chữ vẽ sau -- vỏ không bao giờ đè lên nội dung đặt trong nó
    for op in sorted(P["ops"], key=lambda o: o.get("kind") not in ("shape", "group")):
        if op.get("kind") not in DRAW or not (op.get("html") or op.get("box_style")):
            continue
        g = op_geom(op, M)
        if g is None:
            continue
        b = g["rect"]
        # 1em = cỡ chữ model vẽ ở các dòng của lệnh, hoặc các dòng NẰM TRONG khung lệnh (vỏ: cỡ chữ model vẽ trong vỏ);
        # không có dòng nào: nửa cao khung; ô chỉ gồm chi tiết I#: cao ô (icon / hàng sao không ghi cỡ lấp đúng ô)
        Ls = [M["by"][m] for m in op.get("marks") or [] if m.startswith("L") and m in M["by"]] or \
            [m for m in M["L"] if _ov(m["box"], g["aabb"]) >= 0.6 and (m["box"][2] - m["box"][0]) * (m["box"][3] - m["box"][1]) > 0]
        size = float(np.median([m["size"] for m in Ls])) if Ls else 0.5 * (b[3] - b[1])
        mk = op.get("marks") or []
        if mk and all(m.startswith("I") for m in mk):
            size = b[3] - b[1]
        color = next((m["color"] for m in Ls if m.get("color")), "#161616")
        # MẶC ĐỊNH khi VLM không ghi (i-font / CSS trong lệnh thắng): tiêu đề dùng display_font, độ đậm = độ đậm đo trên nháp
        # (07/10: lệnh không ghi weight dựng ở 400 -> tiêu đề / tên đậm trong nháp thành chữ mảnh)
        fam = dfont if med and Ls and size >= HEAD * med else tfont
        st_w = [m["stroke"] / m["size"] for m in Ls if m.get("stroke") and m.get("size")]
        wt = WEIGHT(float(np.median(st_w))) if st_w else None
        # VLM ghi cỡ bằng px poster (đọc thẳng từ số đo dòng / khung); đổi sang em theo cỡ khung để vòng co vẫn co đều mọi thứ
        html = re.sub(r"(\d+(?:\.\d+)?)px\b", lambda m: f"{float(m.group(1)) / max(size, 1.0):.4f}em", op.get("html") or "")
        h, lg = expand(html, icons, used)
        log += [f"{op['id']}: {x}" for x in lg]
        bs, lg = expand(op.get("box_style") or "", icons, used)
        log += [f"{op['id']}: {x}" for x in lg]
        if re.search(r"text-align\s*:\s*(left|start)", bs) and "justify-content" not in bs:   # căn trái mà khối vẫn ở giữa ô
            bs += ";justify-content:flex-start"
        elif re.search(r"text-align\s*:\s*(right|end)", bs) and "justify-content" not in bs:
            bs += ";justify-content:flex-end"
        bs = (f"font-weight:{wt};" if wt else "") + bs
        els.append({"id": op["id"], "box": b, "html": h, "style": bs, "size": size, "color": color, "font": family(fam),
                    "fit": op.get("fit") or "shrink", "angle": g["angle"]})
    log += _align_left(els)
    page.set_viewport_size({"width": W, "height": H})
    page.set_content(f"<html><head><style>{faces_for(tuple(sorted(used)))} *{{margin:0;padding:0;box-sizing:border-box}} #c{{position:relative;"
                     f"width:{W}px;height:{H}px;overflow:hidden;background:url('{to_data_uri(plate)}') 0 0/{W}px {H}px no-repeat}}"
                     f"</style></head><body><div id=c></div></body></html>")
    res = page.evaluate(RENDER_JS, {"els": els, "floor": floor or 0.012 * min(W, H)})
    for r in res:
        if r["shrink"] < 0.7:
            log.append(f"{r['id']}: co còn {r['shrink']:.2f} để vừa khung")
    img = np.asarray(Image.open(io.BytesIO(page.locator("#c").screenshot())).convert("RGB"))
    # NỀN THẬT sau chữ (bản xoá + vỏ + nền khung lệnh, chữ trong suốt) -> tỉ lệ tương phản chữ / nền (check: low_contrast)
    page.evaluate("""() => { for (const x of document.querySelectorAll('#c *')) { const s = x.style;
        s.setProperty('color', 'transparent', 'important'); s.setProperty('-webkit-text-fill-color', 'transparent', 'important');
        s.setProperty('text-shadow', 'none', 'important'); s.setProperty('-webkit-text-stroke-width', '0', 'important'); } }""")
    bg = np.asarray(Image.open(io.BytesIO(page.locator("#c").screenshot())).convert("RGB"))
    for r in res:
        r["low"] = _low_contrast(r, bg)
    return img, res, log


CONTRAST_PX = 2.0   # điểm nền mà màu chữ trên đó có tỉ lệ tương phản WCAG < 2 = chữ chìm (chữ thân cần >= 4.5, chữ lớn >= 3)
CONTRAST_FRAC = 0.25   # > 25% nền sau khối chữ là điểm chìm = lỗi low_contrast (khối chữ không có bóng / viền)


def _lum(rgb: np.ndarray) -> np.ndarray:
    c = rgb.astype(np.float32) / 255.0
    c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return c @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def _low_contrast(r: dict, bg: np.ndarray) -> float | None:
    """Phần nền (trong khung nét chữ của lệnh) mà màu chữ của lệnh chìm trên đó."""
    m = re.match(r"rgba?\(([\d.]+),\s*([\d.]+),\s*([\d.]+)(?:,\s*([\d.]+))?", r.get("fg") or "")
    if not m or (m.group(4) is not None and float(m.group(4)) < 0.5) or r.get("empty"):
        return None
    H, W = bg.shape[:2]
    x0, y0, x1, y1 = r["ink"]
    x0, y0, x1, y1 = int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    lf = float(_lum(np.array([[float(m.group(i)) for i in (1, 2, 3)]])[0]))
    lb = _lum(bg[y0:y1, x0:x1].reshape(-1, 3))
    ratio = (np.maximum(lf, lb) + 0.05) / (np.minimum(lf, lb) + 0.05)
    return round(float((ratio < CONTRAST_PX).mean()), 2)


def _align_left(els: list[dict]) -> list[str]:
    """Cột căn trái (các mục của một danh sách / thực đơn): khung OCR mỗi dòng thò thụt vài px -> dựng ra mép trái lởm chởm
    (08/10 r01, h02). Dòng căn trái, cùng cỡ (+-25%), mép trái lệch <= 0.6 cỡ, cách nhau <= 3 cỡ theo chiều dọc = một cột:
    mép trái cả cột = mép trái nhỏ nhất (khung chỉ nới sang trái, không đổi ô đo)."""
    L = [e for e in els if not e["angle"] and re.search(r"justify-content\s*:\s*(flex-)?start", e["style"] or "")]
    par = list(range(len(L)))

    def root(i):
        while par[i] != i:
            i = par[i]
        return i
    for i, a in enumerate(L):
        for j in range(i + 1, len(L)):
            b, s = L[j], max(a["size"], L[j]["size"])
            gap = max(a["box"][1] - b["box"][3], b["box"][1] - a["box"][3], 0)
            if max(a["size"], b["size"]) / max(1.0, min(a["size"], b["size"])) <= 1.25 and \
                    abs(a["box"][0] - b["box"][0]) <= 0.6 * s and gap <= 3 * s:
                par[root(i)] = root(j)
    groups, log = {}, []
    for i, e in enumerate(L):
        groups.setdefault(root(i), []).append(e)
    for g in groups.values():
        x = min(e["box"][0] for e in g)
        moved = [e["id"] for e in g if e["box"][0] - x >= 1]
        for e in g:
            e["box"] = [x] + list(e["box"][1:])
        if len(g) >= 2 and moved:
            log.append(f"căn thẳng mép trái {[e['id'] for e in g]} ở x={x:.0f}")
    return log


def _ov(a, b) -> float:
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    s = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return w * h / s if w > 0 and h > 0 and s > 0 else 0.0


def _poly_ov(a, b) -> float:
    """Phần giao hai đa giác lồi (khung có thể nghiêng) / diện tích đa giác nhỏ hơn."""
    import cv2
    pa, pb = np.asarray(a, np.float32), np.asarray(b, np.float32)
    s = min(cv2.contourArea(pa), cv2.contourArea(pb))
    if s <= 0:
        return 0.0
    inter, _ = cv2.intersectConvexConvex(pa, pb)
    return float(inter) / s


def check(P: dict, M: dict, res: list | None = None, shrink_min: float | None = 0.7) -> list[dict]:
    """LỖI KHÁCH QUAN cho vòng sửa (không cần mắt): ô chưa thuộc lệnh nào; ô dùng ở hai lệnh vẽ; khung hai lệnh chồng nhau; (sau
    dựng) nội dung hai lệnh chồng nhau, nội dung tràn khung dù đã co tới sàn, phải co dưới shrink_min (nhồi quá nhiều chữ).
    Lệnh nằm trong khung một vỏ (shape / group) không tính là chồng; vỏ rỗng (chỉ CSS) không đo tràn / chồng nội dung."""
    draw = [op for op in P["ops"] if op.get("kind") in DRAW and (op.get("html") or op.get("box_style"))]
    errs, seen = [], {}
    if M.get("S") is not None:   # mọi ô phải thuộc một lệnh -- ô bỏ sót = chữ / chi tiết biến mất không ai quyết
        used = {m for op in P["ops"] for m in op.get("marks") or []}
        left = [m["id"] for m in M["L"] + M["S"] + M["I"] if m["id"] not in used]
        if left:
            errs.append({"type": "unassigned", "ops": [], "slots": left})
    for op in draw:
        for m in op.get("marks") or []:
            if m in seen and seen[m] != op["id"]:
                errs.append({"type": "mark_twice", "ops": [seen[m], op["id"]], "mark": m})
            seen.setdefault(m, op["id"])
    geoms = {op["id"]: op_geom(op, M) for op in draw}
    boxes = {i: (g["aabb"] if g else None) for i, g in geoms.items()}
    polys = {i: (g["poly"] if g else None) for i, g in geoms.items()}
    kinds = {op["id"]: op.get("kind") for op in draw}
    pairs = set()
    for i, a in enumerate(draw):
        for b in draw[i + 1:]:
            ba, bb = boxes[a["id"]], boxes[b["id"]]
            if not ba or not bb:
                continue
            # phần tử nằm gọn trong một VỎ (shape / group) không phải lỗi: vỏ chứa chữ / icon là bình thường
            o = _poly_ov(polys[a["id"]], polys[b["id"]])
            inside = bool({kinds[a["id"]], kinds[b["id"]]} & {"group", "shape"}) and o >= 0.95
            if o > OVERLAP and not inside:
                errs.append({"type": "boxes_overlap", "ops": [a["id"], b["id"]], "frac": round(o, 2)})
                pairs.add((a["id"], b["id"]))
    for i, a in enumerate(res or []):
        for b in (res or [])[i + 1:]:
            if a.get("empty") or b.get("empty"):
                continue
            if (a["id"], b["id"]) not in pairs and _ov(a["ink"], b["ink"]) > OVERLAP:
                errs.append({"type": "content_overlap", "ops": [a["id"], b["id"]], "frac": round(_ov(a["ink"], b["ink"]), 2)})
        if a["over"] > 1 and boxes.get(a["id"]):
            errs.append({"type": "overflow", "ops": [a["id"]], "px": round(a["over"])})
        elif shrink_min and a.get("shrink", 1) < shrink_min:   # nhồi quá nhiều chữ so với chỗ model vẽ: chữ bé đi -> xấu
            errs.append({"type": "too_small", "ops": [a["id"]], "shrink": round(a["shrink"], 2)})
        if (a.get("low") or 0) > CONTRAST_FRAC and not a.get("fx"):   # chữ chìm vào nền thật sau nó
            errs.append({"type": "low_contrast", "ops": [a["id"]], "frac": a["low"], "color": a.get("fg")})
    errs += _uneven(P, M, res or [])
    return errs


UNEVEN = 1.25   # cùng cụm (model vẽ cùng cỡ, thẳng cột / cùng hàng, gần nhau) mà cỡ dựng lệch > 25% = lỗi uneven_size


def _uneven(P: dict, M: dict, res: list) -> list[dict]:
    """Các dòng cùng vai (model vẽ cùng cỡ +-20%, thẳng mép trái / tâm hoặc cùng hàng, cách nhau <= 4 cỡ) phải dựng cùng cỡ
    (08/10 r01: các mục của một danh sách cỡ lệch nhau ~2 lần vì mỗi lệnh tự co / tự ghi cỡ)."""
    fs = {r["id"]: r["fs"] for r in res if r.get("fs")}
    ff = {r["id"]: r.get("ff") for r in res}   # khác font thì cỡ px không so được (chữ viết tay cần cỡ khác chữ in)
    T = []
    for op in P["ops"]:
        Ls = [M["by"][m] for m in op.get("marks") or [] if m.startswith("L") and m in M["by"]]
        if op.get("kind") != "text" or not Ls or op["id"] not in fs:
            continue
        b = [min(m["box"][0] for m in Ls), min(m["box"][1] for m in Ls), max(m["box"][2] for m in Ls), max(m["box"][3] for m in Ls)]
        T.append((op["id"], float(np.median([m["size"] for m in Ls])), b))
    par = {t[0]: t[0] for t in T}

    def root(x):
        while par[x] != x:
            x = par[x]
        return x
    for i, (ia, sa, ba) in enumerate(T):
        for ib, sb, bb in T[i + 1:]:
            s = max(sa, sb)
            if max(sa, sb) / max(1.0, min(sa, sb)) > 1.2 or ff[ia] != ff[ib]:
                continue
            col = abs(ba[0] - bb[0]) <= 0.6 * s or abs((ba[0] + ba[2]) - (bb[0] + bb[2])) / 2 <= 0.6 * s
            row = abs((ba[1] + ba[3]) - (bb[1] + bb[3])) / 2 <= 0.3 * s
            gap = max(ba[1] - bb[3], bb[1] - ba[3], 0)
            if (col or row) and gap <= 4 * s:
                par[root(ia)] = root(ib)
    groups = {}
    for t in T:
        groups.setdefault(root(t[0]), []).append(t[0])
    out = []
    for g in groups.values():
        sz = [fs[i] for i in g]
        if len(g) >= 2 and max(sz) / max(1.0, min(sz)) > UNEVEN:
            out.append({"type": "uneven_size", "ops": g, "sizes": {i: round(fs[i]) for i in g}})
    return out


KIND_COL = {"text": (230, 30, 30), "icon": (30, 120, 230), "stars": (240, 170, 0), "shape": (160, 40, 200), "group": (160, 40, 200),
            "keep": (0, 160, 80), "skip": (170, 170, 170)}


def ops_image(img, M, P):
    """Nháp mờ + khung từng lệnh (màu theo loại) + nhãn: loại và chữ / icon sẽ vẽ."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray((img * 0.55 + 115).astype(np.uint8)).convert("RGB")
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("arial.ttf", max(11, img.shape[1] // 90))
    except OSError:
        f = ImageFont.load_default()
    for op in P["ops"]:
        b = op_box(op, M)
        if not b:
            continue
        col = KIND_COL.get(op.get("kind"), (0, 0, 0))
        d.rectangle(b, outline=col, width=3)
        lab = f"{op.get('kind')}: {plain(op.get('html', ''))[:40]}"
        d.text((b[0] + 3, b[1] + 2), lab, fill=col, font=f)
    return im
