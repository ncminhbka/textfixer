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
CONDENSE = 0.85   # chữ dài hơn ô theo chiều ngang: nén ngang (scaleX) tới 0.85 TRƯỚC khi co cỡ -- chữ giữ chiều cao như nháp
ROT_MIN = 2.0   # độ: dòng model vẽ nghiêng >= 2 độ thì dựng nghiêng đúng góc đó (dưới 2 độ là nhiễu đo của OCR)
PARA_WORDS = 10   # đoạn văn: >= 10 chữ trên >= 3 dòng nháp -> tự ngắt dòng
LH_ACCENT = 1.15   # line-height tối thiểu khi dòng sau có chữ hoa mang dấu trên (dấu không đâm vào dòng trên)
EMOJI = re.compile("[\U0001F000-\U0001FAFF️‍]")   # emoji (không có font emoji trong bộ font)
ACCENT_UP = {"̀", "́", "̃", "̉", "̂", "̆"}   # huyền sắc ngã hỏi, mũ, trăng (dấu nằm TRÊN chữ)


def _accent_below(h: str) -> bool:
    """HTML nhiều dòng (<br> / khối) mà một dòng SAU dòng đầu có chữ HOA mang dấu trên."""
    import unicodedata
    parts = re.split(r"<\s*br\b[^>]*>|<\s*/?(?:div|p|li)\b[^>]*>", h, flags=re.I)
    lines = [re.sub(r"<[^>]+>", "", p).strip() for p in parts]
    lines = [x for x in lines if x]
    return any(ch.isupper() and any(c in ACCENT_UP for c in unicodedata.normalize("NFD", ch)[1:])
               for x in lines[1:] for ch in x)
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
    theo góc đó (ôm sát dòng nghiêng), div dựng thẳng rồi xoay. -> {rect (chưa xoay), angle, poly (4 góc thật), aabb}.
    Lệnh có fix.box (autofix cắt / thu khung): rect = fix.box (hệ trục của dòng), góc giữ như đo."""
    g = _geom(op, M)
    fb = (op.get("fix") or {}).get("box")
    if g is None or not fb:
        return g
    t = np.radians(g["angle"])
    c, sn = np.cos(t), np.sin(t)
    cx, cy, w, h = (fb[0] + fb[2]) / 2, (fb[1] + fb[3]) / 2, fb[2] - fb[0], fb[3] - fb[1]
    corners = [[cx + u * c - v * sn, cy + u * sn + v * c] for u, v in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))]
    xs, ys = [p[0] for p in corners], [p[1] for p in corners]
    W, H = M["size"]
    return {"rect": list(fb), "angle": g["angle"], "poly": corners,
            "aabb": [max(0.0, min(xs)), max(0.0, min(ys)), min(float(W), max(xs)), min(float(H), max(ys))]}


def _geom(op: dict, M: dict) -> dict | None:
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


def _clip_text_shadow(h: str) -> str:
    """Chữ tô gradient (background-clip:text, màu chữ trong suốt): text-shadow vẽ DƯỚI lớp tô và lộ qua chữ trong suốt -> chữ
    vàng thành nâu (08/10 h08_s1). Đổi bóng sang filter:drop-shadow (bóng của cả khối chữ đã tô), viền -webkit-text-stroke
    vẽ sau phần tô (paint-order) để viền chỉ ở mép ngoài."""
    def fix(m):
        st = m.group(2)
        if not re.search(r"background-clip\s*:\s*text", st, flags=re.I):
            return m.group(0)
        sh = re.search(r"(?<![-\w])text-shadow\s*:\s*([^;]+);?", st, flags=re.I)
        if sh:
            parts = [x.strip() for x in re.split(r",(?![^(]*\))", sh.group(1)) if x.strip() and x.strip() != "none"]
            st = st.replace(sh.group(0), "") + (";filter:" + " ".join(f"drop-shadow({x})" for x in parts) if parts else "")
        if "text-stroke" in st:
            st += ";paint-order:stroke fill"
        return f"style={m.group(1)}{st}{m.group(1)}"
    return re.sub(r"""style=(["'])(.*?)\1""", fix, h, flags=re.S)


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
    h = _clip_text_shadow(h)
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
    if (e.cls) d.className = e.cls;
    // CỘT FLEX trong thẻ font: VLM viết <i-font><span style="flex:1">tên</span><span style="flex:0 0 30%">giá</span></i-font> --
    // span bọc font (inline) là con flex duy nhất, tên và giá dính liền "Bún bò tái40.000đ" (09/10 dev2 d05) -> bọc font
    // display:contents, các span flex thành con flex thật của khung (vẫn thừa hưởng font)
    for (const x of d.querySelectorAll('[style*="flex"]')) {
      // (con trực tiếp của khung flex bị "blockify": display tính ra là block, không phải inline -- xét theo thẻ span)
      for (let p = x.parentElement; p && p !== d && p.tagName === 'SPAN' && !/flex|grid/.test(getComputedStyle(p).display);
           p = p.parentElement)
        p.style.display = 'contents';
    }
    // dòng nghiêng: renderer tự xoay đúng góc đo (bên dưới) -- transform VLM ghi ở khung (rotate(-14deg)) bỏ trước khi đo,
    // không thì "đo trước khi xoay" là đo khối đã xoay (tràn / nét sai)
    if (e.angle) d.style.transform = 'none';
    c.appendChild(d);
    const R0 = c.getBoundingClientRect();
    // TRÀN = nội dung thật (mọi con, kể cả SVG) vượt khung ở BẤT KỲ phía nào (nội dung căn giữa tràn cả trái -- scrollWidth chỉ
    // đếm phía phải: p20 hàng sao thò ra ngoài huy hiệu bên trái)
    // KHUNG Ô ôm NÉT MỰC (cả dấu thanh), còn hộp chữ của trình duyệt = ascent + descent của font (đệm trên / dưới nét): so
    // thẳng thì mọi dòng "tràn dọc" và bị co ~0.6-0.8 (07/10: 80% lỗi too_small, chữ nhỏ / nhạt hơn nháp). Trừ phần đệm đó --
    // đo bằng measureText trên chính chữ của khối, theo font của PHẦN CHỮ CHIẾM NHIỀU NHẤT (i-font / span bên trong có font, cỡ
    // riêng -- đo theo font của khung thì sai: 08/10 h02 khung Plus Jakarta, chữ Baloo dấu cao -> "tràn dọc" oan, co còn 0.6),
    // tính theo đơn vị cỡ của khung (tỉ lệ theo cỡ, đo một lần ở 100px).
    const pad = (() => {
      const t = d.textContent.trim();
      if (!t) return {top: 0, bot: 0};
      let el = d, best = 0;
      const w = document.createTreeWalker(d, NodeFilter.SHOW_TEXT);
      for (let n = w.nextNode(); n; n = w.nextNode()) {
        const k = n.textContent.trim().length;
        if (k > best) { best = k; el = n.parentElement; }
      }
      const cs = getComputedStyle(el), cx = document.createElement('canvas').getContext('2d');
      const r = parseFloat(cs.fontSize) / (parseFloat(getComputedStyle(d).fontSize) || 1);
      cx.font = `${cs.fontStyle} ${cs.fontWeight} 100px ${cs.fontFamily}`;
      const mt = cx.measureText(t);
      return {top: r * Math.max(0, mt.fontBoundingBoxAscent - mt.actualBoundingBoxAscent) / 100,
              bot: r * Math.max(0, mt.fontBoundingBoxDescent - mt.actualBoundingBoxDescent) / 100};
    })();
    let inner = d;   // phần đo nội dung (sau khi nén ngang: bên trong span nén -- hộp span cao theo line-height, đo nó là tràn dọc oan)
    const over = () => {
      const rr = document.createRange(); rr.selectNodeContents(inner);
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
    let s = e.size, k = 0, m = over(), sx = 1;
    const m0q = m.q, m0b = d.getBoundingClientRect(), fs0 = parseFloat(d.style.fontSize) || 0;   // tràn lúc CHƯA co: ngang / dọc
    const o0 = [Math.max(0, m0b.left - m0q.left, m0q.right - m0b.right),
                Math.max(0, m0b.top - m0q.top - pad.top * fs0, m0q.bottom - m0b.bottom - pad.bot * fs0)];
    // NÉN NGANG trước khi co: chữ dài hơn ô (tràn ngang) -> scaleX tới o.condense, giữ cỡ chữ (chiều cao) như nháp -- co cỡ
    // làm chữ bé hẳn so với nháp (08/10 server: too_small 33 / 27 ảnh). Chỉ khối chữ dòng (không div / p / li bên trong: bọc
    // lại sẽ đổi bố cục flex của VLM); tâm nén theo căn lề của khung.
    if (((e.fit !== 'grow' && e.fit !== 'none') || e.even) && o0[0] > 1 && !d.querySelector('div,p,ul,ol,li,table,section,[style*="flex"]')) {
      sx = Math.max(e.condense || o.condense, Math.min(1, m0b.width / Math.max(1, m0q.width)));
      const j = getComputedStyle(d).justifyContent;
      const org = /start|left/.test(j) ? 'left' : /end|right/.test(j) ? 'right' : 'center';
      d.innerHTML = `<span style="display:inline-block;flex:none;white-space:inherit;transform:scaleX(${sx});` +
                    `transform-origin:${org} center">${d.innerHTML}</span>`;
      inner = d.firstElementChild;
      m = over();
    }
    if (e.fit !== 'grow' && e.fit !== 'none') {   // co nội dung cho vừa khung (em theo font-size của khung)
      while (m.v > 1 && s > o.floor && k < 80) { s *= 0.95; d.style.fontSize = s + 'px'; k++; m = over(); }
    }
    const m0 = m.v;   // tràn đo khi CHƯA xoay (sau khi xoay, khung thẳng ôm ngoài lệch so với nội dung)
    let ink0 = null;   // chữ xoay: nét mực CHƯA xoay + tâm xoay -> đa giác nét thật (hộp thẳng ôm chữ xoay chồng oan dòng kề)
    if (e.angle) {
      const q0 = m.q, b0 = d.getBoundingClientRect(), f0 = parseFloat(d.style.fontSize) || 0;
      ink0 = {r: [q0.left - R0.left, q0.top - R0.top + Math.min(pad.top * f0, q0.height / 3),
                  q0.right - R0.left, q0.bottom - R0.top - Math.min(pad.bot * f0, q0.height / 3)],
              c: [(b0.left + b0.right) / 2 - R0.left, (b0.top + b0.bottom) / 2 - R0.top]};
    }
    if (e.angle) { d.style.transformOrigin = '50% 50%'; d.style.transform = `rotate(${e.angle}deg)`; m = over(); m.v = m0; }
    const q = m.q, b = d.getBoundingClientRect();
    // NÉT MỰC = hộp chữ trừ phần đệm trên / dưới của font (dòng chồng sát nhau như FLASH / SALE thì hộp font chồng 0.2-0.3 dù
    // nét không chạm: 08/10 server 15/27 ảnh báo content_overlap oan, vòng sửa chạy vì nó). Chữ xoay: giữ hộp ngoài.
    const fsn = parseFloat(d.style.fontSize) || 0;
    const it = e.angle ? 0 : Math.min(pad.top * fsn, (q.bottom - q.top) / 3), ib = e.angle ? 0 : Math.min(pad.bot * fsn, (q.bottom - q.top) / 3);
    // MÀU CHỮ thật (màu tính được của phần tử chứa nhiều chữ nhất) + có hiệu ứng tách nền không (bóng / viền / chữ tô gradient)
    let fg = null, fx = false, best = 0, fs = null, ff = null, fw = null;
    const runs = [];   // từng đoạn chữ (màu / hiệu ứng riêng): khối hai màu -- "Giảm" trắng + "40%" cam trên nền đỏ (09/10 dev2 d11)
    const tw = document.createTreeWalker(d, NodeFilter.SHOW_TEXT);
    for (let t = tw.nextNode(); t; t = tw.nextNode()) {
      const n = t.textContent.trim().length, cs = getComputedStyle(t.parentElement);
      const tfx = cs.textShadow !== 'none' || parseFloat(cs.webkitTextStrokeWidth) > 0 || cs.backgroundClip === 'text' ||
          cs.webkitBackgroundClip === 'text';
      if (tfx) fx = true;
      if (n > best) { best = n; fg = cs.webkitTextFillColor || cs.color; fs = parseFloat(cs.fontSize); ff = cs.fontFamily; fw = parseInt(cs.fontWeight); }
      if (n >= 2) {
        const rg = document.createRange(); rg.selectNodeContents(t);
        const rc = rg.getBoundingClientRect();
        runs.push({n, fg: cs.webkitTextFillColor || cs.color, fx: tfx,
                   ink: [rc.left - R0.left, rc.top - R0.top, rc.right - R0.left, rc.bottom - R0.top]});
      }
    }
    out.push({id: e.id, size: s, shrink: e.size ? s / e.size : 1, over: m.v, o0, fg, fx, fs, ff, fw, sx: +sx.toFixed(2), ink0, angle: e.angle, runs,
              ink: [q.left - R0.left, q.top - R0.top + it, q.right - R0.left, q.bottom - R0.top - ib],
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
    els, log, force = [], [], []
    # LỚP: vỏ (shape / group) vẽ trước, chi tiết / chữ vẽ sau -- vỏ không bao giờ đè lên nội dung đặt trong nó
    for op in sorted(P["ops"], key=lambda o: o.get("kind") not in ("shape", "group")):
        if op.get("kind") not in DRAW or not (op.get("html") or op.get("box_style")):
            continue
        g = op_geom(op, M)
        if g is None:
            continue
        fix = op.get("fix") or {}
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
        # chữ là TOÀN BỘ nội dung một vỏ (pill / nút một dòng): căn giữa dọc theo vỏ -- khung dòng nắn theo nét (dấu thanh, mép
        # sáng của vỏ) lệch lên, chữ căn giữa khung dòng trông "nảy lên" khỏi tâm vỏ (08/10 m03)
        lm = {m for m in mk if m.startswith("L")}
        sh = next((s for s in M.get("S") or [] if lm and set(s.get("lines") or []) == lm and not s.get("icons")), None)
        if sh and op.get("kind") == "text" and not g["angle"]:
            cy, hh = (sh["box"][1] + sh["box"][3]) / 2, min(b[3] - b[1], 0.9 * (sh["box"][3] - sh["box"][1]))
            b = [b[0], cy - hh / 2, b[2], cy + hh / 2]
        # font-size ghi ở box_style = CỠ GỐC của khung (để nguyên px thì nó đè cỡ khung: vòng co nhảy cỡ, cân cỡ cùng vai hỏng)
        bs_raw = op.get("box_style") or ""
        fsz = re.search(r"(?<![-\w])font-size\s*:\s*(\d+(?:\.\d+)?)px\s*;?", bs_raw)
        if fsz and not (mk and all(m.startswith("I") for m in mk)):
            size, bs_raw = float(fsz.group(1)), bs_raw.replace(fsz.group(0), "")
        # VLM ghi cỡ bằng px poster (đọc thẳng từ số đo dòng / khung); đổi sang em theo cỡ khung để vòng co vẫn co đều mọi thứ
        html = re.sub(r"(\d+(?:\.\d+)?)px\b", lambda m: f"{float(m.group(1)) / max(size, 1.0):.4f}em", op.get("html") or "")
        h, lg = expand(html, icons, used)
        log += [f"{op['id']}: {x}" for x in lg]
        bs, lg = expand(bs_raw, icons, used)
        log += [f"{op['id']}: {x}" for x in lg]
        # EMOJI: không font nào trong bộ font có hình emoji -> ô vuông (09/10 dev2 d10 "MINIGAME 🎁"); bỏ, ghi log
        h2 = EMOJI.sub("", h)
        if h2 != h:
            log.append(f"{op['id']}: bỏ emoji (không có font emoji)")
            h = h2
        # DẤU CHỒNG DÒNG TRÊN: nhiều dòng, dòng sau có chữ HOA mang dấu trên (Á, Ỗ, Ế...) mà line-height < LH_ACCENT -> dấu đâm
        # vào chân dòng trên (09/10 dev2 d06: "DROP MỚI<br>THÁNG 11" line-height 1.0, dấu sắc của Á thành đuôi chữ O -> "DRQP")
        if _accent_below(h):
            h, bs = (re.sub(r"(line-height\s*:\s*)(\d*\.?\d+)(?=\s*[;\"']|\s*$)",
                            lambda m: m.group(1) + (str(LH_ACCENT) if float(m.group(2)) < LH_ACCENT else m.group(2)), x) for x in (h, bs))
        # ĐOẠN VĂN (>= 3 dòng nháp, >= PARA_WORDS chữ): bỏ <br> chép theo dòng nháp, trình duyệt tự ngắt theo bề ngang khung --
        # dòng nháp và dòng chữ thật dài khác nhau, giữ <br> thì dòng dài tự gãy thêm, một chữ mồ côi mỗi dòng ("lưng", "thể":
        # 09/10 dev2 d17), khối phải co nhỏ cho vừa
        if op.get("kind") == "text" and len(lm) >= 3 and len(re.sub(r"<[^>]+>", " ", h).split()) >= PARA_WORDS \
                and re.search(r"<br", h, flags=re.I) and not re.search(r"<(div|p|li|ul|ol)\b", h, flags=re.I):
            h = re.sub(r"\s*<\s*br\b[^>]*>\s*", " ", h, flags=re.I)
            bs = "text-wrap:pretty;" + bs
            log.append(f"{op['id']}: đoạn văn -- bỏ <br> theo dòng nháp, tự ngắt dòng")
        if re.search(r"text-align\s*:\s*(left|start)", bs) and "justify-content" not in bs:   # căn trái mà khối vẫn ở giữa ô
            bs += ";justify-content:flex-start"
        elif re.search(r"text-align\s*:\s*(right|end)", bs) and "justify-content" not in bs:
            bs += ";justify-content:flex-end"
        bs = (f"font-weight:{wt};" if wt else "") + bs
        # ô MỘT dòng nháp, không <br>: không tự xuống dòng -- trình duyệt bẻ dòng khi chữ dài, chiều cao vượt ô, vòng co thu nhỏ
        # tới khi 2 dòng li ti lọt ô (08/10: 59 / 78 lệnh bị co là "tràn dọc" kiểu này); một dòng co theo chiều ngang to hơn
        if op.get("kind") == "text" and len(lm) == 1 and not re.search(r"<br", op.get("html") or "", flags=re.I):
            bs = "white-space:nowrap;" + bs
        cls = None
        if fix.get("color"):   # SỬA BẰNG CODE (fix): chữ chìm nền -> màu tương phản (+ quầng) cưỡng cho mọi phần tử trong khối
            cls = f"fx{len(els)}"
            force.append(f"#c>.{cls},#c>.{cls} *{{color:{fix['color']}!important;-webkit-text-fill-color:{fix['color']}!important;"
                         + (f"text-shadow:0 0 .08em {fix['halo']},0 0 .16em {fix['halo']},0 0 .3em {fix['halo']}!important;"
                            if fix.get("halo") else "") + "}")
        els.append({"id": op["id"], "box": b, "html": h, "style": bs, "size": size, "color": color, "font": family(fam),
                    "fit": fix.get("fit") or op.get("fit") or "shrink", "angle": g["angle"], "cls": cls,
                    "condense": fix.get("condense"), "icon": op.get("kind") == "icon" and bool(mk)
                    and all(m.startswith("I") for m in mk)})
    log += _align_left(els) + _even_icons(els)
    page.set_viewport_size({"width": W, "height": H})
    html = (f"<html><head><style>{faces_for(tuple(sorted(used)))} *{{margin:0;padding:0;box-sizing:border-box}} #c{{position:relative;"
            f"width:{W}px;height:{H}px;overflow:hidden;background:url('{to_data_uri(plate)}') 0 0/{W}px {H}px no-repeat}}"
            f"{''.join(force)}</style></head><body><div id=c></div></body></html>")
    page.set_content(html)
    res = page.evaluate(RENDER_JS, {"els": els, "floor": floor or 0.012 * min(W, H), "condense": CONDENSE})
    # CÙNG VAI CÙNG CỠ: mỗi dòng tự co cho vừa ô của nó -> các mục một danh sách lệch cỡ (08/10 r01: 20 / 24 / 24 / 19px). Dựng
    # lại cả nhóm ở cỡ nhỏ nhất của nhóm (nhỏ hơn thì chắc chắn vừa ô, không co thêm)
    size0, fs = {e["id"]: e["size"] for e in els}, {r["id"]: r.get("fs") for r in res}
    by_el, by_res, changed = {e["id"]: e for e in els}, {r["id"]: r for r in res}, []
    for g in _role_groups(P, M, res):
        # dòng nhỏ BẤT THƯỜNG (< 0.65 trung vị: chữ quá dài cho ô) không kéo cả nhóm -- để nó lại, too_small báo vòng sửa rút gọn;
        # 0.8 thì danh sách vẫn lệch (08/10 r01_s1: 33 / 33 / 33 / 23 px) -- danh sách ĐỀU quan trọng hơn to
        med = float(np.median([fs[i] for i in g]))
        g = [i for i in g if fs[i] >= 0.65 * med]
        if len(g) < 2:
            continue
        lo = min(fs[i] for i in g)
        for i in g:
            if fs[i] > lo * 1.03:
                # cỡ chung, không co thêm; vẫn được nén ngang (dòng nhỏ nhất nhóm có thể đã nén mới giữ được cỡ đó)
                by_el[i]["size"], by_el[i]["fit"], by_el[i]["even"] = by_res[i]["size"] * lo / fs[i], "none", True
                changed.append(i)
    if changed:
        log.append(f"cùng vai cùng cỡ: dựng lại {changed}")
        page.set_content(html)
        res = page.evaluate(RENDER_JS, {"els": els, "floor": floor or 0.012 * min(W, H), "condense": CONDENSE})
        for r in res:
            r["shrink"] = r["size"] / size0[r["id"]] if size0.get(r["id"]) else 1
            r["evened"] = r["id"] in changed   # nhỏ đi vì theo dòng nhỏ nhất nhóm: lỗi too_small chỉ tính ở dòng đó
    for r in res:
        if r["shrink"] < 0.7:
            log.append(f"{r['id']}: co còn {r['shrink']:.2f} để vừa khung")
    img = np.asarray(Image.open(io.BytesIO(page.locator("#c").screenshot())).convert("RGB"))
    # NỀN THẬT sau chữ (bản xoá + vỏ + nền khung lệnh, chữ trong suốt) -> tỉ lệ tương phản chữ / nền (check: low_contrast)
    page.evaluate("""() => { for (const x of document.querySelectorAll('#c *')) { const s = x.style;
        s.setProperty('color', 'transparent', 'important'); s.setProperty('-webkit-text-fill-color', 'transparent', 'important');
        s.setProperty('text-shadow', 'none', 'important'); s.setProperty('-webkit-text-stroke-width', '0', 'important'); } }""")
    bg = np.asarray(Image.open(io.BytesIO(page.locator("#c").screenshot())).convert("RGB"))
    color0 = {e["id"]: e["color"] for e in els}
    bad = lambda r: (r["low"] or 0) > (0.6 if r.get("px_based") else CONTRAST_FRAC)   # noqa: E731
    for r in res:
        r["low"] = _low_contrast(r, bg, img)
        if len(r.get("runs") or []) > 1 and not bad(r):   # khối nhiều màu: một đoạn chìm là lỗi dù cả khối trung bình đọc được
            for u in r["runs"]:
                u["low"] = _low_contrast(u, bg, img)
                if bad(u):
                    r["low"], r["px_based"] = u["low"], u.get("px_based", False)
                    break
    pal = list(dict.fromkeys(r["fg"] for r in res if r.get("fg") and not bad(r)))   # màu chữ đọc tốt trên poster
    for r in res:
        if bad(r):
            r["recolor"] = _recolor(r, bg, color0.get(r["id"]), pal)
    return img, res, log


CONTRAST_PX = 2.0   # điểm nền mà màu chữ trên đó có tỉ lệ tương phản WCAG < 2 = chữ chìm (chữ thân cần >= 4.5, chữ lớn >= 3)
CONTRAST_FRAC = 0.25   # > 25% nền sau khối chữ là điểm chìm = lỗi low_contrast (khối chữ không có bóng / viền)


def _lum(rgb: np.ndarray) -> np.ndarray:
    c = rgb.astype(np.float32) / 255.0
    c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return c @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def _low_contrast(r: dict, bg: np.ndarray, img: np.ndarray | None = None) -> float | None:
    """Phần nền (trong khung nét chữ của lệnh) mà màu chữ của lệnh chìm trên đó. Chữ có bóng / viền / tô gradient (màu chữ
    không đo được một màu, 08/10 h08_s1: tiêu đề vàng nhạt chìm hẳn vào nền hồng nhạt mà không bị bắt): đo trên ĐIỂM ẢNH THẬT
    -- điểm khác nền (bản dựng so với nền không chữ) là nét chữ; phần nét chữ có tương phản < 1.5 với nền ngay dưới nó."""
    if r.get("empty"):
        return None
    H, W = bg.shape[:2]
    x0, y0, x1, y1 = r["ink"]
    x0, y0, x1, y1 = int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    m = re.match(r"rgba?\(([\d.]+),\s*([\d.]+),\s*([\d.]+)(?:,\s*([\d.]+))?", r.get("fg") or "")
    if r.get("fx") or not m or (m.group(4) is not None and float(m.group(4)) < 0.5):
        if img is None:
            return None
        a, b = img[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32), bg[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
        ink = np.abs(a - b).max(1) > 30
        if ink.sum() < 20:
            return None
        la, lb = _lum(a[ink]), _lum(b[ink])
        ratio = (np.maximum(la, lb) + 0.05) / (np.minimum(la, lb) + 0.05)
        r["px_based"] = True
        return round(float((ratio < 1.5).mean()), 2)
    lf = float(_lum(np.array([[float(m.group(i)) for i in (1, 2, 3)]])[0]))
    lb = _lum(bg[y0:y1, x0:x1].reshape(-1, 3))
    ratio = (np.maximum(lf, lb) + 0.05) / (np.minimum(lf, lb) + 0.05)
    return round(float((ratio < CONTRAST_PX).mean()), 2)


def _rgb(c: str | None) -> tuple | None:
    if not c:
        return None
    m = re.match(r"#([0-9a-f]{6})$", c.strip(), flags=re.I)
    if m:
        return tuple(int(m.group(1)[i:i + 2], 16) for i in (0, 2, 4))
    m = re.match(r"rgba?\(([\d.]+),\s*([\d.]+),\s*([\d.]+)", c)
    return tuple(float(m.group(i)) for i in (1, 2, 3)) if m else None


def _recolor(r: dict, bg: np.ndarray, draft_color: str | None, pal: list | None = None) -> dict | None:
    """Chữ chìm nền: màu chữ thay thế, ưu tiên HỢP BẢNG MÀU (08/10 replay: tiêu đề vàng nhạt -> gần đen, đọc được nhưng thô) --
    màu các chữ đọc tốt khác trên poster, rồi màu nháp tối / sáng dần, cuối cùng trắng / gần đen: màu ĐẦU TIÊN có <= 10% nền
    chìm (tương phản WCAG < CONTRAST_PX). Không màu nào đạt: màu ít chìm nhất + quầng màu ngược độ sáng."""
    H, W = bg.shape[:2]
    x0, y0, x1, y1 = r["ink"]
    x0, y0, x1, y1 = int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    lb = _lum(bg[y0:y1, x0:x1].reshape(-1, 3))
    cands = list(pal or [])   # màu chữ khác của chính poster trước: hợp bảng màu (màu nháp pha đen ra xám đục -- 08/10 h08_s1)
    d = _rgb(draft_color) or _rgb(r.get("fg"))
    if d:
        for t in (0.35, 0.6, 0.8):
            for to in (0, 255):
                cands.append("#%02x%02x%02x" % tuple(int(round(v + (to - v) * t)) for v in d))
    cands += ["#ffffff", "#161616"]
    best = None
    for c in dict.fromkeys(cands):
        rgb = _rgb(c)
        if rgb is None:
            continue
        lf = float(_lum(np.array([rgb], np.float32))[0])
        f = float(((np.maximum(lf, lb) + 0.05) / (np.minimum(lf, lb) + 0.05) < CONTRAST_PX).mean())
        if f <= 0.1:
            best = (c, f, lf)
            break
        if best is None or f < best[1] - 0.02:
            best = (c, f, lf)
    c, f, lf = best
    return {"color": c, "frac": round(f, 2), "halo": ("rgba(0,0,0,.55)" if lf > 0.4 else "rgba(255,255,255,.6)")
            if f > CONTRAST_FRAC else None}


def _even_icons(els: list[dict]) -> list[str]:
    """Icon cùng hàng / cùng cột (đầu dòng một danh sách, hàng icon liên hệ): khung đo từng icon trên nháp to nhỏ khác nhau ->
    dựng ra icon lệch cỡ (08/10 r05, r01). Icon cỡ gần nhau (<= 1.6 lần), thẳng tâm cột / hàng, cách nhau <= 3 cỡ = một nhóm:
    mọi icon = ô vuông cạnh trung vị, giữ tâm của nó."""
    I = [e for e in els if e.get("icon") and not e["angle"]]
    side = lambda e: min(e["box"][2] - e["box"][0], e["box"][3] - e["box"][1])   # noqa: E731
    cen = lambda e: ((e["box"][0] + e["box"][2]) / 2, (e["box"][1] + e["box"][3]) / 2)   # noqa: E731
    par = list(range(len(I)))

    def root(i):
        while par[i] != i:
            i = par[i]
        return i
    for i, a in enumerate(I):
        for j in range(i + 1, len(I)):
            b = I[j]
            s = max(side(a), side(b))
            if s <= 0 or s / max(1.0, min(side(a), side(b))) > 1.6:
                continue
            (ax, ay), (bx, by) = cen(a), cen(b)
            # cột đầu dòng của danh sách thưa: tâm cách nhau tới 6 cỡ icon (08/10 r01_s1: 4 tick, chỉ 2 được nối)
            if (abs(ax - bx) <= 0.4 * s and abs(ay - by) <= 6 * s) or (abs(ay - by) <= 0.4 * s and abs(ax - bx) <= 8 * s):
                par[root(i)] = root(j)
    groups, log = {}, []
    for i, e in enumerate(I):
        groups.setdefault(root(i), []).append(e)
    for g in groups.values():
        if len(g) < 2:
            continue
        m = float(np.median([side(e) for e in g]))
        for e in g:
            cx, cy = cen(e)
            e["box"] = [cx - m / 2, cy - m / 2, cx + m / 2, cy + m / 2]
            e["size"] = m
        log.append(f"icon cùng cỡ {[e['id'] for e in g]}: {m:.0f}px")
    return log


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


def _ink_poly(r: dict) -> list:
    """Đa giác nét mực của kết quả dựng: chữ xoay = nét chưa xoay xoay quanh tâm khung; chữ thẳng = khung nét."""
    i0 = r.get("ink0")
    if not i0 or not r.get("angle"):
        x0, y0, x1, y1 = r["ink"]
        return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    t = np.radians(r["angle"])
    c, sn = np.cos(t), np.sin(t)
    (x0, y0, x1, y1), (cx, cy) = i0["r"], i0["c"]
    return [[cx + (x - cx) * c - (y - cy) * sn, cy + (x - cx) * sn + (y - cy) * c] for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]


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
            o = _poly_ov(_ink_poly(a), _ink_poly(b)) if a.get("angle") or b.get("angle") else _ov(a["ink"], b["ink"])
            if (a["id"], b["id"]) not in pairs and o > OVERLAP:
                errs.append({"type": "content_overlap", "ops": [a["id"], b["id"]], "frac": round(o, 2)})
        if a["over"] > 1 and boxes.get(a["id"]):
            errs.append({"type": "overflow", "ops": [a["id"]], "px": round(a["over"])})
        elif shrink_min and a.get("shrink", 1) < shrink_min and not a.get("evened"):   # nhồi quá nhiều chữ so với chỗ model vẽ: chữ bé đi -> xấu
            errs.append({"type": "too_small", "ops": [a["id"]], "shrink": round(a["shrink"], 2)})
        if (a.get("low") or 0) > (0.6 if a.get("px_based") else CONTRAST_FRAC):   # chữ chìm vào nền thật sau nó
            errs.append({"type": "low_contrast", "ops": [a["id"]], "frac": a["low"], "color": a.get("fg")})
    errs += _uneven(P, M, res or [])
    return errs


FIX_ROUNDS = 2   # sửa bằng code: sửa -> dựng -> kiểm, tối đa 2 lượt (lượt 2 sửa lỗi lượt 1 làm lộ ra)


def _trim(A: list, B: list) -> list | None:
    """Khung A cắt bớt MỘT phía để không còn giao B -- phía giữ được nhiều diện tích A nhất; None khi cắt mất quá nửa."""
    w, h = A[2] - A[0], A[3] - A[1]
    c = [[A[0], A[1], min(A[2], B[0]), A[3]], [max(A[0], B[2]), A[1], A[2], A[3]],
         [A[0], A[1], A[2], min(A[3], B[1])], [A[0], max(A[1], B[3]), A[2], A[3]]]
    c = [x for x in c if x[2] - x[0] > 2 and x[3] - x[1] > 2]
    if not c or w <= 0 or h <= 0:
        return None
    best = max(c, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
    return best if (best[2] - best[0]) * (best[3] - best[1]) >= 0.5 * w * h else None


def autofix(P: dict, M: dict, errs: list[dict], res: list[dict]) -> tuple[dict, list[str]]:
    """SỬA LỖI NẶNG BẰNG CODE (thay vòng duyệt VLM, 08/10): quyết định thiết kế của VLM giữ nguyên, chỉ sửa phần đo được.
      mark_twice      -- ô giữ ở lệnh đầu, bỏ khỏi lệnh sau
      unassigned      -- ô bỏ sót = skip (bản xoá đã sạch; không tự viết chữ không ai quyết)
      low_contrast    -- màu chữ tương phản nhất với nền thật ({màu nháp, trắng, gần đen}), nền loang thêm quầng
      overflow        -- nén ngang sâu hơn (0.75)
      boxes_overlap / content_overlap -- lệnh lấn (nét thò khỏi khung nhiều hơn, hoặc khung to hơn) cắt khung một phía cho
                         hết giao, co nội dung trong khung mới. Lệnh xoay / vỏ (shape, group) không cắt.
    -> (P mới, log). Không lỗi nào sửa được: P y nguyên."""
    import copy
    P2 = copy.deepcopy(P)
    ops = {o["id"]: o for o in P2["ops"]}
    R = {r["id"]: r for r in res}
    log = []
    fx = lambda i: ops[i].setdefault("fix", {})   # noqa: E731
    for e in errs:
        t, ids = e["type"], [i for i in e.get("ops") or [] if i in ops]
        if t == "mark_twice" and len(ids) == 2:
            o = ops[ids[1]]
            o["marks"] = [m for m in o.get("marks") or [] if m != e["mark"]]
            log.append(f"{ids[1]}: bỏ ô {e['mark']} (đã thuộc {ids[0]})")
        elif t == "unassigned" and e.get("slots"):
            P2["ops"].append({"id": f"fix_skip{len(P2['ops'])}", "kind": "skip", "marks": list(e["slots"]), "html": "",
                              "why": "autofix: ô bỏ sót"})
            log.append(f"ô bỏ sót {e['slots']} -> skip")
        elif t == "low_contrast" and ids and (R.get(ids[0]) or {}).get("recolor"):
            rc = R[ids[0]]["recolor"]
            fx(ids[0]).update(color=rc["color"], halo=rc["halo"])
            log.append(f"{ids[0]}: chìm nền {e['frac']} -> màu {rc['color']}" + (" + quầng" if rc["halo"] else ""))
        elif t == "overflow" and ids:
            fx(ids[0])["condense"] = 0.75
            log.append(f"{ids[0]}: tràn {e['px']}px -> nén ngang tới 0.75")
        elif t in ("boxes_overlap", "content_overlap") and len(ids) == 2:
            G = {i: op_geom(ops[i], M) for i in ids}
            if any(G[i] is None for i in ids):
                continue
            box = {i: G[i]["aabb"] for i in ids}

            def spill(i):   # phần nét chữ thò ra ngoài khung của chính lệnh
                r = R.get(i)
                if not r:
                    return 0.0
                a, b = r["ink"], box[i]
                area = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
                return 1 - _ov(a, b) * min(area, (b[2] - b[0]) * (b[3] - b[1])) / area
            cand = [i for i in ids if ops[i].get("kind") not in ("shape", "group")]
            if not cand:
                continue
            area = lambda i: (box[i][2] - box[i][0]) * (box[i][3] - box[i][1])   # noqa: E731
            # lệnh thẳng cắt được một phía; lệnh xoay chỉ thu đều quanh tâm (khung thật nghiêng) -> ưu tiên lệnh thẳng
            i = max(cand, key=lambda i: (not G[i]["angle"], round(spill(i), 2), area(i)))
            j = ids[1] if i == ids[0] else ids[0]
            if G[i]["angle"]:
                r0 = G[i]["rect"]
                cx, cy, w, h = (r0[0] + r0[2]) / 2, (r0[1] + r0[3]) / 2, 0.85 * (r0[2] - r0[0]), 0.85 * (r0[3] - r0[1])
                fx(i).update(box=[round(v, 1) for v in (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)], fit="shrink")
                log.append(f"{i}: chồng {j} ({t} {e.get('frac')}), chữ xoay -> thu khung 0.85 quanh tâm")
                continue
            other = R[j]["ink"] if t == "content_overlap" and R.get(j) else box[j]
            nb = _trim(box[i], other)
            if nb is None:
                log.append(f"{i}: chồng {j} nhưng cắt khung mất quá nửa -> để nguyên")
                continue
            fx(i).update(box=[round(v, 1) for v in nb], fit="shrink")
            log.append(f"{i}: chồng {j} ({t} {e.get('frac')}) -> cắt khung còn {[round(v) for v in nb]}")
    P2["ops"] = [o for o in P2["ops"] if o.get("kind") == "skip" or o.get("marks")]
    return P2, log


UNEVEN = 1.25   # cùng cụm (model vẽ cùng cỡ, thẳng cột / cùng hàng, gần nhau) mà cỡ dựng lệch > 25% = lỗi uneven_size


def _uneven(P: dict, M: dict, res: list) -> list[dict]:
    """Nhóm cùng vai (_role_groups) vẫn lệch cỡ > 25% / độ đậm >= 200 sau khi render đã cân cỡ."""
    fs = {r["id"]: r["fs"] for r in res if r.get("fs")}
    fw = {r["id"]: r.get("fw") or 400 for r in res}
    out = []
    for g in _role_groups(P, M, res):
        sz, wt = [fs[i] for i in g], [fw[i] for i in g]
        # cỡ lệch > 25%, hoặc độ đậm lệch >= 200 (08/10 r01: cùng danh sách mục đậm mục nhạt)
        if max(sz) / max(1.0, min(sz)) > UNEVEN or max(wt) - min(wt) >= 200:
            out.append({"type": "uneven_size", "ops": g, "sizes": {i: round(fs[i]) for i in g}, "weights": {i: fw[i] for i in g}})
    return out


def _role_groups(P: dict, M: dict, res: list, same_font: bool = True) -> list[list[str]]:
    """Các dòng cùng vai: model vẽ cùng cỡ +-15%, thẳng MÉP TRÁI hoặc cùng hàng, cách nhau <= 3 cỡ (+ cùng font khi same_font:
    cỡ px chỉ so được trong cùng font) -> nhóm id lệnh (>= 2)."""
    fs = {r["id"]: r["fs"] for r in res if r.get("fs")}
    ff = {r["id"]: r.get("ff") for r in res}   # khác font thì cỡ px không so được (chữ viết tay cần cỡ khác chữ in)
    T = []
    for op in P["ops"]:
        Ls = [M["by"][m] for m in op.get("marks") or [] if m.startswith("L") and m in M["by"]]
        if op.get("kind") != "text" or not Ls or op["id"] not in fs:
            continue
        b = [min(m["box"][0] for m in Ls), min(m["box"][1] for m in Ls), max(m["box"][2] for m in Ls), max(m["box"][3] for m in Ls)]
        T.append((op["id"], float(np.median([m["size"] for m in Ls])), b))
    # DÒNG CÓ ĐẦU DÒNG (icon / tick ngay bên trái): nháp hay vẽ các mục một danh sách to nhỏ khác nhau (08/10 r01_s1: 34-38 so với
    # 30-32 px) -> cỡ nháp không đủ để nhận cùng vai; hai dòng cùng có đầu dòng và thẳng mép trái là một danh sách dù lệch tới 40%
    bul = {c["near"].split()[-1] for c in M.get("I") or [] if str(c.get("near") or "").startswith("left of")}
    has_bul = {op["id"] for op in P["ops"] if any(m in bul for m in op.get("marks") or [])}
    lim = lambda a, b: 1.4 if a[0] in has_bul and b[0] in has_bul else 1.15   # noqa: E731
    par = {t[0]: t[0] for t in T}

    def root(x):
        while par[x] != x:
            x = par[x]
        return x
    # CỘT trước (thẳng mép trái), rồi HÀNG chỉ giữa các dòng không thuộc cột nào: nối qua hàng thì cột tên món + cột giá thành
    # một nhóm (08/10 h02: cả bảng co theo dòng dài nhất)
    pairs = [(a, b) for i, a in enumerate(T) for b in T[i + 1:]
             if max(a[1], b[1]) / max(1.0, min(a[1], b[1])) <= lim(a, b) and not (same_font and ff[a[0]] != ff[b[0]])]
    gap = lambda ba, bb: max(ba[1] - bb[3], bb[1] - ba[3], 0)   # noqa: E731
    in_col = set()
    for (ia, sa, ba), (ib, sb, bb) in pairs:   # thẳng tâm thì không: tiêu đề / nút / chân trang xếp giữa là các vai khác nhau
        far = 4 if ia in has_bul and ib in has_bul else 3   # danh sách thưa (mục cách xa nhau) vẫn là một danh sách
        if abs(ba[0] - bb[0]) <= 0.6 * max(sa, sb) and gap(ba, bb) <= far * max(sa, sb):
            par[root(ia)] = root(ib)
            in_col |= {ia, ib}
    # NHÓM DO VLM KHAI ("group" của lệnh: cùng vai theo nghĩa -- mục danh sách, tên món, giá...): gộp thẳng, không cần đoán
    # (cùng font thì mới so được cỡ px)
    grp = {}
    for op in P["ops"]:
        if op.get("group") and op["id"] in par:
            grp.setdefault(str(op["group"]).strip().lower(), []).append(op["id"])
    for ids in grp.values():
        for b in ids[1:]:
            if not (same_font and ff[ids[0]] != ff[b]):
                par[root(ids[0])] = root(b)
    for (ia, sa, ba), (ib, sb, bb) in pairs:
        if ia not in in_col and ib not in in_col and abs((ba[1] + ba[3]) - (bb[1] + bb[3])) / 2 <= 0.3 * max(sa, sb) and \
                abs(ba[0] - bb[0]) <= 8 * max(sa, sb):
            par[root(ia)] = root(ib)
    groups = {}
    for t in T:
        groups.setdefault(root(t[0]), []).append(t[0])
    return [g for g in groups.values() if len(g) >= 2]


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
