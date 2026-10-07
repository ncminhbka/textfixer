"""Mặt nạ XOÁ CHỮ mọc theo NÉT MỰC (07/10): đa giác OCR chỉ ôm thân dòng -- nét bay / đuôi chữ nghiêng, viết tay thò ra ngoài
khung, gạch đầu dòng là mẩu mực rời bên trái: xoá theo khung thì sót nét.

  m = text_mask(img, lines)   # lines: ocr.read_lines -> bool HxW

Mỗi dòng: nền = trung vị màu Lab ở vành quanh đa giác; màu mực = <= 3 cụm màu trong đa giác khác nền rõ (nét, viền, bóng);
MỰC = điểm gần một màu mực và khác nền rõ. Lấy:
  1. đa giác OCR nới GROW cao dòng (như cũ);
  2. mọi mảng mực CHẠM đa giác, kể cả phần thò ra ngoài, nếu mảng không quá to (<= BIG cao dòng: to hơn là vật trong ảnh trùng
     màu chữ, không phải nét);
  3. mảng mực NHỎ (<= SMALL cao dòng) CÙNG HÀNG với dòng, cách đầu / cuối <= NEAR cao dòng: gạch đầu dòng, chấm, nét rời
  (q05: nụ hoa cùng màu chữ cùng hàng xa hơn thì không lấy).
  4. mảng mực nhỏ PHÍA TRÊN dòng (dấu thanh / mũ tiếng Việt OCR không ôm).
Mảng chạm mép cửa sổ xét (vật lớn chạy ra ngoài) không lấy.
Mảng lấy được nới thêm HALO cao dòng (viền khử răng cưa, bóng mờ).
"""

from __future__ import annotations

import numpy as np

GROW = 0.3      # nới đa giác OCR (bóng / viền sát nét)
REACH = 1.5     # cửa sổ xét quanh dòng (bội số cao dòng)
INK_FRAC = 0.45  # mực = khác nền >= 45% khoảng cách màu chữ - nền
INK_MIN = 12.0  # ... và >= 12 ΔE (nền có vân nhẹ không thành mực)
INK_TOL = 0.5   # điểm thuộc một màu mực khi cách màu đó <= 50% khoảng cách màu đó - nền ...
INK_TOL_MAX = 18.0  # ... và <= 18 ΔE (q05: hoa kem / khung viền vàng gần màu chữ vàng)
BIG = 1.0       # mảng chạm dòng mà thò ra ngoài đa giác > 1 cao dòng = vật trong ảnh trùng màu chữ, bỏ
SMALL = 0.6     # mảng rời nhỏ (gạch đầu dòng, chấm, nét rời) cao / rộng <= 0.6 cao dòng
NEAR = 0.8      # ... CÙNG HÀNG với dòng (tâm lệch trục dòng <= ROW cao dòng) và cách đầu / cuối dòng <= NEAR cao dòng
ROW = 0.6
ACCENT = 0.9    # dấu thanh / mũ tiếng Việt: mảng nhỏ PHÍA TRÊN dòng, trong bề ngang dòng, cách mép trên <= 0.9 cao dòng
HALO = 0.08     # nới mảng mực


def _line_h(poly) -> float:
    """Cao dòng = cạnh ngắn của đa giác 4 góc (dòng nghiêng: không lấy cao khung thẳng)."""
    p = np.asarray(poly, float)
    return float(max(2.0, min(np.linalg.norm(p[1] - p[0]), np.linalg.norm(p[3] - p[0]))))


def line_mask(img: np.ndarray, lab: np.ndarray, poly) -> tuple[np.ndarray, tuple[int, int]]:
    """Mặt nạ một dòng trong cửa sổ quanh nó -> (mask, (X0, Y0))."""
    import cv2
    H, W = img.shape[:2]
    h = _line_h(poly)
    P = np.asarray(poly, np.float32)
    r = REACH * h
    X0, Y0 = int(max(0, P[:, 0].min() - r)), int(max(0, P[:, 1].min() - r))
    X1, Y1 = int(min(W, P[:, 0].max() + r)), int(min(H, P[:, 1].max() + r))
    L = lab[Y0:Y1, X0:X1]
    core = np.zeros(L.shape[:2], np.uint8)
    cv2.fillPoly(core, [np.round(P - [X0, Y0]).astype(np.int32)], 1)

    def disk(k):
        k = max(1, int(round(k)))
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))

    out = cv2.dilate(core, disk(GROW * h))
    ring = cv2.dilate(core, disk(0.8 * h)).astype(bool) & ~cv2.dilate(core, disk(0.35 * h)).astype(bool)
    if ring.sum() < 20 or core.sum() < 20:
        return out.astype(bool), (X0, Y0)
    bg = np.median(L[ring], axis=0)
    d = np.linalg.norm(L - bg, axis=2)
    cm = core.astype(bool)
    dc = d[cm]
    ink_d = float(np.percentile(dc, 90))   # độ khác nền của nét chữ
    if ink_d < INK_MIN:
        return out.astype(bool), (X0, Y0)
    # MÀU MỰC = các cụm màu trong lõi khác nền rõ (nét; viền / bóng nếu có). Mực = điểm GẦN một màu mực -- không phải "khác
    # nền" (hoa, người, toà nhà chạm dòng cũng khác nền: p20 / q05 07/10 lan sang ảnh)
    px = L[cm & (d >= max(INK_MIN, INK_FRAC * ink_d))].astype(np.float32)
    if len(px) < 10:
        return out.astype(bool), (X0, Y0)
    k = min(3, len(px))
    cv2.setRNGSeed(0)
    _, _, ctr = cv2.kmeans(px, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5), 2, cv2.KMEANS_PP_CENTERS)
    ink = np.zeros(L.shape[:2], bool)
    for c in ctr:
        tol = min(INK_TOL_MAX, INK_TOL * float(np.linalg.norm(c - bg)))
        ink |= np.linalg.norm(L - c, axis=2) <= tol
    ink = (ink & (d >= max(INK_MIN, INK_FRAC * ink_d))).astype(np.uint8)
    n, cc, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    if n <= 1:
        return out.astype(bool), (X0, Y0)
    touch = np.zeros(n, bool)
    touch[np.unique(cc[cm & (cc > 0)])] = True
    # trục dòng (đúng cả dòng nghiêng): u dọc dòng, v vuông góc; toạ độ trong cửa sổ
    Q = P - [X0, Y0]
    u = Q[1] - Q[0]
    u = u / max(1e-6, float(np.linalg.norm(u)))
    v = np.array([-u[1], u[0]])
    mid = Q.mean(0)
    half = float(np.abs((Q - mid) @ u).max())
    ys, xs = np.nonzero(core)
    cx0, cy0, cx1, cy1 = xs.min(), ys.min(), xs.max(), ys.max()
    Hc, Wc = L.shape[:2]
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, w, hh, area = st[i]
        if x == 0 or y == 0 or x + w >= Wc or y + hh >= Hc:   # chạm mép cửa sổ = vật lớn chạy ra ngoài, không phải nét
            continue
        if touch[i]:
            over = max(cx0 - x, x + w - cx1, cy0 - y, y + hh - cy1, 0)
            keep[i] = over <= BIG * h
        elif max(w, hh) <= SMALL * h and area >= 3:
            c = np.array([x + w / 2, y + hh / 2]) - mid
            along, perp = float(c @ u), float(c @ v)
            same_row = abs(perp) <= ROW * h and abs(along) <= half + NEAR * h
            # v hướng XUỐNG khi dòng đọc trái -> phải (ảnh: trục y đi xuống): dấu ở phía trên = perp âm
            accent = -(0.5 + ACCENT) * h <= perp < 0 and abs(along) <= half
            keep[i] = same_row or accent
    grown = cv2.dilate(keep[cc].astype(np.uint8), disk(HALO * h + 1))
    return (out.astype(bool) | grown.astype(bool)), (X0, Y0)


def text_mask(img: np.ndarray, lines: list[dict]) -> np.ndarray:
    import cv2
    H, W = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100 / 255   # L về thang 0-100 (ΔE xấp xỉ)
    lab[..., 1:] -= 128
    m = np.zeros((H, W), bool)
    for l in lines:
        lm, (x0, y0) = line_mask(img, lab, l["poly"])
        m[y0:y0 + lm.shape[0], x0:x0 + lm.shape[1]] |= lm
    return m


def box_mask(img: np.ndarray, lines: list[dict], grow: float = GROW) -> np.ndarray:
    """Mặt nạ cũ (so sánh): đa giác OCR nới grow cao dòng."""
    import cv2
    H, W = img.shape[:2]
    m = np.zeros((H, W), np.uint8)
    for l in lines:
        one = np.zeros((H, W), np.uint8)
        cv2.fillPoly(one, [np.asarray(l["poly"], np.int32)], 1)
        k = max(1, int(grow * _line_h(l["poly"])))
        m |= cv2.dilate(one, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    return m.astype(bool)
