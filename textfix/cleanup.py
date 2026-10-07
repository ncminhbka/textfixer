"""DỌN CHỮ SÓT trên bản xoá (07/10): bước xoá FLUX ngẫu nhiên, ~8% dòng chữ còn nguyên (chữ trong pill / badge màu, logo, dòng
liên hệ, hàng sao). OCR bản xoá; dòng nào TRÙNG VỊ TRÍ một dòng của nháp = chữ chưa xoá -> xoá: chỉ các điểm NÉT (màu có
trong dòng, hiếm ở vành quanh dòng), inpaint từ nền quanh nét. Chỗ này designer sẽ vẽ chữ mới đè lên nên chỉ cần sạch.

  plate2, info = clean(draft, plate)            # info: {"lines": [chữ OCR đã xoá], "px": số điểm đã lấp}

Chạy TRƯỚC khi tách ô: chỗ vừa dọn trở thành lớp phủ (nháp - bản đã dọn) -> thành ô như mọi chữ khác (hàng sao "★★★★" OCR
đọc được -> thành chi tiết I, designer vẽ lại <i-stars>). Chữ trên sản phẩm còn sót cũng bị dọn ở đây, nhưng designer chọn
"keep" thì bước dựng dán lại pixel nháp. Chấp nhận (không xử lý): mẩu dấu lẻ OCR không đọc ra dòng, icon méo còn trong bản xoá.
"""

from __future__ import annotations

import numpy as np

MATCH = 0.5        # dòng bản xoá trùng dòng nháp: phần giao >= 50% khung một trong hai
MIN_SCORE = 0.3    # độ tin OCR tối thiểu (chữ giun đọc ra độ tin thấp vẫn là chữ)
RATIO = 0.35       # màu NÉT: tỉ lệ điểm gần màu đó ở vành SÁT dòng < 0.35 lần tỉ lệ trong lõi (nền / vỏ chiếm vành) ...
CORE_MIN = 0.05    # ... và chiếm >= 5% lõi
TOL = 22.0         # điểm thuộc màu nét: cách tâm màu <= 22 ΔE
HALO = 0.06        # nới mặt nạ nét 0.06 cao dòng (>= 2 px): viền khử răng cưa, bóng sát nét
TEXTURE = 10.0     # nền quanh nét: lệch trung vị tới màu trung vị (ΔE) > 10 = nền có vân / ảnh -> FLUX xoá lại vùng cắt (inpaint
                   # nhoè thành mảng, m03 07/10); <= 10 = phẳng / chuyển màu đều (pill, thẻ, badge: <= 9 trên 36 cặp) -> inpaint
CROP_MIN = 384     # vùng cắt cho FLUX: cạnh >= 384 px ảnh gốc (đủ ngữ cảnh), phóng để cạnh dài 768


def _cov(a, b) -> float:
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h) / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))


def leftovers(draft_lines: list[dict], plate_lines: list[dict]) -> list[dict]:
    return [l for l in plate_lines if l.get("score", 1) >= MIN_SCORE
            and any(_cov(l["box"], d["box"]) >= MATCH or _cov(d["box"], l["box"]) >= MATCH for d in draft_lines)]


def _lab(x):
    import cv2
    a = cv2.cvtColor(x, cv2.COLOR_RGB2LAB).astype(np.float32)
    a[..., 0] *= 100 / 255
    a[..., 1:] -= 128
    return a


def ink_mask(img: np.ndarray, lab: np.ndarray, poly) -> tuple[np.ndarray, tuple[int, int], float]:
    """Mặt nạ NÉT của một dòng (cửa sổ quanh dòng) -> (mask, (X0, Y0), cao dòng). Màu nét = cụm màu trong lõi (đa giác OCR)
    hiếm ở vành quanh dòng; nét = điểm gần màu nét, thuộc mảng chạm lõi. Không lấy cả khung dòng: khung nới tràn ra ngoài
    vỏ nhỏ (huy hiệu sao nổ, h02 07/10) làm hỏng mép vỏ."""
    import cv2
    from .textmask import _line_h
    H, W = img.shape[:2]
    h = _line_h(poly)
    P = np.asarray(poly, np.float32)
    r = int(0.9 * h) + 2
    X0, Y0 = int(max(0, P[:, 0].min() - r)), int(max(0, P[:, 1].min() - r))
    X1, Y1 = int(min(W, P[:, 0].max() + r)), int(min(H, P[:, 1].max() + r))
    L = lab[Y0:Y1, X0:X1]
    core = np.zeros(L.shape[:2], np.uint8)
    cv2.fillPoly(core, [np.round(P - [X0, Y0]).astype(np.int32)], 1)
    disk = lambda k: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max(1, int(k)) + 1,) * 2)   # noqa: E731
    near = cv2.dilate(core, disk(0.35 * h)).astype(bool)          # lõi + dấu thanh / nét bay sát dòng
    # vành SÁT dòng (0.12-0.4 cao dòng): chữ trong pill / huy hiệu nhỏ -> vành vẫn nằm trong vỏ (vành rộng chạm nền trang cùng
    # màu chữ: chữ trắng trong pill xanh trên trang trắng, h06 07/10)
    ring = cv2.dilate(core, disk(0.4 * h)).astype(bool) & ~cv2.dilate(core, disk(0.12 * h)).astype(bool)
    cm = core.astype(bool)
    empty = (np.zeros(L.shape[:2], bool), (X0, Y0), h)
    if cm.sum() < 20 or ring.sum() < 20:
        return empty
    px = L[cm].astype(np.float32)
    k = min(4, len(px))
    cv2.setRNGSeed(0)
    _, _, ctr = cv2.kmeans(px, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5), 2, cv2.KMEANS_PP_CENTERS)
    R = L[ring]
    frac = lambda X, c: float((np.linalg.norm(X - c, axis=1) <= TOL).mean())   # noqa: E731
    ink_c = [c for c in ctr if frac(px, c) >= CORE_MIN and frac(R, c) < RATIO * frac(px, c)]
    if not ink_c:   # không tách được màu nét: inpaint khung dòng nới nhẹ (ít tràn hơn khung nới rộng)
        return cv2.dilate(core, disk(0.1 * h)).astype(bool), (X0, Y0), h
    ink = np.zeros(L.shape[:2], bool)
    for c in ink_c:
        ink |= np.linalg.norm(L - c, axis=2) <= TOL
    ink &= near
    n, cc, _, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    touch = np.unique(cc[cm & (cc > 0)])
    ink = np.isin(cc, touch[touch > 0])
    ink = cv2.dilate(ink.astype(np.uint8), disk(max(2, HALO * h))).astype(bool)
    return ink, (X0, Y0), h


def _texture(lab: np.ndarray, m: np.ndarray, x0: int, y0: int, h: float) -> float:
    """Độ vân của nền quanh nét: lệch trung vị (ΔE) tới màu trung vị ở vành 2 px .. 0.3 cao dòng quanh mặt nạ nét."""
    import cv2
    mu = m.astype(np.uint8)
    k = max(3, int(0.3 * h))
    ring = cv2.dilate(mu, np.ones((2 * k + 1,) * 2, np.uint8)).astype(bool) & ~cv2.dilate(mu, np.ones((5, 5), np.uint8)).astype(bool)
    L = lab[y0:y0 + m.shape[0], x0:x0 + m.shape[1]][ring]
    return float(np.median(np.linalg.norm(L - np.median(L, 0), axis=1))) if len(L) else 0.0


def _reerase(out: np.ndarray, full: np.ndarray, h: float, erase) -> None:
    """FLUX xoá lại VÙNG CẮT quanh mặt nạ full (toạ độ ảnh), dán về đúng trong mặt nạ nới 0.15 cao dòng, mép làm mờ."""
    import cv2
    H, W = full.shape
    ys, xs = np.nonzero(full)
    pad = max(int(0.8 * h), 32)
    b = [xs.min() - pad, ys.min() - pad, xs.max() + pad + 1, ys.max() + pad + 1]
    for lo, hi, lim in ((0, 2, W), (1, 3, H)):   # nới tới CROP_MIN quanh tâm, dời vào trong ảnh
        need = max(0, min(CROP_MIN, lim) - (b[hi] - b[lo]))
        b[lo] -= need // 2
        b[hi] += need - need // 2
        sh = max(0, -b[lo]) - max(0, b[hi] - lim)
        b[lo], b[hi] = max(0, b[lo] + sh), min(lim, b[hi] + sh)
    x0, y0, x1, y1 = (int(v) for v in b)
    crop = out[y0:y1, x0:x1]
    z = 768 / max(crop.shape[:2])
    cw, ch = max(16, int(crop.shape[1] * z) // 16 * 16), max(16, int(crop.shape[0] * z) // 16 * 16)
    er = erase(cv2.resize(crop, (cw, ch), interpolation=cv2.INTER_CUBIC if z > 1 else cv2.INTER_AREA))
    er = cv2.resize(np.asarray(er), (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_AREA if z > 1 else cv2.INTER_CUBIC)
    k = max(2, int(0.15 * h))
    a = cv2.dilate(full[y0:y1, x0:x1].astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1,) * 2))
    a = cv2.GaussianBlur(a.astype(np.float32), (0, 0), max(1.0, k / 2))[..., None]
    out[y0:y1, x0:x1] = (er * a + crop * (1 - a)).round().clip(0, 255).astype(np.uint8)


def clean(draft: np.ndarray, plate: np.ndarray, draft_lines: list[dict] | None = None, erase=None) -> tuple[np.ndarray, dict]:
    """-> (bản xoá đã dọn, {"lines", "inpaint", "reerase", "px"}). erase(ảnh) -> ảnh: FLUX xoá (lời dặn A) cho vùng cắt nền
    có vân; None (không có FLUX): mọi dòng inpaint."""
    import cv2
    from .ocr import read_lines
    dl = draft_lines if draft_lines is not None else read_lines(draft)
    left = leftovers(dl, read_lines(plate))
    info = {"lines": [l["text"] for l in left], "inpaint": [], "reerase": [], "px": 0}
    if not left:
        return plate, info
    out = plate.copy()
    lab = _lab(plate)
    H, W = plate.shape[:2]
    rough = []   # dòng nền có vân -> gom thành vùng cắt cho FLUX
    for l in left:   # từng dòng (nền mỗi dòng một kiểu)
        m, (x0, y0), h = ink_mask(plate, lab, l["poly"])
        if not m.any():
            continue
        info["px"] += int(m.sum())
        if erase is not None and _texture(lab, m, x0, y0, h) > TEXTURE:
            full = np.zeros((H, W), bool)
            full[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = m
            rough.append((full, h, l["text"]))
            continue
        sub = np.ascontiguousarray(out[y0:y0 + m.shape[0], x0:x0 + m.shape[1]])
        fixed = cv2.inpaint(sub, m.astype(np.uint8), max(3, int(0.12 * h)), cv2.INPAINT_TELEA)
        sub[m] = fixed[m]
        out[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = sub
        info["inpaint"].append(l["text"])
    while rough:   # gom dòng nền vân gần nhau (nới 1 cao dòng chạm nhau) vào MỘT vùng cắt: một lần gọi FLUX
        full, h, texts = rough.pop(0)
        grew = True
        while grew:
            grew = False
            k = int(h)
            zone = cv2.dilate(full.astype(np.uint8), np.ones((2 * k + 1,) * 2, np.uint8)).astype(bool)
            for r in list(rough):
                if (zone & r[0]).any():
                    full, h, texts = full | r[0], max(h, r[1]), f"{texts} | {r[2]}"
                    rough.remove(r)
                    grew = True
        try:
            _reerase(out, full, h, erase)
            info["reerase"].append(texts)
        except Exception as e:   # FLUX lỗi: inpaint
            out = cv2.inpaint(out, full.astype(np.uint8), max(3, int(0.12 * h)), cv2.INPAINT_TELEA)
            info["inpaint"].append(f"{texts} (FLUX lỗi: {type(e).__name__})")
    return out, info
