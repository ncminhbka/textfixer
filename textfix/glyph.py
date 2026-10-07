"""Đo nét chữ trong khung một dòng: mặt nạ nét, đường chân, cao chữ, mép, màu, độ dày nét, tương phản nét, hiệu ứng
(bóng đổ / viền), ký hiệu đầu dòng."""

from __future__ import annotations

import numpy as np

# TẠM: cụm thuộc CHỮ khi độ tập trung >= max(ENRICH_ABS, ENRICH_REL x cao nhất). Nền có mật độ lõi ~ dải bên -> ~0.5;
# nét / bóng / viền / răng cưa chỉ ở lõi -> ~1 (q01 01/10: chữ vàng 0.78 vì khung OCR hẹp hơn mực, bóng 1.0, nền 0.34)
ENRICH_ABS, ENRICH_REL = 0.65, 0.7
EFFECT_MIN = 0.15    # TẠM: cụm hiệu ứng phải >= 15% số điểm cụm nét (ít hơn là vụn)
PROFILE_DROP = 0.5   # TẠM: hàng có mật độ mực < 0.5 mật độ thân = trên đỉnh thân chữ
ALT_H = True        # đo cao bằng mặt cắt (True) hay trung vị đỉnh nét (False) -- để so trên bộ đáp án
MARKS = "•·●○◦▪■-–—*✓✔►▶➤★☆"


def _seg_dist(p, a, b) -> float:
    """Khoảng cách điểm p tới đoạn a-b (Lab), chia cho |a-b|."""
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(1e-6, np.dot(ab, ab)), 0, 1))
    return float(np.linalg.norm(p - (a + t * ab)) / max(1e-6, np.linalg.norm(ab)))


SHADOW_MAX = 0.15   # TẠM: bóng đổ lệch tối đa 0.15 cao dòng
SHADOW_ASYM = 1.5   # TẠM: trùng ở độ dịch d phải >= 1.5 lần ở -d mới là bóng LỆCH; không thì viền / vầng sáng đối xứng


def shadow_offset(fill: np.ndarray, eff: np.ndarray, lh: float) -> tuple[float, float]:
    """Độ lệch bóng đổ = độ dịch d (|d| <= SHADOW_MAX cao dòng) làm NÉT dịch đi trùng nhiều nhất với cụm hiệu ứng (tương quan
    chéo FFT). Trùng ở d không hơn hẳn ở -d -> viền / vầng sáng đối xứng -> (0, 0). Bản trước lấy hiệu hai TÂM cụm: vầng sáng /
    viền phân bố lệch cho 55-100px trên chữ cao 60-80px (02/10 q01 '50%', q09) -> in bóng thành chữ 'bóng ma' thứ hai."""
    h, w = fill.shape
    r = max(2, int(SHADOW_MAX * lh))
    H2, W2 = h + 2 * r, w + 2 * r
    F = np.fft.rfft2(fill.astype(np.float32), (H2, W2))
    E = np.fft.rfft2(eff.astype(np.float32), (H2, W2))
    corr = np.fft.irfft2(E * np.conj(F), (H2, W2))   # corr[dy, dx] = trùng của eff với fill dịch (dx, dy) (vòng)
    best, bd = -1.0, (0, 0)
    for ddy in range(-r, r + 1):
        for ddx in range(-r, r + 1):
            v = corr[ddy % H2, ddx % W2]
            if v > best and (ddx or ddy):
                best, bd = v, (ddx, ddy)
    opp = corr[(-bd[1]) % H2, (-bd[0]) % W2]
    if best < SHADOW_ASYM * max(1.0, opp):
        return 0.0, 0.0
    return float(bd[0]), float(bd[1])


def glyph_mask(img: np.ndarray, box, k: int = 3, others=()) -> dict:
    """k-means k cụm màu Lab trong khung dòng (nở 25% chiều cao).
    - Cụm thuộc CHỮ = cụm TẬP TRUNG trong lõi (khung OCR): nền có cả ở lõi lẫn vành nở, nét chữ hầu như chỉ ở lõi. (Bản đầu
      lấy "cụm xa màu viền": nền có vân thì màu viền là màu trộn, chọn trúng nền -- bộ đáp án tổng hợp.)
    - Trong các cụm thuộc chữ: cụm màu PHA nằm trên đoạn nét-nền là viền khử răng cưa; cụm còn lại là HIỆU ỨNG (bóng đổ /
      viền chữ). Nét chính = cụm DÀY hơn (bóng / viền chỉ lộ thành dải mảnh quanh nét). Mặt nạ xoá = mọi cụm thuộc chữ.
    (FLASH SALE trắng bóng đỏ sẫm, 01/10: bản trước lấy bóng làm nét -> in chữ đỏ sẫm, xoá sót nét trắng.)"""
    import cv2
    H, W = img.shape[:2]
    x0, y0, x1, y1 = box
    p = max(2, int(0.25 * (y1 - y0)))
    X0, Y0, X1, Y1 = max(0, int(x0) - p), max(0, int(y0) - p), min(W, int(x1) + p), min(H, int(y1) + p)
    crop = img[Y0:Y1, X0:X1]
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    cv2.setRNGSeed(0)   # k-means khởi tạo ngẫu nhiên -> cố định để số đo lặp lại được (01/10: cùng ảnh, hai lần chạy, mặt nạ khác)
    _, lbl, ctr = cv2.kmeans(lab, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5), 3,
                             cv2.KMEANS_PP_CENTERS)
    lab2 = lab.reshape(crop.shape[0], crop.shape[1], 3)
    lbl = lbl.reshape(crop.shape[:2])
    # mảng NỀN cùng màu chữ: mảng liền chạm mép vùng cắt và to hơn hẳn một con chữ (cao > 1.3 / rộng > 2 cao dòng) -> gán nhãn
    # riêng (k), không tính vào cụm chữ. (q01 01/10: '50%' vàng vắt qua đường chéo đỏ / vàng -- nền vàng cùng cụm với chữ
    # vàng, chữ bị coi là nền.) Con chữ bị khung OCR cắt ngang (OCR đọc thiếu) vẫn giữ vì cỡ con chữ.
    lh_ = max(1.0, y1 - y0)
    for c in range(k):
        n_, cc_, st_, _ = cv2.connectedComponentsWithStats((lbl == c).astype(np.uint8), 8)
        for i in range(1, n_):
            x_, y_, w_, h_ = st_[i, :4]
            edge = x_ == 0 or y_ == 0 or x_ + w_ >= crop.shape[1] or y_ + h_ >= crop.shape[0]
            if edge and (h_ > 1.3 * lh_ or w_ > 2 * lh_):
                lbl[cc_ == i] = k
    ctr = np.vstack([ctr, np.median(lab2[lbl == k], axis=0) if (lbl == k).any() else ctr[:1]])
    core = np.zeros(crop.shape[:2], bool)
    core[int(y0) - Y0:int(np.ceil(y1)) - Y0, int(x0) - X0:int(np.ceil(x1)) - X0] = True
    # độ tập trung = mật độ trong lõi / (mật độ lõi + mật độ VÀNH nở), vành bỏ các khung OCR KHÁC: dòng kề sát (FLASH / SALE)
    # hay mảnh cùng hàng ('50%' cạnh 'GIẢM ĐẾ') cùng màu chữ không phải nền (01/10: cụm chữ bị coi là nền, lấy nhầm bóng).
    # Không dùng riêng dải trái-phải: OCR hay đọc thiếu một phần dòng -> phần chưa đọc nằm ngay dải bên (bộ tổng hợp).
    busy = np.zeros_like(core)
    for b in others:
        bx0, by0, bx1, by1 = (int(b[0]) - X0, int(b[1]) - Y0, int(np.ceil(b[2])) - X0, int(np.ceil(b[3])) - Y0)
        if bx1 > 0 and by1 > 0 and bx0 < core.shape[1] and by0 < core.shape[0]:
            busy[max(0, by0):max(0, by1), max(0, bx0):max(0, bx1)] = True
    side = ~core & ~busy
    if side.sum() < 0.05 * core.sum():   # vành bị các khung khác phủ gần hết
        side = ~core
    a_core, a_side = max(1, core.sum()), max(1, side.sum())
    enrich = []
    for c in range(k):
        n_core = ((lbl == c) & core).sum()
        if n_core < 0.03 * core.sum():
            enrich.append(-1)
            continue
        dc, ds = n_core / a_core, ((lbl == c) & side).sum() / a_side
        enrich.append(dc / (dc + ds))
    out_px = lab2[~core & ~busy] if (~core & ~busy).any() else lab2[~core]
    ref = np.median(out_px, axis=0) if len(out_px) else np.median(lab2.reshape(-1, 3), axis=0)
    top = max(enrich)
    textual = [c for c in range(k) if enrich[c] >= max(ENRICH_ABS, ENRICH_REL * top)] or [int(np.argmax(enrich))]
    # nét chính mặc định = cụm có (độ tập trung x khoảng cách màu tới vành ngoài) lớn nhất: lòng chữ / khe giữa chữ cùng màu
    # NỀN cũng chỉ nằm trong lõi (độ tập trung ~1) nhưng màu ~ nền (p13 01/10: lấy lòng chữ tối làm nét, in chữ đen trên nền đen)
    dmax = max(1e-6, max(np.linalg.norm(ctr[c] - ref) for c in textual))
    far = max(textual, key=lambda c: enrich[c] * np.linalg.norm(ctr[c] - ref) / dmax)
    fill, effect = far, None
    effc = [c for c in textual if c != far and (lbl == c).sum() >= EFFECT_MIN * (lbl == far).sum()]
    # cụm khác trên đoạn (cụm xa nhất -> nền): răng cưa, bỏ qua; lệch khỏi đoạn: hiệu ứng
    effc = [c for c in effc if _seg_dist(ctr[c], ctr[far], ref) > 0.25]
    # bóng / viền phải NẰM SÁT nét: >= 60% điểm cách cụm xa nhất <= 0.15 cao dòng (mảng nền nằm gọn trong lõi thì không --
    # bộ tổng hợp: thiếu điều kiện này, mảng nền dày bị lấy làm nét, mặt nạ trùng ~0)
    near = cv2.dilate((lbl == far).astype(np.uint8), np.ones((2 * max(1, int(0.15 * (y1 - y0))) + 1,) * 2, np.uint8)) > 0
    effc = [c for c in effc if (near & (lbl == c)).sum() >= 0.6 * (lbl == c).sum()]
    if effc:
        o = effc[0]
        # nét chính = cụm NHIỀU điểm hơn: bóng đổ / viền chỉ lộ thành dải mảnh quanh nét (q01: '50%' vàng bóng đỏ sẫm --
        # luật "dày hơn 1.5x" để bóng làm nét)
        fill, eff = (o, far) if (lbl == o).sum() > (lbl == far).sum() else (far, o)
        dx, dy = shadow_offset(lbl == fill, lbl == eff, y1 - y0)
        effect = {"cluster": eff, "dx": dx, "dy": dy}
    m = (lbl == fill).astype(np.uint8)
    er = np.isin(lbl, textual).astype(np.uint8)
    ring = lab2[~core & ~er.astype(bool)]
    bg = np.median(ring, axis=0) if len(ring) else np.median(lab2[~er.astype(bool)], axis=0)

    def color_of(mask):
        d = np.linalg.norm(lab2 - bg, axis=2)[mask.astype(bool)]
        px = crop[mask.astype(bool)]
        return np.median(px[d >= np.median(d)], axis=0) if len(px) else np.array([255, 255, 255])

    if effect is not None:
        effect["color"] = "#%02x%02x%02x" % tuple(int(v) for v in color_of(lbl == effect.pop("cluster")))
    if effect is not None and k == 3:
        # có hiệu ứng (bóng / viền): chữ nhiều lớp màu (nét + viền + bóng + nền) -> phân thêm 4 cụm, mặt nạ XOÁ = hợp các
        # cụm thuộc chữ của cả hai lần (02/10 s08: chữ 3D viền trắng bóng tối xoá sót viền)
        g4 = glyph_mask(img, box, k=4, others=others)
        er = (er | g4["erase"]).astype(np.uint8)
    return {"fill": m, "erase": er, "origin": (X0, Y0), "color": color_of(m), "bg": bg, "effect": effect}


def stroke_stats(m: np.ndarray, dist: np.ndarray) -> dict:
    """Độ dày nét (2 x p90 khoảng cách tới mép trên xương chữ) + tương phản nét (p90 / p25): chữ có chân kiểu Playfair
    dày-mảnh rõ, chữ không chân đều nét."""
    import cv2
    sk = (dist > 0) & (dist >= cv2.dilate(dist, np.ones((3, 3), np.uint8)))  # xương chữ ~ đỉnh cục bộ của khoảng cách
    v = dist[sk] if sk.sum() > 20 else dist[m.astype(bool)]
    v = v[v > 0]
    return {"stroke": float(2 * np.percentile(v, 90)) if len(v) else 1.0,
            "contrast": float(np.percentile(v, 90) / max(0.5, np.percentile(v, 25))) if len(v) else 1.0}


def main_stats(m: np.ndarray, lh: float | None = None) -> dict | None:
    """Phép đo CHUNG cho nét nháp và nét in thử: nét chính = mảnh cao >= 0.3 cao dòng; đường chân = trung vị đáy nét chính,
    đỉnh = trung vị đỉnh, mép = mọi mảnh trừ hạt nhiễu. lh = cao dòng (khung OCR); None -> cao vùng mực."""
    import cv2
    n, cc, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if lh is None:
        ys = np.nonzero(m.any(1))[0]
        if not len(ys):
            return None
        lh = float(ys[-1] - ys[0] + 1)
    main = [i for i in range(1, n) if st[i, cv2.CC_STAT_HEIGHT] >= 0.3 * lh and st[i, cv2.CC_STAT_AREA] >= 0.01 * lh * lh]
    if not main:
        return None
    tops = [st[i, cv2.CC_STAT_TOP] for i in main]
    bots = [st[i, cv2.CC_STAT_TOP] + st[i, cv2.CC_STAT_HEIGHT] for i in main]
    keep = np.isin(cc, [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 0.002 * lh * lh])  # nét + dấu, bỏ hạt nhiễu
    ys, xs = np.nonzero(keep)
    base, top = float(np.median(bots)), float(np.median(tops))
    # cao THÂN chữ theo mặt cắt mật độ hàng: từ giữa thân đi lên tới hàng mật độ < PROFILE_DROP x mật độ thân. Trung vị đỉnh
    # nét chập chờn giữa cao 'x' và cao nét lên tuỳ dòng nhiều chữ nào (q04 01/10: 4 món cùng cỡ đo lệch 25%); thân chữ thì
    # mọi chữ cái đều phủ -> ổn định hơn. Cùng phép đo cho nét nháp và nét in thử.
    mm = np.isin(cc, main)
    occ = mm.sum(1).astype(float)
    h0 = base - top
    lo, hi = int(max(0, base - 0.75 * h0)), int(max(1, base - 0.25 * h0))
    body = float(np.median(occ[lo:hi])) if hi > lo else float(occ.max())
    r = int(base - 0.5 * h0)
    while r > 0 and occ[r - 1] >= PROFILE_DROP * body:
        r -= 1
    return {"base": base, "top": top, "h": base - r if ALT_H else h0, "h_med": h0, "x0": float(xs.min()), "x1": float(xs.max() + 1),
            "parts": len(main)}


def _drop_lead_mark(g: dict, box, lh: float) -> bool:
    """Ký hiệu đầu dòng (•, -, ✓...) OCR gộp vào dòng: cột mực ĐẦU TIÊN hẹp (< 0.8 cao dòng) và cách phần sau một khe
    (>= 0.2 cao dòng) -> bỏ khỏi mặt nạ nét VÀ mặt nạ xoá (giữ ký hiệu model vẽ, như icon). Trả True nếu đã bỏ."""
    m = g["fill"]
    col = m.any(0).astype(np.int8)
    d = np.diff(np.concatenate([[0], col, [0]]))
    st, en = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
    if len(st) < 2 or en[0] - st[0] >= 0.8 * lh or st[1] - en[0] < 0.2 * lh:
        return False
    cut = int((en[0] + st[1]) / 2)
    g["fill"] = g["fill"].copy()
    g["fill"][:, :cut] = 0
    g["erase"] = g["erase"].copy()
    g["erase"][:, :cut] = 0
    return True


def measure(img: np.ndarray, box, ocr_text: str | None = None, others=()) -> dict | None:
    """Số đo một dòng. None nếu không tách được nét chính (khung rỗng / chỉ nhiễu). ocr_text bắt đầu bằng ký hiệu đầu dòng
    -> giữ ký hiệu (không xoá, không tính vào mép trái)."""
    import cv2
    g = glyph_mask(img, box, others=others)
    lh = box[3] - box[1]
    mark = bool(ocr_text) and ocr_text.lstrip()[:1] in MARKS and _drop_lead_mark(g, box, lh)
    m = g["fill"]
    ox, oy = g["origin"]
    s = main_stats(m, lh)
    if s is None:
        return None
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    return {"base": oy + s["base"], "top": oy + s["top"], "h": s["h"], "x0": ox + s["x0"], "x1": ox + s["x1"],
            "color": "#%02x%02x%02x" % tuple(int(v) for v in g["color"]), "bg_lab": [float(v) for v in g["bg"]],
            **stroke_stats(m, dist), "parts": s["parts"], "mask": (m, ox, oy), "erase_mask": (g["erase"], ox, oy),
            "effect": g["effect"], "lead_mark": mark}
