"""HÌNH HỌC CHỮ (09/10, đợt nâng cấp chữ cong / một dòng nhiều cỡ) -- bổ sung cho ô L của slots.build:

  arcs(ink, L, I, W, H)  -> chữ chạy theo CUNG TRÒN (tiêu đề cong, vòng con dấu, chữ "nụ cười"): OCR chỉ cho khung thẳng, cắt
                            cung thành mẩu xoay lệch hoặc bỏ hai đầu cung (thành chi tiết I). Dò từ NÉT: chuỗi mảnh nét cùng cỡ
                            nối nhau, khớp đường tròn qua tâm các mảnh; cung rõ (võng >= 0.25 cao chữ, lệch khớp nhỏ) -> MỘT ô L
                            {curve: cx, cy, r (bán kính ĐƯỜNG CHÂN), a0, a1 (độ, toạ độ ảnh: 0 phải, 90 dưới), up}.
  runs(L)                -> MỘT DÒNG NHIỀU CỠ ("GIẢM" nhỏ + "50%" to): OCR tách thành các ô riêng; các ô liền nhau cùng đường
                            chân, cỡ lệch >= 1.4 lần -> MỘT ô L {runs: [{text, size, x0, x1}], base}.
Bộ thử: scripts/geo_synth.py (đáp án tuyệt đối trên nền bản xoá thật).
"""

from __future__ import annotations

import math
import os

import numpy as np

ENABLED = os.environ.get("TEXTFIX_GEO", "off") == "on"   # bật khi B3 (mô tả) + B4 (dựng theo đường chân) xong

CHAIN_GAP = 0.8      # hai mảnh nét cùng chuỗi: khe <= 0.8 cao mảnh lớn hơn ...
CHAIN_H = 1.8        # ... và cao lệch <= 1.8 lần
MARK_H = 0.4         # mảnh thấp hơn 0.4 cao trung vị chuỗi = dấu thanh / chấm (không dựng chuỗi, gộp vào khung sau)
ARC_MIN_PARTS = 5    # chuỗi >= 5 mảnh mới khớp cung
ARC_FIT = 0.15       # lệch khớp đường tròn (trung vị khoảng cách tâm mảnh tới đường tròn) <= 0.15 cao chữ
ARC_SAG = 0.3        # độ võng của cung (dây cung^2 / 8R) >= 0.3 cao chữ: thấp hơn là dòng thẳng / nghiêng (OCR lo)
ARC_LINE = 0.18      # và đường thẳng khớp tệ: lệch trung vị >= 0.18 cao chữ
ARC_BLIND = 7        # cung >= 7 mảnh mà OCR không đọc được chữ nào vẫn là chữ (OCR hay trượt hẳn chữ cong)
RUN_SIZE = 1.4       # một dòng nhiều cỡ: hai ô liền nhau cỡ lệch >= 1.4 lần ...
RUN_BASE = 0.15      # ... đường chân lệch <= 0.15 cỡ lớn ...
RUN_GAP = 0.7        # ... khe ngang <= 0.7 cỡ lớn
RUN_MIN = 24.0       # ... và cỡ lớn >= 24 px (chữ li ti trang trí -- dãy số nhị phân -- không phải dòng nhiều cỡ)


def _circle(pts: np.ndarray) -> tuple[float, float, float]:
    """Khớp đường tròn đại số (Kåsa)."""
    x, y = pts[:, 0], pts[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    b = x ** 2 + y ** 2
    (cx, cy, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(cx), float(cy), float(math.sqrt(max(c + cx ** 2 + cy ** 2, 1e-6)))


def _line_res(pts: np.ndarray) -> float:
    c = pts - pts.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    return float(np.median(np.abs(c @ vt[1])))


NPATH = 16           # số điểm đường chân ghi vào ô
WAVE_DEV = 0.3       # đường cong tự do: lệch khỏi đường thẳng >= 0.3 cao chữ ...
WAVE_FIT = 0.15      # ... và khớp đa thức bậc 3 (theo trục chính) lệch trung vị <= 0.15 cao chữ
GROW = 1.35          # chữ TO DẦN / NHỎ DẦN: cao chữ hai đầu lệch >= 1.35 lần, đổi đều (tương quan với vị trí >= 0.8)


def _scale(u: np.ndarray, ext: np.ndarray, uq: np.ndarray) -> list[float] | None:
    """Hồ sơ cỡ dọc dòng (tỉ lệ so với cỡ trung vị) tại các điểm uq -- chỉ khi chữ to / nhỏ dần rõ rệt."""
    if len(u) < 6 or np.ptp(u) <= 0:
        return None
    k, b0 = np.polyfit(u, ext, 1)
    e0, e1 = k * u.min() + b0, k * u.max() + b0
    if min(e0, e1) <= 0 or max(e0, e1) / min(e0, e1) < GROW or abs(np.corrcoef(u, ext)[0, 1]) < 0.8:
        return None
    med = float(np.median(ext))
    return [round(float((k * x + b0) / med), 3) for x in uq]


def _baseline(center: np.ndarray, half: np.ndarray) -> list[list[float]]:
    """Điểm giữa cao chữ -> điểm đường chân: dịch nửa cao chữ theo pháp tuyến "xuống" của hướng đọc (trái -> phải)."""
    t = np.gradient(center, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-6)
    n = np.c_[-t[:, 1], t[:, 0]]
    return [[round(float(x), 1), round(float(y), 1)] for x, y in center + n * half[:, None]]


def _fit_circle(pts, ext, hm):
    cx, cy, r = _circle(pts)
    res = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r)
    keep = res <= max(2 * np.median(res), 0.1 * hm)   # khớp lại không có điểm lệch (dấu dính, nét chữ Q / g thò xuống)
    if keep.sum() >= ARC_MIN_PARTS:
        pts, ext = pts[keep], ext[keep]
        cx, cy, r = _circle(pts)
        res = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r)
    chord = float(np.hypot(*(pts.max(0) - pts.min(0))))
    sag = chord ** 2 / (8 * r) if r > 0 else 0.0
    if np.median(res) > ARC_FIT * hm or sag < ARC_SAG * hm or _line_res(pts) < ARC_LINE * hm or r > 8 * chord:
        return None
    up = cy > pts[:, 1].mean()   # tâm dưới chuỗi: cung lồi phía trên
    ang = np.degrees(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx))
    # góc liền mạch quanh đỉnh (cung trên: 180..360) / quanh đáy (cung dưới: 0..180)
    ang = np.where(ang < 0, ang + 360, ang) if up else np.where(ang < -90, ang + 360, ang)
    da = math.degrees(0.5 * hm / max(r, 1))   # nới hai đầu nửa cao chữ (tâm lát đầu / cuối cách mép chữ)
    if up:   # đi trái -> phải theo chiều kim đồng hồ quanh đỉnh
        a0, a1 = float(ang.min()) - da, float(ang.max()) + da
    else:    # cung dưới: trái (góc lớn) -> phải (góc nhỏ)
        a0, a1 = float(ang.max()) + da, float(ang.min()) - da
    # đường chân: chữ cung trên đứng ngoài đường tròn (chân phía tâm), cung dưới ngược lại
    rb = r - hm / 2 if up else r + hm / 2
    aq = np.linspace(a0, a1, NPATH)
    path = [[round(float(cx + rb * math.cos(math.radians(a))), 1), round(float(cy + rb * math.sin(math.radians(a))), 1)] for a in aq]
    u = ang if up else -ang
    return {"geo": "arc", "path": path, "scale": _scale(u, ext, aq if up else -aq),
            "curve": {"cx": round(cx, 1), "cy": round(cy, 1), "r": round(rb, 1), "a0": round(a0, 1), "a1": round(a1, 1),
                      "up": bool(up)}}


def _axis(pts):
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    e1 = vt[0] if vt[0][0] >= 0 else -vt[0]   # hướng đọc trái -> phải
    e2 = np.array([-e1[1], e1[0]])            # pháp tuyến "xuống" của hướng đọc
    return c, e1, e2


def _fit_wave(pts, ext, hm):
    c, e1, e2 = _axis(pts)
    u, v = (pts - c) @ e1, (pts - c) @ e2
    if np.ptp(u) < 3 * hm or len(u) < 8:
        return None
    p3 = np.polyfit(u, v, 3)
    res = np.median(np.abs(np.polyval(p3, u) - v))
    lin = np.polyval(np.polyfit(u, v, 1), u)
    dev = float(np.max(np.abs(np.polyval(p3, u) - lin)))
    if res > WAVE_FIT * hm or dev < WAVE_DEV * hm:
        return None
    uq = np.linspace(u.min() - 0.5 * hm, u.max() + 0.5 * hm, NPATH)
    center = c + np.outer(uq, e1) + np.outer(np.polyval(p3, uq), e2)
    sc = _scale(u, ext, uq)
    half = 0.5 * hm * (np.array(sc) if sc else np.ones(NPATH))
    return {"geo": "wave", "path": _baseline(center, half), "scale": sc}


def _fit_grow(pts, ext, hm):
    """Dòng thẳng (thẳng / nghiêng) mà chữ to dần / nhỏ dần: đường chân thẳng qua chân các lát + hồ sơ cỡ."""
    c, e1, e2 = _axis(pts)
    u = (pts - c) @ e1
    uq = np.linspace(u.min() - 0.5 * hm, u.max() + 0.5 * hm, NPATH)
    sc = _scale(u, ext, uq)
    if not sc:
        return None
    bot = pts + np.outer(ext / 2, e2)   # chân các lát
    ub, vb = (bot - c) @ e1, (bot - c) @ e2
    k, b0 = np.polyfit(ub, vb, 1)
    center = c + np.outer(uq, e1) + np.outer(k * uq + b0, e2)
    return {"geo": "grow", "path": [[round(float(x), 1), round(float(y), 1)] for x, y in center], "scale": sc}


def _chains(comps: list[dict]) -> list[list[int]]:
    """Nối mảnh nét thành chuỗi (khe nhỏ, cao gần nhau)."""
    n = len(comps)
    par = list(range(n))

    def f(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i
    for i in range(n):
        a = comps[i]
        for j in range(i + 1, n):
            b = comps[j]
            hi, lo = max(a["h"], b["h"]), min(a["h"], b["h"])
            if hi > CHAIN_H * lo:
                continue
            gx = max(b["x0"] - a["x1"], a["x0"] - b["x1"], 0)
            gy = max(b["y0"] - a["y1"], a["y0"] - b["y1"], 0)
            # nối CẠNH nhau (cùng dòng): hai mảnh chồng nhau theo chiều dọc >= 0.3 cao mảnh thấp (cung nghiêng tới ~45 độ vẫn
            # chồng); dòng trên / dưới sát nhau (khe dòng ~0.3 cao chữ) không chồng dọc -> không nối
            ov = min(a["y1"], b["y1"]) - max(a["y0"], b["y0"])
            if ov >= 0.3 * lo and math.hypot(gx, gy) <= CHAIN_GAP * hi:
                par[f(i)] = f(j)
    g = {}
    for i in range(n):
        g.setdefault(f(i), []).append(i)
    return list(g.values())


def arcs(ink: np.ndarray, W: int, H: int, dbg: list | None = None) -> list[dict]:
    """Các cung chữ trong mặt nạ nét. -> [{box, curve, size, parts}]"""
    import cv2
    m = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, cc, st, cen = cv2.connectedComponentsWithStats(m, 8)
    comps = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 12 or h < 8 or w > 0.5 * W or h > 0.4 * H:
            continue
        comps.append({"i": i, "x0": x, "y0": y, "x1": x + w, "y1": y + h, "h": h, "c": cen[i], "a": a})
    if not comps:
        return []
    hs = np.array([c["h"] for c in comps])
    out = []
    for ch in _chains(comps):
        if len(ch) < 2:
            continue
        hm = float(np.median(hs[ch]))
        body = [k for k in ch if comps[k]["h"] >= MARK_H * hm]
        if len(body) < 2:
            continue
        # ĐIỂM KHỚP: tâm nét theo LÁT dọc rộng ~0.4 cao chữ trên từng mảnh (chữ đậm / có viền dính liền thành vài mảnh to)
        pts, ext = [], []
        for k in body:
            c = comps[k]
            sub = cc[c["y0"]:c["y1"], c["x0"]:c["x1"]] == c["i"]
            step = max(3, int(0.4 * hm))
            for x in range(0, sub.shape[1], step):
                sl = sub[:, x:x + step]
                ys = np.nonzero(sl.any(1))[0]
                if len(ys) and sl.sum() >= 0.05 * step * hm:
                    pts.append((c["x0"] + x + sl.shape[1] / 2, c["y0"] + (ys[0] + ys[-1]) / 2))
                    ext.append(ys[-1] - ys[0] + 1)
        pts = np.array(pts, np.float64)
        if len(ext):   # cao chữ theo lát (mảnh dính cả từ trên cung nghiêng cao phồng)
            hm = float(np.median(ext))
        if len(pts) < ARC_MIN_PARTS + 1 or np.ptp(pts[:, 0]) < 3 * hm:
            if dbg is not None and len(pts):
                dbg.append({"box": [float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())],
                            "why": "ít điểm / chuỗi ngắn"})
            continue
        ext = np.array(ext, np.float64)
        g = _fit_circle(pts, ext, hm) or _fit_wave(pts, ext, hm) or _fit_grow(pts, ext, hm)
        if g is None:
            if dbg is not None:
                dbg.append({"box": [float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())],
                            "why": "không cong / không đổi cỡ", "line": round(_line_res(pts) / hm, 2), "n": len(pts)})
            continue
        sel = np.isin(cc, [comps[k]["i"] for k in ch])
        ys, xs = np.nonzero(sel)
        out.append({"box": [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)], **g,
                    "h": hm, "mask": sel, "parts": len(body)})
    return out


def apply_arcs(L: list[dict], I: list[dict], found: list[dict], cap_em: float) -> tuple[list[dict], list[dict], list[str]]:
    """Thay các ô L / I nằm trên một cung bằng MỘT ô L cung. Chữ OCR gợi ý = các mẩu OCR theo thứ tự trái -> phải."""
    log = []
    for a in found:
        mk = a["mask"]
        H, W = mk.shape

        def on(b):
            x0, y0, x1, y1 = (int(max(0, v)) for v in b)
            sub = mk[y0:min(H, y1), x0:min(W, x1)]
            return sub.size > 0 and sub.mean() >= 0.08
        cov_L = [l for l in L if on(l["box"])]
        cov_I = [c for c in I if on(c["box"])]
        if not cov_L and a["parts"] < ARC_BLIND:
            continue   # cung ngắn không có chữ OCR nào: có thể là hoạ tiết tròn (hạt, chấm) -- không phải chữ
        cov_L.sort(key=lambda l: (l["box"][0] + l["box"][2]) / 2)
        if cov_L:
            first = cov_L[0]
        else:   # OCR không đọc được chữ cong (cung dài, nhiều mảnh cùng cỡ đều nhau = chữ): ô mới, chữ OCR rỗng
            first = {"id": f"L{1 + max([int(l['id'][1:]) for l in L] or [0])}", "ocr": "", "caps": True, "color": None,
                     "stroke": None, "shell": None, "base": None}
            L = L + [first]
        caps = all(l.get("caps") for l in cov_L) if cov_L else True
        new = {**first, "box": a["box"], "ocr_box": a["box"], "ocr": " ".join(l["ocr"] for l in cov_L),
               "size": float(a["h"] / (cap_em if caps else 0.6)), "angle": 0.0, "geo": a["geo"], "path": a["path"],
               "scale": a.get("scale"), **({"curve": a["curve"]} if a.get("curve") else {}),
               "poly": [[a["box"][0], a["box"][1]], [a["box"][2], a["box"][1]], [a["box"][2], a["box"][3]], [a["box"][0], a["box"][3]]]}
        gone = {id(l) for l in cov_L}
        L = [new if l is first else l for l in L if id(l) not in gone or l is first]
        I = [c for c in I if c not in cov_I]
        log.append(f"chữ {a['geo']}: {[l['id'] for l in cov_L]} + {[c['id'] for c in cov_I]} -> {first['id']}"
                   + (f" {a['curve']}" if a.get("curve") else "") + (f" cỡ {a['scale'][0]}..{a['scale'][-1]}" if a.get("scale") else ""))
    return L, I, log


def runs(L: list[dict]) -> tuple[list[dict], list[str]]:
    """Gộp các ô liền nhau CÙNG đường chân, cỡ lệch nhiều thành MỘT ô có runs (một dòng nhiều cỡ)."""
    log = []
    cand = [l for l in L if l.get("base") is not None and abs(l.get("angle") or 0) < 2 and not l.get("path")]
    cand.sort(key=lambda l: l["box"][0])
    groups, used = [], set()
    for a in cand:
        if id(a) in used:
            continue
        g = [a]
        while True:
            last = g[-1]
            nxt = None
            for b in cand:
                if id(b) in used or b in g or b["box"][0] < last["box"][0]:
                    continue
                big = max(last["size"], b["size"])
                if abs(b["base"] - last["base"]) <= RUN_BASE * big and 0 <= b["box"][0] - last["box"][2] <= RUN_GAP * big \
                        and (nxt is None or b["box"][0] < nxt["box"][0]):
                    nxt = b
            if nxt is None:
                break
            g.append(nxt)
        sz = [x["size"] for x in g]
        if len(g) >= 2 and max(sz) >= RUN_SIZE * min(sz) and max(sz) >= RUN_MIN:
            groups.append(g)
            used |= {id(x) for x in g}
    for g in groups:
        first = g[0]
        box = [min(x["box"][0] for x in g), min(x["box"][1] for x in g), max(x["box"][2] for x in g), max(x["box"][3] for x in g)]
        new = {**first, "box": box, "ocr_box": box, "ocr": " ".join(x["ocr"] for x in g), "size": max(x["size"] for x in g),
               "runs": [{"text": x["ocr"], "size": round(x["size"], 1), "x0": round(x["box"][0]), "x1": round(x["box"][2])} for x in g],
               "poly": [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]]}
        gone = {id(x) for x in g[1:]}
        L = [new if x is first else x for x in L if id(x) not in gone]
        log.append(f"một dòng nhiều cỡ: {[x['id'] for x in g]} -> {first['id']} {[r['size'] for r in new['runs']]}")
    return L, log
