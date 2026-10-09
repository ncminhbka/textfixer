"""BẢN ĐỒ LỚP PHỦ (07/10): nháp - bản xoá. Bản xoá FLUX (lời dặn ngắn "Remove all text...") bỏ đúng LỚP PHỦ -- chữ, vỏ (thẻ bình
luận, pill, dải, nút), icon, avatar, sao -- và giữ ảnh gần như từng pixel (probe: đổi ngoài chữ ~0%). Nên chỗ nháp khác bản xoá
= lớp phủ, đúng vị trí, không cần DINO; ảnh / người / sản phẩm không bao giờ bị khoanh (bản xoá giữ chúng).

  O = overlay(draft, plate, lines)   # lines: ocr.read_lines(draft)
  O["S"]: vỏ  {id, box, fill (#hex), radius (px), lines [chỉ số dòng trong vỏ], icons [id I], _mask, _fill_lab}
  O["I"]: chi tiết nhỏ {id, box, color, shell (id S | None)}
  O["mask"]: lớp phủ (bool), O["text"]: mặt nạ chữ (bool)

Cách tách:
  1. chênh ΔE(nháp, bản xoá) > DIFF, mở hình thái (bỏ viền mảnh của ảnh hơi lệch) = lớp phủ;
  2. mảng lớp phủ TO, GỌN, ĐẶC (>= 80% điểm đổi: vỏ bị xoá cả lòng; chữ trần chỉ đổi nét) và THÒ RA ngoài khung dòng chữ
     (nét chữ rất đậm cũng đặc nhưng nằm gọn trong khung dòng) = VỎ, màu lòng = màu chiếm nhiều nhất;
     lớp phủ ngoài vỏ trừ chữ (mặt nạ nét mực, textmask) = chi tiết, bỏ mẩu lọt trong dòng chữ;
  3. trong vỏ: điểm khác màu lòng vỏ rõ, không phải chữ = chi tiết nằm trong vỏ (avatar, sao, tick...).
"""

from __future__ import annotations

import numpy as np

DIFF = 20.0        # ΔE nháp - bản xoá coi là lớp phủ
OPEN = 5           # mở hình thái (px): bỏ viền 1-2 px của ảnh hơi lệch khi xoá -- mặt nạ cho VỎ và nắn khung CHỮ
DETAIL_OPEN = 3    # ... riêng cho CHI TIẾT: 5 px xoá mất icon nét mảnh 2-3 px (vụn thêm ra do mở nhẹ: lọc theo cỡ ở slots)
SHELL_MIN = 0.004  # vỏ: diện tích >= 0.4% ảnh ...
SHELL_SOLID = 0.55  # ... hình gọn (diện tích / khung >= 55%) ...
SHELL_OUT = 0.15    # vỏ THÒ RA ngoài khung dòng chữ >= 15% diện tích (phần đệm); nét chữ đậm nằm gọn trong khung dòng
SHELL_DENSE = 0.8   # ... và ĐẶC: >= 80% điểm trong mảng là lớp phủ (vỏ: cả lòng vỏ bị xoá; chữ trần: chỉ nét đổi, ~30-60%)
ICON_MIN = 0.00005  # chi tiết: >= 0.005% ảnh (bỏ vụn)
INNER_DE = 25.0    # trong vỏ: khác màu lòng vỏ >= 25 ΔE = chi tiết


def _lab(x):
    import cv2
    a = cv2.cvtColor(x, cv2.COLOR_RGB2LAB).astype(np.float32)
    a[..., 0] *= 100 / 255
    a[..., 1:] -= 128
    return a


def _radius(m: np.ndarray) -> float:
    """Bo góc ước theo khoảng hở ở 4 góc khung trên đường chéo: hình chữ nhật bo r hở r(√2-1) ở mỗi góc."""
    h, w = m.shape
    gaps = []
    for flip_y, flip_x in ((0, 0), (0, 1), (1, 0), (1, 1)):
        mm = m[::-1] if flip_y else m
        mm = mm[:, ::-1] if flip_x else mm
        n = min(h, w)
        d = next((i for i in range(n) if mm[i, i]), n)
        gaps.append(d)
    g = float(np.median(gaps))
    return float(min(g / (np.sqrt(2) - 1), min(h, w) / 2))


SOFT_DIFF = 7.0    # VỎ NHẠT quanh dòng chữ: pill trắng trên nền kem / be chỉ khác bản xoá ΔE 10-18 (< DIFF, 09/10 dev2: 6 / 31
                   # ca mất pill, VLM vẽ lại pill ôm sát chữ); nền không vỏ quanh chữ ΔE 1-4
SOFT_PAD = (4.0, 2.5)   # vùng tìm quanh dòng: ngang / dọc theo cao dòng (vỏ chạm mép vùng = mảng nền lớn, không phải vỏ của dòng)
SOFT_MIN = 0.0015       # vỏ nhạt >= 0.15% ảnh (nút "Mua ngay" nhỏ)
SOFT_STD = 14.0         # lòng vỏ phẳng màu: độ lệch chuẩn ΔE quanh màu lòng
SOFT_EDGE = 0.6         # bản xoá CÒN mép vỏ (độ sắc mép bản xoá / nháp dọc đường viền >= 0.6) = vỏ chưa bị xoá, không phải ô:
                        # vỏ bị xoá 0.01-0.39 (dev2), pill bản xoá giữ chỉ nhạt màu 1.05 (gt m03_s1)


def _edge_kept(A, B, mask, text) -> float:
    """Độ sắc mép (Sobel trên L) của bản xoá / của nháp, dọc dải 5 px quanh đường viền vỏ (trừ chữ)."""
    import cv2
    m = mask.astype(np.uint8)
    band = (cv2.dilate(m, np.ones((5, 5), np.uint8)) - cv2.erode(m, np.ones((5, 5), np.uint8))).astype(bool) & ~text
    if band.sum() < 20:
        return 0.0
    def g(L):
        L = L.astype(np.float32)
        return float(np.hypot(cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3))[band].mean())
    return g(B[..., 0]) / max(1e-3, g(A[..., 0]))


def _ring(draft, A, lo, text, win, lb, W, H) -> dict | None:
    """Vỏ VIỀN quanh dòng (nút / khung chỉ có nét viền, lòng trong suốt: lòng không đổi khi xoá, chỉ nét viền đổi -- 09/10 dev2
    d02 "Shop now", d12 khung ngày): một đường viền KÍN bao trọn dòng, lòng (trừ chữ) gần như không đổi, nằm gọn trong vùng tìm."""
    import cv2
    X0, Y0, X1, Y1 = win
    x0, y0, x1, y1 = lb
    h = y1 - y0
    t = text[Y0:Y1, X0:X1]
    m = (lo.astype(bool) & ~t).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        x, y, w, hh = cv2.boundingRect(c)
        if not (x <= x0 - X0 - 0.1 * h and y <= y0 - Y0 - 0.1 * h and x + w >= x1 - X0 + 0.1 * h and y + hh >= y1 - Y0 + 0.1 * h):
            continue   # không bao trọn dòng
        if x == 0 or y == 0 or x + w >= X1 - X0 or y + hh >= Y1 - Y0 or w * hh < SOFT_MIN * W * H:
            return None
        F = cv2.drawContours(np.zeros_like(m), [c], -1, 1, cv2.FILLED).astype(bool)
        if F.sum() / float(w * hh) < SHELL_SOLID:
            return None
        k = max(3, int(0.25 * h))
        inner = cv2.erode(F.astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8)).astype(bool) & ~t
        edge = F & ~cv2.erode(F.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        if inner.sum() < max(50, 0.03 * F.sum()) or m.astype(bool)[inner].mean() > 0.15 or m.astype(bool)[edge].mean() < 0.7:
            return None   # lòng cũng đổi (vỏ đặc -- nhánh khác lo) hoặc viền hở
        ring = m.astype(bool) & F & ~inner
        col = np.median(draft[Y0:Y1, X0:X1][ring], axis=0)
        bw = float(ring.sum()) / max(1.0, cv2.arcLength(c, True))
        sm = np.zeros((H, W), bool)
        sm[Y0:Y1, X0:X1] = F
        return {"box": [float(X0 + x), float(Y0 + y), float(X0 + x + w), float(Y0 + y + hh)], "fill": "none",
                "border": "#%02x%02x%02x" % tuple(int(v) for v in col), "border_px": round(min(bw, 0.3 * h), 1),
                "radius": round(_radius(F[y:y + hh, x:x + w]), 1), "_mask": sm,
                "_fill_lab": np.median(A[Y0:Y1, X0:X1][inner], axis=0),   # màu NỀN trong lòng (chữ / viền = khác màu này)
                "soft": True}
    return None


def _soft_shells(draft, A, B, lines, text, taken) -> list[dict]:
    """Vỏ NHẠT (pill / thẻ màu gần nền) quanh từng dòng chưa nằm trong vỏ: lớp phủ ngưỡng thấp SOFT_DIFF, chỉ trong vùng quanh
    dòng; mảng chứa dòng phải GỌN, ĐẶC, THÒ RA ngoài dòng, LÒNG PHẲNG MÀU và nằm gọn trong vùng tìm."""
    import cv2
    H, W = draft.shape[:2]
    e = np.linalg.norm(A - B, axis=2)
    out = []
    for l in lines:
        x0, y0, x1, y1 = l["box"]
        h = y1 - y0
        cx, cy = int((x0 + x1) / 2), int((y0 + y1) / 2)
        if h < 6 or taken[min(H - 1, cy), min(W - 1, cx)] or any(s["_mask"][min(H - 1, cy), min(W - 1, cx)] for s in out):
            continue
        X0, Y0 = int(max(0, x0 - SOFT_PAD[0] * h)), int(max(0, y0 - SOFT_PAD[1] * h))
        X1, Y1 = int(min(W, x1 + SOFT_PAD[0] * h)), int(min(H, y1 + SOFT_PAD[1] * h))
        lo = (cv2.GaussianBlur(e[Y0:Y1, X0:X1], (0, 0), 1.0) > SOFT_DIFF).astype(np.uint8)
        # nét viền mảnh 2-3 px: mở 5 px xoá mất -> nhánh viền dùng mặt nạ thô, chỉ đóng nhẹ cho kín
        ring = _ring(draft, A, cv2.morphologyEx(lo, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)), text, (X0, Y0, X1, Y1), (x0, y0, x1, y1), W, H)   # vỏ chỉ có VIỀN (lòng trong suốt)
        lo = cv2.morphologyEx(lo, cv2.MORPH_OPEN, np.ones((OPEN, OPEN), np.uint8))
        lo = cv2.morphologyEx(lo, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        lo[int(y0 - Y0):int(y1 - Y0), int(x0 - X0):int(x1 - X0)] = 1   # dòng chữ thuộc vỏ
        n, cc, st, _ = cv2.connectedComponentsWithStats(lo, 8)
        k = cc[cy - Y0, cx - X0]
        x, y, w, hh, area = st[k]
        # nhỏ quá / chạm mép vùng tìm (trừ mép ảnh): mảng nền lớn / dải tràn khung, không phải vỏ đặc của riêng dòng này
        if area < SOFT_MIN * W * H or \
                (x == 0 and X0 > 0) or (y == 0 and Y0 > 0) or (x + w == X1 - X0 and X1 < W) or (y + hh == Y1 - Y0 and Y1 < H):
            if ring:
                out.append(ring)
            continue
        comp = (cc[y:y + hh, x:x + w] == k).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = cv2.drawContours(np.zeros_like(comp), cnts, -1, 1, cv2.FILLED).astype(bool)
        line_box = np.zeros_like(filled)
        line_box[max(0, int(y0 - Y0) - y):max(0, int(y1 - Y0) - y), max(0, int(x0 - X0) - x):max(0, int(x1 - X0) - x)] = True
        if filled.sum() / float(w * hh) < SHELL_SOLID or (comp.astype(bool) & filled).mean() < SHELL_DENSE \
                or (filled & ~line_box).sum() < SHELL_OUT * filled.sum():
            if ring:
                out.append(ring)
            continue
        sm = np.zeros((H, W), bool)
        sm[Y0 + y:Y0 + y + hh, X0 + x:X0 + x + w] = filled
        body = sm & ~text
        # lòng vỏ phải BỊ XOÁ thật (bản xoá còn giữ pill, chỉ xoá chữ: chỉ quầng quanh nét đổi -- bench m03_s1)
        if body.sum() < 0.2 * sm.sum() or np.median(e[body]) <= SOFT_DIFF:
            continue
        lab = A[body]
        fill_lab = np.median(lab, axis=0)
        if np.linalg.norm(lab - fill_lab, axis=1).std() > SOFT_STD:   # lòng loang / có hoạ tiết: cảnh, không phải vỏ
            continue
        fill = np.median(draft[body], axis=0)
        out.append({"box": [float(X0 + x), float(Y0 + y), float(X0 + x + w), float(Y0 + y + hh)],
                    "fill": "#%02x%02x%02x" % tuple(int(v) for v in fill), "radius": round(_radius(filled), 1),
                    "_mask": sm, "_fill_lab": fill_lab, "soft": True})
    return [s for s in out if _edge_kept(A, B, s["_mask"], text) < SOFT_EDGE]


def overlay(draft: np.ndarray, plate: np.ndarray, lines: list[dict]) -> dict:
    import cv2
    from .textmask import text_mask
    H, W = draft.shape[:2]
    A, B = _lab(draft), _lab(plate)
    e = cv2.GaussianBlur(np.linalg.norm(A - B, axis=2), (0, 0), 1.0)
    ov = (e > DIFF).astype(np.uint8)
    ov_fine = cv2.morphologyEx(ov, cv2.MORPH_OPEN, np.ones((DETAIL_OPEN, DETAIL_OPEN), np.uint8)).astype(bool)
    ov = cv2.morphologyEx(ov, cv2.MORPH_OPEN, np.ones((OPEN, OPEN), np.uint8))
    text = text_mask(draft, lines)
    rest = ov.astype(bool) & ~text   # lớp phủ không phải chữ
    # VỎ: mảng lớp phủ (đóng nhẹ, lấp lỗ) TO, GỌN và ĐẶC -- đo trên lớp phủ thô, không dựa mặt nạ chữ (pill / dải mà chữ chiếm gần
    # hết: màu vỏ bị coi là màu mực, 07/10 dải đỏ p11, pill p13)
    full = cv2.morphologyEx(ov, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    boxes = np.zeros((H, W), np.uint8)   # khung các dòng chữ (đa giác OCR nới 0.1 cao dòng)
    for l in lines:
        one = np.zeros((H, W), np.uint8)
        cv2.fillPoly(one, [np.asarray(l["poly"], np.int32)], 1)
        k = max(1, int(0.1 * (l["box"][3] - l["box"][1])))
        boxes |= cv2.dilate(one, np.ones((2 * k + 1, 2 * k + 1), np.uint8))
    boxes = boxes.astype(bool)
    n, cc, st, _ = cv2.connectedComponentsWithStats(full, 8)
    S, I = [], []
    shell_mask = np.zeros((H, W), bool)
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if area < SHELL_MIN * W * H:
            continue
        comp = (cc[y:y + h, x:x + w] == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = cv2.drawContours(np.zeros_like(comp), cnts, -1, 1, cv2.FILLED).astype(bool)
        dense = ov[y:y + h, x:x + w][filled].mean()
        out = 1 - boxes[y:y + h, x:x + w][filled].mean()
        if filled.sum() / float(w * h) >= SHELL_SOLID and dense >= SHELL_DENSE and out >= SHELL_OUT:
            sm = np.zeros((H, W), bool)
            sm[y:y + h, x:x + w] = filled
            body = sm & rest
            body = body if body.sum() > 0.2 * sm.sum() else sm
            # màu lòng vỏ = màu chiếm nhiều nhất trong mảng (trung vị dễ lẫn màu chữ khi chữ chiếm gần hết vỏ)
            q = (draft[body] // 16).astype(np.int32)
            key = q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2]
            mode = np.bincount(key).argmax()
            near = draft[body][key == mode]
            fill = np.median(near, axis=0)
            S.append({"id": f"S{len(S) + 1}", "box": [float(x), float(y), float(x + w), float(y + h)],
                      "fill": "#%02x%02x%02x" % tuple(int(v) for v in fill), "radius": round(_radius(filled), 1),
                      "_mask": sm, "_fill_lab": _lab(fill.reshape(1, 1, 3).astype(np.uint8))[0, 0]})
            shell_mask |= sm
    for s in _soft_shells(draft, A, B, lines, text, shell_mask):
        s["id"] = f"S{len(S) + 1}"
        S.append(s)
        shell_mask |= s["_mask"]
        ov |= s["_mask"].astype(np.uint8)
    # chi tiết nhỏ: (a) mảng lớp phủ ngoài vỏ, không phải chữ; (b) trong vỏ, khác màu lòng vỏ, không phải chữ
    cand = (ov_fine & ~text & ~shell_mask).astype(np.uint8)
    for s in S:
        if s.get("border"):   # vỏ viền: lòng là nền ảnh, không tìm chi tiết trong lòng
            continue
        inner = s["_mask"] & ~text & ov_fine & (np.linalg.norm(A - s["_fill_lab"], axis=2) > INNER_DE)
        inner = cv2.erode(s["_mask"].astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) & inner   # bỏ viền / bóng mép vỏ
        cand |= inner.astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, cc, st, _ = cv2.connectedComponentsWithStats(cand, 8)
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if area < ICON_MIN * W * H or max(w, h) < 8:
            continue
        cx, cy = x + w / 2, y + h / 2
        # mẩu nằm LỌT trong một dòng chữ hoặc vùng dấu thanh ngay trên dòng (nét / dấu sót mặt nạ) không phải chi tiết
        if any(l["box"][0] - 0.25 * (l["box"][3] - l["box"][1]) <= cx <= l["box"][2] + 0.25 * (l["box"][3] - l["box"][1])
               and l["box"][1] - 0.9 * (l["box"][3] - l["box"][1]) <= cy <= l["box"][3] + 0.25 * (l["box"][3] - l["box"][1])
               and max(w, h) <= 0.8 * (l["box"][3] - l["box"][1]) for l in lines):
            continue
        sel = cc == i
        # nét MẢNH (chỉ có trên mặt nạ mở nhẹ): icon nét mảnh designer đặt luôn đi kèm một dòng chữ cùng hàng (icon đầu dòng,
        # icon liên hệ); mảnh mảnh đứng lẻ giữa cảnh = cảnh FLUX vẽ lại (confetti, vệt sáng, mép vật)
        if ov.astype(bool)[sel].mean() < 0.2 and not any(
                l["box"][1] - 0.5 * (l["box"][3] - l["box"][1]) <= cy <= l["box"][3] + 0.5 * (l["box"][3] - l["box"][1])
                and max(l["box"][0] - (x + w), x - l["box"][2]) <= 2 * (l["box"][3] - l["box"][1]) for l in lines):
            continue
        shell = next((s["id"] for s in S if s["_mask"][sel].mean() > 0.5), None)
        col = np.median(draft[sel], axis=0)
        core = sel & ov.astype(bool)   # khung theo mặt nạ mở 5 px nếu chi tiết có nét đậm (mặt nạ mịn lấy thêm viền)
        if core.sum() >= 0.5 * sel.sum():
            ys, xs = np.nonzero(core)
            x, y, w, h = int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
        I.append({"id": f"I{len(I) + 1}", "box": [float(x), float(y), float(x + w), float(y + h)],
                  "color": "#%02x%02x%02x" % tuple(int(v) for v in col), "shell": shell})
    for s in S:
        s["lines"] = [j for j, l in enumerate(lines) if s["_mask"][int((l["box"][1] + l["box"][3]) / 2) % H,
                                                                  int((l["box"][0] + l["box"][2]) / 2) % W]]
        s["icons"] = [c["id"] for c in I if c["shell"] == s["id"]]
    # _mask / _fill_lab (mặt nạ vỏ, màu lòng Lab) giữ lại cho bước nắn khung chữ (slots); bỏ khi ghi JSON
    return {"S": S, "I": I, "mask": ov.astype(bool), "text": text}
