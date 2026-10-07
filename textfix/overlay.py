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
    # chi tiết nhỏ: (a) mảng lớp phủ ngoài vỏ, không phải chữ; (b) trong vỏ, khác màu lòng vỏ, không phải chữ
    cand = (ov_fine & ~text & ~shell_mask).astype(np.uint8)
    for s in S:
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
