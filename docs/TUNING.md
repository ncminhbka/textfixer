# Theo dõi tune từng bước

Mỗi lượt tune ghi một dòng: ngày, tập dữ liệu, lỗi thấy, thay đổi, số đo trước -> sau, trạng thái.
Trạng thái: **đo** (có số đo trên tập) · **thử** (chỉ thử vài ca / dựng lại) · **chờ server** (chưa chạy thật) · **đề xuất** (chưa làm).
Tập: `dev2` = bench/dev2.json (chạy server 09/10, đủ 62 ảnh) · `gt` = bench/slots_gt (23 ảnh, đáp án ô).

## Các bước

| Bước | Việc | Mã | Núm chỉnh chính | Đo bằng |
|---|---|---|---|---|
| B0 Brief | LLM: intent (poster / ảnh thường), câu khách nguyên văn, prompt FLUX, các hướng thiết kế; code giữ cam kết | `textfix/brief.py` | `SYSTEM`, `FIELD_ROLE`, `ORIENT`, luật nguyên văn / ngoặc kép / trường form | brief.json: câu khách đủ / thừa / sai, `log` |
| B1 Nháp | FLUX.2-klein vẽ cả poster có chữ | `textfix/flux.py` | prompt (B0), seed, số bước | mắt: bố cục, số hàng danh sách đủ |
| B2 Xoá | FLUX xoá lớp phủ + dọn chữ sót (OCR bản xoá, inpaint / FLUX vùng cắt) | `flux.py`, `textfix/cleanup.py` | `ERASE_PROMPT`, ngưỡng dọn | `scripts/erase_eval.py`, mắt: chữ sót, vết loang |
| B2.5 Ô | nháp − bản xoá = lớp phủ -> L dòng chữ / S vỏ / I chi tiết | `textfix/overlay.py`, `slots.py` (build) | `DIFF`, `SHELL_*`, `SOFT_*`, `_ring` | `scripts/slots_gt.py eval --vs base` (F1 L/S/I) |
| B3 Designer | VLM điền ô: text / shape / icon / keep / skip; code chặn chữ | `slots.py` (plan, validate, _guard) | `SYSTEM` designer, `PRINTED`, luật _guard | plan.json: `missing`, log chặn chữ, `uncovered` |
| B4 Dựng + kiểm | Chromium dựng, đo lỗi, sửa bằng code (vòng duyệt VLM tắt) | `textfix/render.py`, `engine.py` | `CONDENSE`, `FIX_ROUNDS`, `LH_ACCENT`, `PARA_WORDS`, `CONTRAST_*` | `scripts/runs_report.py`: lỗi lượt đầu / cuối, sửa code giữ |

## Mốc dev2 (09/10, mã 3fe4022 trên server)

51 ảnh · lỗi lượt đầu: too_small 42, low_contrast 9, uneven_size 8, boxes_overlap 5, overflow 5, unassigned 3 ·
sửa bằng code 17 lần (giữ 12) · lỗi cuối: too_small 42, uneven_size 8, overflow 5, boxes_overlap 1 ·
câu khách thiếu: VLM khai 9 ảnh, đo bằng mã thêm 5 ảnh.

## Nhật ký

### B0 Brief

| Ngày | Tập | Lỗi thấy | Thay đổi | Trước -> sau | Trạng thái |
|---|---|---|---|---|---|
| 09/10 | dev2 | "tạo ảnh con mèo" báo lỗi "không tìm ra chữ" | `intent`; không câu khách -> chỉ FLUX vẽ | lỗi -> ảnh thường | thử (LLM giả) · chờ server (d33-d36) |
| 09/10 | — | form đủ + prompt sơ sài có thể làm LLM bỏ form | dặn: prompt sơ sài cạnh form vẫn là poster, mọi trường là chữ | — | chờ server (d37-d40) |
| 09/10 | dev2 d16, d32 | khung 16:9 mà prompt viết "vertical poster" | khung ảnh nói bằng lời (`ORIENT`) | — | thử (LLM giả) · chờ server |
| 09/10 | dev2 d06 | phong cách "Y2K", "vintage" trong ngoặc kép thành chữ khách -> thiếu câu | chuỗi LLM dùng làm từ tả (không ngoặc) trong prompt_en -> không thêm | 2 câu thừa -> 0 | thử (LLM giả) |
| 09/10 | dev2 d32 | LLM viết "Apply deadline: 15/12/2026" -> bị bỏ, mất hạn nộp | thêm lại giá trị trường form không có trong câu khách | mất 1 trường -> 0 | thử (LLM giả) |
| 09/10 | dev2 d01 | ngoặc kép cong “…” không được bắt lại khi LLM sót | nhận cả ngoặc cong | — | thử |
| 09/10 | dev2 d23 | form "Giảm 20%" và prompt "GIẢM 30%" mâu thuẫn: cả hai thành câu khách, poster không vẽ câu prompt | **prompt thắng** (quyết 09/10): LLM ghi trường form bị thay vào `overridden`, code bỏ giá trị đó, không thêm lại | 2 câu form thừa -> 0 | thử (LLM giả) · chờ server |

### B1 Nháp

| Ngày | Tập | Lỗi thấy | Thay đổi | Trước -> sau | Trạng thái |
|---|---|---|---|---|---|
| 09/10 | dev2 d19, d20 | hướng dẫn 5 / 8 bước: nháp vẽ 4 / 6-7 hàng -> mất bước, số bước nhảy (1 2 4 5) | **chấp nhận** (quyết 09/10): hạn chế của FLUX; câu thiếu đã được báo trên UI | — | chấp nhận |

### B2 Xoá

| Ngày | Tập | Lỗi thấy | Thay đổi | Trước -> sau | Trạng thái |
|---|---|---|---|---|---|
| — | dev2 d09, d22 | vết loang chỗ logo / huy hiệu sau dọn (poster thường che được) | — | — | theo dõi |

### B2.5 Ô

| Ngày | Tập | Lỗi thấy | Thay đổi | Trước -> sau | Trạng thái |
|---|---|---|---|---|---|
| 09/10 | dev2 | pill trắng / xám trên nền gần màu (ΔE 10-18 < 20) không thành S -> VLM vẽ pill ôm sát chữ, bé tí (d10, d13, d17) | lượt "vỏ nhạt" quanh từng dòng (`SOFT_*`) | dev2: +4 vỏ đúng ở 4 ảnh, 0 vỏ mất | đo (dev2) |
| 09/10 | dev2 d02 | nút chỉ có viền ("Shop now") không thành S | `_ring`: viền kín bao dòng, lòng không đổi | dev2: +2 vỏ đúng | đo (dev2) |
| 09/10 | gt m03_s1 | vỏ nhạt bắt nhầm pill mà bản xoá còn giữ (chỉ nhạt màu, ΔE lòng ~10 như vỏ thật) | bỏ vỏ khi bản xoá còn mép vỏ (`SOFT_EDGE`: độ sắc mép bản xoá / nháp >= 0.6; vỏ thật 0.01-0.39, ca nhầm 1.05) | gt S F1 0.800 -> 0.842 (= trước lượt vỏ nhạt: không thêm sai) | đo (gt + dev2) |
| — | dev2 d26, d12 | khung liên hệ tối trên nền tối, khung ngày nhiều dòng chưa thành S | dò vỏ quanh CỤM dòng | — | đề xuất |
| 09/10 | dev2 (62) | too_small: chữ dựng ở cỡ nháp tràn ngang ngay từ đầu (VLM ghi cỡ = cỡ nháp, trung vị 0.99) vì **cỡ nháp đo phồng**: dòng chữ thường nhiều chữ số đo ra cao chữ hoa rồi chia cao chữ thường ("Hotline 0866 777 888" 1.37 lần) | dòng không toàn hoa, có >= 15% ký tự cao: cỡ = cao nét cao (phân vị 90) / 0.72 | bộ chữ biết cỡ (315 dòng, 7 font): p90 cỡ ước / thật 1.37 -> 1.18, trung vị 1.02 | đo |
| 09/10 | dev2 (62) | too_small: khung dòng nắn theo lớp phủ co về một mẩu khi chữ nhỏ / nhạt ngoài vỏ ("25 Nguyễn Huệ, TP. Huế" 281 -> 30 px; 16 dòng) | khung nắn ngoài vỏ < 0.6 bề ngang OCR -> giữ bề ngang OCR (`SNAP_MIN_W`) | gt L F1 0.967 -> 0.976 | đo |
| 09/10 | dev2 (62) | (hai dòng trên gộp) | replay: lệnh VLM cũ, cỡ px nhân theo cỡ mới / cũ | too_small 50 -> 39, ảnh có lỗi 35 -> 28 | đo (replay) |

#### Đợt HÌNH HỌC CHỮ (mở 09/10): chữ cong / lượn sóng / to dần / một dòng nhiều cỡ

Mô hình chung: mỗi dòng = **đường chân** (chuỗi điểm, trái -> phải) + **hồ sơ cỡ**. OCR vẫn lo đọc chữ, khung, góc dòng thẳng /
nghiêng; `textfix/geo.py` dò hình dạng từ NÉT (chuỗi mảnh nét cạnh nhau cùng cỡ -> điểm theo lát -> khớp đường tròn / đa thức /
đường thẳng + cỡ đổi đều) và gộp các mẩu OCR / chi tiết I trên cùng đường chân thành một ô. Gộp mẩu cùng đường chân khác cỡ
(OCR tách "GIẢM" | "50%") thành ô có `runs`. Đo: `python scripts/geo_synth.py make --n 90 && python scripts/geo_synth.py eval`
(nền = bản xoá thật dev2, chữ vẽ bằng Chromium, đáp án tuyệt đối). **Tắt mặc định** (`TEXTFIX_GEO=on`) tới khi B3 + B4 xong.

| Ngày | Tập | Lỗi thấy | Thay đổi | Trước -> sau | Trạng thái |
|---|---|---|---|---|---|
| 09/10 | synth 60 | OCR cắt cung thành mẩu thẳng / bỏ hai đầu cung; tách dòng nhiều cỡ | `geo.arcs` (cung tròn) + `geo.runs` | cung: nhận 0% -> 59%, đúng 1 ô 30% -> 76%; nhiều cỡ: đúng 1 ô 5% -> 73% | đo (synth) |
| 09/10 | synth 90 | thêm lượn sóng, to dần | mô hình đường chân chung (cung / đa thức bậc 3 / thẳng + hồ sơ cỡ) | cung: có đường chân 60%, lệch 0.05 cỡ · sóng: 42%, lệch 0.08 (đa số coi là cung) · to dần: 25%, chiều đúng 100% · chéo: 100% đúng 1 ô, sai góc 0.5° · nhiều cỡ: 67% | đo (synth) |
| 09/10 | gt, dev2 | không được làm hỏng dữ liệu thật | — | gt L / S / I không đổi; dev2: nhận đúng cung d10 v0 (gộp L1 + 2 mẩu I đầu cung), 1 nhận nhầm nhiều cỡ (chữ trang trí 10-16 px -> chặn cỡ >= 24 px) | đo |
| — | — | dựng chữ theo đường chân (SVG textPath, cỡ từng ký tự), mô tả cho VLM | B4 + B3 | — | đang làm |
| — | dev3_geo | kiểm trên nháp FLUX thật | chạy server `bench/dev3_geo.json` | — | chờ server |

### B3 Designer

| Ngày | Tập | Lỗi thấy | Thay đổi | Trước -> sau | Trạng thái |
|---|---|---|---|---|---|
| 09/10 | dev2 d24 | chép chữ méo của nháp: "CUỐI TỬA TUẦN" | chặn chữ: bỏ chữ không có trong câu khách nào | dev2: 1 lệnh sửa, 0 nhầm / 51 ảnh | đo (dev2) |
| 09/10 | dev2 d25, d06 | chữ in trên sản phẩm VLM cho skip (tự ghi "printed logo on product") -> mất con dấu | skip có lý do "printed / on product" -> keep | dev2: 2 lệnh đổi, cả 2 đúng | đo (dev2) |
| 09/10 | dev2 d05 | danh sách bỏ trống hàng giữa (nháp vẽ thừa hàng) | dặn: điền hàng từ trên xuống, chỉ bỏ hàng thừa cuối | — | chờ server |
| 09/10 | dev2 | VLM bỏ câu mà không khai `missing` | đo câu thiếu bằng mã (`slots.uncovered`), cộng vào `missing` | +5 ảnh được báo thiếu đúng, 0 báo oan | đo (dev2) |

### B4 Dựng + kiểm

| Ngày | Tập | Lỗi thấy | Thay đổi | Trước -> sau | Trạng thái |
|---|---|---|---|---|---|
| 09/10 | dev2 d06 | "DROP" thành "DRQP": dấu sắc dòng dưới đâm vào chữ dòng trên (line-height 1.0) | line-height >= 1.15 khi dòng sau có chữ hoa mang dấu | hết | thử (dựng lại) |
| 09/10 | dev2 d05 | bảng giá "Bún bò tái40.000đ" (cột flex trong thẻ font) | thẻ bọc `display:contents`; không nén ngang khối flex | hai cột đúng, chữ to bằng nháp | thử (dựng lại) |
| 09/10 | dev2 d10 | emoji 🎁 thành ô vuông | bỏ emoji | hết | thử (dựng lại) |
| 09/10 | dev2 d11 | "40%" cam trên nền đỏ gần vô hình, không bị bắt | tương phản đo từng đoạn chữ | không bắt -> bắt + đổi màu | thử (dựng lại) |
| 09/10 | dev2 d17 | trích dẫn dài: mỗi dòng một chữ mồ côi, chữ co nhỏ | đoạn văn bỏ `<br>` theo nháp, tự ngắt | chữ to hơn, hết mồ côi | thử (dựng lại) |
| — | dev2 d12 v2, d10 | chữ uốn vòng cung bị cắt thành mảnh xoay lệch | dựng chữ cong (SVG textPath) khi góc OCR đổi dọc dòng | — | đề xuất |
| — | dev2 | too_small 42 / 51 ảnh vẫn lớn nhất | — | — | theo dõi |
| — | dev2 | overflow: sửa bằng code 0/5 | — | — | theo dõi |

## Lượt chạy server tiếp theo

Cập nhật mã (docs/SERVER.md), khởi động lại máy chủ và kernel, `%run scripts/dev_batch.py --resubmit` -> 8 ca mới d33-d40
(ảnh thường, prompt sơ sài) + các ca lỗi. Để so trước / sau trên cả 32 ca cũ: xoá `output/dev/dev2_manifest.json` rồi chạy lại
(seed cố định).
