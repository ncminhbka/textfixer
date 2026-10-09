# Theo dõi tune từng bước

Mỗi lượt tune ghi một dòng: ngày, tập, lỗi thấy, thay đổi, số đo trước -> sau, trạng thái.
Trạng thái: **đo** (có số đo trên cả tập) · **thử** (chỉ thử vài ca / dựng lại / LLM giả) · **chờ server** (cần LLM / VLM / FLUX
thật để biết) · **đề xuất** (chưa làm) · **chấp nhận** (quyết không sửa).
Loại sửa: **[code]** luật / số đo trong mã · **[prompt LLM]** system prompt B0 · **[prompt VLM]** system prompt designer B3.
Tập: `dev2` = bench/dev2.json (chạy server 09/10, đủ 62 ảnh / 32 ca) · `gt` = bench/slots_gt (23 ảnh, đáp án ô) · `synth` =
scripts/geo_synth.py (chữ cong / nghiêng / nhiều cỡ tổng hợp, đáp án tuyệt đối).

## Các bước

| Bước | Việc | Mã | Có AI | Núm chỉnh chính | Đo bằng |
|---|---|---|---|---|---|
| B0 Brief | intent (poster / ảnh thường), câu khách nguyên văn, prompt FLUX, các hướng thiết kế; code giữ cam kết | `textfix/brief.py` | LLM | `SYSTEM`, `FIELD_ROLE`, `ORIENT`, luật nguyên văn / ngoặc kép / trường form / `overridden` | brief.json: câu khách đủ / thừa / sai, `log` |
| B1 Nháp | FLUX.2-klein vẽ cả poster có chữ | `textfix/flux.py` | FLUX | prompt (B0), seed, số bước | mắt: bố cục, số hàng danh sách |
| B2 Xoá | FLUX xoá lớp phủ + dọn chữ sót (OCR bản xoá, inpaint / FLUX vùng cắt, 2 lượt) | `flux.py`, `textfix/cleanup.py` | FLUX + OCR | `ERASE_PROMPT`, `MATCH`, `CONFIRM`, `TEXTURE` | OCR bản xoá đã dọn: chữ còn sót; mắt: vết loang |
| B2.5 Ô | nháp − bản xoá = lớp phủ -> L dòng chữ / S vỏ / I chi tiết (+ hình học chữ, đang tắt) | `textfix/overlay.py`, `slots.py` (lines, build), `glyph.py`, `geo.py` | OCR | `DIFF`, `SHELL_*`, `SOFT_*`, `_ring`, `SNAP_MIN_W`, `TALL_MIN` | `scripts/slots_gt.py eval --vs base`, `scripts/geo_synth.py eval` |
| B3 Designer | VLM điền ô: text / shape / icon / keep / skip; code chặn chữ | `slots.py` (plan, validate, _guard, uncovered) | VLM | `SYSTEM` designer, `PRINTED`, luật `_guard` | plan.json: `missing`, log chặn chữ |
| B4 Dựng + kiểm | Chromium dựng, đo lỗi, sửa bằng code (vòng duyệt VLM tắt) | `textfix/render.py`, `engine.py` | — | `CONDENSE`, `FIX_ROUNDS`, `LH_ACCENT`, `PARA_WORDS`, `CONTRAST_*` | `scripts/runs_report.py`: lỗi lượt đầu / cuối |

## Mốc dev2 (09/10, mã 3fe4022 trên server, đủ 62 ảnh)

| | Số |
|---|---|
| Lỗi lượt đầu | too_small 50 · low_contrast 12 · overflow 8 · uneven_size 8 · boxes_overlap 6 · unassigned 3 |
| Sửa bằng code | chạy 20 lần, giữ 14 (vòng duyệt VLM tắt: 0) |
| Lỗi cuối | too_small 50 · overflow 7 · uneven_size 8 · boxes_overlap 1 |
| Câu khách thiếu | VLM tự khai ở 9 ảnh; đo bằng mã thêm 5 ảnh VLM bỏ mà không khai |
| Thời gian trung vị (s) | LLM 6.8 · nháp 2.7 · xoá 4.8 · dọn 1.0 · ô 0.6 · VLM 9.6 · dựng 1.8 |

Xem bằng mắt cả 62 ảnh (nháp | bản xoá | poster). Lỗi theo bước, số ảnh dính (một ảnh có thể dính nhiều):

| Bước | Lỗi | Ảnh |
|---|---|---|
| B0 | khung 16:9 viết "vertical poster" (d16, d32) · chữ phong cách thành câu khách (d06) · trường form bị viết lại rồi bị bỏ (d32) · form / prompt mâu thuẫn (d23) | 6 |
| B1 | hướng dẫn vẽ thiếu hàng (d19, d20) | 3 |
| B2 | bản xoá còn sót chữ, chữ mới dựng chồng lên (d18 v1 "Gửi CV…", d27 v1 "-36") | 2 |
| B2.5 | pill / nút nhạt hoặc chỉ có viền không thành ô -> pill vẽ lại bé tí (d02, d10, d13, d17) · khung dòng chữ nhỏ co về một mẩu + cỡ nháp đo phồng -> chữ co nhỏ (too_small, 28-35 ảnh) · chữ cong bị cắt mẩu (d10, d12 v2) · khung tối nhiều dòng không thành ô (d26, d12) | ~35 |
| B3 | chép chữ méo của nháp (d24 "TỬA") · viết chữ lên màn hình sản phẩm (d27 "00:36", "GIÁ", "VND") · viết tắt địa chỉ (d12 "Q.3") · bỏ chữ in trên sản phẩm (d25 con dấu) · bỏ câu không khai (5 ảnh) · bỏ trống hàng giữa danh sách (d05) · thẻ trong thẻ (d18) · đổi căn giữa thành căn trái (d30) · mất hiệu ứng vàng kim (d02 v1) | ~15 |
| B4 | dấu chồng dòng trên "DRQP" (d06) · cột tên / giá dính (d05) · emoji ô vuông (d10) · chữ chìm nền không bắt (d11) · trích dẫn mồ côi chữ (d17) | 5 |

## Đã siết prompt LLM / VLM chưa

| Prompt | Đã thêm (09/10) | Bằng chứng dev2 | Kiểm |
|---|---|---|---|
| **LLM B0** | `intent` poster / image; prompt sơ sài cạnh form đã điền vẫn là poster, mọi trường là chữ; prompt mâu thuẫn form -> prompt thắng, ghi `overridden`; khung ảnh nói bằng lời (gợi ý trong tin nhắn) | d23, d16, d32, ca d33-d40 | LLM giả · **chờ server** |
| **VLM B3** | điền hàng danh sách từ trên xuống, chỉ bỏ hàng thừa cuối; màn hình / nhãn / bao bì sản phẩm luôn keep, không viết chữ / số lên đó; dữ kiện không viết tắt ("Quận 3" không thành "Q.3"); dòng trong vỏ không tự thêm nền / viền (không thẻ trong thẻ); giữ căn lề của nháp từng khối | d05, d27, d12, d18, d30 | **chờ server** (VLM ở máy ngoài, không chạy local) |
| VLM B3, chưa đổi | luật "chữ model tự bịa: viết theo vai trò" vẫn giữ -- cho phép nhãn chung như "Khai Trương" khi khách không ghi tiêu đề (d16, nhìn ổn); dữ kiện vẫn cấm | d16 | **cần quyết** nếu muốn cấm hẳn |

Phần lớn lỗi VLM được chặn bằng **[code]** (chắc chắn, đo lại được trên plan cũ); prompt chỉ thêm khi code không biết được (vùng
nào là màn hình sản phẩm, căn lề mong muốn).

## Nhật ký

### B0 Brief

| Ngày | Tập | Lỗi thấy | Thay đổi | Trước -> sau | Trạng thái |
|---|---|---|---|---|---|
| 09/10 | — | "tạo ảnh con mèo" báo lỗi "không tìm ra chữ" | [prompt LLM] `intent`; [code] không câu khách -> chỉ FLUX vẽ | lỗi -> ảnh thường | thử (LLM giả) · chờ server (d33-d36) |
| 09/10 | — | form đủ + prompt sơ sài có thể làm LLM bỏ form | [prompt LLM] prompt sơ sài cạnh form vẫn là poster, mọi trường là chữ | — | chờ server (d37-d40) |
| 09/10 | dev2 d16, d32 | khung 16:9 mà prompt viết "vertical poster" | [code] khung ảnh nói bằng lời trong tin nhắn (`ORIENT`) | — | thử · chờ server |
| 09/10 | dev2 d06 | phong cách "Y2K", "vintage" trong ngoặc kép thành câu khách -> báo thiếu câu | [code] chuỗi LLM dùng làm từ tả (không ngoặc) trong prompt_en -> không thêm lại | 2 câu thừa -> 0 | thử (LLM giả) |
| 09/10 | dev2 d32 | LLM viết "Apply deadline: 15/12/2026" -> bị bỏ, mất hạn nộp | [code] thêm lại giá trị trường form không có trong câu khách | mất 1 trường -> 0 | thử (LLM giả) |
| 09/10 | dev2 d01 | ngoặc kép cong “…” không được bắt lại khi LLM sót | [code] nhận cả ngoặc cong | — | thử |
| 09/10 | dev2 d23 | form "Giảm 20%", prompt "GIẢM 30%": cả hai thành câu khách, poster không vẽ câu prompt | **prompt thắng** (quyết 09/10): [prompt LLM] ghi trường bị thay vào `overridden`; [code] bỏ giá trị đó, không thêm lại | 2 câu form thừa -> 0 | thử (LLM giả) · chờ server |

### B1 Nháp

| Ngày | Tập | Lỗi thấy | Thay đổi | Trước -> sau | Trạng thái |
|---|---|---|---|---|---|
| 09/10 | dev2 d19, d20 | hướng dẫn 5 / 8 bước: nháp vẽ 4 / 6-7 hàng -> mất bước, số bước nhảy (1 2 4 5) | hạn chế của FLUX; câu thiếu đã báo trên UI | 3/4 ảnh hướng dẫn | **chấp nhận** (quyết 09/10) |

### B2 Xoá

| Ngày | Tập | Lỗi thấy | Thay đổi | Trước -> sau | Trạng thái |
|---|---|---|---|---|---|
| 09/10 | dev2 d18 v1 | OCR nháp sót dòng "Gửi CV: tuyendung@…" -> chữ còn trên bản xoá không bị coi là chữ sót -> poster có chữ rác | [code] dòng bản xoá không khớp dòng nháp: OCR lại vùng đó trên nháp (cắt, phóng 2×) thấy cùng chữ -> chữ sót (`CONFIRM`); tắt khi khách tải ảnh sản phẩm | dev2: +3 dòng sót được dọn (d18, d09 số điện thoại, d19) | đo (bản xoá thô, inpaint) · chờ server (FLUX) |
| 09/10 | dev2 d27 v1 | inpaint "-36%" trên huy hiệu không sạch, còn "-36" -> chữ mới dựng chồng | [code] lượt 2: OCR lại bản đã dọn, chữ còn -> FLUX vùng cắt | — | chờ server (lượt 2 cần FLUX) |
| — | dev2 d09, d22 | vết loang chỗ logo / huy hiệu sau dọn (poster thường che được) | — | — | theo dõi |

### B2.5 Ô

| Ngày | Tập | Lỗi thấy | Thay đổi | Trước -> sau | Trạng thái |
|---|---|---|---|---|---|
| 09/10 | dev2 | pill trắng / xám trên nền gần màu (ΔE 10-18 < 20) không thành S -> VLM vẽ pill ôm sát chữ, bé tí (d10, d13, d17, d32) | [code] lượt "vỏ nhạt" quanh từng dòng (`SOFT_*`) | dev2: +5 vỏ đúng, 0 vỏ mất | đo (dev2) |
| 09/10 | dev2 d02 | nút chỉ có viền ("Shop now") không thành S | [code] `_ring`: viền kín bao dòng, lòng không đổi | dev2: +2 vỏ đúng | đo (dev2) |
| 09/10 | gt m03_s1 | vỏ nhạt bắt nhầm pill mà bản xoá còn giữ (chỉ nhạt màu) | [code] bỏ vỏ khi bản xoá còn mép vỏ (`SOFT_EDGE` 0.6; vỏ thật 0.01-0.39, ca nhầm 1.05) | gt S F1 0.800 -> 0.842 (= trước lượt vỏ nhạt) | đo (gt + dev2) |
| 09/10 | dev2 (62) | too_small: VLM ghi cỡ = cỡ nháp (trung vị 0.99) mà chữ vẫn tràn ngang ngay từ đầu vì **cỡ nháp đo phồng**: dòng chữ thường nhiều chữ số đo ra cao chữ hoa rồi chia cao chữ thường ("Hotline 0866 777 888" 1.37×) | [code] dòng không toàn hoa có >= 15% ký tự cao: cỡ = cao nét cao (p90) / 0.72 | synth chữ biết cỡ (315 dòng, 7 font): p90 cỡ ước / thật 1.37 -> 1.18 | đo |
| 09/10 | dev2 (62) | too_small: khung dòng nắn theo lớp phủ co về một mẩu khi chữ nhỏ / nhạt ngoài vỏ ("25 Nguyễn Huệ, TP. Huế" 281 -> 30 px; 16 dòng) | [code] khung nắn ngoài vỏ < 0.6 bề ngang OCR -> giữ bề ngang OCR (`SNAP_MIN_W`) | gt L F1 0.967 -> 0.976 | đo |
| 09/10 | dev2 (62) | (hai dòng trên gộp lại) | replay: lệnh VLM cũ, cỡ px nhân theo cỡ mới / cũ | too_small 50 -> 39, ảnh có lỗi 35 -> 28 | đo (replay) |
| — | dev2 d26, d12 | khung tối chứa nhiều dòng / khung ngày không thành S | dò vỏ quanh CỤM dòng | — | đề xuất |
| — | dev2 d06 v0 | vỏ dính chữ to bên cạnh (pill "Freeship" + "199K" thành một S) -> VLM vẽ thẻ trắng phủ cả giá | tách vỏ khỏi dòng chữ lớn kề | — | đề xuất |

#### Đợt HÌNH HỌC CHỮ (mở 09/10): chữ cong / lượn sóng / to dần / một dòng nhiều cỡ

Mô hình chung: mỗi dòng = **đường chân** (chuỗi điểm, trái -> phải) + **hồ sơ cỡ**. OCR vẫn lo đọc chữ, khung, góc dòng thẳng /
nghiêng; `textfix/geo.py` dò hình dạng từ NÉT (chuỗi mảnh nét cạnh nhau cùng cỡ -> điểm theo lát -> khớp đường tròn / đa thức /
đường thẳng + cỡ đổi đều) và gộp các mẩu OCR / chi tiết I trên cùng đường chân thành một ô; gộp mẩu cùng đường chân khác cỡ (OCR
tách "GIẢM" | "50%") thành ô có `runs`. **Tắt mặc định** (`TEXTFIX_GEO=on`) tới khi B3 + B4 xong.

| Ngày | Tập | Lỗi thấy | Thay đổi | Trước -> sau | Trạng thái |
|---|---|---|---|---|---|
| 09/10 | synth 60 | OCR cắt cung thành mẩu thẳng / bỏ hai đầu cung; tách dòng nhiều cỡ | [code] `geo.arcs` + `geo.runs` | cung nhận 0% -> 59%, đúng 1 ô 30% -> 76%; nhiều cỡ đúng 1 ô 5% -> 73% | đo (synth) |
| 09/10 | synth 90 | thêm lượn sóng, to dần | [code] mô hình đường chân chung | cung: có đường chân 60%, lệch 0.05 cỡ · sóng 42% (lệch 0.08, đa số coi là cung) · to dần 25% (chiều đúng 100%) · chéo 100% đúng 1 ô, sai góc 0.5° · nhiều cỡ 67% | đo (synth) |
| 09/10 | gt, dev2 | không được làm hỏng dữ liệu thật | — | gt L / S / I không đổi; dev2: nhận đúng cung d10 v0; 1 nhận nhầm nhiều cỡ (chữ trang trí 10-16 px -> chặn cỡ >= 24 px) | đo |
| — | — | dựng chữ theo đường chân (SVG textPath, cỡ từng ký tự), mô tả cho VLM | B4 + [prompt VLM] | — | đang làm |
| — | dev3_geo | kiểm trên nháp FLUX thật | `%run scripts/dev_batch.py bench/dev3_geo.json` | — | chờ server |

### B3 Designer

| Ngày | Tập | Lỗi thấy | Thay đổi | Trước -> sau | Trạng thái |
|---|---|---|---|---|---|
| 09/10 | dev2 d24 | chép chữ méo của nháp: "CUỐI TỬA TUẦN" | [code] lệnh viết câu khách: bỏ chữ không có trong câu khách nào | 1 lệnh sửa, 0 nhầm / 62 ảnh | đo (dev2) |
| 09/10 | dev2 d25, d06 | chữ in trên sản phẩm VLM cho skip (tự ghi "printed logo on product") -> mất con dấu | [code] skip có lý do "printed / on product" -> keep | 2 lệnh đổi, cả 2 đúng | đo (dev2) |
| 09/10 | dev2 d27 v0 | VLM viết "00:36", "GIÁ", "VND" lên màn hình nồi cơm | [code] số trong lệnh không khai câu khách phải trùng NGUYÊN số của câu khách ("00" không còn khớp chuỗi con của "1.590.000đ"); [prompt VLM] màn hình / nhãn / bao bì sản phẩm luôn keep / skip | "00:36" bị chặn; "GIÁ", "VND" chờ prompt | đo (code) · chờ server (prompt) |
| 09/10 | dev2 d12 v1 | viết tắt địa chỉ "Quận 3" -> "Q.3" | [prompt VLM] dữ kiện không viết tắt; [code] đo câu thiếu bắt được (báo "Không vẽ") | — | chờ server |
| 09/10 | dev2 | VLM bỏ câu mà không khai `missing` | [code] đo câu thiếu bằng mã (`slots.uncovered`), cộng vào `missing` | +5 ảnh được báo đúng, 0 báo oan | đo (dev2) |
| 09/10 | dev2 d05 | danh sách bỏ trống hàng giữa (nháp vẽ thừa hàng) | [prompt VLM] điền hàng từ trên xuống, chỉ bỏ hàng thừa cuối | — | chờ server |
| 09/10 | dev2 d18 v0 | thẻ trắng lồng trong thẻ trắng | [prompt VLM] dòng trong vỏ không tự thêm nền / viền | — | chờ server |
| 09/10 | dev2 d30 v0 | khối chữ căn giữa ở nháp thành căn trái | [prompt VLM] giữ căn lề từng khối | — | chờ server |
| — | dev2 d02 v1 | tiêu đề vàng kim thành trắng phẳng | luật đã có ("background-clip:text … gold"), VLM không theo | — | theo dõi |
| — | dev2 d16 | VLM tự viết nhãn "Khai Trương" (khách không ghi tiêu đề) | luật hiện cho phép nhãn chung, cấm dữ kiện | — | **cần quyết** |

### B4 Dựng + kiểm

| Ngày | Tập | Lỗi thấy | Thay đổi | Trước -> sau | Trạng thái |
|---|---|---|---|---|---|
| 09/10 | dev2 d06 | "DROP" thành "DRQP": dấu sắc dòng dưới đâm vào chữ dòng trên (line-height 1.0) | [code] line-height >= 1.15 khi dòng sau có chữ hoa mang dấu | hết | thử (dựng lại) |
| 09/10 | dev2 d05 | bảng giá "Bún bò tái40.000đ" (cột flex trong thẻ font) | [code] thẻ bọc `display:contents`; không nén ngang khối flex | hai cột đúng, chữ to bằng nháp | thử (dựng lại) |
| 09/10 | dev2 d10 | emoji 🎁 thành ô vuông | [code] bỏ emoji | hết | thử (dựng lại) |
| 09/10 | dev2 d11 | "40%" cam trên nền đỏ gần vô hình, không bị bắt | [code] tương phản đo từng đoạn chữ | không bắt -> bắt + đổi màu | thử (dựng lại) |
| 09/10 | dev2 d17 | trích dẫn dài: mỗi dòng một chữ mồ côi, chữ co nhỏ | [code] đoạn văn bỏ `<br>` theo nháp, tự ngắt | chữ to hơn, hết mồ côi | thử (dựng lại) |
| — | dev2 | overflow: sửa bằng code 0/7 | — | — | theo dõi |

## Lượt chạy server tiếp theo

Cập nhật mã (docs/SERVER.md), khởi động lại máy chủ **và kernel**, rồi:
1. `%run scripts/dev_batch.py bench/dev3_geo.json` -- tập hình học chữ (baseline cho đợt chữ cong).
2. Để so trước / sau dev2 (prompt LLM / VLM mới, B2 hai lượt, cỡ / khung mới): xoá `output/dev/dev2_manifest.json` rồi
   `%run scripts/dev_batch.py` -- chạy lại cả 40 ca (seed cố định; d33-d40 là ca mới).
