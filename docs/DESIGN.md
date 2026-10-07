# Cách làm

## Nguyên tắc

1. **Model quyết bố cục và hình.** FLUX.2-klein vẽ tốt chi tiết lớn (bố cục, ảnh, người, sản phẩm, vỏ to); vẽ sai chi tiết nhỏ
   cần độ chính xác cao (chữ, icon, avatar, sao). Ta giữ cái đúng, chỉ vẽ lại cái sai.
2. **Thẩm mỹ trước chính xác.** Khi nháp và prompt / câu khách lệch nhau, theo nháp. Chỗ model chừa ít chữ thì designer rút gọn
   câu khách cho vừa; dữ kiện (tên, giá, số, ngày, điện thoại, địa chỉ) không bao giờ sai, chỉ có thể bỏ (báo "thiếu").
3. **Code giữ toạ độ, VLM giữ ý nghĩa.** VLM không đưa toạ độ, chỉ trỏ id ô; code đo mọi khung.
4. **Tổng quát, không vá theo ca.** Mọi luật trong code và lời dặn phải là nguyên tắc chung.

## Các bước

### 1. Brief (`brief.py`)

Một lời gọi LLM: form / prompt -> `prompt_en` theo hướng dẫn nâng prompt FLUX.2 của BFL (mọi chữ trong ngoặc kép, đúng nguyên văn)
+ danh sách câu khách kèm vai trò. Code bỏ câu không có nguyên văn trong dữ liệu người dùng, thêm lại chuỗi trong ngoặc kép bị sót,
nối câu chưa có trong ngoặc kép của `prompt_en`, bỏ tỉ lệ / độ phân giải khỏi prompt.

### 2. Nháp và bản xoá (`flux.py`)

Nháp: FLUX.2-klein distill (4 bước); ảnh sản phẩm người dùng tải lên làm ảnh tham chiếu (FLUX giữ sản phẩm và chữ in trên đó rất
sát). Bản xoá: FLUX.2-klein sửa ảnh, nháp làm ảnh tham chiếu, lời dặn **ngắn, không nhắc tên vật**:
`Remove all text from this image. Keep everything else exactly the same.` Bản xoá được căn theo nháp (ECC affine).

**Bước xoá ngẫu nhiên** (đo 36 cặp `bench/prompts_gt.json`, 07/10): phần ảnh giữ gần như nguyên; nhưng
- ~8% dòng chữ còn nguyên (khó: 11%): chữ trong pill / badge màu, logo chữ, dòng liên hệ, hàng sao; đôi khi mẩu dấu ("^");
- vỏ (thẻ, badge, vòng tròn số, pill) **thường được giữ**, đôi khi bị xoá cả vỏ, không nhất quán giữa hai vỏ giống nhau;
- chữ in trên sản phẩm (từ ảnh tham chiếu) bị xoá ở 4/5 sản phẩm thử.

Thử 6 lời dặn khác (kể thêm nút / badge / icon / sao / logo, "giữ chữ trên sản phẩm", xoá hai lượt): không cái nào tốt hơn lời
dặn ngắn (lời dặn kể tên vật làm bản distill vẽ thêm / xoá sai). **Chốt lời dặn ngắn, luôn hậu xử lý** (bước 2b, 4).

### 2b. Dọn chữ sót (`cleanup.py`)

OCR bản xoá; dòng trùng vị trí một dòng của nháp = chữ chưa xoá. Xoá đúng các điểm **nét** (màu có trong lõi dòng, hiếm ở vành
sát dòng: chữ trắng trên huy hiệu đỏ, sao vàng trên thẻ trắng), không xoá cả khung dòng (khung nới tràn ra mép vỏ nhỏ). Nền quanh
nét phẳng / chuyển màu đều (pill, thẻ, badge) -> inpaint; nền có vân / ảnh -> FLUX xoá lại **vùng cắt** quanh các dòng đó (lời
dặn ngắn, phóng cạnh dài 768), dán về đúng vùng nét, mép làm mờ. Làm **trước** khi tách ô: chỗ vừa dọn thành lớp phủ, thành ô như
mọi chữ khác (hàng sao còn sót -> ô I -> `<i-stars>`). Chấp nhận: mẩu dấu lẻ OCR không đọc thành dòng, icon méo còn trong bản xoá.
36 cặp: 30 dòng sót -> 0 (OCR).

### 3. Ô (`overlay.py`, `slots.py`)

**nháp − bản xoá = lớp phủ**, đúng toạ độ, không cần bộ dò vật thể: ảnh / người / sản phẩm không bao giờ bị khoanh (bản xoá giữ
chúng).

- **Vỏ S#**: mảng lớp phủ to (≥ 0.4% ảnh), gọn, **đặc** (≥ 80% điểm đổi: cả lòng vỏ bị xoá; chữ trần chỉ đổi nét) và **thò ra**
  ngoài khung dòng chữ ≥ 15% (nét chữ rất đậm nằm gọn trong khung dòng). Đo màu lòng (màu chiếm nhiều nhất), bo góc (khoảng hở
  góc trên đường chéo), dòng / chi tiết nằm trong.
- **Chi tiết I#**: lớp phủ ngoài vỏ trừ chữ, và trong vỏ những điểm khác màu lòng vỏ. Bỏ mẩu lọt trong dòng / vùng dấu thanh,
  chi tiết đè lên chữ, chi tiết lồng nhau. Ghi quan hệ với dòng gần nhất (avatar cạnh tên, sao dưới tên, icon đầu dòng).
- **Dòng L#**: OCR (RapidOCR, đa giác 4 góc, góc nghiêng) cho cấu trúc dòng, chữ giun, số ký tự (sức chứa); **khung nắn theo lớp
  phủ** (mảng chạm đa giác OCR + mẩu cỡ dấu trong vùng nới: nét bay, dấu thanh, gạch đầu dòng; không lấy nét của dòng khác). Cỡ
  chữ từ cao thân chữ (chữ hoa 0.72 em, chữ thường 0.53 em), màu từ nét.

### 4. Designer (`slots.py`)

Một lời gọi VLM: prompt FLUX, câu khách, danh sách ô kèm số đo, 6 ảnh (nháp, nháp đánh dấu ô, 4 góc phóng 2×). VLM trả mỗi ô một
lệnh:

| Ô | Lệnh |
|---|---|
| S# | `shape` + CSS vỏ giống nháp (từ màu / bo góc đo được), hoặc `skip` |
| I# | `icon`: một thẻ `<i-icon name>` (Lucide, 1544 tên) / avatar `circle-user` / `<i-stars n>`, lấp đúng ô; hoặc `skip` (rác, chi tiết thuộc cảnh) |
| L# | `text` (được gộp các dòng liền nhau một khối), `keep` (chữ in trên sản phẩm / màn hình: dán lại pixel nháp), `skip` |
| mọi ô | `keep`: thứ thuộc ảnh (chữ / logo / hình in trên sản phẩm, biển trong cảnh) bị bản xoá xoá nhầm -> dán lại pixel NHÁP: khung ô + trọn các mảng lớp phủ chạm khung (vòng con dấu, phần logo ngoài dòng) |

Người dùng tải ảnh sản phẩm: lời nhắn designer nói rõ chữ / logo trên sản phẩm là `keep`, không vẽ lại. Chữ sản phẩm lấy từ
**nháp** (không dán ảnh tham chiếu: sản phẩm trong nháp đã được vẽ lại); nháp vẽ sai thì chấp nhận hạn chế model.

Kèm `style` (font tiêu đề, font thân trong bộ OFL), `missing` (câu khách không vẽ / bị bỏ).

### 5. Dựng và kiểm (`render.py`)

Nền = bản xoá (+ pixel nháp ở dòng `keep`). Mỗi lệnh một div đúng khung các ô của nó (dòng nghiêng: dựng thẳng rồi xoay), lớp vỏ ->
chi tiết -> chữ. VLM ghi cỡ px, code đổi sang em để vòng co cho vừa co đều. Thẻ tiện ích -> SVG Lucide / SVG sao / `@font-face`
nhúng sẵn. HTML được làm sạch (bỏ script, `on*`, URL, img).

Kiểm (khách quan, không cần mắt): ô chưa thuộc lệnh nào, ô dùng hai lần, khung / nội dung hai lệnh chồng nhau, tràn khung, phải co
dưới 70% (nhồi quá nhiều chữ). Có lỗi: **một** vòng VLM chỉ vá các lệnh / ô lỗi (kèm ảnh phóng vùng lỗi), giữ bản ít lỗi hơn.

## Giới hạn đã biết

- Vỏ **nửa trong suốt** trên nền gần cùng màu chênh quá ít, không thành S#: chữ trong đó vẫn được vẽ, nhưng thiếu vỏ; chữ sáng có
  thể chìm trên nền sáng (chưa có bước kiểm tương phản).
- Vỏ dính liền vỏ khác (thẻ + pill trong thẻ) thành một mảng không gọn có thể không được nhận.
- Bản xoá xoá cả chữ / nhãn thuộc cảnh và trên sản phẩm: chỉ cứu được qua `keep` của designer (OCR phải bắt được dòng).
- Dòng OCR không bắt được thì không thành ô (chưa tách dòng từ lớp phủ còn thừa).
- Chưa có mã QR.
