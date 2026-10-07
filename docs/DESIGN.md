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

Nháp: FLUX.2-klein distill (4 bước). Bản xoá: FLUX.2-klein sửa ảnh, nháp làm ảnh tham chiếu, lời dặn **ngắn, không nhắc tên vật**:
`Remove all text from this image. Keep everything else exactly the same.` Lời dặn dài kể "sản phẩm, người, thẻ, badge…" làm model
distill vẽ thêm đúng thứ được nhắc (thử 07/10: thiệp cưới hiện sản phẩm lạ, tờ tuyển dụng hiện người). Lời dặn ngắn trên 12 nháp ×
2 seed: phần ảnh ngoài chữ đổi ~0%; bản xoá bỏ toàn bộ lớp phủ (chữ, thẻ, pill, dải, icon, avatar, sao), giữ ảnh. Bản xoá được
căn theo nháp (ECC affine).

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
- Bản xoá xoá cả chữ / nhãn nhỏ thuộc cảnh (nhãn trên đồ vật): hiện thành I#, designer nên `skip`.
- Dòng OCR không bắt được thì không thành ô (chưa tách dòng từ lớp phủ còn thừa).
- Chưa có mã QR.
