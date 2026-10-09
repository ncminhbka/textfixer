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

LLM cũng phân loại `intent`: `poster` (có chữ cần in) hay `image` (chỉ muốn ảnh: "tạo ảnh con mèo"). Code quyết theo câu khách:
có câu khách -> `mode: poster`, chạy cả dây chuyền; không có -> `mode: image`, **chỉ FLUX vẽ, trả nháp** (bỏ xoá / ô / VLM / dựng;
không bịa chữ). LLM nói `poster` mà không có câu khách ("làm poster cho quán của tôi"): prompt thêm "no text", UI gợi ý ghi câu
trong ngoặc kép.

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

Luật chung (chỉnh trên dev, kiểm một lần trên test của `bench/slots_gt`, chia theo prompt; docs/SLOTS_GT.md):
- "dòng" OCR không có chữ cái / chữ số (hàng sao) không phải chữ -> lớp phủ tách thành chi tiết;
- tách dòng OCR thành mục riêng tại khoảng không nét >= 1 cao dòng, hoặc tại chỗ OCR chèn >= 2 dấu cách (khối hẹp kẹp giữa =
  icon, để riêng);
- chi tiết tìm trên mặt nạ mở nhẹ 3 px (giữ icon nét mảnh), vỏ / khung chữ trên mặt nạ 5 px; nét mảnh đứng lẻ (không kèm dòng
  chữ cùng hàng trong 2 cao dòng) = cảnh vẽ lại, bỏ; khung chi tiết theo mặt nạ 5 px khi có nét đậm;
- chi tiết giống nhau cùng hàng cách nhau <= 0.5 cao = một chi tiết (hàng sao); chi tiết nhỏ hơn 0.5 cao dòng trung vị = vụn.
Chữ số to OCR bỏ sót (số bước trong vòng tròn) thành chi tiết: designer được dùng lệnh `text` cho ô I.

| Lớp (IoU >= 0.5) | dev F1 trước -> sau | test F1 trước -> sau |
|---|---|---|
| L | 0.950 -> 0.954 | 0.917 -> 0.926 |
| S | 0.842 -> 0.842 | 1.000 -> 1.000 |
| I | 0.505 -> 0.681 | 0.419 -> 0.717 |

### 4. Designer (`slots.py`)

Một lời gọi VLM: prompt FLUX, câu khách, danh sách ô kèm số đo, 3 ảnh: nháp, nháp đánh dấu ô, **bản xoá đã dọn** (nền thật sẽ dựng lên: vỏ nào còn, nền sau mỗi dòng). Chốt 07/10 bằng scripts/vlm_probe.py trên 23 ảnh dev, so 4 cấu hình (6 ảnh cũ / +bản xoá / 3 ảnh / vùng cắt quanh cụm ô): 3 ảnh ít lỗi nhất (56 so với 61-71), giữ chữ sản phẩm tốt nhất (29/30 ô P được keep), rẻ nhất (~12k token so với ~16k), giữ dáng chữ nháp tốt hơn; ảnh phóng 2× không giúp. VLM trả mỗi ô một
lệnh:

| Ô | Lệnh |
|---|---|
| S# | `shape` + CSS vỏ giống nháp (từ màu / bo góc đo được), hoặc `skip` |
| I# | `icon`: một thẻ `<i-icon name>` (Lucide, 1544 tên) / avatar `circle-user` / `<i-stars n>`, lấp đúng ô; hoặc `skip` (rác, chi tiết thuộc cảnh) |
| L# | `text` (được gộp các dòng liền nhau một khối), `keep` (chữ in trên sản phẩm / màn hình: dán lại pixel nháp), `skip` |
| mọi ô | `keep`: thứ thuộc ảnh (chữ / logo / hình in trên sản phẩm, biển trong cảnh) bị bản xoá xoá nhầm -> dán lại pixel NHÁP: khung ô + trọn các mảng lớp phủ chạm khung (vòng con dấu, phần logo ngoài dòng) |

Người dùng tải ảnh sản phẩm: lời nhắn designer nói rõ chữ / logo trên sản phẩm là `keep`, không vẽ lại. Chữ sản phẩm lấy từ
**nháp** (không dán ảnh tham chiếu: sản phẩm trong nháp đã được vẽ lại); nháp vẽ sai thì chấp nhận hạn chế model.

Lời nhắn kèm kiểu chữ đo trên nháp từng dòng (độ đậm theo độ dày nét / cỡ, nét dày-mảnh, viết hoa, bóng / viền: glyph.py). System prompt dặn TÁI TẠO phong cách nháp (font cùng dáng, CSS độ đậm / nghiêng / giãn chữ / viền / bóng / chữ chuyển màu-kim loại), bảng màu 3-5 màu rút từ nháp (mọi màu lấy từ đó), phân cấp tiêu đề > giá / ưu đãi / nút > thân, tương phản với nền thật (bản xoá). Kèm `style` (font tiêu đề, font thân trong bộ OFL), `missing` (câu khách không vẽ / bị bỏ).

### 5. Dựng và kiểm (`render.py`)

Nền = bản xoá (+ pixel nháp ở dòng `keep`). Mỗi lệnh một div đúng khung các ô của nó (dòng nghiêng: dựng thẳng rồi xoay), lớp vỏ ->
chi tiết -> chữ. VLM ghi cỡ px, code đổi sang em để vòng co cho vừa co đều. Thẻ tiện ích -> SVG Lucide / SVG sao / `@font-face`
nhúng sẵn. HTML được làm sạch (bỏ script, `on*`, URL, img).

Đo tràn: khung ô ôm nét mực, hộp chữ trình duyệt = ascent + descent của font -> trừ phần đệm trên / dưới (measureText trên chính chữ của khối), không thì mọi dòng bị co ~0.6-0.8 (lỗi 07/10: 80% lỗi too_small, chữ nhỏ hơn nháp). Nạp hết font trước khi đo.

Mặc định khi VLM không ghi: dòng cao >= 1.4 cỡ trung vị dùng `display_font`, độ đậm = độ đậm đo trên nháp (trước 08/10 dựng 400:
tiêu đề đậm thành mảnh); khoá font viết sai / tên họ font khớp về khoá gần nhất, font ghi thẳng bằng CSS cũng được nạp; không giả
đậm font thiếu độ đậm. Dòng căn trái cùng cỡ thẳng cột: mép trái cả cột = mép nhỏ nhất (khung OCR thò thụt).

Chặn chữ trước khi dựng (`slots.validate` + câu khách, 08/10): chữ khác câu khách chỉ ở dấu (TRƯỞNG / TRƯỜNG cho TRƯƠNG) -> thay đúng
chữ khách; lệnh khai câu khách mà không chữ nào có trong câu khách (chép mẩu OCR vỡ) -> skip; hai lệnh cùng một chữ mà câu khách
không lặp (nháp vẽ một dòng hai lần) -> giữ lệnh có OCR giống nhất. Câu khách ngắn (<= 3 chữ: lương, giá, nhãn) không được cắt
bớt (vòng sửa từng bỏ "triệu" cho vừa ô) -> trả nguyên câu.

Độ rộng chữ: mỗi dòng ghi độ rộng nét chữ đo trên nháp (em / ký tự), danh mục font ghi độ rộng từng font (`fonts.WIDTH`, đo bằng
`scripts/font_widths.py`) -- chọn font rộng hơn chữ nháp là nguyên nhân chính của too_small (08/10). Chữ tô gradient: text-shadow đổi
thành filter:drop-shadow (bóng lộ qua chữ trong suốt làm vàng thành nâu).

Chọn bản sau vòng sửa (`slots.score`): nhiều câu khách được viết hơn trước, rồi ít lỗi hơn (vòng sửa từng bỏ tiêu đề để hết lỗi).

Kiểm (khách quan, không cần mắt): ô chưa thuộc lệnh nào, ô dùng hai lần, khung / nội dung hai lệnh chồng nhau, tràn khung, phải co
dưới 70% (nhồi quá nhiều chữ), **tương phản** (dựng lại lần hai với chữ trong suốt = nền thật sau chữ; > 25% nền có tỉ lệ WCAG < 2
với màu chữ, khối không có bóng / viền -> low_contrast), **cỡ không đều** (dòng cùng vai: model vẽ cùng cỡ, cùng font, thẳng cột /
cùng hàng, gần nhau; cỡ dựng lệch > 25% -> uneven_size). Renderer tự sửa trước khi duyệt: dòng cùng vai dựng cùng cỡ (cỡ nhỏ nhất nhóm), icon cùng hàng / cột cùng cỡ, chữ là toàn bộ
nội dung một vỏ thì căn giữa dọc theo vỏ, font-size ở box_style là cỡ gốc của khung.

**Nén ngang** (`render.CONDENSE` = 0.85): chữ dài hơn ô theo chiều ngang được scaleX tới 0.85 trước khi co cỡ (giữ chiều cao
chữ như nháp; replay probe4: cỡ dựng / cỡ nháp trung vị 0.79 -> 0.84, too_small 31 -> 24).

**Sửa bằng code** (`render.autofix`, 08/10 -- thay vòng duyệt VLM): lượt đầu có lỗi nặng -> sửa phần đo được, dựng lại, giữ khi
bớt lỗi (tối đa `FIX_ROUNDS` = 2). mark_twice: ô giữ ở lệnh đầu; unassigned: ô bỏ sót = skip; low_contrast: màu chữ thay thế
(màu chữ khác của poster, rồi màu nháp tối / sáng dần, rồi trắng / gần đen; nền loang thêm quầng); overflow: nén ngang tới 0.75;
boxes_overlap / content_overlap: lệnh lấn cắt khung một phía (chữ xoay: thu khung 0.85 quanh tâm). Chồng nội dung của chữ xoay đo
bằng đa giác nét thật (hộp thẳng ôm chữ xoay chồng oan dòng kề). Replay probe4: lỗi nặng 5 -> 1 (còn lại: VLM gộp dòng xen kẽ vào
hai lệnh, code không gỡ được).

**Vòng duyệt** (`slots.review`). Từ 08/10 mặc định `REVIEW_MODE = "off"` (lỗi nặng do code sửa; bật lại `TEXTFIX_REVIEW=errors`
nếu quan sát thấy code không cứu được). Chế độ `"errors"`: MỘT lượt, chỉ khi lượt đầu có lỗi nặng (ô bỏ sót, chồng,
tràn, chìm nền), chỉ sửa đúng các lỗi đó -- lượt 3-4 (08/10) cho thấy duyệt thẩm mỹ tốn ~20 s / poster mà ~2 / 15 poster đẹp hơn,
~5 xấu đi. Chế độ `"full"` (`TEXTFIX_REVIEW=full`, `vlm_probe.py --review full`) giữ lại để thử: luôn chạy, tối đa `REVIEW_ROUNDS` = 2 lượt (dừng khi VLM không
sửa gì). VLM là giám đốc nghệ thuật xem bản dựng thật: nháp, bản dựng, bản dựng có id lệnh, ảnh so sánh nháp | bản dựng ở các cụm
ô (phóng), danh sách lệnh kèm cỡ nháp -> cỡ / độ đậm / font dựng thật, **nhật ký thay đổi của code** (chặn chữ, căn cột, cân cỡ...:
không được đảo ngược, chỉ sửa nguyên nhân) và lỗi đo được. Trả `findings` (thấy gì) rồi `patches` (sửa html / box_style, gộp ô,
chuyển câu sang ô khác, đổi loại ô, xoá lệnh mồ côi không mang câu khách). Giữ bản duyệt khi không viết ít câu khách hơn và không
thêm lỗi nặng (ô bỏ sót, chồng, tràn, chìm nền) -- `slots.accept`; hỏng ở vài lệnh thì chỉ hoàn lại các lệnh đó
(`slots.salvage`), xoá lệnh = skip. Qua rào đo lường rồi còn **giám khảo** (`slots.judge`): VLM so trước / sau cạnh nhau với nháp
(thứ tự xáo), chỉ giữ bản sau khi nó đẹp hơn -- lượt 4 (08/10) cho thấy bản duyệt có thể qua rào mà xấu đi.

## Giới hạn đã biết

- Vỏ **nửa trong suốt** trên nền gần cùng màu chênh quá ít, không thành S#: chữ trong đó vẫn được vẽ, nhưng thiếu vỏ; chữ sáng có
  thể chìm trên nền sáng (kiểm low_contrast báo cho vòng sửa).
- Vỏ dính liền vỏ khác (thẻ + pill trong thẻ) thành một mảng không gọn có thể không được nhận.
- Bản xoá xoá cả chữ / nhãn thuộc cảnh và trên sản phẩm: chỉ cứu được qua `keep` của designer (OCR phải bắt được dòng).
- Dòng OCR không bắt được thì không thành ô (chưa tách dòng từ lớp phủ còn thừa).
- Chưa có mã QR.
