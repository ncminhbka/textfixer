# Hướng dẫn gán nhãn ô (S / I / L / P)

Bộ đáp án dùng để **đo bộ tách ô** (`overlay.py`, `slots.build`) bằng số: mỗi lần sửa code biết ngay tốt lên hay tệ đi. Đáp án là
**cái designer cần nhận**: mỗi thứ model vẽ ở lớp phủ thành đúng một ô, khung ôm sát. Ta gán **sự thật trên ảnh**, không gán theo
cái bộ đo hiện tại làm được.

Bộ ảnh: [bench/prompts_gt.json](../bench/prompts_gt.json), 20 prompt theo 6 loại form của giao diện và 4 tỉ lệ ảnh, tổng 36 ảnh.

| Mức | Số ảnh | Nội dung |
|---|---|---|
| Dễ (`e01`–`e04`) | 4 | 2–3 dòng chữ trần, không vỏ, không icon |
| Trung bình (`m01`–`m05`) | 10 | 4–7 dòng, huy hiệu, pill, nút, icon đầu dòng, một thẻ bình luận, các bước đánh số |
| Khó (`h01`–`h03`, `h06`–`h08`) | 12 | nhiều thẻ, menu 2 cột, giá gạch, chân trang có icon, vỏ nửa trong suốt, nhiều dòng liên hệ |
| Ảnh tham chiếu (`r01`–`r05`) | 10 | người dùng tải ảnh sản phẩm lên: **chữ in trên sản phẩm** (chai Dove, bánh pía có chữ tiếng Việt, lon Sapporo, lon Coca-Cola, gói Nescafé) phải giữ nguyên; kèm kính mờ, huy hiệu sao nổ, nhãn dán nghiêng |

Chữ trên sản phẩm hầu như chỉ xuất hiện khi người dùng tải ảnh sản phẩm lên (FLUX giữ ảnh tham chiếu rất sát, còn tự vẽ thì
không ra chữ trên sản phẩm), nên các ca P đều dùng ảnh thật trong [bench/refs/](../bench/refs/SOURCES.md).

**Ai gán:** Claude gán lượt đầu (xem ảnh để quyết có gì, lớp nào; mép khung do code ôm theo mặt nạ lớp phủ, không ước bằng mắt).
Người duyệt mở công cụ, sửa chỗ sai, bấm Enter. Hướng dẫn dưới đây là luật chung cho cả hai.

---

## 1. Chuẩn bị (một lần)

1. **Máy chủ**, trong ô notebook JupyterLab, thư mục làm việc là repo:
   ```python
   %run scripts/make_pairs.py
   ```
   Máy chủ vẽ 36 nháp và 36 bản xoá (ca `r` dùng ảnh tham chiếu trong `bench/refs/`), gói thành các `pairs_NN.zip` (mỗi cái
   < 24 MB) rồi tự tải về.
2. **Máy cá nhân**: giải nén **mọi** `pairs_NN.zip` vào `output/pairs/` của repo (cùng một thư mục, không tạo thư mục con).
3. Điền sẵn khung của bộ đo hiện tại để chỉ phải sửa, không vẽ từ đầu:
   ```bash
   python scripts/slots_gt.py prefill
   ```
4. Mở `tools/annotate.html` bằng **Chrome hoặc Edge** (kéo tệp vào trình duyệt), bấm **Mở thư mục repo**, chọn thư mục gốc repo,
   cho phép **sửa**. Công cụ ghi thẳng vào `bench/slots_gt/<key>.json`.

## 2. Giao diện

- **Trái trên**: danh sách ảnh và trạng thái: `prefill` (chưa động vào), `editing` (đang sửa), `done` (xong).
- **Trái dưới**: mức khó, điểm khó của ảnh và **các câu prompt yêu cầu**. Model có thể vẽ sai chính tả, thiếu hoặc thêm câu: gán
  theo cái **thấy trên ảnh**, danh sách chỉ để hiểu ý. Câu tô tím là chữ cảnh, gán **P**.
- **Giữa**: ảnh nháp và các khung. Màu theo lớp: **L đỏ**, **S xanh dương**, **I cam**, **P tím nét đứt**.

| Thao tác | Cách làm |
|---|---|
| Vẽ khung mới | Kéo chuột ở chỗ trống, hoặc trong một khung **chưa chọn** (vẽ L bên trong S được). Lớp = lớp đang bật |
| Chọn lớp | Phím `1` L · `2` S · `3` I · `4` P. Đang chọn khung thì đổi luôn lớp của khung đó |
| Chọn / dời / đổi cỡ | Bấm vào khung (khung nhỏ nhất dưới chuột được chọn) · kéo bên trong để dời · kéo 8 ô vuông trắng ở mép để đổi cỡ |
| Chỉnh từng điểm ảnh | Mũi tên: nhích 1 px (Shift: 10 px) · Alt + mũi tên: nới mép theo hướng đó |
| Xoá / bỏ chọn | `Delete` / `Esc` |
| So bản xoá | **Giữ `Space`**: thấy bản xoá. Thứ còn lại trong bản xoá thuộc ảnh, thứ biến mất thuộc lớp phủ |
| Tô lớp phủ | `D`: tô hồng chỗ nháp khác bản xoá. Rất nhanh để thấy vỏ, icon, dòng chữ |
| Phóng / dời ảnh | Lăn chuột để phóng tại con trỏ · kéo chuột phải hoặc giữa để dời · `F` vừa màn hình |
| Ẩn khung | `H` (nhìn ảnh gốc cho rõ) |
| Hoàn tác | `Ctrl+Z` / `Ctrl+Y` |
| Lưu / chuyển ảnh | `Ctrl+S` · `[` `]` ảnh trước / sau (tự lưu) · **`Enter` = xong ảnh này + sang ảnh sau** |

## 3. Quy trình mỗi ảnh

1. **Nhìn toàn cảnh** (`F`), đọc ô thông tin bên trái. Bấm `D` xem lớp phủ gồm những gì.
2. **Dọn khung prefill sai**: khung bao rác, khung gộp hai thứ, khung trùng nhau → `Delete`.
3. **L**: đi từ trên xuống, mỗi dòng chữ một khung. Phóng to (lăn chuột) để ôm sát dấu thanh.
4. **S**: mọi thẻ, pill, nút, dải, huy hiệu. Nhiều vỏ nửa trong suốt prefill bỏ sót: giữ `Space` để thấy mép.
5. **I**: icon, avatar, hàng sao, tick, mũi tên.
6. **P**: chữ in trên sản phẩm, bao bì, màn hình, biển trong cảnh.
7. **Kiểm** (mục 6), rồi **`Enter`**.

## 4. Luật theo lớp

Chỉ gán thứ thuộc **lớp phủ** (thứ một designer đặt lên ảnh). Phép thử: giữ `Space`, thứ đó **biến mất** trong bản xoá → lớp
phủ. **Còn nguyên** → thuộc ảnh: không gán, trừ chữ thì gán P. Bản xoá đôi khi xoá nhầm cả nhãn sản phẩm: lúc đó dựa vào ý
nghĩa (chữ nằm **trên vật** → P).

### L: dòng chữ

- **Một dòng nhìn thấy = một khung.** Câu dài model ngắt 3 dòng thì 3 khung. Hai cột cạnh nhau (tên món … giá) thì **hai khung
  riêng**, kể cả khi cùng hàng.
- Chữ "giun", sai chính tả, vô nghĩa nhưng rõ ràng là một dòng chữ: **vẫn là L**.
- **Ôm sát nét mực**: gồm dấu thanh, mũ (Ế, Ộ, Ữ…), nét bay của chữ viết tay, **viền chữ** (chữ có outline). Không gồm bóng mờ
  loang hoặc vầng sáng.
- Gạch đầu dòng **đơn giản** (chấm, gạch ngang, ô vuông nhỏ) sát đầu dòng: **gồm trong L**. Icon có hình rõ (lá, tick, điện thoại,
  ghim bản đồ) → **I riêng**, L bắt đầu từ chữ.
- **Giá gạch**: khung L gồm cả đường gạch ngang.
- Số trong vòng tròn bước (①): con số là **L**, vòng tròn đặc là **S**.
- Chữ nghiêng (nhãn dán xoay): khung thẳng **ôm ngoài** toàn bộ chữ (công cụ chỉ có khung thẳng).
- Logo dạng chữ (tên cửa hàng viết kiểu): L. Logo dạng hình: I.

### S: vỏ

- Hình đặc hoặc có viền **chứa hoặc nằm dưới** chữ / icon: thẻ, pill, nút, dải băng, huy hiệu tròn / sao nổ, ô giá, dải chân trang,
  khung kính mờ.
- **Ôm sát mép vỏ**, không gồm bóng đổ. Vỏ có viền: tính cả viền.
- **Vỏ lồng vỏ** (pill lương trong hàng, nút trong thẻ): gán **cả hai**.
- **Nửa trong suốt** (kính mờ, dải tối mờ): vẫn là S nếu mắt thấy mép. Đây là chỗ bộ đo đang yếu, nhất định phải có trong đáp án.
- Vỏ **không chứa gì** nhưng là khối đồ hoạ rõ (một pill trống, ô màu): vẫn là S.
- Dải băng / vỏ chạy ra **ngoài mép ảnh**: khung cắt tại mép ảnh.

### I: chi tiết nhỏ

- Icon, avatar (cả khung tròn ảnh người), **hàng sao: một khung cho cả hàng**, dấu tick, mũi tên nối các bước, nút ✕, emoji, logo
  hình nhỏ, đường kẻ phân cách ngắn do designer đặt.
- Hình trang trí lớn thuộc nền (hoạ tiết hình học, confetti, bóng bay): **không gán** (thường bản xoá còn giữ).
- Chi tiết nằm trong vỏ (avatar trong thẻ): I riêng **và** S của thẻ.

### P: chữ thuộc cảnh

- Chữ in trên vật trong ảnh: nhãn chai (`r01`), dấu chữ trên bánh (`r02`), lon (`r03`, `r04`), bao bì (`r05`), biển hiệu trong
  cảnh, chữ trên áo. Ô thông tin bên trái liệt kê chữ có trên ảnh tham chiếu.
- **Một khung cho cả mảng chữ** trên vật (nhãn 3 dòng → 1 khung P).
- Đây là chữ pipeline phải **giữ nguyên**, không vẽ lại. Bộ chấm đo xem bộ tách ô có lấy nhầm nó thành L không.

### Không gán

Người, sản phẩm, đồ vật, nền; mẩu nét rác vô nghĩa không thành dòng; thứ nhỏ hơn khoảng 8 px; bóng đổ.

## 5. Mẫu theo kiểu poster

**Thẻ bình luận** (`m04`, `h01`, mỗi thẻ):
```
┌ S ─────────────────────────────────┐
│ (I avatar)  [L Chị Lan, Hà Nội]    │
│             (I ★★★★★ cả hàng)      │
│ [L "Nhân viên chu đáo, da mình]    │
│ [L căng bóng sau buổi đầu tiên!"]  │
└────────────────────────────────────┘
```
3 thẻ → 3 S, 3 I avatar, 3 I hàng sao, mỗi tên / mỗi dòng trích dẫn một L.

**Danh sách có icon** (`m02`, `r01`, `r05`, `h07`): mỗi hàng = I (icon) + L (chữ). Hàng nằm trên kính mờ thì thêm một S cho tấm kính.

**Các bước** (`m05`, `h03`): vòng tròn số = S + L (chữ số). Thẻ bước = S chứa I (icon) + L tiêu đề + L mô tả. Mũi tên giữa thẻ = I.

**Giá** (`h02`, `r01`, `r02`): giá mới một L, giá gạch một L riêng (gồm đường gạch). Ô giá có nền → S.

**Menu 2 cột** (`h02`): tên món và giá là **hai L riêng** dù cùng hàng.

**Huy hiệu / nhãn dán** (`m01`, `r02`, `r04`, `h08`): vòng tròn hoặc sao nổ = S, chữ trong = L. Nhãn dán nghiêng: S và L đều là khung thẳng
ôm ngoài.

**Chân trang liên hệ** (`m03`, `h01`, `h02`, `h06`, `h08`, `r03`): mỗi icon một I, mỗi dòng liên hệ một L, dải nền (nếu có) một S.

**Chữ trên sản phẩm** (`r01`–`r05`): nhãn trên chai / bánh / lon / gói = P. Chữ poster đặt **cạnh** sản phẩm vẫn là L.
FLUX có thể chép sản phẩm nhiều lần (nhiều lon, nhiều gói): mỗi sản phẩm một khung P.

## 6. Kiểm trước khi bấm Enter

- [ ] Bấm `D`: mọi mảng hồng đều nằm trong một khung (hoặc là rác / bóng đổ cố ý bỏ).
- [ ] Không còn khung prefill bao rác hoặc gộp hai dòng.
- [ ] Dấu thanh ở dòng chữ hoa trên cùng nằm trong khung (phóng to kiểm 1–2 dòng).
- [ ] Mỗi hàng sao là một I, không phải từng ngôi sao.
- [ ] Chữ trên sản phẩm là P, không phải L.
- [ ] Thanh trên cùng: số L / S / I / P hợp lý với những gì thấy.

## 7. Chấm

```bash
python scripts/slots_gt.py eval --save base     # mốc, sau khi đã "done" đủ ảnh
python scripts/slots_gt.py eval --vs base       # sau mỗi lần sửa bộ đo
```

Từng lớp L / S / I: khớp theo IoU lớn nhất trước, ngưỡng **IoU ≥ 0.5**. Báo precision, recall, F1, IoU trung bình của các cặp khớp,
số thiếu, số sai. Dòng L của bộ đo trúng một khung P được đếm riêng ("lấy nhầm chữ cảnh"). Ảnh so khớp ở `output/slots_eval/`:
xanh lá đúng, đỏ sai, xanh dương thiếu, tím trúng chữ cảnh.

Ảnh (`output/pairs/`) không vào git. Tệp đáp án `bench/slots_gt/*.json` **vào git**: commit sau mỗi buổi gán nhãn.
