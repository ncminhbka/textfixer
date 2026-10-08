# Chạy TextFix trên máy chủ GPU (2× A30, JupyterLab)

Máy chủ **chỉ cài được bằng pip**: không ra HuggingFace, không apt, không tải được Chromium của Playwright. Mọi thứ khác phải có
sẵn trong repo hoặc `~/persistent-data`. Mọi lệnh gõ trong Terminal của JupyterLab.

## Tóm tắt

```bash
# 1. mã: tải zip repo, giải nén vào ~/work
mkdir -p ~/work && cd ~/work && unzip -q <repo>.zip && cd <thư mục repo>
# 2. thư viện
pip install -r requirements.txt
# 3. biến môi trường (mục 3), lưu vào ~/textfix_env.sh
source ~/textfix_env.sh
# 4. kiểm: phải in "SẴN SÀNG."
python scripts/check_server.py --fix
# 5. chạy (cổng 8088), log: tail -f server.log
bash scripts/start_server.sh
# 6. mở https://<địa-chỉ-jupyter>/user/<tên>/proxy/8088/   (nhớ dấu / cuối)
```

## Cập nhật mã (không git pull được)

Tải zip mới của repo (GitHub: Code -> Download ZIP), tải lên `~/work`, rồi **chép đè** lên thư mục repo đang dùng. Zip không có
`output/` (đã gitignore) nên kết quả cũ, `.env` giữ nguyên:

```bash
cd ~/work && rm -rf _new && mkdir _new && unzip -q <tên zip>.zip -d _new
cp -a _new/*/. ~/work/<thư mục repo>/     # đè mã, giữ output/ và .env
rm -rf _new <tên zip>.zip
```

Không xoá thư mục repo cũ rồi giải nén lại: mất `output/`.

## 1. Cần có trong `~/persistent-data`

| Thứ | Đường dẫn | Ghi chú |
|---|---|---|
| FLUX.2-klein 4B distill: DiT, `vae/`, `text_encoder/` (Qwen3-4B), `tokenizer/` | `~/persistent-data/FLUX.2-klein-4B/` | Tự dò. Chỗ khác: `TEXTFIX_WEIGHTS` |
| FLUX.2-klein 4B base (tuỳ chọn, `--model base`) | `~/persistent-data/FLUX.2-klein-base-4B/` | |
| Chromium (Chrome for Testing, linux64) | `~/persistent-data/chrome-linux64.zip` | `check_server.py --fix` giải nén + ghi đường dẫn |

OCR (RapidOCR) có mô hình ONNX trong gói pip, không cần tải thêm.

## 2. Thư viện

`pip install -r requirements.txt`. Cảnh báo kiểu `notebook ... requires jupyterlab ...` là xung đột có sẵn của image notebook,
không liên quan, **đừng nâng jupyterlab** (đang chạy chính giao diện notebook).

## 3. Biến môi trường

Lưu vào `~/textfix_env.sh` (ngoài repo, không bao giờ commit khoá), rồi `echo 'source ~/textfix_env.sh' >> ~/.bashrc`:

```bash
export TEXTFIX_LLM_URL=http://<host>:<port>/v1     # LLM / VLM nội bộ (OpenAI-compatible)
export TEXTFIX_MODEL=<tên model nhìn được ảnh>     # một model làm cả prompt FLUX lẫn designer
export TEXTFIX_LLM_API_KEY=<khoá>
# export TEXTFIX_GPUS=0,1                          # máy chủ: card dùng để VẼ (mặc định mọi card, mỗi card một FLUX)
# export TEXTFIX_DESIGN_WORKERS=4                  # máy chủ: số luồng THIẾT KẾ song song (mỗi luồng một Chromium)
export PLAYWRIGHT_CHROME_PATH=/home/jovyan/persistent-data/chrome-linux64/chrome
```

Máy chủ VLM phải nhận **ít nhất 12 ảnh / lời gọi** (designer gửi 6, vòng sửa tới 12) và ngữ cảnh **≥ 32k token**
(vLLM: `--limit-mm-per-prompt '{"image": 16}' --max-model-len 32768`). `check_server.py` gọi thử đúng 12 ảnh.

Chạy từ ô notebook thay vì terminal: `%env TEXTFIX_MODEL=...` cho từng biến.

## 4. Tự kiểm tra

```bash
python scripts/check_server.py --fix 2>&1 | tee check.log
```

Kiểm: thư mục, gói pip (in một lệnh pip cho gói thiếu, không đụng torch), GPU, trọng số FLUX, font / icon, Chromium (giải nén
zip, `ldd` báo thư viện hệ thống thiếu, mở thử), OCR, LLM / VLM (danh sách model, JSON, 1 và 12 ảnh), cổng.
Mục nào `[LỖI]` thì làm theo dòng `SỬA:` ngay dưới.

## 5. Chạy

```bash
bash scripts/start_server.sh               # distill, 4 bước
bash scripts/start_server.sh --model base  # base, 50 bước, CFG 4 (chậm khoảng 25 lần)
tail -f server.log
curl -s localhost:8088/api/health          # phải thấy "ready": true
```

Khởi động nạp **một FLUX trên mỗi card** (song song) và OCR: khoảng 1 phút; Chromium mở khi luồng thiết kế dùng lần đầu.

Dây chuyền: mỗi phiên bản qua trạm VẼ (GPU: nháp ~2 s, xoá ~4 s, dọn chữ sót ~1-4 s) rồi trạm THIẾT KẾ (đo ô ~2 s, VLM
~11 s, dựng ~2 s, vòng sửa chỉ khi có lỗi nặng ~10 s). Các phiên bản và các yêu cầu gối đầu nhau: 2 card vẽ 2 phiên bản cùng lúc,
VLM gọi song song. Chọn nhiều ảnh: LLM viết thêm các **hướng thiết kế khác nhau** (cùng chữ khách), mỗi ảnh một hướng; ảnh xong
trước hiện trước trên UI. 4 ảnh ~35-45 s (một ảnh ~25-30 s).

UI có ô **Debug** (góc trên kết quả): nháp FLUX, bản xoá FLUX, bản xoá đã dọn, **Ô + OCR** (khung L / S / I vẽ đè lên nháp, di
chuột xem chữ OCR đọc và lệnh designer), lệnh designer, lượt đầu trước vòng sửa, bảng OCR -> ô -> lệnh, lỗi, nhật ký.

Kết quả: `output/runs/<run_id>/` gồm `request.json`, `brief.json` (kèm các hướng thiết kế), và mỗi ảnh `v<i>/`: `draft.png`,
`plate_raw.png` (bản xoá FLUX), `plate.png` (đã dọn), `slots.jpg`, `poster.png`, `poster_first.png` (khi vòng sửa đổi),
`steps.jpg` (nháp | ô | bản xoá | poster), `ops.jpg` (lệnh designer), `plan.json` (ô, lệnh, lỗi còn lại, nhật ký, câu thiếu,
thời gian).

## 6. Thử engine không qua giao diện

Mọi script chạy trên máy chủ tự gói kết quả thành các zip **độc lập, mỗi zip < 24 MB** (giải nén từng cái, không cần ghép) và
**tự tải về** khi chạy bằng ô notebook (`%run`). Chạy từ terminal thì zip vẫn được gói, tải tay từ cây thư mục JupyterLab.
Lần đầu trình duyệt có thể hỏi "cho phép tải nhiều tệp": chọn cho phép.

```python
# ô notebook, thư mục làm việc = thư mục repo (%cd ~/work/<thư mục repo>)
%run scripts/probe.py                          # 12 prompt bench/prompts_C.json: FLUX vẽ + xoá, VLM thật -> probe_NN.zip
%run scripts/probe.py --only p11_distill_s4    # một ca
%run scripts/make_pairs.py                     # cặp nháp + bản xoá cho bộ đáp án ô (docs/SLOTS_GT.md) -> pairs_NN.zip
```

Biến môi trường trong notebook: `%env` từng biến, hoặc chạy `source ~/textfix_env.sh` trước khi mở JupyterLab.

**Dùng cả 2 card:** `make_pairs.py`, `erase_probe.py` mặc định chạy mỗi A30 một tiến trình (mỗi tiến trình một bản FLUX ≈ 17 GB,
chia đều việc, nhật ký có tiền tố `[gpu k]`), nhanh gần gấp đôi. Tiến trình con bỏ qua `TEXTFIX_DEVICE_*` và đặt mọi phần
FLUX lên card của nó. `--gpus 1` để chạy một tiến trình như cũ (vd. khi card kia đang bận).

`probe.py` ra `output/probe/<key>.jpg` (nháp | ô | bản xoá | poster | lệnh), `<key>_plan.json`, `<key>_poster.png`.

## Sự cố

| Hiện tượng | Xử lý |
|---|---|
| `/api/health` có `error` | Đọc `server.log`: thường thiếu trọng số, chưa export biến LLM, hoặc hết VRAM |
| `CUDA out of memory` | `nvidia-smi`: card có bị người khác chiếm không (bộ nhớ đầy mà không có tiến trình trong container = dịch vụ ngoài, báo quản trị). Đặt `TEXTFIX_DEVICE_TE=cuda:1` |
| Lỗi 803 / `cuInit` | `start_server.sh` và `check_server.py` tự bỏ `.../compat` khỏi `LD_LIBRARY_PATH`; chạy tay thì làm tương tự |
| VLM `400` / `too many images` / `maximum context length` | Tăng `--limit-mm-per-prompt` / `--max-model-len` phía máy chủ VLM |
| Poster thiếu câu (giao diện báo "Không vẽ") | Model không chừa chỗ cho câu đó, hoặc designer rút gọn cho vừa: đổi seed, thêm ảnh |
