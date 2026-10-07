# TextFix

Poster tiếng Việt từ FLUX.2-klein. Model vẽ tốt bố cục và hình ảnh, nhưng vẽ sai chi tiết nhỏ: chữ thành "giun", icon, avatar, sao
méo. TextFix giữ nguyên bố cục và hình của model, rồi vẽ lại sạch toàn bộ lớp chi tiết đó.

## Luồng

```
form / prompt
  -> LLM: prompt FLUX.2 (theo hướng dẫn nâng prompt của BFL) + câu của khách          textfix/brief.py
  -> FLUX.2-klein vẽ nháp                                                            textfix/flux.py
  -> FLUX.2-klein xoá lớp phủ (lời dặn ngắn: "Remove all text from this image...")    textfix/flux.py
  -> nháp - bản xoá = lớp phủ -> các Ô: S# vỏ, I# chi tiết nhỏ, L# dòng chữ           textfix/overlay.py, slots.py
  -> VLM designer: quyết mỗi ô là gì, điền nội dung + CSS (không đưa toạ độ)          textfix/slots.py
  -> dựng HTML đúng khung từng ô trên bản xoá (Chromium), kiểm lỗi, một vòng sửa      textfix/render.py
  -> poster
```

Chi tiết cách làm và giới hạn đã biết: [docs/DESIGN.md](docs/DESIGN.md).

## Chạy

Máy chủ GPU (2x A30, JupyterLab): [docs/SERVER.md](docs/SERVER.md).

```bash
pip install -r requirements.txt
export TEXTFIX_LLM_URL=http://<host>:<port>/v1 TEXTFIX_MODEL=<model nhìn được ảnh> TEXTFIX_LLM_API_KEY=<khoá>
python scripts/check_server.py --fix        # kiểm môi trường, phải in "SẴN SÀNG."
bash scripts/start_server.sh                # giao diện ở cổng 8088
python scripts/probe.py                     # thử engine trên bench/prompts_C.json, nhìn output/probe/*.jpg
```

## Cấu trúc

| Đường dẫn | Việc |
|---|---|
| `textfix/engine.py` | Nối toàn bộ luồng: `Engine.brief`, `make` (vẽ + sửa), `fix` (sửa nháp có sẵn) |
| `textfix/brief.py` | LLM đầu luồng: prompt FLUX + câu khách, code giữ cam kết không bịa chữ |
| `textfix/flux.py` | FLUX.2-klein: vẽ nháp, xoá lớp phủ, căn bản xoá theo nháp |
| `textfix/overlay.py` | Bản đồ lớp phủ: tách vỏ (S#) và chi tiết nhỏ (I#), đo màu, bo góc |
| `textfix/slots.py` | Ô (dòng chữ nắn theo lớp phủ), lời dặn designer, gọi VLM, vòng sửa, nền |
| `textfix/render.py` | Dựng HTML đúng khung ô (thẻ tiện ích icon / sao / font), kiểm lỗi |
| `textfix/ocr.py`, `glyph.py`, `textmask.py` | OCR dòng, đo nét (cỡ, màu), mặt nạ nét chữ |
| `textfix/fonts.py`, `icons.py` | Bộ font OFL đủ dấu tiếng Việt, bộ icon Lucide |
| `server/` | Máy chủ FastAPI + giao diện form |
| `flux2/` | Mã suy luận FLUX.2 của Black Forest Labs (Apache-2.0), nguyên trạng |
| `assets/` | Font (OFL) và icon Lucide (ISC) |

## Giấy phép tài nguyên

- `flux2/`: Apache-2.0 (Black Forest Labs), xem `flux2/LICENSE.md`.
- Font trong `assets/fonts/`: SIL Open Font License.
- Icon trong `assets/icons/lucide/`: ISC (Lucide).
