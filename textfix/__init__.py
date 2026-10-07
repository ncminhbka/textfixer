"""TextFix: poster tiếng Việt từ FLUX.2-klein, chữ / chi tiết nhỏ vẽ lại bằng VLM designer theo các ô đo từ lớp phủ.

Luồng (textfix.engine.Engine):
  form / prompt --LLM--> prompt FLUX + câu khách        (brief)
  FLUX.2-klein vẽ nháp                                   (flux.Flux.generate)
  FLUX.2-klein xoá lớp phủ bằng lời dặn ngắn             (flux.erase)
  nháp - bản xoá = lớp phủ -> ô S# / I# / L#             (slots.build: overlay + OCR)
  VLM điền từng ô -> dựng đúng khung ô -> kiểm -> sửa     (slots.plan / render.render / render.check / slots.repair)
"""
