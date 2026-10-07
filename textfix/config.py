"""Cấu hình qua biến môi trường (export trong terminal, hoặc tệp .env ở gốc repo -- biến đã export thắng .env).

  TEXTFIX_LLM_URL, TEXTFIX_MODEL, TEXTFIX_LLM_API_KEY     LLM / VLM (OpenAI-compatible, vLLM); một model nhìn được ảnh làm cả hai
  TEXTFIX_VLM_URL, TEXTFIX_VLM_MODEL, TEXTFIX_VLM_API_KEY  VLM riêng (không đặt: dùng LLM)
  TEXTFIX_LLM_TIMEOUT                                      giây chờ mỗi lời gọi (mặc định 300)
  TEXTFIX_WEIGHTS, TEXTFIX_DEVICE_DIT / _TE / _AE          FLUX.2-klein (xem flux.py)
  PLAYWRIGHT_CHROME_PATH                                   Chromium có sẵn (máy chủ không tải được bản của Playwright)
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_env() -> None:
    """Nạp .env vào biến môi trường (không ghi đè biến đã có, không in giá trị)."""
    f = ROOT / ".env"
    if not f.exists():
        return
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def models() -> dict:
    """{llm: {url, model, key_env}, vlm: {...}} từ biến môi trường."""
    url, model = os.environ.get("TEXTFIX_LLM_URL"), os.environ.get("TEXTFIX_MODEL")
    vurl = os.environ.get("TEXTFIX_VLM_URL") or url
    vmodel = os.environ.get("TEXTFIX_VLM_MODEL") or model
    return {"llm": {"url": url, "model": model, "key_env": "TEXTFIX_LLM_API_KEY"},
            "vlm": {"url": vurl, "model": vmodel, "key_env": "TEXTFIX_VLM_API_KEY"}}
