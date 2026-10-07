"""Gọi LLM / VLM qua API OpenAI-compatible (vLLM): trả JSON, nhiều ảnh, có cache theo nội dung (~/.cache/textfix/<tag>).

  call = json_call(url, model, key_env, tag="designer", schema=None)
  out = call(system, [text | png_bytes, ...])   # -> dict
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path


def client(base_url: str, key_env: str = "TEXTFIX_LLM_API_KEY"):
    from openai import OpenAI
    key = os.environ.get(key_env) or os.environ.get("TEXTFIX_LLM_API_KEY") or "none"
    return OpenAI(base_url=base_url, api_key=key, timeout=float(os.environ.get("TEXTFIX_LLM_TIMEOUT", "300")), max_retries=1)


def extra_body() -> dict:
    """vLLM + Qwen: tắt chế độ suy nghĩ (JSON ép khuôn, không cần chuỗi suy luận; nhanh hơn nhiều)."""
    return {"chat_template_kwargs": {"enable_thinking": False}}


def json_call(base_url: str, model: str, key_env: str = "TEXTFIX_LLM_API_KEY", tag: str = "llm", schema: dict | None = None):
    if not base_url or not model:
        raise RuntimeError("chưa đặt LLM / VLM: export TEXTFIX_LLM_URL và TEXTFIX_MODEL (docs/SERVER.md)")
    cache = Path.home() / ".cache" / "textfix" / tag
    cl = None

    def call(system: str, parts: list) -> dict:
        nonlocal cl
        cache.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha1((model + base_url + system + json.dumps(schema or {})).encode())
        for p in parts:
            h.update(p if isinstance(p, bytes) else str(p).encode())
        f = cache / f"{h.hexdigest()}.json"
        if f.exists():
            return json.loads(f.read_text(encoding="utf-8"))
        cl = cl or client(base_url, key_env)
        content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(p).decode()}}
                   if isinstance(p, bytes) else {"type": "text", "text": str(p)} for p in parts]
        fmt = ({"type": "json_schema", "json_schema": {"name": tag, "schema": schema, "strict": True}} if schema
               else {"type": "json_object"})
        r = cl.chat.completions.create(model=model, messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
                                       response_format=fmt, extra_body=extra_body())
        out = json.loads(r.choices[0].message.content)   # trả lời hỏng: ném lỗi, không vào cache
        f.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        return out
    return call
