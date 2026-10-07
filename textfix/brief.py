"""LLM ĐẦU LUỒNG (một lời gọi): form / prompt người dùng -> prompt tiếng Anh cho FLUX.2 + CÂU KHÁCH.

  prompt_en  theo hướng dẫn nâng prompt chính thức của BFL (flux2/system_messages.py, SYSTEM_MESSAGE_UPSAMPLING_T2I): giữ chủ đề,
             thêm chi tiết hình, MỌI chữ trong ngoặc kép đúng nguyên văn tiếng Việt; không bịa giá / % / khuyến mãi / ngày / số
             điện thoại / địa chỉ / bảo hành; không ghi tỉ lệ / độ phân giải.
  texts      chữ poster phải có: CHỈ chữ của người dùng, nguyên văn, kèm vai trò (designer dùng để hiểu từng chỗ).
Code giữ cam kết: câu phải có NGUYÊN VĂN trong form / prompt (sai -> bỏ); chuỗi trong ngoặc kép của prompt người dùng luôn có mặt;
câu chưa nằm trong ngoặc kép của prompt_en thì nối vào cuối prompt_en (model chỉ vẽ được chữ nó thấy).
"""

from __future__ import annotations

import json
import re

ROLES = ["headline", "subheadline", "body", "price", "old_price", "offer", "date", "info", "contact", "cta", "list_item",
         "review_quote", "review_name", "label"]
TECH_KEYS = {"image_base64", "seed", "num_images", "aspect_ratio", "primary_color", "style_pref", "has_product_image"}
QUOTED = re.compile(r'"([^"]+)"')

SYSTEM = """You are the copy planner and prompt engineer of a Vietnamese poster generator built on FLUX.2 by Black Forest Labs.
Input: the user's poster form (category + filled fields, JSON) and/or a free prompt. Output ONE JSON object with:

1. prompt_en -- an English prompt for FLUX.2 that draws the WHOLE poster including its text. Follow the official FLUX.2 prompt-upsampling guidelines:
   - Strictly preserve the user's subject and intent. Convert the request into a detailed paragraph.
   - Add concrete visual specifics: subject, form, materials, textures, lighting (quality, direction, color), shadows, spatial relationships, environment, mood, and a clean professional poster layout (where the headline, price / offer, list and footer sit; flat graphic text, badges, pills or cards where suitable).
   - Text in images: put EVERY text of the poster in double quotes, EXACTLY as written in "texts" below (Vietnamese with full diacritics), and say its role, typographic style and its place (e.g. headline "..." in a bold condensed sans-serif at the top center). Never add any other text.
   - Put texts on calm areas, never on the main subject.
   - Never invent commercial claims: no price, discount, promotional offer, deadline / date, phone number, address or warranty that the user did not give.
   - Never write aspect ratios, resolutions or sizes (e.g. 9:16, 4k, 1080p).
2. texts -- the texts the poster must show. ONLY the user's own words: copy every text EXACTLY character for character from the form values or from the quoted strings / explicit texts of the prompt (same case, accents, punctuation). Never invent, translate, shorten or rewrite text. Every filled content field appears in full (store name, phone, address, website included). A comma-separated list of highlights / features the user wrote becomes one text per item ONLY if the user separated them; copy each item exactly.
   - role: one of """ + ", ".join(ROLES) + """ (old_price = an original price shown crossed out next to the new price).
Answer only the JSON object."""

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["prompt_en", "texts"],
          "properties": {"prompt_en": {"type": "string"},
                         "texts": {"type": "array", "items": {
                             "type": "object", "additionalProperties": False, "required": ["text", "role"],
                             "properties": {"text": {"type": "string"}, "role": {"type": "string", "enum": ROLES}}}}}}

# gợi ý thiết kế từ UI (không phải chữ)
STYLES = {"hien_dai": "modern, minimal, clean", "sang_trong": "luxury, premium, elegant", "nang_dong": "energetic, sporty, dynamic",
          "am_cung": "warm, cozy, friendly", "le_hoi": "festive, vibrant, colorful"}
CATEGORIES = {"promo": "promotion / advertising poster", "product_intro": "product introduction poster",
              "opening": "grand opening banner", "feedback": "customer feedback / testimonial poster",
              "recruitment": "recruitment / hiring poster", "guide": "step-by-step process / how-to guide poster"}


def _norm(s: str) -> str:
    return " ".join(s.split())


def make_brief(llm, prompt: str | None = None, form: dict | None = None, design: dict | None = None,
               product_image: bool = False) -> dict:
    """llm = llm.json_call(..., schema=SCHEMA). -> {prompt_en, texts: [{text, role}], log}."""
    form = {k: v for k, v in (form or {}).items() if k not in TECH_KEYS and v not in (None, "", [], False)}
    if form.get("category") in CATEGORIES:
        form["category"] = CATEGORIES[form["category"]]
    user = []
    if form:
        user.append("Form (JSON):\n" + json.dumps(form, ensure_ascii=False, indent=1))
    if prompt:
        user.append("Prompt:\n" + prompt)
    hints = []
    st = (design or {}).get("style")
    if st and st != "auto":
        hints.append(f"visual style: {STYLES.get(st, st)}")
    if (design or {}).get("color"):
        hints.append(f"main brand color: {design['color']} (use it as the dominant accent color of the design)")
    if product_image:
        hints.append("the user uploaded a photo of their product, given to the image model as reference image 1: the poster must "
                     "feature exactly that product (say 'the product from image 1' in prompt_en)")
    if hints:
        user.append("Design hints (not texts, never draw them as words):\n- " + "\n- ".join(hints))
    raw = llm(SYSTEM, ["\n\n".join(user)])
    sources = [_norm(prompt or "")] + [_norm(v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)) for k, v in form.items()
                                       if k != "category"]
    must = list(dict.fromkeys(QUOTED.findall(prompt or "")))
    out, log, seen = [], {"dropped": [], "added": []}, set()
    for t in raw.get("texts", []):
        text = _norm(t.get("text", ""))
        if not text or text in seen:
            continue
        if not any(text in s for s in sources):   # không nguyên văn từ người dùng -> bỏ (không bịa chữ)
            log["dropped"].append(text)
            continue
        seen.add(text)
        out.append({"text": text, "role": t.get("role") if t.get("role") in ROLES else "body"})
    for q in must:   # chuỗi người dùng viết trong ngoặc kép mà LLM bỏ sót -> thêm lại nguyên văn (trừ khi đã tách thành các ý)
        qn = _norm(q)
        parts = [t for t in seen if t in qn]
        if qn not in seen and not (parts and sum(len(p) for p in parts) >= 0.8 * len(qn)):
            out.append({"text": qn, "role": "body"})
            seen.add(qn)
            log["added"].append(qn)
    # không tỉ lệ / độ phân giải trong prompt -- chỉ NGOÀI ngoặc kép (chữ người dùng như "8:30" giữ nguyên)
    parts = raw.get("prompt_en", "").split('"')
    bad = re.compile(r"\b(?:1:1|9:16|16:9|4:5|5:4|2:3|3:2|4:3|3:4|21:9)\b|\b\d{3,4}p\b|\b[248]k\b", re.I)
    pe = '"'.join(bad.sub("", p) if i % 2 == 0 else p for i, p in enumerate(parts))
    have = set(QUOTED.findall(pe))
    miss = [t["text"] for t in out if t["text"] not in have]
    if miss:
        pe += " The poster shows exactly these texts: " + ", ".join(f'"{t}"' for t in miss) + "."
        log["quoted_added"] = miss
    return {"prompt_en": " ".join(pe.split()), "texts": out, "log": log}
