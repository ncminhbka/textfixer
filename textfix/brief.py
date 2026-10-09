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
CURLY = re.compile(r"“([^”]+)”")   # ngoặc kép cong (gõ trên điện thoại / Word)
# vai mặc định của trường form (thêm lại trường LLM bỏ sót)
FIELD_ROLE = {"title": "headline", "discount": "offer", "applied_product": "subheadline", "date_start": "date", "date_end": "date",
              "product_name": "headline", "product_desc": "subheadline", "highlights": "list_item", "price": "price",
              "opening_date": "date", "opening_promo": "offer", "booking_contact": "contact", "feedback_target": "label",
              "feedback_quote": "review_quote", "customer_name": "review_name", "feedback_highlights": "list_item",
              "special_offer": "offer", "job_position": "headline", "job_desc": "body", "apply_deadline": "date",
              "apply_method": "contact", "guide_steps": "list_item", "store_name": "info", "phone": "contact",
              "address": "contact", "website_link": "contact"}
# khung ảnh (tỉ lệ người dùng chọn) bằng lời: LLM mặc định "vertical poster" cả khi khung ngang (09/10 dev2 d16, d32 16:9)
ORIENT = {"1:1": "a square canvas", "4:5": "a portrait (slightly tall) canvas", "3:4": "a portrait canvas",
          "9:16": "a tall vertical canvas (phone story)", "16:9": "a wide horizontal landscape canvas (web banner)",
          "4:3": "a landscape canvas", "2:3": "a tall portrait canvas"}

SYSTEM = """You are the copy planner and prompt engineer of a Vietnamese poster generator built on FLUX.2 by Black Forest Labs.
Input: the user's poster form (category + filled fields, JSON) and/or a free prompt. Output ONE JSON object with:

0. intent -- "poster" when the user wants words on the image: any quoted string, any filled form field, or a poster / banner / flyer / ad / menu / sign whose content (name, offer, price, date, slogan, ...) the user gave. "image" when the user only wants a picture (photo, illustration, artwork, wallpaper, ...) and gives no words to show. Never "image" when there is any user text to show. Most users are not prompt writers: a short vague prompt ("tạo poster", "làm banner cho quán tôi", "đẹp nhé") next to a filled form is "poster", and it never cancels the form -- every filled form field is still a text.
   For "image": texts = [], and prompt_en (and every variant) is a plain FLUX.2 image prompt following the same upsampling guidelines (subject, details, materials, lighting, mood, composition) WITHOUT poster layout and without adding any text; the rules about texts below do not apply.

1. prompt_en -- an English prompt for FLUX.2 that draws the WHOLE poster including its text. Follow the official FLUX.2 prompt-upsampling guidelines:
   - Strictly preserve the user's subject and intent. Convert the request into a detailed paragraph.
   - Add concrete visual specifics: subject, form, materials, textures, lighting (quality, direction, color), shadows, spatial relationships, environment, mood, and a clean professional poster layout (where the headline, price / offer, list and footer sit; flat graphic text, badges, pills or cards where suitable).
   - Text in images: put EVERY text of the poster in double quotes, EXACTLY as written in "texts" below (Vietnamese with full diacritics), and say its role, typographic style and its place (e.g. headline "..." in a bold condensed sans-serif at the top center). Never add any other text.
   - Put texts on calm areas, never on the main subject.
   - Never invent commercial claims: no price, discount, promotional offer, deadline / date, phone number, address or warranty that the user did not give.
   - Never write aspect ratios, resolutions or sizes (e.g. 9:16, 4k, 1080p).
2. texts -- the texts the poster must show. ONLY the user's own words: copy every text EXACTLY character for character from the form values or from the quoted strings / explicit texts of the prompt (same case, accents, punctuation). Never invent, translate, shorten or rewrite text. Every filled content field appears in full (store name, phone, address, website included). A comma-separated list of highlights / features the user wrote becomes one text per item ONLY if the user separated them; copy each item exactly.
   - role: one of """ + ", ".join(ROLES) + """ (old_price = an original price shown crossed out next to the new price).
3. variants -- ALTERNATIVE DESIGN DIRECTIONS for the same poster, exactly as many as asked below (empty list when none are asked).
   Each: name (2-4 words, e.g. "big type centered", "split photo / text", "card layout", "dark premium") and its own complete
   prompt_en following every rule of 1. Same subject, same product, same texts (every text in double quotes, exactly the same),
   same user style / color hints -- but a clearly DIFFERENT composition: layout (where the text block sits, centered / left /
   split / top band / framed card), scale of the subject, background treatment, typographic mood and palette within the hints.
   The directions must differ from prompt_en and from each other, and all must stay professional and suit the category.
Answer only the JSON object."""

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["intent", "prompt_en", "texts", "variants"],
          "properties": {"intent": {"type": "string", "enum": ["poster", "image"]},
                         "prompt_en": {"type": "string"},
                         "variants": {"type": "array", "items": {
                             "type": "object", "additionalProperties": False, "required": ["name", "prompt_en"],
                             "properties": {"name": {"type": "string"}, "prompt_en": {"type": "string"}}}},
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
               product_image: bool = False, n_variants: int = 0, aspect: str | None = None) -> dict:
    """llm = llm.json_call(..., schema=SCHEMA). -> {mode, prompt_en, texts: [{text, role}], variants: [{name, prompt_en}], log}.
    mode: "poster" (có câu khách -> cả dây chuyền) | "image" (không chữ khách -> chỉ FLUX vẽ, trả nháp).
    n_variants: số hướng thiết kế KHÁC (người dùng chọn nhiều ảnh: ảnh 1 = prompt_en, ảnh 2.. = các hướng này; cùng chữ khách).
    Mỗi prompt (chính và từng hướng) qua cùng các bước giữ cam kết: bỏ tỉ lệ / độ phân giải, chữ khách thiếu ngoặc kép thì nối."""
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
    if aspect in ORIENT:
        hints.append(f"canvas: {ORIENT[aspect]} -- compose the whole layout for this shape and describe it with words like "
                     "horizontal / vertical / square, never as a ratio")
    if hints:
        user.append("Design hints (not texts, never draw them as words):\n- " + "\n- ".join(hints))
    user.append(f"Alternative design directions asked (variants): {max(0, int(n_variants))}")
    raw = llm(SYSTEM, ["\n\n".join(user)])
    sources = [_norm(prompt or "")] + [_norm(v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)) for k, v in form.items()
                                       if k != "category"]
    must = list(dict.fromkeys(QUOTED.findall(prompt or "") + CURLY.findall(prompt or "")))
    out, log, seen = [], {"dropped": [], "added": []}, set()
    pe_raw = raw.get("prompt_en", "")
    pe_open = QUOTED.sub(" ", pe_raw).casefold()   # prompt_en ngoài các chuỗi trong ngoặc kép
    for t in raw.get("texts", []):
        text = _norm(t.get("text", ""))
        if not text or text in seen:
            continue
        # không nguyên văn từ người dùng -> bỏ (không bịa chữ); so không phân biệt hoa / thường (08/10: "Khai trương cửa hàng ..."
        # viết hoa chữ đầu, prompt khách viết thường -> bị bỏ oan, tiêu đề mất khỏi câu khách -> VLM chép chữ nháp "TRƯỞNG")
        if not any(text.casefold() in s.casefold() for s in sources):
            log["dropped"].append(text)
            continue
        seen.add(text)
        out.append({"text": text, "role": t.get("role") if t.get("role") in ROLES else "body"})
    for q in must:   # chuỗi người dùng viết trong ngoặc kép mà LLM bỏ sót -> thêm lại nguyên văn (trừ khi đã tách thành các ý)
        qn = _norm(q)
        parts = [t for t in seen if t in qn]
        if qn not in seen and not (parts and sum(len(p) for p in parts) >= 0.8 * len(qn)):
            # LLM dùng chuỗi đó làm TỪ TẢ trong prompt_en (không ngoặc kép): khách đặt ngoặc cho phong cách, không phải chữ in
            # (09/10 dev2 d06: phong cách "Y2K", "vintage" -> "Y2K-style t-shirt with a vintage aesthetic") -> không thêm lại
            if re.search(r"(?<!\w)" + re.escape(qn.casefold()) + r"(?!\w)", pe_open):
                log.setdefault("quoted_as_style", []).append(qn)
                continue
            out.append({"text": qn, "role": "body"})
            seen.add(qn)
            log["added"].append(qn)
    # TRƯỜNG FORM đã điền mà không có trong câu khách (LLM viết lại thành câu khác -- "Apply deadline: 15/12/2026" -- nên bị bỏ ở
    # bước nguyên văn, 09/10 dev2 d32 mất hạn nộp) -> thêm nguyên giá trị trường
    for k, v in form.items():
        if k == "category":
            continue
        for x in (v if isinstance(v, list) else [v]):
            xn = _norm(str(x))
            if not xn or any(xn.casefold() in t.casefold() for t in seen):
                continue
            parts = [t for t in seen if t.casefold() in xn.casefold()]
            if parts and sum(len(p) for p in parts) >= 0.8 * len(xn):   # đã tách thành các ý / các dòng
                continue
            out.append({"text": xn, "role": FIELD_ROLE.get(k, "info")})
            seen.add(xn)
            log.setdefault("form_added", []).append(xn)
    pe, miss = _fix_prompt(raw.get("prompt_en", ""), out)
    gone = _quoted_extra(raw.get("prompt_en", ""), out)
    if gone:
        log["quoted_removed"] = gone
    if miss:
        log["quoted_added"] = miss
    variants = []
    for v in (raw.get("variants") or [])[:max(0, int(n_variants))]:
        if isinstance(v, dict) and (v.get("prompt_en") or "").strip():
            vp, vm = _fix_prompt(v["prompt_en"], out)
            variants.append({"name": " ".join(str(v.get("name") or "").split())[:60], "prompt_en": vp})
            if vm:
                log.setdefault("variant_quoted_added", []).append(vm)
    # ẢNH THƯỜNG (09/10): không có câu khách nào -> chỉ FLUX vẽ, bỏ xoá / ô / VLM / dựng (trước đây báo lỗi "không tìm ra chữ").
    # Có câu khách thì luôn là poster dù LLM nói "image" (chữ khách phải lên ảnh). LLM nói "poster" mà không có câu khách: prompt
    # có thể tả chỗ tiêu đề / nút -> FLUX vẽ chữ vô nghĩa, dặn thêm không chữ.
    intent = raw.get("intent") if raw.get("intent") in ("poster", "image") else "poster"
    mode = "poster" if out else "image"
    log["intent"] = intent
    if mode == "image" and intent == "poster":
        tail = " No text, letters, words or logos anywhere in the image."
        pe += tail
        variants = [{**v, "prompt_en": v["prompt_en"] + tail} for v in variants]
    return {"mode": mode, "prompt_en": pe, "texts": out, "variants": variants, "log": log}


def _approved(q: str, texts: list[dict]) -> bool:
    """Chuỗi trong ngoặc kép của prompt_en là chữ khách: trùng một câu, nằm trong một câu, hoặc ghép từ các câu khách."""
    q = _norm(q).casefold()
    tx = [t["text"].casefold() for t in texts]
    if any(q in t for t in tx):
        return True
    parts = [t for t in tx if t in q]
    return bool(parts) and sum(len(t) for t in parts) >= 0.8 * len(q)


def _sentences(pe: str) -> list[str]:
    """Tách câu, không tách trong ngoặc kép ("TP. HCM", "Mở cửa 7:00!")."""
    out, cur, inq = [], "", False
    for i, ch in enumerate(pe):
        cur += ch
        if ch == '"':
            inq = not inq
        elif ch in ".!?" and not inq and (i + 1 == len(pe) or pe[i + 1].isspace()):
            out.append(cur)
            cur = ""
    return out + ([cur] if cur.strip() else [])


def _quoted_extra(pe: str, texts: list[dict]) -> list[str]:
    return [q for q in dict.fromkeys(QUOTED.findall(pe)) if not _approved(q, texts)]


def _fix_prompt(pe: str, texts: list[dict]) -> tuple[str, list[str]]:
    """Không tỉ lệ / độ phân giải trong prompt -- chỉ NGOÀI ngoặc kép (chữ người dùng như "8:30" giữ nguyên); chữ khách chưa nằm
    trong ngoặc kép thì nối vào cuối (model chỉ vẽ được chữ nó thấy). Chữ trong ngoặc kép KHÔNG phải chữ khách (câu LLM bịa, đã
    bị bỏ khỏi texts) -> bỏ cả câu chứa nó: FLUX vẫn vẽ, VLM chép lại thành chữ không ai duyệt (08/10: địa chỉ bịa "123 Đường Số
    1, Quận 1, TP. HCM" bị bỏ khỏi texts nhưng còn trong prompt -> lên cả 4 poster)."""
    bad_q = set(_quoted_extra(pe, texts))
    if bad_q:
        keep = [s for s in _sentences(pe) if not set(QUOTED.findall(s)) & bad_q]
        pe = "".join(keep)
    parts = pe.split('"')
    bad = re.compile(r"\b(?:1:1|9:16|16:9|4:5|5:4|2:3|3:2|4:3|3:4|21:9)\b|\b\d{3,4}p\b|\b[248]k\b", re.I)
    pe = '"'.join(bad.sub("", p) if i % 2 == 0 else p for i, p in enumerate(parts))
    have = set(QUOTED.findall(pe))
    miss = [t["text"] for t in texts if t["text"] not in have]
    if miss:
        pe += " The poster shows exactly these texts: " + ", ".join(f'"{t}"' for t in miss) + "."
    return " ".join(pe.split()), miss
