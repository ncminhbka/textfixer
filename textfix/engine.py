"""ENGINE end to end: form / prompt -> poster.

  with Engine(flux) as E:                                    # flux = flux.Flux("distill").load() (None: chỉ fix() nháp có sẵn)
      B = E.brief(prompt=..., form=..., design=...)          # LLM: prompt FLUX + câu khách
      r = E.make(B, W, H, seed)                              # FLUX vẽ nháp -> fix()
      r = E.fix(draft, B, seed, plate=None)                  # xoá (FLUX) -> dọn chữ sót -> ô -> VLM điền ô -> dựng -> kiểm -> sửa
  r: {draft, plate, poster, steps (ảnh 4 cột), plan (JSON: ô, lệnh, lỗi, nhật ký, câu thiếu), timing}
"""

from __future__ import annotations

import time

import numpy as np

from . import config


class Engine:
    def __init__(self, flux=None, progress=None, llm=None, vlm=None):
        """llm / vlm: hàm gọi tự truyền (thử nghiệm, VLM giả); None: lấy từ biến môi trường (config.models)."""
        self.flux = flux
        self.progress = progress or (lambda msg: None)
        self.llm, self.vlm = llm, vlm

    def __enter__(self) -> "Engine":
        from .brief import SCHEMA
        from .browser import Browser
        from .llm import json_call
        config.load_env()
        self.models = config.models()
        m = self.models
        if self.llm is None:
            self.llm = json_call(m["llm"]["url"], m["llm"]["model"], m["llm"]["key_env"], tag="brief", schema=SCHEMA)
        if self.vlm is None:
            self.vlm = json_call(m["vlm"]["url"], m["vlm"]["model"], m["vlm"]["key_env"], tag="designer")
        self.browser = Browser().__enter__()
        return self

    def __exit__(self, *exc) -> None:
        self.browser.__exit__(*exc)

    def brief(self, prompt: str | None = None, form: dict | None = None, design: dict | None = None,
              product_image: bool = False) -> dict:
        from .brief import make_brief
        return make_brief(self.llm, prompt, form, design, product_image)

    def make(self, B: dict, W: int, H: int, seed: int, product: np.ndarray | None = None) -> dict:
        self.progress("FLUX vẽ nháp")
        t0 = time.time()
        draft = self.flux.generate(B["prompt_en"], W, H, seed, refs=[product] if product is not None else None)
        t_draft = round(time.time() - t0, 1)
        r = self.fix(draft, B, seed, product=product is not None)
        r["timing"] = {"ve_nhap": t_draft, **r["timing"]}
        return r

    def fix(self, draft: np.ndarray, B: dict, seed: int = 0, plate: np.ndarray | None = None, product: bool = False) -> dict:
        """product: người dùng tải ảnh sản phẩm (chữ / logo in trên sản phẩm trong nháp -> designer "keep")."""
        from . import cleanup, render, slots
        from .flux import ERASE_PROMPT, erase
        T, t0 = {}, time.time()
        if plate is None:
            self.progress("FLUX xoá lớp phủ")
            plate = erase(self.flux, draft, seed)
            T["xoa"] = time.time() - t0
            t0 = time.time()
        # bước xoá ngẫu nhiên: dọn chữ còn sót (OCR bản xoá) TRƯỚC khi tách ô -> chỗ dọn thành lớp phủ, thành ô như mọi chữ khác
        self.progress("dọn chữ sót")
        fe = (lambda im: self.flux.edit(im, ERASE_PROMPT, seed)) if self.flux is not None else None   # noqa: E731
        plate, cinfo = cleanup.clean(draft, plate, erase=fe)
        T["don"] = time.time() - t0
        t0 = time.time()
        self.progress("đo ô (OCR + lớp phủ)")
        M = slots.build(draft, plate)
        T["o"] = time.time() - t0
        t0 = time.time()
        self.progress("VLM thiết kế")
        P, log = slots.validate(slots.plan(draft, M, B["prompt_en"], B["texts"], self.vlm, product=product, base=plate), M)
        if cinfo["lines"]:
            log.insert(0, f"dọn chữ sót: inpaint {cinfo['inpaint']}, FLUX vùng cắt {cinfo['reerase']}")
        T["vlm"] = time.time() - t0
        t0 = time.time()
        self.progress("dựng + kiểm")
        poster, res, rlog = render.render(self.browser.page, slots.base_image(draft, plate, M, P), P, M)
        errs = render.check(P, M, res)
        log += rlog + [f"LỖI: {e}" for e in errs]
        T["dung"] = time.time() - t0
        if errs:
            t0 = time.time()
            self.progress("VLM sửa lỗi")
            try:
                P2, vlog2 = slots.validate(slots.repair(draft, poster, M, P, errs, B["prompt_en"], B["texts"], self.vlm,
                                                        product=product), M)
                poster2, res2, rlog2 = render.render(self.browser.page, slots.base_image(draft, plate, M, P2), P2, M)
                errs2 = render.check(P2, M, res2)
                log += ["-- vòng sửa lỗi --", f"VLM: {P2.get('repair_why')}"] + vlog2 + rlog2 + [f"LỖI CÒN: {e}" for e in errs2]
                bad = lambda E: (len(E), sum(e.get("frac", 0) + e.get("px", 0) / 100 for e in E))   # noqa: E731  ít lỗi, rồi ít chồng
                if bad(errs2) < bad(errs):
                    poster, P, errs = poster2, P2, errs2
                    log.append("-> giữ bản đã sửa")
                else:
                    log.append("-> bản sửa không tốt hơn, giữ bản lượt đầu")
            except Exception as e:   # vòng sửa hỏng (VLM trả sai, quá giờ): giữ bản lượt đầu
                log.append(f"vòng sửa lỗi hỏng: {type(e).__name__}: {str(e)[:200]}")
            T["sua"] = time.time() - t0
        tid = lambda m: int(m[1:]) if isinstance(m, str) and m[1:].isdigit() else m   # noqa: E731  "T5" -> 5
        missing = [B["texts"][i]["text"] for i in map(tid, P.get("missing") or []) if isinstance(i, int) and 0 <= i < len(B["texts"])]
        plan = {"style": P.get("style"), "ops": P["ops"], "missing": missing, "notes": P.get("notes"), "errors": errs, "log": log,
                "slots": slots.public(M), "cleanup": cinfo}
        steps = _row([draft, slots.marked_image(draft, M), plate, poster])
        return {"draft": draft, "plate": plate, "poster": poster, "steps": steps, "plan": plan,
                "timing": {k: round(v, 1) for k, v in T.items()}}


def _row(ims, h: int = 640) -> np.ndarray:
    """nháp | ô | bản xoá | poster."""
    from PIL import Image
    ims = [Image.fromarray(np.asarray(x)).convert("RGB") for x in ims]
    ims = [im.resize((int(im.width * h / im.height), h)) for im in ims]
    row = Image.new("RGB", (sum(im.width + 10 for im in ims), h), "white")
    x = 0
    for im in ims:
        row.paste(im, (x, 0))
        x += im.width + 10
    return np.asarray(row)
