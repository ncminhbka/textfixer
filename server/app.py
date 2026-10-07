#!/usr/bin/env python3
"""MÁY CHỦ TEXTFIX: form / prompt -> LLM (prompt FLUX + câu khách) -> FLUX.2-klein vẽ nháp -> FLUX.2-klein xoá lớp phủ -> ô (OCR +
lớp phủ) -> VLM designer điền ô -> dựng (Chromium) -> kiểm -> một vòng sửa -> poster.

  python server/app.py --port 8088                  # bản distill (4 bước)
  python server/app.py --port 8088 --model base     # bản base (50 bước, CFG 4)

API (giao diện server/ui.html dùng job + hỏi tiến độ; script dùng /api/generate chờ xong):
  POST /api/jobs         GenerateRequest -> {job_id}
  GET  /api/jobs/{id}    {status: queued|running|done|error, progress, result, error}
  POST /api/generate     GenerateRequest -> kết quả (chờ xong)
  GET  /api/health       model đã nạp, LLM / VLM đang dùng, hàng đợi
Ảnh từng lần chạy: output/runs/<run_id>/ (request.json, brief.json, v<i>/draft.png, plate.png, poster.png, steps.jpg, ops.jpg,
plan.json), phục vụ ở /outputs/<run_id>/...

Mọi việc GPU + Chromium chạy trên MỘT luồng riêng (Playwright bản đồng bộ không chạy trong vòng lặp asyncio; GPU dùng tuần tự):
các yêu cầu xếp hàng.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import shutil
import sys
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", line_buffering=True)

from textfix.config import load_env  # noqa: E402

load_env()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("textfix.server")

RUNS = Path(os.environ.get("TEXTFIX_RUNS", ROOT / "output" / "runs"))
RUNS.mkdir(parents=True, exist_ok=True)
MAX_RUNS = int(os.environ.get("TEXTFIX_MAX_RUNS", "300"))
UI = Path(__file__).resolve().parent / "ui.html"
EXEC = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")
STATE: dict[str, Any] = {"engine": None, "flux": None, "error": None, "args": None, "started": None, "job": None}
JOBS: dict[str, dict] = {}
ORDER: list[str] = []   # hàng đợi (job chưa xong), để báo vị trí

# ---------------------------------------------------------------------------------------------------- form
# trường NỘI DUNG từng loại form (khớp UI); trường chung cửa hàng gửi cho mọi loại
CATEGORY_FIELDS = {
    "promo": ["title", "discount", "applied_product", "date_start", "date_end"],
    "product_intro": ["title", "product_name", "product_desc", "highlights", "price"],
    "opening": ["title", "opening_date", "opening_promo", "booking_contact"],
    "feedback": ["title", "feedback_target", "feedback_quote", "customer_name", "feedback_highlights", "special_offer"],
    "recruitment": ["title", "job_position", "job_desc", "apply_deadline", "apply_method"],
    "guide": ["title"],
}
REQUIRED = {"promo": ["discount"], "product_intro": ["product_name", "product_desc"], "opening": ["opening_date"],
            "feedback": ["feedback_target", "feedback_quote"], "recruitment": ["job_position", "apply_deadline", "apply_method"],
            "guide": ["guide_steps"]}
COMMON = ["store_name", "phone", "address", "website_link"]


class GenerateRequest(BaseModel):
    category: str = "promo"
    # thông tin chung theo loại
    title: Optional[str] = ""
    discount: Optional[str] = ""
    applied_product: Optional[str] = ""
    date_start: Optional[str] = ""
    date_end: Optional[str] = ""
    product_name: Optional[str] = ""
    product_desc: Optional[str] = ""
    highlights: Optional[str] = ""
    price: Optional[str] = ""
    opening_date: Optional[str] = ""
    opening_promo: Optional[str] = ""
    booking_contact: Optional[str] = ""
    feedback_target: Optional[str] = ""
    feedback_quote: Optional[str] = ""
    customer_name: Optional[str] = ""
    feedback_highlights: Optional[str] = ""
    special_offer: Optional[str] = ""
    job_position: Optional[str] = ""
    job_desc: Optional[str] = ""
    apply_deadline: Optional[str] = ""
    apply_method: Optional[str] = ""
    guide_steps: Optional[List[str]] = None
    prompt: Optional[str] = ""
    image_base64: Optional[str] = None   # ảnh sản phẩm tải lên (data URI hoặc base64)
    # cửa hàng
    store_name: Optional[str] = ""
    phone: Optional[str] = ""
    address: Optional[str] = ""
    website_link: Optional[str] = ""
    # hiển thị / thiết kế
    style_pref: Optional[str] = "auto"
    primary_color: Optional[str] = None
    aspect_ratio: str = "1:1"
    num_images: int = 1
    seed: Optional[int] = None


def form_of(req: GenerateRequest) -> dict:
    """Chỉ trường thuộc loại form đang chọn + trường cửa hàng, bỏ ô trống. Các bước hướng dẫn -> step_1, step_2..."""
    cat = req.category if req.category in CATEGORY_FIELDS else "promo"
    out: dict = {"category": cat}
    for k in CATEGORY_FIELDS[cat] + COMMON:
        v = (getattr(req, k) or "").strip()
        if v:
            out[k] = v
    if cat == "guide":
        for i, s in enumerate([s.strip() for s in req.guide_steps or [] if s and s.strip()], 1):
            out[f"step_{i}"] = s
    return out


def missing_fields(req: GenerateRequest, form: dict) -> list[str]:
    """Trường bắt buộc -- chỉ khi KHÔNG có prompt (có prompt thì prompt quyết)."""
    if (req.prompt or "").strip():
        return []
    cat = form["category"]
    if cat == "guide":
        return [] if any(k.startswith("step_") for k in form) else ["guide_steps"]
    return [k for k in REQUIRED.get(cat, []) if k not in form]


def decode_image(b64: str | None) -> np.ndarray | None:
    if not b64:
        return None
    from PIL import Image
    raw = base64.b64decode(b64.split(",", 1)[1] if "," in b64 else b64)
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    im.thumbnail((1024, 1024))
    return np.asarray(im)


# ---------------------------------------------------------------------------------------------------- việc (luồng GPU)
def init_models(args) -> None:
    """Chạy TRÊN luồng GPU: FLUX + Engine (Chromium, LLM / VLM) + làm nóng OCR."""
    from textfix.engine import Engine
    from textfix.flux import Flux
    from textfix.ocr import _engine
    t0 = time.time()
    if not os.environ.get("TEXTFIX_LLM_URL") or not os.environ.get("TEXTFIX_MODEL"):
        raise RuntimeError("chưa đặt TEXTFIX_LLM_URL / TEXTFIX_MODEL (export trong terminal, docs/SERVER.md)")
    F = Flux(args.model, steps=args.steps, weights_dir=args.weights_dir).load()
    log.info("FLUX sẵn sàng: %s (%.0fs)", F.info(), time.time() - t0)

    def progress(msg: str) -> None:
        job = STATE.get("job")
        if job is not None:
            job["progress"] = job.get("tag", "") + msg

    STATE["engine"] = Engine(F, progress=progress).__enter__()
    _engine()(np.full((64, 256, 3), 255, np.uint8))   # làm nóng OCR (lần đầu nạp mô hình ONNX)
    STATE["flux"] = F
    log.info("Engine sẵn sàng (Chromium, OCR); LLM %s | VLM %s (%.0fs)", STATE["engine"].models["llm"]["model"],
             STATE["engine"].models["vlm"]["model"], time.time() - t0)
    try:
        import torch
        for i in range(torch.cuda.device_count()):
            log.info("  cuda:%d dùng %.1f GB", i, torch.cuda.memory_allocated(i) / 2 ** 30)
    except Exception:
        pass


def save(img: np.ndarray, path: Path, quality: int | None = None) -> None:
    from PIL import Image
    Image.fromarray(np.asarray(img)).save(path, **({"quality": quality} if quality else {}))


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return None
    return o


def run_job(job: dict, req: GenerateRequest) -> dict:
    """Một yêu cầu: LLM một lần, rồi mỗi ảnh: nháp -> xoá -> ô -> VLM -> dựng -> kiểm / sửa. Lỗi một ảnh không dừng ảnh khác."""
    from textfix.flux import SIZES
    from textfix.render import ops_image
    E = STATE["engine"]
    STATE["job"] = job
    t_all = time.time()
    run_id = job["run_id"]
    rd = RUNS / run_id
    rd.mkdir(parents=True, exist_ok=True)
    form = form_of(req)
    prompt = (req.prompt or "").strip() or None
    product = decode_image(req.image_base64)
    design = {"style": req.style_pref or "auto", "color": req.primary_color or None}
    (rd / "request.json").write_text(json.dumps({**req.model_dump(exclude={"image_base64"}), "has_product_image": product is not None},
                                                ensure_ascii=False, indent=1), encoding="utf-8")
    if product is not None:
        save(product, rd / "product.png")

    job["tag"], job["progress"] = "", "LLM viết prompt cho FLUX"
    t0 = time.time()
    B = E.brief(prompt=prompt, form=form if len(form) > 1 else None, design=design, product_image=product is not None)
    t_llm = round(time.time() - t0, 1)
    (rd / "brief.json").write_text(json.dumps(B, ensure_ascii=False, indent=1), encoding="utf-8")
    if not B["texts"]:
        raise ValueError("LLM không tìm ra chữ nào để vẽ (form trống?)")

    W, H = SIZES.get(req.aspect_ratio, SIZES["1:1"])
    n = max(1, min(4, int(req.num_images or 1)))
    base_seed = req.seed if req.seed is not None else int(time.time()) % 100000
    posters = []
    for i in range(n):
        seed = base_seed + i * 9973
        vd = rd / f"v{i}"
        vd.mkdir(exist_ok=True)
        url = f"outputs/{run_id}/v{i}"
        job["tag"] = f"Ảnh {i + 1}/{n}: "
        try:
            r = E.make(B, W, H, seed, product)
            for k in ("draft", "plate", "poster"):
                save(r[k], vd / f"{k}.png")
            save(r["steps"], vd / "steps.jpg", 88)
            P = r["plan"]
            M = {**P["slots"], "size": [W, H], "by": {m["id"]: m for m in P["slots"]["L"] + P["slots"]["S"] + P["slots"]["I"]}}
            save(np.asarray(ops_image(r["draft"], M, P)), vd / "ops.jpg", 88)
            timing = {"llm": t_llm, **r["timing"]}
            (vd / "plan.json").write_text(json.dumps(jsonable({"seed": seed, "timing": timing, **P}), ensure_ascii=False, indent=1),
                                          encoding="utf-8")
            posters.append({"index": i, "seed": seed, "poster_url": f"{url}/poster.png", "draft_url": f"{url}/draft.png",
                            "plate_url": f"{url}/plate.png", "steps_url": f"{url}/steps.jpg", "ops_url": f"{url}/ops.jpg",
                            "timing": timing, "missing": P["missing"], "errors": len(P["errors"]),
                            "ops": sum(o.get("kind") not in ("skip", "keep") for o in P["ops"])})
            log.info("%s v%d xong %s", run_id, i, timing)
        except Exception as e:
            log.error("%s v%d lỗi: %s", run_id, i, traceback.format_exc())
            posters.append({"index": i, "seed": seed, "error": f"{type(e).__name__}: {e}",
                            **({"draft_url": f"{url}/draft.png"} if (vd / "draft.png").exists() else {})})
    STATE["job"] = None
    ok = [p for p in posters if "error" not in p]
    if not ok:
        raise RuntimeError(posters[0]["error"])
    prune()
    return {"status": "success", "run_id": run_id, "elapsed_seconds": round(time.time() - t_all, 1), "width": W, "height": H,
            "final_poster_url": ok[0]["poster_url"], "posters": posters,
            "brief": {"prompt_en": B["prompt_en"], "texts": B["texts"], "log": B["log"]}}


def prune() -> None:
    ds = sorted([d for d in RUNS.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime)
    for d in ds[:-MAX_RUNS]:
        shutil.rmtree(d, ignore_errors=True)


def new_job(req: GenerateRequest) -> dict:
    form = form_of(req)
    miss = missing_fields(req, form)
    if miss:
        raise HTTPException(422, {"error": "missing_required_fields", "category": form["category"], "fields": miss})
    if STATE["flux"] is None:
        raise HTTPException(503, f"model chưa sẵn sàng: {STATE['error'] or 'đang nạp'}")
    jid = uuid.uuid4().hex[:12]
    job = {"id": jid, "run_id": f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{jid[:4]}", "status": "queued",
           "progress": "đang xếp hàng", "created": time.time(), "result": None, "error": None}
    JOBS[jid] = job
    ORDER.append(jid)
    return job


def work(job: dict, req: GenerateRequest) -> None:
    job["status"], job["started"] = "running", time.time()
    try:
        job["result"] = run_job(job, req)
        job["status"] = "done"
    except Exception as e:
        log.error("job %s lỗi: %s", job["id"], traceback.format_exc())
        job["status"], job["error"] = "error", f"{type(e).__name__}: {e}"
    finally:
        job["progress"] = ""
        if job["id"] in ORDER:
            ORDER.remove(job["id"])
        for k in [k for k, j in JOBS.items() if j["status"] in ("done", "error") and time.time() - j["created"] > 6 * 3600]:
            JOBS.pop(k, None)


# ---------------------------------------------------------------------------------------------------- HTTP
@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["started"] = time.time()
    try:
        await asyncio.get_running_loop().run_in_executor(EXEC, init_models, STATE["args"])
    except BaseException as e:   # gồm SystemExit của flux2.util khi thiếu trọng số: máy chủ vẫn lên, /api/health báo lỗi
        if isinstance(e, KeyboardInterrupt):
            raise
        STATE["error"] = f"{type(e).__name__}: {e}"
        log.error("KHÔNG nạp được model: %s", traceback.format_exc())
    yield
    if STATE["engine"] is not None:
        EXEC.submit(STATE["engine"].__exit__, None, None, None).result()


app = FastAPI(title="TextFix", lifespan=lifespan)
app.mount("/outputs", StaticFiles(directory=str(RUNS)), name="outputs")


@app.get("/")
async def ui():
    return FileResponse(UI, media_type="text/html")


@app.get("/api/health")
async def health():
    E, F = STATE["engine"], STATE["flux"]
    return {"ready": F is not None and E is not None, "error": STATE["error"],
            "flux": F.info() if F else None,
            "models": {k: {"url": v["url"], "model": v["model"]} for k, v in E.models.items()} if E else None, "queue": len(ORDER),
            "uptime_s": round(time.time() - (STATE["started"] or time.time()))}


@app.post("/api/jobs")
async def create_job(req: GenerateRequest):
    job = new_job(req)
    asyncio.get_running_loop().run_in_executor(EXEC, work, job, req)
    return {"job_id": job["id"], "run_id": job["run_id"], "queue": len(ORDER)}


@app.get("/api/jobs/{jid}")
async def get_job(jid: str):
    job = JOBS.get(jid)
    if job is None:
        raise HTTPException(404, "không có job này (máy chủ đã khởi động lại?)")
    pos = ORDER.index(jid) if jid in ORDER else None
    return {k: job[k] for k in ("id", "run_id", "status", "progress", "result", "error")} | \
        {"position": pos, "elapsed": round(time.time() - job.get("started", job["created"]), 1)}


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    job = new_job(req)
    await asyncio.get_running_loop().run_in_executor(EXEC, work, job, req)
    if job["status"] != "done":
        return JSONResponse({"status": "error", "error": job["error"], "run_id": job["run_id"]}, status_code=500)
    return job["result"]


def main() -> None:
    import uvicorn
    ap = argparse.ArgumentParser(description="Máy chủ TextFix")
    ap.add_argument("--host", default=os.environ.get("TEXTFIX_HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("TEXTFIX_PORT", "8088")))
    ap.add_argument("--model", choices=["distill", "base"], default=os.environ.get("TEXTFIX_FLUX", "distill"))
    ap.add_argument("--steps", type=int, default=int(os.environ["TEXTFIX_STEPS"]) if os.environ.get("TEXTFIX_STEPS") else None,
                    help="số bước khử nhiễu (mặc định BFL: distill 4, base 50)")
    ap.add_argument("--weights-dir", default=os.environ.get("TEXTFIX_WEIGHTS"),
                    help="thư mục trọng số FLUX.2-klein (nếu không ở ~/persistent-data)")
    STATE["args"] = ap.parse_args()
    uvicorn.run(app, host=STATE["args"].host, port=STATE["args"].port)


if __name__ == "__main__":
    main()
