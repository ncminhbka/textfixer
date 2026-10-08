#!/usr/bin/env python3
"""MÁY CHỦ TEXTFIX: form / prompt -> LLM (prompt FLUX + câu khách + các hướng thiết kế) -> mỗi phiên bản: FLUX.2-klein vẽ nháp ->
FLUX.2-klein xoá lớp phủ -> dọn chữ sót -> ô (OCR + lớp phủ) -> VLM designer điền ô -> dựng (Chromium) -> kiểm -> vòng sửa -> poster.

  python server/app.py --port 8088                  # bản distill (4 bước)
  python server/app.py --port 8088 --model base     # bản base (50 bước, CFG 4)

API (giao diện server/ui.html dùng job + hỏi tiến độ; script dùng /api/generate chờ xong):
  POST /api/jobs         GenerateRequest -> {job_id}
  GET  /api/jobs/{id}    {status: queued|running|done|error, progress, result, error}
  POST /api/generate     GenerateRequest -> kết quả (chờ xong)
  GET  /api/health       model đã nạp, LLM / VLM đang dùng, hàng đợi
Ảnh từng lần chạy: output/runs/<run_id>/ (request.json, brief.json, v<i>/draft.png, plate_raw.png (bản xoá FLUX), plate.png (đã dọn),
slots.jpg, poster.png, poster_first.png (lượt đầu, khi vòng sửa đổi), steps.jpg, ops.jpg, plan.json), phục vụ ở /outputs/<run_id>/...

DÂY CHUYỀN: mỗi card GPU một FLUX + một luồng VẼ (chung hàng đợi, TEXTFIX_GPUS="0,1" mặc định mọi card), nhóm luồng THIẾT KẾ
(TEXTFIX_DESIGN_WORKERS=4, mỗi luồng một Chromium -- Playwright đồng bộ không chạy trong vòng lặp asyncio). Nhiều phiên bản / nhiều
yêu cầu gối đầu nhau; job ghi kết quả từng phiên bản ngay khi xong (UI hiện dần).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import queue
import shutil
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
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
STATE: dict[str, Any] = {"flux": [], "llm": None, "vlm": None, "models": None, "error": None, "args": None, "started": None}
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


# ---------------------------------------------------------------------------------------------------- dây chuyền
# Mỗi phiên bản đi qua hai trạm, các phiên bản (và các yêu cầu) gối đầu nhau:
#   VẼ (GPU)        mỗi card một FLUX + một luồng, chung một hàng đợi (vào trước ra trước)  nháp -> xoá -> dọn chữ sót  ~7 s
#   THIẾT KẾ        nhóm luồng, mỗi luồng một Chromium; VLM ở máy khác nên gọi song song    ô -> VLM -> dựng -> kiểm   ~15 s
# LLM viết brief (một lời gọi / yêu cầu, kèm các hướng thiết kế) chạy ở luồng điều phối của yêu cầu.
DRAW_Q: "queue.Queue" = queue.Queue()
DESIGN = ThreadPoolExecutor(max_workers=int(os.environ.get("TEXTFIX_DESIGN_WORKERS", "4")), thread_name_prefix="design")
JOBRUN = ThreadPoolExecutor(max_workers=int(os.environ.get("TEXTFIX_JOB_WORKERS", "8")), thread_name_prefix="job")
LOCAL = threading.local()   # Chromium của từng luồng thiết kế (Playwright đồng bộ: mỗi luồng một bản)


def gpu_list() -> list[int]:
    env = os.environ.get("TEXTFIX_GPUS")
    if env:
        return [int(x) for x in env.split(",") if x.strip()]
    import torch
    return list(range(max(1, torch.cuda.device_count())))


def gpu_worker(k: int, args, ready: threading.Event) -> None:
    """Luồng của card k: nạp FLUX lên card đó, rồi lấy việc VẼ từ hàng đợi chung."""
    import torch
    from textfix.engine import draw
    from textfix.flux import Flux
    try:
        torch.cuda.set_device(k)
        t0 = time.time()
        F = Flux(args.model, steps=args.steps, weights_dir=args.weights_dir, device=f"cuda:{k}").load()
        STATE["flux"].append(F)
        log.info("FLUX cuda:%d sẵn sàng (%.0fs, %.1f GB)", k, time.time() - t0, torch.cuda.memory_allocated(k) / 2 ** 30)
    except BaseException as e:   # gồm SystemExit của flux2.util khi thiếu trọng số
        STATE["error"] = f"cuda:{k}: {type(e).__name__}: {e}"
        log.error("KHÔNG nạp được FLUX cuda:%d: %s", k, traceback.format_exc())
        return
    finally:
        ready.set()
    while True:
        task = DRAW_Q.get()
        if task is None:
            return
        v, fut = task["v"], task["fut"]
        if fut.set_running_or_notify_cancel():
            try:
                v["gpu"] = k
                fut.set_result(draw(F, task["B"], task["W"], task["H"], v["seed"], task["product"], progress=task["progress"]))
            except BaseException as e:
                fut.set_exception(e)


def init_models(args) -> None:
    """LLM / VLM (gọi qua mạng), một luồng VẼ cho mỗi card, làm nóng OCR. Chạy ở luồng phụ (lúc khởi động)."""
    from textfix import config
    from textfix.brief import SCHEMA
    from textfix.llm import json_call
    from textfix.ocr import _engine
    if not os.environ.get("TEXTFIX_LLM_URL") or not os.environ.get("TEXTFIX_MODEL"):
        raise RuntimeError("chưa đặt TEXTFIX_LLM_URL / TEXTFIX_MODEL (export trong terminal, docs/SERVER.md)")
    m = config.models()
    STATE["models"] = m
    STATE["llm"] = json_call(m["llm"]["url"], m["llm"]["model"], m["llm"]["key_env"], tag="brief", schema=SCHEMA)
    STATE["vlm"] = json_call(m["vlm"]["url"], m["vlm"]["model"], m["vlm"]["key_env"], tag="designer")
    _engine()(np.full((64, 256, 3), 255, np.uint8))   # làm nóng OCR (lần đầu nạp mô hình ONNX)
    gpus = gpu_list()
    evs = []
    for k in gpus:
        ev = threading.Event()
        threading.Thread(target=gpu_worker, args=(k, args, ev), name=f"gpu{k}", daemon=True).start()
        evs.append(ev)
    for ev in evs:
        ev.wait()
    if not STATE["flux"]:
        raise RuntimeError(STATE["error"] or "không nạp được FLUX trên card nào")
    log.info("Sẵn sàng: %d card vẽ %s, %d luồng thiết kế; LLM %s | VLM %s", len(STATE["flux"]), gpus, DESIGN._max_workers,
             m["llm"]["model"], m["vlm"]["model"])


def page():
    """Chromium của luồng thiết kế hiện tại (mở lần đầu dùng)."""
    if getattr(LOCAL, "browser", None) is None:
        from textfix.browser import Browser
        LOCAL.browser = Browser().__enter__()
    return LOCAL.browser.page


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


def design_one(job: dict, v: dict, d: dict, Bv: dict, product: bool, vd: Path, url: str, t_llm: float) -> None:
    """TRẠM THIẾT KẾ cho một phiên bản + lưu ảnh / JSON debug. Lỗi -> v["error"] (không dừng phiên bản khác)."""
    from textfix import slots
    from textfix.engine import design
    from textfix.render import ops_image
    try:
        for k in ("draft", "plate_raw", "plate"):   # có ngay khi vẽ xong: UI xem được nháp / bản xoá trong lúc thiết kế
            save(d[k], vd / f"{k}.png")
        v.update(draft_url=f"{url}/draft.png", plate_raw_url=f"{url}/plate_raw.png", plate_url=f"{url}/plate.png")
        r = design(d, Bv, STATE["vlm"], page(), product=product, progress=lambda m: v.update(stage=m))
        save(r["poster"], vd / "poster.png")
        save(r["steps"], vd / "steps.jpg", 88)
        P = r["plan"]
        M = {**P["slots"], "size": [d["draft"].shape[1], d["draft"].shape[0]],
             "by": {m["id"]: m for m in P["slots"]["L"] + P["slots"]["S"] + P["slots"]["I"]}}
        save(np.asarray(ops_image(r["draft"], M, P)), vd / "ops.jpg", 88)
        save(np.asarray(slots.marked_image(r["draft"], M)), vd / "slots.jpg", 88)
        first = r["poster_first"] is not r["poster"]
        if first:
            save(r["poster_first"], vd / "poster_first.png")
        timing = {"llm": t_llm, **r["timing"]}
        (vd / "plan.json").write_text(json.dumps(jsonable({"seed": v["seed"], "direction": v.get("direction"), "prompt_en": Bv["prompt_en"],
                                                           "timing": timing, **P}), ensure_ascii=False, indent=1), encoding="utf-8")
        v.update(status="done", stage="", poster_url=f"{url}/poster.png", steps_url=f"{url}/steps.jpg", ops_url=f"{url}/ops.jpg",
                 slots_url=f"{url}/slots.jpg", plan_url=f"{url}/plan.json", timing=timing, missing=P["missing"],
                 errors=len(P["errors"]), ops=sum(o.get("kind") not in ("skip", "keep") for o in P["ops"]),
                 **({"poster_first_url": f"{url}/poster_first.png"} if first else {}))
        log.info("%s v%d xong (gpu %s) %s", job["run_id"], v["index"], v.get("gpu"), timing)
    except Exception as e:
        log.error("%s v%d lỗi thiết kế: %s", job["run_id"], v["index"], traceback.format_exc())
        v.update(status="error", stage="", error=f"{type(e).__name__}: {e}")


def run_job(job: dict, req: GenerateRequest) -> dict:
    """Một yêu cầu: LLM một lần (brief + các hướng thiết kế), rồi n phiên bản vào dây chuyền VẼ -> THIẾT KẾ. Kết quả từng phiên
    bản ghi vào job["result"]["posters"] ngay khi xong (UI hiện dần)."""
    from textfix.brief import make_brief
    from textfix.flux import SIZES
    t_all = time.time()
    run_id = job["run_id"]
    rd = RUNS / run_id
    rd.mkdir(parents=True, exist_ok=True)
    form = form_of(req)
    prompt = (req.prompt or "").strip() or None
    product = decode_image(req.image_base64)
    design_hint = {"style": req.style_pref or "auto", "color": req.primary_color or None}
    (rd / "request.json").write_text(json.dumps({**req.model_dump(exclude={"image_base64"}), "has_product_image": product is not None},
                                                ensure_ascii=False, indent=1), encoding="utf-8")
    if product is not None:
        save(product, rd / "product.png")
    n = max(1, min(4, int(req.num_images or 1)))
    job["progress"] = "LLM viết prompt" + (f" + {n - 1} hướng thiết kế" if n > 1 else "")
    t0 = time.time()
    B = make_brief(STATE["llm"], prompt, form if len(form) > 1 else None, design_hint, product is not None, n_variants=n - 1)
    t_llm = round(time.time() - t0, 1)
    (rd / "brief.json").write_text(json.dumps(B, ensure_ascii=False, indent=1), encoding="utf-8")
    if not B["texts"]:
        raise ValueError("LLM không tìm ra chữ nào để vẽ (form trống?)")
    W, H = SIZES.get(req.aspect_ratio, SIZES["1:1"])
    base_seed = req.seed if req.seed is not None else int(time.time()) % 100000
    directions = [{"name": "chính", "prompt_en": B["prompt_en"]}] + B.get("variants", [])
    posters = []
    job["result"] = {"status": "running", "run_id": run_id, "width": W, "height": H, "posters": posters,
                     "brief": {"prompt_en": B["prompt_en"], "texts": B["texts"], "variants": B.get("variants", []), "log": B["log"]}}
    futs = []
    for i in range(n):
        dirn = directions[i] if i < len(directions) else directions[0]   # thiếu hướng: dùng prompt chính, seed khác
        v = {"index": i, "seed": base_seed + i * 9973, "direction": dirn["name"], "status": "drawing", "stage": "chờ GPU"}
        posters.append(v)
        vd = rd / f"v{i}"
        vd.mkdir(exist_ok=True)
        Bv = {**B, "prompt_en": dirn["prompt_en"]}
        fut: Future = Future()
        DRAW_Q.put({"v": v, "fut": fut, "B": Bv, "W": W, "H": H, "product": product,
                    "progress": lambda m, v=v: v.update(stage=m)})

        def drawn(f, v=v, Bv=Bv, vd=vd, i=i):
            if f.exception() is not None:
                log.error("%s v%d lỗi vẽ: %s", run_id, i, f.exception())
                v.update(status="error", stage="", error=f"{type(f.exception()).__name__}: {f.exception()}")
                return
            v.update(status="designing", stage="chờ thiết kế")
            futs.append(DESIGN.submit(design_one, job, v, f.result(), Bv, product is not None, vd, f"outputs/{run_id}/v{i}", t_llm))
        fut.add_done_callback(drawn)
        futs.append(fut)
    while True:   # chờ mọi phiên bản (danh sách futs lớn dần khi bản vẽ xong được đưa sang thiết kế)
        pending = [f for f in list(futs) if not f.done()]
        if not pending and all(v["status"] in ("done", "error") for v in posters):
            break
        done = sum(v["status"] == "done" for v in posters)
        job["progress"] = f"{done}/{n} xong · " + " · ".join(f"ảnh {v['index'] + 1}: {v['stage']}" for v in posters
                                                                if v["status"] not in ("done", "error"))
        time.sleep(0.3)
    ok = [p for p in posters if p["status"] == "done"]
    if not ok:
        raise RuntimeError(posters[0].get("error") or "mọi phiên bản đều lỗi")
    prune()
    job["result"].update(status="success", elapsed_seconds=round(time.time() - t_all, 1), final_poster_url=ok[0]["poster_url"])
    return job["result"]


def prune() -> None:
    ds = sorted([d for d in RUNS.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime)
    for d in ds[:-MAX_RUNS]:
        shutil.rmtree(d, ignore_errors=True)


def new_job(req: GenerateRequest) -> dict:
    form = form_of(req)
    miss = missing_fields(req, form)
    if miss:
        raise HTTPException(422, {"error": "missing_required_fields", "category": form["category"], "fields": miss})
    if not STATE["flux"] or STATE.get("vlm") is None:
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
        await asyncio.get_running_loop().run_in_executor(None, init_models, STATE["args"])
    except BaseException as e:   # máy chủ vẫn lên, /api/health báo lỗi
        if isinstance(e, KeyboardInterrupt):
            raise
        STATE["error"] = STATE["error"] or f"{type(e).__name__}: {e}"
        log.error("KHÔNG nạp được model: %s", traceback.format_exc())
    yield
    for _ in STATE["flux"]:
        DRAW_Q.put(None)


app = FastAPI(title="TextFix", lifespan=lifespan)
app.mount("/outputs", StaticFiles(directory=str(RUNS)), name="outputs")


@app.get("/")
async def ui():
    return FileResponse(UI, media_type="text/html")


@app.get("/api/health")
async def health():
    F, m = STATE["flux"], STATE.get("models")
    return {"ready": bool(F) and STATE.get("vlm") is not None, "error": STATE["error"],
            "flux": F[0].info() if F else None, "gpus": [f.dev_dit for f in F],
            "models": {k: {"url": v["url"], "model": v["model"]} for k, v in m.items()} if m else None,
            "queue": len(ORDER), "draw_queue": DRAW_Q.qsize(), "design_workers": DESIGN._max_workers,
            "uptime_s": round(time.time() - (STATE["started"] or time.time()))}


@app.post("/api/jobs")
async def create_job(req: GenerateRequest):
    job = new_job(req)
    JOBRUN.submit(work, job, req)
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
    await asyncio.wrap_future(JOBRUN.submit(work, job, req))
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
