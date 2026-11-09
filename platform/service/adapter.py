"""Voice Guard 课程平台 HTTP 适配服务。

符合 lym-lab/algorithm_research_frontend 的算法接入规范（docs/算法接入规范.md）：

    POST /process   multipart/form-data
        defend_audio   文件     必填，要保护的音频（wav/mp3/flac 等）
        adv_audio      文件     可选，扰动方向音频（缺省自动选 least-likely 说话人）
        method_id      字符串   方法 ID
        parameters     JSON 字符串 例如 {"eps":0.05,"fast_mode":true,"target_speaker":"p225"}

    返回 200 JSON：
        {"message": "...", "artifacts": [{"kind":"audio","label":"保护后音频",
          "filename":"defended.wav","url":"http://host/files/<job>/defended.wav"}, ...]}

其他端点：
    GET  /health    健康检查
    GET  /files/... 结果文件（CORS 已放开，供浏览器预览/下载）

启动（GPU）：
    python platform/service/adapter.py \
        --model-dir checkpoints/adain-vc-official \
        --registry outputs/registry.npz --attack-registry outputs/registry_vc.npz \
        --vctk data/VCTK22 --device cuda --host 0.0.0.0 --port 8020
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import shutil
import sys
import time
import traceback
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fastapi import FastAPI, File, Form, Request, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from starlette.concurrency import run_in_threadpool  # noqa: E402
from vguard.audio import load_audio  # noqa: E402
from vguard.evaluation import identity_verdict  # noqa: E402

ARGS = None
PIPE = None
SPEAKER_FIRST: dict[str, str] = {}
GPU_LOCK = asyncio.Lock()


def _build_pipeline(args):
    from vguard.pipeline import Pipeline
    pipe = Pipeline(args.model_dir, args.registry, args.attack_registry,
                    ecapa_dir=args.ecapa_dir, device=args.device)
    if args.vctk and os.path.isdir(args.vctk):
        for s in pipe.speakers:
            d = os.path.join(args.vctk, s)
            if os.path.isdir(d):
                wavs = sorted(f for f in os.listdir(d) if f.endswith(".wav"))
                if wavs:
                    SPEAKER_FIRST[s] = os.path.join(d, wavs[0])
    return pipe


def _auto_speaker(pipe, wav_path: str) -> str:
    """用评估 ASV（ECAPA）判上传音频的说话人，作为被保护说话人。"""
    wav = load_audio(wav_path, pipe.pp["sample_rate"])
    res = pipe._ecapa_asv(wav)  # noqa: SLF001
    return res["pred"]


app = FastAPI(title="Voice Guard 算法服务", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok" if PIPE is not None else "not_ready", "service": "voice-guard-adapter"}


def parse_parameters(params):
    eps = float(params.get("eps", .05))
    if not math.isfinite(eps) or not .02 <= eps <= .3:
        raise ValueError("eps 必须在 0.02–0.3 之间")
    fast = params.get("fast_mode", True)
    context = params.get("preserve_context", False)
    if not isinstance(fast, bool) or not isinstance(context, bool):
        raise ValueError("fast_mode/preserve_context 必须是布尔值")
    method = params.get("method", "vg")
    if method not in ("vg", "vg-projected"):
        raise ValueError("method 必须为 vg 或 vg-projected")
    return eps, fast, context, method


@app.post("/process")
async def process(
    request: Request,
    defend_audio: UploadFile | None = File(default=None),
    adv_audio: UploadFile | None = File(default=None),
    method_id: str = Form(default="group_xx_voice_guard"),
    parameters: str = Form(default="{}"),
):
    try:
        params = json.loads(parameters or "{}")
        if not isinstance(params, dict):
            raise ValueError("parameters 必须是 JSON 对象")
        eps, fast, context, method = parse_parameters(params)
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=400, content={"error": f"parameters 解析失败：{e}"})

    if defend_audio is None:
        return JSONResponse(status_code=400, content={"error": "缺少必填素材 defend_audio"})
    if PIPE is None:
        return JSONResponse(status_code=503, content={"error": "模型尚未就绪"})

    job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
    job_dir = os.path.join(ARGS.out_dir, job_id)
    in_dir = os.path.join(job_dir, "in")
    os.makedirs(in_dir, exist_ok=True)

    def _save(uf: UploadFile, name: str) -> str:
        ext = os.path.splitext(uf.filename or "")[1].lower() or ".wav"
        dst = os.path.join(in_dir, name + ext)
        with open(dst, "wb") as f:
            shutil.copyfileobj(uf.file, f)
        return dst

    defend_path = _save(defend_audio, "defend")
    adv_path = _save(adv_audio, "adv") if adv_audio is not None else None

    it1, it2 = (300, 200) if fast else (ARGS.iters1, ARGS.iters2)

    def compute():
        nonlocal adv_path
        target = str(params.get("target_speaker") or "").strip()
        if target not in PIPE.speakers:
            target = _auto_speaker(PIPE, defend_path)

        if adv_path is None:
            y = PIPE.least_likely(target)
            if y not in SPEAKER_FIRST:
                raise ValueError(f"缺少 least-likely 说话人 {y} 的音频，请上传 adv_audio")
            adv_path = SPEAKER_FIRST[y]

        rep = PIPE.protect(defend_path, target, adv_path, eps=eps,
                           iters1=it1, iters2=it2, out_dir=job_dir, progress=False,
                           refine_steps=1000 if method == "vg-projected" else 0,
                           preserve_context=context)
        return target, rep

    try:
        async with GPU_LOCK:
            target, rep = await run_in_threadpool(compute)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"处理失败：{type(e).__name__}: {e}"})

    base = str(request.base_url).rstrip("/")

    def url_of(fn: str) -> str:
        return f"{base}/files/{job_id}/{fn}"

    files = rep["files"]
    artifacts = []
    for key, label in (("original", "原始音频（完整上下文）" if context else "原始音频（trim 后）"),
                       ("defended", "保护后音频（含时域扰动）"),
                       ("delta", "扰动本身（可试听）"),
                       ("conv_clean", "对照：干净音频经 VC 转换"),
                       ("conv_def", "被保护音频经 VC 转换（需核查身份与质量）")):
        fp = files.get(key)
        if fp and os.path.exists(fp):
            artifacts.append({"kind": "audio", "label": label,
                              "filename": os.path.basename(fp), "url": url_of(os.path.basename(fp))})
    rp = os.path.join(job_dir, "report.json")
    if os.path.exists(rp):
        artifacts.append({"kind": "json", "label": "处理报告（参数/判定/相似度）",
                          "filename": "report.json", "url": url_of("report.json")})

    asv = rep.get("asv", {})
    imp_pred = asv.get("defended", {}).get("pred")
    conv_pred = asv.get("conv_def", {}).get("pred")
    assessed = identity_verdict(asv, target)
    ok = assessed["screened_identity_ok"]
    verdict = "身份维通过，质量尚需独立评估" if ok else "未通过筛选后的身份维检查（详见 report.json）"

    return {
        "message": f"处理完成：说话人 {target}，ε={eps}，{verdict}",
        "artifacts": artifacts,
        "meta": {"job_id": job_id, "method_id": method_id, "target_speaker": target,
                 "eps": eps, "iters": [it1, it2], "elapsed_s": rep.get("elapsed_s"),
                 "defended_pred": imp_pred, "converted_pred": conv_pred, "defense_ok": ok,
                 "identity_verdict": assessed, "quality_evaluated": False,
                 "method": method, "preserve_context": context},
    }


def main():
    global ARGS, PIPE
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--attack-registry", required=True)
    ap.add_argument("--vctk", default="")
    ap.add_argument("--ecapa-dir", default="checkpoints/ecapa-local")
    ap.add_argument("--out-dir", default="outputs/platform_jobs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--iters1", type=int, default=1500)
    ap.add_argument("--iters2", type=int, default=750)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8020)
    ARGS = ap.parse_args()

    os.makedirs(ARGS.out_dir, exist_ok=True)
    PIPE = _build_pipeline(ARGS)
    app.mount("/files", StaticFiles(directory=ARGS.out_dir), name="files")

    import uvicorn
    uvicorn.run(app, host=ARGS.host, port=ARGS.port)


if __name__ == "__main__":
    main()
