import os, uuid, time, asyncio, subprocess, random
from pathlib import Path
from typing import Optional
import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import uvicorn

COMFY_DIR  = "/u01/vt_media/stable-diffusion-webui/ComfyUI"
COMFY_VENV = f"{COMFY_DIR}/venv_311/bin/python"
COMFY_URL  = "http://127.0.0.1:8188"
CKPT_NAME  = "ltx-2.3-22b-distilled-fp8.safetensors"
TE_NAME    = "gemma_3_12B_it_fp8_scaled.safetensors"
OUTPUT_DIR = Path(f"{COMFY_DIR}/output")
API_PORT   = 7299
NEG = "low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly"

jobs: dict = {}
comfy_proc: Optional[subprocess.Popen] = None


def start_comfyui():
    global comfy_proc
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"
    log = open("/u01/vt_media/comfyui_api.log", "w")
    comfy_proc = subprocess.Popen(
        [COMFY_VENV, "main.py", "--listen", "127.0.0.1", "--port", "8188"],
        cwd=COMFY_DIR, env=env, stdout=log, stderr=log,
    )
    print(f"[LTX] ComfyUI PID={comfy_proc.pid}")


def wait_comfyui(timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"{COMFY_URL}/system_stats", timeout=3)
            if r.status_code == 200:
                print("[LTX] ComfyUI ready")
                return
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError("ComfyUI did not start in time")


def fix_frames(frames):
    frames = max(9, frames)
    if (frames - 1) % 8 != 0:
        frames = ((frames - 1) // 8) * 8 + 1
    return frames


def wf_t2v(prompt, neg, w, h, frames, steps, cfg, seed):
    frames = fix_frames(frames)
    return {
        "client_id": str(uuid.uuid4()),
        "prompt": {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": CKPT_NAME}
            },
            "2": {
                "class_type": "LTXAVTextEncoderLoader",
                "inputs": {
                    "text_encoder": TE_NAME,
                    "ckpt_name": CKPT_NAME,
                    "device": "default"
                }
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["2", 0], "text": prompt}
            },
            "4": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["2", 0], "text": neg}
            },
            "5": {
                "class_type": "LTXVConditioning",
                "inputs": {
                    "positive": ["3", 0],
                    "negative": ["4", 0],
                    "frame_rate": 25.0
                }
            },
            "6": {
                "class_type": "EmptyLTXVLatentVideo",
                "inputs": {
                    "width": w, "height": h,
                    "length": frames, "batch_size": 1
                }
            },
            "7": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["1", 0],
                    "positive": ["5", 0],
                    "negative": ["5", 1],
                    "latent_image": ["6", 0],
                    "seed": seed, "steps": steps, "cfg": cfg,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0
                }
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["7", 0], "vae": ["1", 2]}
            },
            "9": {
                "class_type": "CreateVideo",
                "inputs": {"images": ["8", 0], "fps": 25.0}
            },
            "10": {
                "class_type": "SaveVideo",
                "inputs": {
                    "video": ["9", 0],
                    "filename_prefix": "ltx_t2v",
                    "format": "mp4",
                    "codec": "h264"
                }
            }
        }
    }


def wf_i2v(prompt, neg, img, w, h, frames, steps, cfg, seed):
    frames = fix_frames(frames)
    return {
        "client_id": str(uuid.uuid4()),
        "prompt": {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": CKPT_NAME}
            },
            "2": {
                "class_type": "LTXAVTextEncoderLoader",
                "inputs": {
                    "text_encoder": TE_NAME,
                    "ckpt_name": CKPT_NAME,
                    "device": "default"
                }
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["2", 0], "text": prompt}
            },
            "4": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["2", 0], "text": neg}
            },
            "5": {
                "class_type": "LTXVConditioning",
                "inputs": {
                    "positive": ["3", 0],
                    "negative": ["4", 0],
                    "frame_rate": 25.0
                }
            },
            "6": {
                "class_type": "LoadImage",
                "inputs": {"image": img}
            },
            "7": {
                "class_type": "LTXVImgToVideo",
                "inputs": {
                    "positive": ["5", 0],
                    "negative": ["5", 1],
                    "vae": ["1", 2],
                    "image": ["6", 0],
                    "width": w, "height": h,
                    "length": frames, "batch_size": 1
                }
            },
            "8": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["1", 0],
                    "positive": ["7", 0],
                    "negative": ["7", 1],
                    "latent_image": ["7", 2],
                    "seed": seed, "steps": steps, "cfg": cfg,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0
                }
            },
            "9": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["8", 0], "vae": ["1", 2]}
            },
            "10": {
                "class_type": "CreateVideo",
                "inputs": {"images": ["9", 0], "fps": 25.0}
            },
            "11": {
                "class_type": "SaveVideo",
                "inputs": {
                    "video": ["10", 0],
                    "filename_prefix": "ltx_i2v",
                    "format": "mp4",
                    "codec": "h264"
                }
            }
        }
    }


async def submit(wf: dict) -> str:
    async with httpx.AsyncClient() as c:
        r = await c.post(f"{COMFY_URL}/prompt", json=wf, timeout=30)
        if r.status_code != 200:
            raise Exception(f"{r.status_code}: {r.text}")
        return r.json()["prompt_id"]


async def poll(prompt_id: str, job_id: str, timeout=900):
    deadline = time.time() + timeout
    async with httpx.AsyncClient() as c:
        while time.time() < deadline:
            await asyncio.sleep(5)
            try:
                r = await c.get(f"{COMFY_URL}/history/{prompt_id}", timeout=10)
                data = r.json()
                if prompt_id not in data:
                    continue
                hist = data[prompt_id]
                st = hist.get("status", {})
                if st.get("completed"):
                    for node_out in hist.get("outputs", {}).values():
                        for key in ("videos", "gifs", "images"):
                            for item in node_out.get(key, []):
                                subfolder = item.get("subfolder", "")
                                fname = item.get("filename", "")
                                p = OUTPUT_DIR / subfolder / fname if subfolder else OUTPUT_DIR / fname
                                if p.exists():
                                    jobs[job_id].update(status="done", output=str(p))
                                    return
                    jobs[job_id].update(status="error", error="output not found")
                    return
                if st.get("status_str") in ("error", "failed"):
                    jobs[job_id].update(status="error", error=str(st))
                    return
            except Exception as e:
                print(f"[poll] {e}")
    jobs[job_id].update(status="error", error="timeout")


app = FastAPI(title="LTX-2.3 Video API", version="2.1.0")


@app.on_event("startup")
async def startup():
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(f"{COMFY_URL}/system_stats", timeout=3)
            if r.status_code == 200:
                print("[LTX] ComfyUI already up")
                return
    except Exception:
        pass
    start_comfyui()
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, wait_comfyui)


@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(f"{COMFY_URL}/system_stats", timeout=3)
            ok = r.status_code == 200
    except Exception:
        ok = False
    return {"status": "ok", "comfyui_backend": ok, "active_jobs": len(jobs)}


@app.post("/t2v")
async def t2v(
    bg: BackgroundTasks,
    prompt: str = Form(...),
    neg_prompt: str = Form(default=NEG),
    width: int = Form(default=768),
    height: int = Form(default=512),
    num_frames: int = Form(default=25),
    steps: int = Form(default=20),
    cfg: float = Form(default=3.0),
    seed: int = Form(default=-1),
):
    if seed == -1:
        seed = random.randint(0, 2**31)
    jid = str(uuid.uuid4())
    jobs[jid] = {"status": "queued", "output": None}
    wf = wf_t2v(prompt, neg_prompt, width, height, num_frames, steps, cfg, seed)
    try:
        pid = await submit(wf)
    except Exception as e:
        raise HTTPException(502, f"ComfyUI error: {e}")
    jobs[jid]["status"] = "running"
    bg.add_task(poll, pid, jid)
    return {"job_id": jid, "status": "running", "poll": f"/status/{jid}"}


@app.post("/i2v")
async def i2v(
    bg: BackgroundTasks,
    prompt: str = Form(...),
    image: UploadFile = File(...),
    neg_prompt: str = Form(default=NEG),
    width: int = Form(default=768),
    height: int = Form(default=512),
    num_frames: int = Form(default=25),
    steps: int = Form(default=20),
    cfg: float = Form(default=3.0),
    seed: int = Form(default=-1),
):
    if seed == -1:
        seed = random.randint(0, 2**31)
    img_name = f"{uuid.uuid4()}_{image.filename}"
    img_path = Path(f"{COMFY_DIR}/input/{img_name}")
    img_path.parent.mkdir(parents=True, exist_ok=True)
    img_path.write_bytes(await image.read())
    jid = str(uuid.uuid4())
    jobs[jid] = {"status": "queued", "output": None}
    wf = wf_i2v(prompt, neg_prompt, img_name, width, height, num_frames, steps, cfg, seed)
    try:
        pid = await submit(wf)
    except Exception as e:
        raise HTTPException(502, f"ComfyUI error: {e}")
    jobs[jid]["status"] = "running"
    bg.add_task(poll, pid, jid)
    return {"job_id": jid, "status": "running", "poll": f"/status/{jid}"}


@app.get("/status/{jid}")
async def status(jid: str):
    if jid not in jobs:
        raise HTTPException(404, "Job not found")
    j = jobs[jid]
    return {
        "job_id": jid,
        "status": j["status"],
        "download": f"/result/{jid}" if j["status"] == "done" else None,
        "error": j.get("error"),
    }


@app.get("/result/{jid}")
async def result(jid: str, bg: BackgroundTasks):
    if jid not in jobs:
        raise HTTPException(404, "Job not found")
    j = jobs[jid]
    if j["status"] != "done":
        raise HTTPException(400, f"Not ready: {j['status']}")
    p = j.get("output")
    if not p or not Path(p).exists():
        raise HTTPException(404, "File missing")
    mt = "video/mp4" if str(p).endswith(".mp4") else "video/webm"

    def cleanup():
        try:
            os.remove(p)
        except Exception:
            pass
        jobs.pop(jid, None)
        print(f"[LTX] deleted {p} job={jid}")

    bg.add_task(cleanup)
    return FileResponse(p, media_type=mt, filename=Path(p).name)


if __name__ == "__main__":
    uvicorn.run("ltx_api:app", host="0.0.0.0", port=API_PORT, log_level="info")
