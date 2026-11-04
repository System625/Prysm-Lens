"""Lens on a ZeroGPU Space: the same pipeline as service/app.py, with the encoder and the 575,973-row search on a GPU.

On the 2-vCPU Docker Space the Perception Encoder took ~2.2 s per image, 35-70 s a clip. ZeroGPU lends a GPU only
inside @spaces.GPU calls, billed to the caller's daily quota (the web demo calls through its server with the owner's
PRO token: anonymous visitors get 2 minutes and a small cap on runs), so the
encoder and the similarity search move in there and everything else (ffmpeg, SAM 3.1 and Muse Spark over HTTP,
the voting) stays on CPU.

API (what the web demo calls through @gradio/client): /identify(video file, url) -> JSON list of stage events,
growing with each yield; the events are the ones service/app.py streams as NDJSON.
"""
import hashlib, os, shutil, tempfile
import spaces  # must come before torch touches CUDA
import gradio as gr
import numpy as np
import torch
from huggingface_hub import snapshot_download

DATA = os.environ.setdefault("LENS_DATA", os.path.expanduser("~/data"))
if not os.path.exists(f"{DATA}/index-q8.npy"):
    snapshot_download("System625/prysm-lens-index", repo_type="dataset", local_dir=DATA, token=os.environ.get("HF_TOKEN"))

from indexer.embed import encode  # noqa: E402
from service import app as svc  # noqa: E402
from service.search import HUB_WEIGHT  # noqa: E402

index, (model, preprocess), games = svc.load()  # the model lands on cuda: ZeroGPU emulates it outside GPU calls
VECS = torch.from_numpy(index.vecs).half().to("cuda")
HUB = torch.from_numpy(index.hub.astype(np.float32)).half().to("cuda") if index.hub is not None else None
K = 300
_neighbours: dict[str, list] = {}

def key(q: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(q).tobytes()).hexdigest()

@spaces.GPU(duration=15)  # a call is admitted only if this much quota is left; the work takes ~3 s
def gpu_embed(batch: torch.Tensor):
    """Encode the views and find each one's K nearest screenshots in one GPU slot."""
    q = encode(model, batch)
    sims = torch.from_numpy(q).to("cuda").half() @ VECS.T
    if HUB is not None:
        sims = sims - HUB_WEIGHT * HUB
    vals, idx = sims.float().topk(K, dim=1)
    return q, idx.cpu().numpy(), vals.cpu().numpy()

def embed(ims: list) -> np.ndarray:
    q, idx, vals = gpu_embed(torch.stack([preprocess(v) for v in ims]))
    for row, i, v in zip(q, idx, vals):  # per view: the search may use only some rows (drop_webcam)
        _neighbours[key(row)] = (i, v)
    while len(_neighbours) > 1024:  # a filmed-screen run searches its crops, so its full-frame rows are never read
        _neighbours.pop(next(iter(_neighbours)))
    return q

def neighbours(q: np.ndarray, k: int):
    got = [_neighbours.get(key(row)) for row in q]
    if k != K or any(g is None for g in got):  # not from embed(): fall back to the CPU search
        return type(index).neighbours(index, q, k)
    return got

index.neighbours = neighbours

def run(video: str | None, url: str | None):
    if not video and not (url or "").strip():
        yield [{"stage": "error", "message": "Send a video file or a link."}]
        return
    if not svc.allow():
        yield [{"stage": "error", "message": "The demo's daily limit is used up; try again tomorrow (UTC)."}]
        return
    work = tempfile.mkdtemp(prefix="lens-")
    events: list[dict] = []
    try:
        src = video or svc.fetch(url.strip(), f"{work}/fetched.mp4")
        clip = svc.normalise(src, f"{work}/clip.mp4")
        for ev in svc.identify(clip, work, embed=embed):
            events.append(ev)
            yield events
    except Exception as e:
        events.append({"stage": "error", "message": str(e)[:400]})
        yield events
    finally:
        shutil.rmtree(work, ignore_errors=True)

with gr.Blocks(title="Prysm Lens") as demo:
    gr.Markdown("# Prysm Lens\nName the video game in a gameplay clip: SAM 3.1 finds a filmed screen, Meta Perception "
                "Encoder searches 575,973 game screenshots, Muse Spark 1.3 names the game with its evidence. "
                "The full demo: https://github.com/System625/Prysm-Lens")
    with gr.Row():
        video = gr.File(label="Gameplay clip", file_types=["video"], type="filepath")
        url = gr.Textbox(label="...or a link to one")
    go = gr.Button("Identify", variant="primary")
    out = gr.JSON(label="Stages")
    go.click(run, [video, url], out, api_name="identify", concurrency_limit=4)

if __name__ == "__main__":
    demo.queue(max_size=32).launch(max_file_size="200mb")
