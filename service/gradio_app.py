"""Lens on a Hugging Face Gradio Space: the same pipeline as service/app.py. On ZeroGPU hardware the encoder and the
575,973-row search run on a GPU; on free CPU hardware (cpu-basic, from 11 Nov 2026) they run on the Space's 2 vCPUs.

On the 2-vCPU Docker Space the Perception Encoder took ~2.2 s per image, 35-70 s a clip. ZeroGPU lends a GPU only
inside @spaces.GPU calls, billed to the caller's daily quota (the web demo calls through its server with the owner's
PRO token: anonymous visitors get 2 minutes and a small cap on runs), so the
encoder and the similarity search move in there and everything else (ffmpeg, SAM 3.1 and Muse Spark over HTTP,
the voting) stays on CPU.

API (what prysmblast.com/lens calls through @gradio/client): /identify(video file, url, ticket) -> JSON list of stage
events, growing with each yield; the events are the ones service/app.py streams as NDJSON.

The Space is public, so anyone could call /identify straight and spend the Meta API credit, skipping the website's
free-run count. With LENS_SECRET set, a run needs a ticket the website signs after counting it (see ticket_ok).
"""
import hashlib, hmac, os, shutil, tempfile, time
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

GPU = torch.cuda.is_available()  # ZeroGPU emulates CUDA outside GPU calls; cpu-basic has none
index, (model, preprocess), games = svc.load()  # on ZeroGPU the model lands on cuda
VECS = torch.from_numpy(index.vecs).half().to("cuda") if GPU else None
HUB = torch.from_numpy(index.hub.astype(np.float32)).half().to("cuda") if GPU and index.hub is not None else None
K = 300
_neighbours: dict[str, list] = {}

SECRET = os.environ.get("LENS_SECRET", "").encode()
TICKET_TTL = 600  # seconds; the website signs a ticket just before it calls
_spent: dict[str, float] = {}  # nonce -> expiry, so one ticket buys one run

def ticket_ok(ticket: str | None) -> bool:
    """`<expiry unix s>.<nonce>.<hex HMAC-SHA256 of "run:<expiry>.<nonce>">`, signed by prysmblast.com with LENS_SECRET.
    Spent nonces live in memory: a restart forgets them, but a ticket expires within TICKET_TTL anyway."""
    if not SECRET:
        return True  # local runs and the old demo
    try:
        exp, nonce, sig = (ticket or "").split(".")
        exp_s = int(exp)
    except ValueError:
        return False
    now = time.time()
    good = hmac.new(SECRET, f"run:{exp}.{nonce}".encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(good, sig) or not now < exp_s <= now + TICKET_TTL or nonce in _spent:
        return False
    for n in [n for n, e in _spent.items() if e < now]:
        del _spent[n]
    _spent[nonce] = exp_s
    return True

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

if GPU:
    index.neighbours = neighbours

def run(video: str | None, url: str | None, ticket: str | None = None):
    if not ticket_ok(ticket):
        yield [{"stage": "error", "message": "Use Lens at https://www.prysmblast.com/lens"}]
        return
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
        for ev in svc.identify(clip, work, embed=embed if GPU else svc.cpu_embed):
            events.append(ev)
            yield events
    except Exception as e:
        events.append({"stage": "error", "message": str(e)[:400]})
        yield events
    finally:
        shutil.rmtree(work, ignore_errors=True)

# Uploaded clips sit in Gradio's cache until deleted: every 30 minutes, delete any older than an hour (the privacy
# policy at prysmblast.com/privacy says clips are deleted within hours).
with gr.Blocks(title="Prysm Lens", delete_cache=(1800, 3600)) as demo:
    gr.Markdown("# Prysm Lens\nName the video game in a gameplay clip: SAM 3.1 finds a filmed screen, Meta Perception "
                "Encoder searches 575,973 game screenshots, Muse Spark 1.3 names the game with its evidence. "
                "Try it at https://www.prysmblast.com/lens. Code: https://github.com/System625/Prysm-Lens")
    with gr.Row():
        video = gr.File(label="Gameplay clip", file_types=["video"], type="filepath")
        url = gr.Textbox(label="...or a link to one")
    ticket = gr.Textbox(visible=False)
    go = gr.Button("Identify", variant="primary")
    out = gr.JSON(label="Stages")
    # 2 vCPUs on cpu-basic: a second run at once already halves both runs' speed
    go.click(run, [video, url, ticket], out, api_name="identify", concurrency_limit=4 if GPU else 1)

if __name__ == "__main__":
    demo.queue(max_size=32).launch(max_file_size="200mb")
