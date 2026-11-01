"""Lens as a web service: POST a clip (or a link to one), get every stage back as it finishes.

  POST /identify   multipart `file` (any video ffmpeg reads: mp4, mov, webm from a phone camera) or form `url`
  GET  /           health + today's usage

The response is NDJSON, one event per line, so the web demo can draw each stage as it lands:
  {"stage": "frames",     "frames": [jpeg data URLs], "skipped": flat frames left out}
  {"stage": "screen",     "found": bool, "boxes": {frame: [x1,y1,x2,y2] as 0..1}, "crops": [jpeg data URLs]}
  {"stage": "candidates", "items": [{"id", "title", "year", "score", "editions", "shots": [IGDB image URLs]}]}
  {"stage": "answer",     "own_guess", "pick", "game_title", "confidence", "evidence", "game": {...} | null,
                          "catalog": {...} | null}
    game = the search candidate Muse Spark picked. catalog = with pick "none", the catalog entry its own answer
    names (year + that game's screenshots closest to the clip), so the answer can be checked by eye.
  {"stage": "done",       "ms": {...}, "usd": float}
  {"stage": "error",      "message": str}

Pipeline = the benchmark's best single-call setup: 16 frames -> SAM 3.1 screen crop (filmed screens only) ->
Perception Encoder -> top 10 of 575,973 IGDB screenshots -> Muse Spark 1.3, own answer first.

Env: MODEL_API_KEY (Meta Model API), LENS_DATA (folder with the index + catalog, default data),
LENS_INDEX (index base name, default index-q8), LENS_DAILY_CAP (identify calls per UTC day, default 400:
each costs about $0.03 of Meta credit, so a public demo can't drain it).
"""
import base64, io, json, os, re, subprocess, tempfile, threading, time, unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import numpy as np, torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image

from indexer.embed import load_model, encode
from service.search import Index, extract_frames, keep_frames, views
from service.screen import crop_to_screen, find_screen
from service.verify import build_request, call, prepare_clip, LABELS

DATA = os.environ.get("LENS_DATA", "data")
INDEX = os.environ.get("LENS_INDEX", "index-q8")
DAILY_CAP = int(os.environ.get("LENS_DAILY_CAP", 400))
MAX_UPLOAD = 200_000_000
SHOT_URL = "https://images.igdb.com/igdb/image/upload/{size}/{id}.jpg"

def cpu_quota() -> int:
    """CPUs this container may use. os.cpu_count() reports the host's cores, and torch starting that many threads
    on a 2-vCPU allowance made the encoder ~4x slower than 2 threads on a laptop (1 Nov)."""
    try:
        q, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if q != "max":
            return max(1, round(int(q) / int(period)))
    except (OSError, ValueError):
        pass
    return os.cpu_count() or 1

THREADS = int(os.environ.get("LENS_THREADS", cpu_quota()))
torch.set_num_threads(THREADS)

app = FastAPI(title="Prysm Lens")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

state = {"index": None, "model": None, "games": None, "titles": None, "indexed": None, "day": None, "count": 0}
busy = threading.Semaphore(2)  # the encoder and the 2.4 GB search are CPU-bound; queue the rest

loading = threading.Lock()

def load():
    with loading:  # startup warms this in the background; an early request waits for it instead of racing it
        if state["index"] is None:
            state["games"] = {g["igdb_id"]: g for g in json.load(open(f"{DATA}/catalog-games.json"))}
            state["model"] = load_model()
            index = Index(f"{DATA}/{INDEX}", f"{DATA}/catalog-games.json", f"{DATA}/catalog-families.json")
            state["indexed"] = set(np.unique(index.games).tolist())
            state["titles"] = title_table(state["games"].values(), state["indexed"])
            state["index"] = index
    return state["index"], state["model"], state["games"]

def title_table(games, indexed: set[int]) -> dict[str, list[dict]]:
    """Normalised title -> the games with it (remakes share the original's title: Resident Evil 4 is three games).
    Games with screenshots in the index first, then the most-rated."""
    by: dict[str, list[dict]] = {}
    for g in sorted(games, key=lambda g: (g["igdb_id"] not in indexed, -(g.get("total_rating_count") or 0))):
        by.setdefault(title_key(g["title"]), []).append(g)
    return by

def title_key(t: str) -> str:
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode().lower().replace("&", " and ")  # Ragnarök
    t = re.sub(r"\((19|20)\d\d\)", " ", t)  # "Elden Ring (2022)"
    return re.sub(r"^the ", "", re.sub(r"[^a-z0-9]+", " ", t).strip())

NOT_A_TITLE = re.compile(r"^\s*(unknown|unidentified|unclear|none|n/a|not\b|no\b)", re.I)

def lookup(title: str) -> dict | None:
    """The catalog game Muse Spark's own answer names, or None. A shared title (Resident Evil 4 is three games)
    gives the most-rated one, or the newest when the answer says remake/remaster: screenshot similarity can't tell
    them apart on a filmed screen (row 111, the 2023 remake, sits closer to the 2011 HD release).
    On the benchmark's 102 pick=none answers: 52 land on the right game, 28 on the (wrong) game Spark named,
    22 name none ("Unknown - no gameplay visible") or a spelling the catalog lacks ("Phantom Blade 0")."""
    if not title or NOT_A_TITLE.match(title):
        return None
    tries = (title, re.sub(r"\([^)]*\)", " ", title), title.split(" / ")[0],  # "(Shutoko mod)", "A / B"
             re.sub(r"\s+(remake|remastered|remaster)\b.*$", "", title, flags=re.I))
    named = next((state["titles"][k] for k in map(title_key, tries) if k and k in state["titles"]), [])
    if not named:
        return None
    if re.search(r"\b(remake|remastered|remaster)\b", title, re.I):
        return max(named, key=lambda g: g.get("release_date") or "")
    return named[0]

@app.on_event("startup")
def warm():
    threading.Thread(target=load, daemon=True).start()

def allow() -> bool:
    day = datetime.now(timezone.utc).date().isoformat()
    if state["day"] != day:
        state["day"], state["count"] = day, 0
    if state["count"] >= DAILY_CAP:
        return False
    state["count"] += 1
    return True

def thumb(im: Image.Image, h: int = 240) -> str:
    im = im.convert("RGB")
    im.thumbnail((h * 2, h))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=70)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

def fetch(url: str, out: str) -> str:
    """A shared post's video via yt-dlp (first 3 minutes, <= 720p). Some sites block data-centre IPs: upload instead."""
    r = subprocess.run(["yt-dlp", "-q", "--no-playlist", "-f", "b[height<=720]/bv*[height<=720]+ba/b", "--download-sections",
                        "*0-180", "--merge-output-format", "mp4", "-o", out, url], capture_output=True, text=True, timeout=150)
    if r.returncode != 0 or not os.path.exists(out):
        raise RuntimeError("Couldn't fetch that link from the server (the site may block it). Upload the clip instead. "
                           + (r.stderr.strip().splitlines() or [""])[-1][:200])
    return out

def duration(path: str) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0

def normalise(src: str, out: str) -> str:
    """Use the upload as it is: ffmpeg reads phone formats (mov, webm) directly, and re-encoding a filmed screen
    blurred it enough to lose Resident Evil 4 from the top 10 (1 Nov). Only a file with no readable duration
    (some in-browser camera recordings) is rewritten, near-lossless, so frames can be spread over it."""
    if duration(src) > 0:
        return src
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-t", "180", "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "16", "-c:a", "aac", "-b:a", "96k", out], check=True)
    return out

def cpu_embed(ims: list) -> np.ndarray:
    _, (model, preprocess), _ = load()
    with busy:
        return encode(model, torch.stack([preprocess(v) for v in ims]))

def identify(clip: str, work: str, embed=cpu_embed):
    """The pipeline as a generator of stage events. `embed` turns PIL images into normalised query vectors;
    the GPU Space passes one that runs the encoder (and the index search) on a ZeroGPU slot."""
    index, _, games = load()
    pool = ThreadPoolExecutor(2)
    # Muse Spark's copy of the clip (<= 60 s, 480p) encodes in the background while the search runs.
    small = pool.submit(prepare_clip, clip, f"{work}/verify.mp4")
    ms, t = {}, time.time()
    every = extract_frames(clip, f"{work}/frames")
    if not every:
        raise RuntimeError("No frames could be read from that video.")
    keep = keep_frames(every)  # outros, black fades and title cards are not searched
    frames = [every[i] for i in keep]
    yield {"stage": "frames", "frames": [thumb(Image.open(f)) for f in frames], "skipped": len(every) - len(frames)}
    ms["frames"] = int((time.time() - t) * 1000); t = time.time()

    # SAM is a network call: fingerprint the full frames while it runs, and redo only if it finds a filmed screen.
    sam = pool.submit(find_screen, every, f"{work}/sam.json", keep)
    full = embed([v for f in frames for v in views(Image.open(f))])
    sam_error = None
    try:
        boxes = sam.result(timeout=90)
    except Exception as e:  # SAM down or slow: search the plain frames rather than fail the request
        boxes, sam_error = {}, str(e)[:200]
    crops = crop_to_screen(frames, boxes) if boxes else []
    yield {"stage": "screen", "found": bool(boxes), "boxes": {str(k): v for k, v in boxes.items()},
           "crops": [thumb(c) for c in crops], "error": sam_error}
    ms["sam"] = int((time.time() - t) * 1000); t = time.time()

    query = embed(crops) if crops else full
    ranked = index.rank(query)
    evidence = {g: index.evidence(query, g) for g, _ in ranked}
    items = []
    for g, score in ranked:
        meta = games.get(g, {})
        items.append({"id": g, "title": meta.get("title", str(g)), "year": (meta.get("release_date") or "")[:4] or None,
                      "score": round(score, 3), "editions": [m for m in index.members(g) if m != g][:5],
                      "shots": [SHOT_URL.format(size="t_screenshot_med", id=s) for s in evidence[g]]})
    yield {"stage": "candidates", "items": items}
    ms["search"] = int((time.time() - t) * 1000); t = time.time()

    cands = [{"title": c["title"], "year": c["year"],
              "shots": [SHOT_URL.format(size="t_logo_med", id=s) for s in evidence[c["id"]]]} for c in items]
    res = call(build_request(small.result(), cands, "first"), os.environ["MODEL_API_KEY"])
    pool.shutdown(wait=False)
    usd = 0.01 + res["usd"]  # one SAM video call (50-frame minimum) + Muse Spark
    a = res["answer"] or {}
    pick = a.get("pick")
    game = items[LABELS.index(pick)] if pick and pick != "none" and LABELS.index(pick) < len(items) else None
    catalog = None
    if game is None and (g := lookup(a.get("game_title", ""))):
        shots = index.evidence(query, g["igdb_id"]) if g["igdb_id"] in state["indexed"] else []  # Rerouted has none
        catalog = {"id": g["igdb_id"], "title": g["title"], "year": (g.get("release_date") or "")[:4] or None,
                   "shots": [SHOT_URL.format(size="t_screenshot_med", id=s) for s in shots]}
    yield {"stage": "answer", **a, "game": game and {k: game[k] for k in ("id", "title", "year", "shots")},
           "catalog": catalog, "error": res.get("error")}
    ms["spark"] = int((time.time() - t) * 1000)
    yield {"stage": "done", "ms": ms, "usd": round(usd, 4)}

@app.get("/")
def health():
    return {"ok": True, "loaded": state["index"] is not None, "today": state["count"], "cap": DAILY_CAP,
            "threads": THREADS, "host_cpus": os.cpu_count()}

@app.post("/identify")
async def identify_route(file: UploadFile | None = File(None), url: str | None = Form(None)):
    if not file and not url:
        return JSONResponse({"error": "send a video file or a url"}, status_code=400)
    if not allow():
        return JSONResponse({"error": "the demo's daily limit is used up; try again tomorrow (UTC)"}, status_code=429)
    work = tempfile.mkdtemp(prefix="lens-")
    raw = f"{work}/upload"
    if file:
        size = 0
        with open(raw, "wb") as f:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD:
                    return JSONResponse({"error": "video over 200 MB"}, status_code=413)
                f.write(chunk)

    def stream():
        try:
            clip = normalise(raw if file else fetch(url, f"{work}/fetched.mp4"), f"{work}/clip.mp4")
            for ev in identify(clip, work):
                yield json.dumps(ev) + "\n"
        except Exception as e:
            yield json.dumps({"stage": "error", "message": str(e)[:400]}) + "\n"
        finally:
            subprocess.run(["rm", "-rf", work])

    return StreamingResponse(stream(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
