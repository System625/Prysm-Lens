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
import base64, io, ipaddress, json, os, re, socket, subprocess, tempfile, threading, time, unicodedata
from urllib.parse import urlparse
import requests
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
# IGDB ids of adult games: IGDB theme "Erotic" or an explicit title, 3,821 ids exported 3 Nov from Shelf's games table.
# 3,817 are in the catalog. They never reach the top 10 on the 104 gameplay clips, but a flower time-lapse put
# "Tomboy: Sex in the Forest" and its screenshots on the page (3 Nov), so search results leave them out.
ADULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adult-ids.json")

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
            state["adult"] = set(json.load(open(ADULT))) if os.path.exists(ADULT) else set()
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

NUMBERS = "zero one two three four five six seven eight nine ten".split()

def swap_numbers(t: str) -> str:
    """Spelled-out numbers as digits and digits as words, whole words only: "Phantom Blade 0" <-> "Phantom Blade Zero"."""
    return re.sub(r"\b(\d+|[a-z]+)\b", lambda m: (NUMBERS[int(m[1])] if m[1].isdigit() and int(m[1]) <= 10 else
                                                str(NUMBERS.index(m[1].lower())) if m[1].lower() in NUMBERS else m[1]),
                  t, flags=re.I)

NOT_A_TITLE = re.compile(r"^\s*(unknown|unidentified|unclear|none|n/a|not\b|no\b)", re.I)

def named_games(title: str) -> list[dict]:
    """Every catalog game the title names (indexed first, then most-rated), trying the spellings Spark uses."""
    if not title or NOT_A_TITLE.match(title):
        return []
    tries = (title, re.sub(r"\([^)]*\)", " ", title), title.split(" / ")[0],  # "(Shutoko mod)", "A / B"
             re.sub(r"\s+(remake|remastered|remaster)\b.*$", "", title, flags=re.I))
    tries += tuple(swap_numbers(t) for t in tries)  # Spark wrote "Phantom Blade 0" in 5 of 5 runs on row 51
    return next((state["titles"][k] for k in map(title_key, tries) if k and k in state["titles"]), [])

def lookup(title: str) -> dict | None:
    """The catalog game Muse Spark's own answer names, or None. A shared title (Resident Evil 4 is three games)
    gives the most-rated one, or the newest when the answer says remake/remaster: screenshot similarity can't tell
    them apart on a filmed screen (row 111, the 2023 remake, sits closer to the 2011 HD release).
    On the benchmark's 102 pick=none answers: 52 land on the right game, 28 on the (wrong) game Spark named,
    22 name none ("Unknown - no gameplay visible") or a spelling the catalog lacks ("Phantom Blade 0")."""
    named = named_games(title)
    if not named:
        return None
    if re.search(r"\b(remake|remastered|remaster)\b", title, re.I):
        return max(named, key=lambda g: g.get("release_date") or "")
    return named[0]

NOTE = re.compile(r"\s*[(\[]([^()\[\]]*\s[^()\[\]]*\s[^()\[\]]*\s[^()\[\]]*)[)\]]")  # a bracket of 4+ words

def tidy(a: dict) -> dict:
    """The title fields hold only a name. A bracketed note Muse Spark put in one anyway ("Mirror's Edge Catalyst
    (first-person parkour look, but clip appears to be Mission: Impossible movie/AI footage, not real gameplay)",
    3 Nov) moves to the end of the evidence, so the headline stays a game's name. Short brackets ("(2019)",
    "(Shutoko mod)") stay."""
    a = dict(a)
    for k in ("game_title", "own_guess"):
        if isinstance(a.get(k), str) and (notes := NOTE.findall(a[k])):
            a[k] = NOTE.sub("", a[k]).strip()
            if k == "game_title":
                a["evidence"] = " ".join(filter(None, [a.get("evidence", "").rstrip(), f"({'; '.join(notes)})"]))
    return a

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

def inline(urls: list[str], pool: ThreadPoolExecutor) -> list[str]:
    """IGDB screenshots as data URLs, fetched here. Sent as URLs, Meta's servers fetch them, and one they couldn't
    fetch failed the whole answer ("media_url_not_fetchable ... input[0].content[15]", a candidate's screenshot,
    3 Nov); the same clip worked on a rerun. ~10 KB each at t_logo_med, ~30 a run. One that fails here is left out."""
    def get(u: str) -> str | None:
        for _ in range(2):
            try:
                r = requests.get(u, timeout=8)
                if r.status_code == 200 and r.headers.get("content-type", "").startswith("image/"):
                    return "data:image/jpeg;base64," + base64.b64encode(r.content).decode()
            except requests.RequestException:
                pass
        return None
    return [d for d in pool.map(get, urls) if d]

def thumb(im: Image.Image, h: int = 240) -> str:
    im = im.convert("RGB")
    im.thumbnail((h * 2, h))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=70)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

SITES = {"instagram": "Instagram", "tiktok": "TikTok", "facebook": "Facebook", "fb.watch": "Facebook", "x.com": "X",
         "twitter": "X", "youtu": "YouTube", "reddit": "Reddit", "twitch": "Twitch"}

NOT_A_LINK = "That isn't a link to a post. Copy the post's link (it starts with https://) and paste it here."
HOST = re.compile(r"^[a-z0-9-]+(\.[a-z0-9-]+)*\.[a-z]{2,}$", re.I)

def check_link(url: str) -> str:
    """The link as https, or a plain error. Only public web addresses: "elden ring" took 37 s to fail inside yt-dlp,
    and a link to localhost or a private address would have had the Space fetch from its own network (3 Nov).
    YouTube is refused here too: it turns servers away, and the web page captures YouTube in the visitor's browser."""
    u = url.strip()
    if not re.match(r"^https?://", u, re.I):
        u = "https://" + u
    host = (urlparse(u).hostname or "").lower()
    if not HOST.match(host):
        raise RuntimeError(NOT_A_LINK)
    if host == "youtu.be" or host.endswith("youtube.com"):
        raise RuntimeError("YouTube doesn't let servers download its videos. Open Lens's page and paste the link "
                           "there to capture it in your browser, or screen-record it and upload.")
    try:
        ips = {ipaddress.ip_address(a[4][0]) for a in socket.getaddrinfo(host, None)}
    except (socket.gaierror, ValueError):
        raise RuntimeError(f"Lens couldn't find the site {host}. Check the link.")
    if any(not ip.is_global for ip in ips):
        raise RuntimeError(NOT_A_LINK)
    return u

def fetch(url: str, out: str) -> str:
    """A shared post's video via yt-dlp (first 3 minutes, <= 720p). Some sites refuse data-centre IPs now and then:
    Instagram answered "empty media response" for a reel at 16:37 on 3 Nov and served it at 16:38, so one failure
    gets one retry after a pause. The message says what happened in plain words, not yt-dlp's --cookies advice."""
    url = check_link(url)
    site = next((name for k, name in SITES.items() if k in url.lower()), "That site")
    no_video = f"That {site} post has no video in it. Lens needs a clip." if site != "That site" else \
        "There's no video at that link. Lens needs a clip."
    if "tiktok.com" in url and "/photo/" in url:  # a TikTok slideshow
        raise RuntimeError(no_video)
    cmd = ["yt-dlp", "-q", "--no-playlist", "-f", "b[height<=720]/bv*[height<=720]+ba/b", "--download-sections",
           "*0-180", "--merge-output-format", "mp4", "-o", out, url]
    for attempt in range(2):
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=150)
        if r.returncode == 0 and os.path.exists(out):
            return out
        last = (r.stderr.strip().splitlines() or [""])[-1]
        if "Unsupported URL" in last or "is not a valid URL" in last:
            raise RuntimeError("Lens can't read videos from that site. Download the clip and upload it instead.")
        if re.search(r"no video|no media|not contain any video", last, re.I):
            raise RuntimeError(no_video)
        if attempt == 0:
            time.sleep(4)
    print(f"fetch failed: {url} :: {last[:300]}", flush=True)  # the Space logs keep yt-dlp's own words
    raise RuntimeError(f"{site} didn't let Lens's server fetch that video. The post may be private, or {site} "
                       "is turning servers away for now: try again in a minute, or download the clip and upload it.")

def duration(path: str) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0

UNREADABLE = "Lens couldn't read a video in that file. Try an MP4 or MOV clip."

def has_video(path: str) -> bool:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=codec_type",
                          "-of", "csv=p=0", path], capture_output=True, text=True).stdout
    return "video" in out

def normalise(src: str, out: str) -> str:
    """Use the upload as it is: ffmpeg reads phone formats (mov, webm) directly, and re-encoding a filmed screen
    blurred it enough to lose Resident Evil 4 from the top 10 (1 Nov). Only a file with no readable duration
    (some in-browser camera recordings) is rewritten, near-lossless, so frames can be spread over it.
    An empty, broken or audio-only file stops here with a plain message, not ffmpeg's command line (3 Nov)."""
    if not has_video(src):
        raise RuntimeError(UNREADABLE)
    if duration(src) > 0:
        return src
    # libx264 needs even sides, and a tab or camera recording can be 691x325 (3 Nov: the run died here)
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-t", "180", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-b:a", "96k", out], capture_output=True, text=True)
    if r.returncode != 0 or duration(out) <= 0:
        print(f"normalise failed: {r.stderr.strip()[-300:]}", flush=True)
        raise RuntimeError(UNREADABLE)
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
    ranked = [(g, s) for g, s in index.rank(query, top=40) if g not in state["adult"]][:10]
    evidence = {g: index.evidence(query, g) for g, _ in ranked}
    items = []
    for g, score in ranked:
        meta = games.get(g, {})
        items.append({"id": g, "title": meta.get("title", str(g)), "year": (meta.get("release_date") or "")[:4] or None,
                      "score": round(score, 3), "editions": [m for m in index.members(g) if m != g][:5],
                      "shots": [SHOT_URL.format(size="t_screenshot_med", id=s) for s in evidence[g]]})
    yield {"stage": "candidates", "items": items}
    ms["search"] = int((time.time() - t) * 1000); t = time.time()

    with ThreadPoolExecutor(16) as fetchers:
        cands = [{"title": c["title"], "year": c["year"],
                  "shots": inline([SHOT_URL.format(size="t_logo_med", id=s) for s in evidence[c["id"]]], fetchers)}
                 for c in items]
    # A filmed screen: Muse Spark also gets SAM's crops, or it judges a room with a small monitor in it.
    screen = [thumb(c, 480) for c in crops]
    # 2 tries of <= 100 s: the web route that streams this to the browser is cut off at 300 s, and a run cut off
    # mid-call left the page sitting on step 4 with no answer and no error (3 Nov).
    try:
        res = call(build_request(small.result(), cands, "first", screen=screen), os.environ["MODEL_API_KEY"],
                   tries=2, read_timeout=100)
    except Exception as e:  # the visitor gets plain words; the Space log keeps the API's own
        print(f"spark failed: {str(e)[:300]}", flush=True)
        raise RuntimeError("Muse Spark didn't answer this time. Try again in a minute.") from e
    pool.shutdown(wait=False)
    usd = 0.01 + res["usd"]  # one SAM video call (50-frame minimum) + Muse Spark
    if res.get("error"):
        print(f"spark gave no answer: {res['error']}", flush=True)
    a = tidy(res["answer"] or {})
    pick = a.get("pick")
    game = items[LABELS.index(pick)] if pick and pick != "none" and LABELS.index(pick) < len(items) else None
    if game:
        game = {k: game[k] for k in ("id", "title", "year", "shots")}
        # Spark picks a candidate when its own answer is the same game, and the candidate can be another edition of it:
        # own answer "PUBG: Battlegrounds" (high), pick "PUBG Lite" (3 Nov). Name the edition it recognised. Same-title
        # remakes are left alone: the screenshots can't tell the three Resident Evil 4s apart.
        own = a.get("own_guess") or ""
        family = set(index.members(game["id"]))
        if title_key(own) != title_key(game["title"]) and (ed := next(
                (g for g in named_games(own) if g["igdb_id"] in family and g["igdb_id"] != game["id"]), None)):
            shots = index.evidence(query, ed["igdb_id"]) if ed["igdb_id"] in state["indexed"] else []
            game = {"id": ed["igdb_id"], "title": ed["title"], "year": (ed.get("release_date") or "")[:4] or None,
                    "shots": [SHOT_URL.format(size="t_screenshot_med", id=s) for s in shots] or game["shots"],
                    "candidate": game["id"]}
            a["game_title"] = ed["title"]
    catalog = None
    if game is None and (g := lookup(a.get("game_title", ""))):
        shots = index.evidence(query, g["igdb_id"]) if g["igdb_id"] in state["indexed"] else []  # Rerouted has none
        if g["igdb_id"] in state["adult"]:
            shots = []
        catalog = {"id": g["igdb_id"], "title": g["title"], "year": (g.get("release_date") or "")[:4] or None,
                   "shots": [SHOT_URL.format(size="t_screenshot_med", id=s) for s in shots]}
    yield {"stage": "answer", **a, "game": game, "catalog": catalog,
           "error": "no usable answer" if res.get("error") else None}
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
