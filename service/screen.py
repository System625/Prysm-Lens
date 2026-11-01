"""Find the screen in a filmed-screen clip with SAM 3.1, so Lens fingerprints the game and not the room.

A phone filming a monitor or TV puts a desk, a wall and a bezel around the gameplay; the room then matches
room-like games (Deskape, Sleep Simulator) better than the game on the screen. SAM 3.1 tracks a concept
("screen") through the clip's sampled frames and returns a box per frame; the box crop becomes an extra
query view. Frames go up as one short video (1 frame per second) because SAM bills video per frame with a
50-frame minimum: one call per clip, ~$0.01, instead of 16 image calls (~$0.04).
"""
import base64, glob, json, os, re, subprocess, tempfile, time
import requests

API = "https://api.meta.ai/v1/responses"
BOX = re.compile(r"(\d+)<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)(?:;c=([\d.]+))?\|>")
FRAME = re.compile(r"<(\d+)f>")

def frames_video(frames: list[str], out: str) -> str:
    """The sampled frames as an H.264 mp4 at 1 fps, even dimensions, so frame index N = frames[N]."""
    with tempfile.TemporaryDirectory() as d:
        for i, f in enumerate(frames):
            os.symlink(os.path.abspath(f), f"{d}/{i:03d}.jpg")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "1", "-i", f"{d}/%03d.jpg",
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-r", "1", out], check=True)
    return out

def parse(text: str) -> dict[int, list[dict]]:
    """SAM output lines -> {frame: [{id, box (x1,y1,x2,y2) as 0..1 fractions, area, conf}]} (masks ignored)."""
    out: dict[int, list[dict]] = {}
    for line in text.splitlines():
        m = FRAME.match(line.strip())
        if not m:
            continue
        recs = []
        for oid, x1, y1, x2, y2, w, h, c in BOX.findall(line):
            x1, y1, x2, y2, w, h = map(int, (x1, y1, x2, y2, w, h))
            box = (x1 / w, y1 / h, (x2 + 1) / w, (y2 + 1) / h)
            recs.append({"id": int(oid), "box": box, "area": (box[2] - box[0]) * (box[3] - box[1]),
                         "conf": float(c) if c else None})
        out[int(m.group(1))] = recs
    return out

def segment(video: str, concept: str = "screen", key: str | None = None, tries: int = 3) -> dict:
    """One sam-3.1 call on a local mp4. Returns {"frames": parse(...), "usage": ..., "raw_len": ...}."""
    key = key or os.environ["MODEL_API_KEY"]
    url = "data:video/mp4;base64," + base64.b64encode(open(video, "rb").read()).decode()
    body = {"model": "sam-3.1", "input": [{"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": concept}, {"type": "input_video", "video_url": url}]}],
        "metadata": {"include_confidence": "true", "mask_encoding": "one_bit"}}
    for attempt in range(tries):
        try:
            r = requests.post(API, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=180)
            if r.status_code >= 500 or r.status_code == 429:
                raise requests.HTTPError(f"{r.status_code} {r.text[:200]}")
            r.raise_for_status()
            j = r.json()
            text = "".join(c.get("text", "") for o in j.get("output", []) for c in o.get("content", []) or [])
            return {"frames": parse(text), "usage": j.get("usage"), "raw_len": len(text)}
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
            if attempt == tries - 1 or (isinstance(e, requests.HTTPError) and str(e)[:1] == "4" and not str(e).startswith("429")):
                raise
            time.sleep(5 * (attempt + 1))

MIN_CONF, MIN_AREA, MAX_AREA, MIN_SHARE = 0.85, 0.02, 0.9, 0.5

def screen_boxes(sam_frames: dict, n_frames: int) -> dict[int, tuple]:
    """{frame index: box} when the clip films a screen, else {}.
    A clip counts as filmed only when a confident screen box (conf >= 0.85, 2-90% of the frame) shows on at
    least half its frames: on the 104-clip benchmark that fires on 7 clips (a phone filming a monitor, a TV,
    two phones) and never on gameplay with monitors inside the game, which one-off boxes would. Per frame the
    largest such box wins."""
    fr = {}
    for k, recs in sam_frames.items():
        ok = [r for r in recs if (r["conf"] or 0) >= MIN_CONF and MIN_AREA <= r["area"] <= MAX_AREA]
        if ok:
            fr[int(k)] = tuple(max(ok, key=lambda r: r["area"])["box"])
    return fr if len(fr) >= MIN_SHARE * n_frames else {}

def crop_to_screen(frames: list[str], boxes: dict[int, tuple]):
    """Each frame cropped to its screen box (frames without one borrow the nearest frame's box)."""
    from PIL import Image
    have = sorted(boxes)
    out = []
    for i, f in enumerate(frames):
        b = boxes[min(have, key=lambda j: abs(j - i))]
        im = Image.open(f).convert("RGB")
        W, H = im.size
        out.append(im.crop((int(b[0] * W), int(b[1] * H), int(b[2] * W), int(b[3] * H))))
    return out

def find_screen(frames: list[str], cache: str | None = None) -> dict[int, tuple]:
    """screen_boxes for these frames, from the SAM result cached at `cache` or one live sam-3.1 call (saved there)."""
    if cache and os.path.exists(cache):
        res = json.load(open(cache))
    else:
        with tempfile.TemporaryDirectory() as d:
            res = segment(frames_video(frames, f"{d}/frames.mp4"))
        if cache:
            os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
            json.dump(res, open(cache, "w"))
    return screen_boxes(res["frames"], len(frames))

if __name__ == "__main__":
    import sys
    row, concept = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "screen")
    frames = sorted(glob.glob(f"data/frames/{row}/*.jpg"))
    os.makedirs("data/sam", exist_ok=True)
    vid = frames_video(frames, f"data/sam/{row}.mp4")
    res = segment(vid, concept)
    print(json.dumps(res["usage"]), res["raw_len"])
    for f, recs in sorted(res["frames"].items()):
        print(f, [(r["id"], [round(v, 2) for v in r["box"]], round(r["area"], 2), r["conf"]) for r in recs])
