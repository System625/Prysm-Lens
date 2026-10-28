"""Lens verification: Muse Spark watches the clip next to each candidate's best-matching screenshots and picks one.

Uses Meta's Responses API (api.meta.ai/v1/responses): the clip goes in as `input_video` (base64 data URL),
screenshots as `input_image`, and the answer comes back constrained to a JSON schema via `text.format`.
Modes:
  video  - the clip itself, with its audio (a streamer saying the name counts as evidence)
  frames - FRAMES stills from the clip instead of the video, for when video and images can't share a request
  blind  - the clip and no candidates: what Muse Spark knows on its own, the no-index baseline
"""
import base64, json, os, subprocess, time
import requests

API = "https://api.meta.ai/v1/responses"
MODEL = "muse-spark-1.3"
PRICE_IN, PRICE_OUT = 1.25 / 1e6, 4.25 / 1e6  # USD per token, muse-spark-1.3
CLIP_SECONDS, CLIP_HEIGHT = 60, 480
FRAMES = 8
LABELS = "ABCDEFGHIJ"

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["pick", "game_title", "confidence", "evidence"],
    "properties": {
        "pick": {"type": "string", "enum": [*LABELS, "none"],
                 "description": "The letter of the matching candidate, or none if no candidate is the game in the clip."},
        "game_title": {"type": "string",
                       "description": "The game in the clip: the picked candidate's title, or your own best guess if none match."},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "evidence": {"type": "string",
                     "description": "What in the clip decided it: HUD, characters, locations, art style, on-screen text, spoken words."},
    },
}

INSTRUCTIONS = """You name the video game shown in a short gameplay clip from social media.
The clip may carry a streamer's facecam, subtitles, stickers or a meme caption; judge the gameplay underneath.
Sequels, remasters and mobile ports often look alike: compare the HUD, menus, characters and art style, not only the genre.
Never name a game you can't see evidence for. If unsure, say so with low confidence."""

def prepare_clip(clip: str, out: str, seconds: int = CLIP_SECONDS, height: int = CLIP_HEIGHT) -> str:
    """Up to `seconds` of the clip from 5% in, scaled to at most `height` lines, audio kept. Cached at `out`."""
    if os.path.exists(out):
        return out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", clip],
                               capture_output=True, text=True).stdout.strip() or 0)
    start = dur * 0.05 if dur > seconds else 0
    subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{start:.2f}", "-t", str(seconds), "-i", clip,
                    "-vf", f"scale=-2:'min({height},ih)'", "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
                    "-c:a", "aac", "-b:a", "64k", "-movflags", "+faststart", out, "-y"], check=True)
    return out

def data_url(path: str, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(open(path, 'rb').read()).decode()}"

def build_request(clip: str, candidates: list[dict], mode: str = "video", frames: list[str] = (),
                  effort: str = "low", max_output_tokens: int = 4000) -> dict:
    """candidates: [{"title", "year", "shots": [jpg paths]}], at most 10, best first. Empty in blind mode."""
    content = [{"type": "input_text", "text": "The clip:"}]
    if mode == "frames":
        content += [{"type": "input_image", "image_url": data_url(f, "image/jpeg")} for f in frames[:FRAMES]]
    else:
        content.append({"type": "input_video", "video_url": data_url(clip, "video/mp4")})
    if mode == "blind":
        content.append({"type": "input_text", "text": "Which game is this? There are no candidates: answer pick=none "
                                                      "and put the game in game_title."})
    else:
        content.append({"type": "input_text", "text": "Candidates, each with the screenshots of it that look most like "
                                                      "the clip. Exactly one may be right, or none."})
        for label, c in zip(LABELS, candidates):
            name = f"{c['title']} ({c['year']})" if c.get("year") else c["title"]
            content.append({"type": "input_text", "text": f"{label}: {name}"})
            content += [{"type": "input_image", "image_url": data_url(s, "image/jpeg")} for s in c["shots"]]
        content.append({"type": "input_text", "text": "Which candidate is the game in the clip?"})
    return {"model": MODEL, "instructions": INSTRUCTIONS,
            "input": [{"type": "message", "role": "user", "content": content}],
            "text": {"format": {"type": "json_schema", "name": "game_pick", "schema": SCHEMA, "strict": True}},
            "reasoning": {"effort": effort}, "max_output_tokens": max_output_tokens}

def call(body: dict, key: str, tries: int = 4) -> dict:
    """POST to the Responses API; retries 429/5xx with backoff. Returns {answer, usage, usd, ms, response_id};
    answer is None (and error set) when the model ran but gave no usable answer, so its cost is still counted."""
    for attempt in range(tries):
        t0 = time.time()
        r = requests.post(API, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=300)
        if r.status_code in (429, 500, 502, 503, 504) and attempt < tries - 1:
            time.sleep(2 ** attempt * 5)
            continue
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
        res = r.json()
        text = "".join(c.get("text", "") for o in res.get("output", []) if o.get("type") == "message"
                       for c in o.get("content", []) if c.get("type") == "output_text")
        usage = res.get("usage", {})
        usd = usage.get("input_tokens", 0) * PRICE_IN + usage.get("output_tokens", 0) * PRICE_OUT
        out = {"answer": None, "usage": usage, "usd": round(usd, 5), "ms": int((time.time() - t0) * 1000),
               "response_id": res.get("id")}
        try:
            out["answer"] = json.loads(text) if res.get("status") == "completed" else None
        except json.JSONDecodeError:
            pass
        if out["answer"] is None:
            out["error"] = f"status {res.get('status')}, {res.get('incomplete_details')}, text {text[:200]!r}"
        return out
    raise RuntimeError("unreachable")
