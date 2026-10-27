"""Lens retrieval: clip -> frames -> PE embeddings -> nearest IGDB screenshots -> ranked games."""
import glob, json, os, subprocess
from collections import defaultdict
import numpy as np
from PIL import Image, ImageOps

FRAMES = 16

def extract_frames(clip: str, folder: str, n: int = FRAMES) -> list[str]:
    """n frames spread evenly over the clip, skipping the first and last 5% (intros, end cards)."""
    have = sorted(glob.glob(f"{folder}/*.jpg"))
    if have:
        return have
    os.makedirs(folder, exist_ok=True)
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", clip],
                               capture_output=True, text=True).stdout.strip() or 0)
    for i in range(n):
        t = dur * (0.05 + 0.9 * i / max(n - 1, 1))
        subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", clip, "-frames:v", "1", "-vf", "scale=-2:720",
                        "-q:v", "3", f"{folder}/{i:02d}.jpg", "-y"], check=False)
    return sorted(glob.glob(f"{folder}/*.jpg"))

def trim_bars(img: Image.Image, thresh: int = 16) -> Image.Image:
    """Crop near-black letterbox/pillarbox bars."""
    g = np.asarray(ImageOps.grayscale(img), dtype=np.float32)
    rows, cols = np.where(g.mean(1) > thresh)[0], np.where(g.mean(0) > thresh)[0]
    if len(rows) < 10 or len(cols) < 10:
        return img
    return img.crop((cols[0], rows[0], cols[-1] + 1, rows[-1] + 1))

def views(img: Image.Image) -> list[Image.Image]:
    """The bar-trimmed frame, plus its centre 16:9 band when it is portrait (vertical video with gameplay in the middle)."""
    img = trim_bars(img.convert("RGB"))
    out = [img]
    w, h = img.size
    if h > w:
        bh = int(w * 9 / 16)
        top = (h - bh) // 2
        out.append(img.crop((0, top, w, top + bh)))
    return out

class Index:
    def __init__(self, base: str):
        self.vecs = np.load(f"{base}.npy").astype(np.float32)
        self.rows = json.load(open(f"{base}.json"))
        self.games = np.array([r["game"] for r in self.rows])

    def rank(self, query: np.ndarray, k_neighbours: int = 300, top: int = 10, temperature: float = 0.02):
        """query: (V, D) normalised view embeddings. Per view, each game's similarity is its best screenshot
        among the k nearest; the view then casts one vote, split across games by a softmax over those
        similarities. A game's clip score is its share of the votes (0..1). Returns [(game, score)].
        Voting beat summing raw similarities on the pilot (R@10 75 vs 69 of 89): no single frame can
        outweigh the rest just because a game has many near-duplicate screenshots."""
        sims = query.astype(np.float32) @ self.vecs.T
        score = defaultdict(float)
        for s in sims:
            idx = np.argpartition(-s, k_neighbours)[:k_neighbours]
            best = {}
            for i in idx:
                g = int(self.games[i])
                if s[i] > best.get(g, -1):
                    best[g] = float(s[i])
            gs = list(best)
            v = np.array([best[g] for g in gs])
            w = np.exp((v - v.max()) / temperature)
            for g, wg in zip(gs, w / w.sum()):
                score[g] += float(wg) / len(sims)
        return sorted(score.items(), key=lambda kv: -kv[1])[:top]
