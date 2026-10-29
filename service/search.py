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

HUB_WEIGHT = 0.5  # similarity minus this times the screenshot's hub score
PRIOR_WEIGHT = 0.4  # vote share times 1 + this * log(1 + IGDB total_rating_count)

class Index:
    def __init__(self, base: str, games_path: str | None = None, families_path: str | None = None):
        """<base>-hub.npy (from indexer/hub_scores.py), games_path's rating counts and families_path
        (from indexer/list_families.py) are each optional; without them ranking is the plain vote per game."""
        self.vecs = np.load(f"{base}.npy").astype(np.float32)
        self.rows = json.load(open(f"{base}.json"))
        self.games = np.array([r["game"] for r in self.rows])
        self.hub = np.load(f"{base}-hub.npy") if os.path.exists(f"{base}-hub.npy") else None
        self.prior = {g["igdb_id"]: float(np.log1p(g.get("total_rating_count") or 0))
                      for g in json.load(open(games_path))} if games_path else {}
        self.family = {int(g): r for g, r in json.load(open(families_path)).items()} if families_path else {}
        self.family_members = defaultdict(list)
        for g, r in self.family.items():
            self.family_members[r].append(g)

    def members(self, game: int) -> list[int]:
        """Every game in this game's family (just the game itself when it has no editions)."""
        return self.family_members.get(self.family.get(game, game), [game])

    def rank(self, query: np.ndarray, k_neighbours: int = 300, top: int = 10, temperature: float = 0.02):
        """query: (V, D) normalised view embeddings. Per view, each game's similarity is its best screenshot
        among the k nearest; the view then casts one vote, split across games by a softmax over those
        similarities. A game's clip score is its share of the votes (0..1). Returns [(game, score)].
        Voting beat summing raw similarities on the pilot (R@10 75 vs 69 of 89): no single frame can
        outweigh the rest just because a game has many near-duplicate screenshots.
        On the full 92k-game catalog two corrections lifted R@10 from 67 to 78 of 109 without losing a hard
        clip: a hub penalty (generic screenshots that sit close to everything stop winning votes) and a mild
        popularity prior on the final score.
        With families loaded, a game's editions vote as one (Days Gone + Days Gone Remastered take one slot,
        not two); the returned game is the member whose screenshots matched best, see members() for the rest."""
        sims = query.astype(np.float32) @ self.vecs.T
        if self.hub is not None:
            sims -= HUB_WEIGHT * self.hub
        score, by_member = defaultdict(float), defaultdict(float)
        for s in sims:
            idx = np.argpartition(-s, k_neighbours)[:k_neighbours]
            best, owner = {}, {}
            for i in idx:
                g = int(self.games[i])
                f = self.family.get(g, g)
                if s[i] > best.get(f, -9):
                    best[f], owner[f] = float(s[i]), g
            fs = list(best)
            v = np.array([best[f] for f in fs])
            w = np.exp((v - v.max()) / temperature)
            for f, wf in zip(fs, w / w.sum()):
                score[f] += float(wf) / len(sims)
                by_member[owner[f]] += float(wf)
        for f in score:
            score[f] *= 1 + PRIOR_WEIGHT * max(self.prior.get(g, 0.0) for g in self.members(f))
        ranked = sorted(score.items(), key=lambda kv: -kv[1])[:top]
        return [(max(self.members(f), key=lambda g: by_member.get(g, 0.0)), s) for f, s in ranked]

    def evidence(self, query: np.ndarray, game: int, n: int = 3) -> list[str]:
        """The game's n screenshots closest to any view of the clip: what the verifier is shown for that candidate."""
        rows = np.flatnonzero(self.games == game)
        best = (query.astype(np.float32) @ self.vecs[rows].T).max(0)
        return [self.rows[rows[i]]["image_id"] for i in np.argsort(-best)[:n]]
