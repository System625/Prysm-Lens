"""Embed every screenshot with Meta Perception Encoder (PE-Core-L-14-336) into one index.

Usage: python indexer/embed.py data/pool-screenshots.json data/shots data/index
Writes <out>.npy (N x 1024, fp16, L2-normalised) and <out>.json ([{image_id, game}] row-aligned).
Progress is saved every 6,400 images under <out>.parts/, so a re-run resumes; unreadable images are dropped.
"""
import json, os, sys, time
import numpy as np
import open_clip, torch
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

MODEL = "PE-Core-L-14-336"
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL, pretrained="meta")
    model = model.eval().to(DEVICE)
    return (model.half() if DEVICE != "cpu" else model), preprocess

@torch.no_grad()
def encode(model, batch):
    dtype = next(model.parameters()).dtype
    f = model.encode_image(batch.to(DEVICE, dtype))
    return torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy().astype(np.float16)

CHUNK = int(os.environ.get("EMBED_CHUNK", 6400))

def main(src: str, folder: str, out: str) -> None:
    rows = [{"image_id": s["image_id"], "game": s["game"]} for s in json.load(open(src))
            if os.path.exists(f"{folder}/{s['image_id']}.jpg")]
    parts_dir = f"{out}.parts"
    os.makedirs(parts_dir, exist_ok=True)
    if os.path.exists(f"{parts_dir}/rows.json") and json.load(open(f"{parts_dir}/rows.json")) != rows:
        raise SystemExit(f"{parts_dir} was built from a different screenshot list; delete it to start over")
    json.dump(rows, open(f"{parts_dir}/rows.json", "w"))
    model, preprocess = load_model()

    def load(r):
        try:
            return preprocess(Image.open(f"{folder}/{r['image_id']}.jpg").convert("RGB")), True
        except Exception:
            return torch.zeros(3, 336, 336), False

    t0, done = time.time(), 0
    # Decode on threads; DataLoader's spawned workers hang on macOS here.
    with ThreadPoolExecutor(8) as pool:
        for c in range(0, len(rows), CHUNK):
            path = f"{parts_dir}/{c // CHUNK:05d}.npz"
            if os.path.exists(path):
                continue
            vecs, good = [], []
            for b in range(c, min(c + CHUNK, len(rows)), 64):
                loaded = list(pool.map(load, rows[b:b + 64]))
                vecs.append(encode(model, torch.stack([t for t, _ in loaded])))
                good += [ok for _, ok in loaded]
            np.savez(path, vecs=np.concatenate(vecs), good=np.array(good))
            done += len(good)
            print(f"  {c + len(good)}/{len(rows)}  {done / (time.time() - t0):.1f} img/s", flush=True)
    chunks = [np.load(f"{parts_dir}/{c // CHUNK:05d}.npz") for c in range(0, len(rows), CHUNK)]
    good = np.concatenate([ch["good"] for ch in chunks])
    np.save(f"{out}.npy", np.concatenate([ch["vecs"] for ch in chunks])[good])
    json.dump([r for r, ok in zip(rows, good) if ok], open(f"{out}.json", "w"))
    print(f"done: {int(good.sum())} of {len(rows)} embedded ({len(rows) - int(good.sum())} unreadable) -> {out}.npy")

if __name__ == "__main__":
    main(*sys.argv[1:4])
