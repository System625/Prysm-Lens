"""Embed every screenshot with Meta Perception Encoder (PE-Core-L-14-336) into one index.

Usage: python indexer/embed.py data/pool-screenshots.json data/shots data/index
Writes <out>.npy (N x 1024, fp16, L2-normalised) and <out>.json ([{image_id, game}] row-aligned).
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

def main(src: str, folder: str, out: str) -> None:
    rows = [{"image_id": s["image_id"], "game": s["game"]} for s in json.load(open(src))
            if os.path.exists(f"{folder}/{s['image_id']}.jpg")]
    model, preprocess = load_model()
    load = lambda r: preprocess(Image.open(f"{folder}/{r['image_id']}.jpg").convert("RGB"))
    batches = (rows[i:i + 64] for i in range(0, len(rows), 64))
    parts, t0 = [], time.time()
    # Decode on threads; DataLoader's spawned workers hang on macOS here.
    with ThreadPoolExecutor(8) as pool:
        for i, chunk in enumerate(batches):
            parts.append(encode(model, torch.stack(list(pool.map(load, chunk)))))
            if i % 20 == 0:
                n = sum(len(p) for p in parts)
                print(f"  {n}/{len(rows)}  {n / (time.time() - t0):.1f} img/s", flush=True)
    np.save(f"{out}.npy", np.concatenate(parts))
    json.dump(rows, open(f"{out}.json", "w"))
    print(f"done: {len(rows)} in {time.time() - t0:.0f}s -> {out}.npy")

if __name__ == "__main__":
    main(*sys.argv[1:4])
