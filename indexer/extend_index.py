"""Add screenshots that are on disk but not yet in an index, without re-embedding the rest.

Usage: python indexer/extend_index.py data/catalog-screenshots.json data/shots-full data/index-full
Embeds the new ones into <base>-extra (checkpointed and resumable like embed.py), then appends them to
<base>.npy/.json. Rerun indexer/hub_scores.py afterwards.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from indexer import embed

def main(src: str, folder: str, base: str) -> None:
    have = {r["image_id"] for r in json.load(open(f"{base}.json"))}
    new = [s for s in json.load(open(src)) if s["image_id"] not in have and os.path.exists(f"{folder}/{s['image_id']}.jpg")]
    print(f"{len(have)} indexed, {len(new)} to add")
    if not new:
        return
    extra = f"{base}-extra"
    json.dump(new, open(f"{extra}-list.json", "w"))
    embed.main(f"{extra}-list.json", folder, extra)
    vecs = np.concatenate([np.load(f"{base}.npy"), np.load(f"{extra}.npy")])
    rows = json.load(open(f"{base}.json")) + json.load(open(f"{extra}.json"))
    np.save(f"{base}.npy.tmp.npy", vecs)
    json.dump(rows, open(f"{base}.json.tmp", "w"))
    os.replace(f"{base}.npy.tmp.npy", f"{base}.npy")
    os.replace(f"{base}.json.tmp", f"{base}.json")
    print(f"done: {len(rows)} rows -> {base}.npy")

if __name__ == "__main__":
    main(*sys.argv[1:4])
