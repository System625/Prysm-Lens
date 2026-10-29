"""Hub score per screenshot: its mean similarity to its 10 nearest neighbours in a 20k random sample of the index.

Usage: python indexer/hub_scores.py data/index-full
Writes <base>-hub.npy (float32, row-aligned), which service/search.py subtracts (weighted) from query similarity.
Generic screenshots (dark scenes, menus, stock asset-flip art) sit close to everything and otherwise win votes
for whatever game owns them. Rerun after the index changes.
"""
import sys
import numpy as np

def main(base: str, sample: int = 20000, k: int = 10, chunk: int = 20000) -> None:
    vecs = np.load(f"{base}.npy").astype(np.float32)
    ref = vecs[np.random.default_rng(0).choice(len(vecs), sample, replace=False)]
    hub = np.empty(len(vecs), np.float32)
    for s in range(0, len(vecs), chunk):
        sims = vecs[s:s + chunk] @ ref.T
        hub[s:s + chunk] = np.sort(sims, 1)[:, -k - 1:-1].mean(1)  # skips the top hit: a sampled row matches itself
    np.save(f"{base}-hub.npy", hub)
    print(f"{len(hub)} hub scores: mean {hub.mean():.3f}, p99 {np.percentile(hub, 99):.3f} -> {base}-hub.npy")

if __name__ == "__main__":
    main(sys.argv[1])
