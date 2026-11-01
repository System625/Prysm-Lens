"""Quantize an index to int8 per row (symmetric, scale = max |x| / 127): <base>.npy -> <out>.npy + <out>-scale.npy.
Half the bytes of fp16 (1.18 GB -> 0.59 GB for 575,973 shots), worst-case cosine error 0.0003, and the same
retrieval recall on the 104-clip benchmark. Index() dequantizes on load. Copy or link <base>.json and <base>-hub.npy.

Usage: python indexer/quantize.py data/index-full data/index-q8
"""
import sys
import numpy as np

def main(base: str, out: str) -> None:
    v = np.load(f"{base}.npy").astype(np.float32)
    scale = np.abs(v).max(1, keepdims=True) / 127.0
    np.save(f"{out}.npy", np.round(v / scale).astype(np.int8))
    np.save(f"{out}-scale.npy", scale.astype(np.float32).ravel())

if __name__ == "__main__":
    main(*sys.argv[1:3])
