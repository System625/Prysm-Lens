"""Download IGDB screenshots listed in a screenshots JSON (from the IGDB /screenshots endpoint).

Usage: python indexer/download_shots.py data/pool-screenshots.json data/shots
Files land as <out>/<image_id>.jpg at t_screenshot_med (569x320). Re-runs skip what exists.
"""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
import requests

SIZE = "t_screenshot_med"

def main(src: str, out: str, workers: int = 24) -> None:
    shots = json.load(open(src))
    os.makedirs(out, exist_ok=True)
    todo = [s["image_id"] for s in shots if not os.path.exists(f"{out}/{s['image_id']}.jpg")]
    print(f"{len(shots)} listed, {len(todo)} to fetch")
    session = requests.Session()

    def fetch(image_id: str) -> bool:
        try:
            r = session.get(f"https://images.igdb.com/igdb/image/upload/{SIZE}/{image_id}.jpg", timeout=30)
            if r.status_code != 200:
                return False
            tmp = f"{out}/{image_id}.part"
            open(tmp, "wb").write(r.content)
            os.replace(tmp, f"{out}/{image_id}.jpg")
            return True
        except requests.RequestException:
            return False

    ok = 0
    with ThreadPoolExecutor(workers) as pool:
        for i, good in enumerate(pool.map(fetch, todo), 1):
            ok += good
            if i % 1000 == 0:
                print(f"  {i}/{len(todo)} ({ok} ok)", flush=True)
    print(f"done: {ok}/{len(todo)} fetched")

if __name__ == "__main__":
    main(*sys.argv[1:3])
