"""List IGDB screenshots for a set of games.

Usage: TWITCH_CLIENT_ID=... TWITCH_CLIENT_SECRET=... python indexer/list_screenshots.py <games.json> <out.json>
<games.json> is a list of objects with an `igdb_id`. Output: [{game, image_id, width, height}].
IGDB allows 4 requests/second; this stays under it.
"""
import json, os, sys, time
import requests

def token() -> str:
    r = requests.post("https://id.twitch.tv/oauth2/token", params={
        "client_id": os.environ["TWITCH_CLIENT_ID"], "client_secret": os.environ["TWITCH_CLIENT_SECRET"],
        "grant_type": "client_credentials"})
    r.raise_for_status()
    return r.json()["access_token"]

def main(src: str, out: str) -> None:
    ids = sorted({g["igdb_id"] for g in json.load(open(src)) if g.get("igdb_id")})
    headers = {"Client-ID": os.environ["TWITCH_CLIENT_ID"], "Authorization": f"Bearer {token()}"}
    shots, last = [], 0.0
    for n, i in enumerate(range(0, len(ids), 100)):
        chunk, offset = ids[i:i + 100], 0
        while True:
            time.sleep(max(0.0, 0.3 - (time.time() - last)))
            last = time.time()
            body = f"fields game,image_id,width,height; where game=({','.join(map(str, chunk))}); sort id asc; limit 500; offset {offset};"
            r = requests.post("https://api.igdb.com/v4/screenshots", headers=headers, data=body, timeout=60)
            if r.status_code == 429:
                time.sleep(2)
                continue
            r.raise_for_status()
            page = r.json()
            shots += page
            if len(page) < 500:
                break
            offset += 500
        if n % 50 == 0:
            print(f"  {i + len(chunk)}/{len(ids)} games, {len(shots)} screenshots", flush=True)
    json.dump(shots, open(out, "w"))
    print(f"done: {len(shots)} screenshots for {len({s['game'] for s in shots})}/{len(ids)} games -> {out}")

if __name__ == "__main__":
    main(*sys.argv[1:3])
