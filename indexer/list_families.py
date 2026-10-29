"""Group catalog games into families: an original plus its editions, remasters, ports and standalone modes.

Usage: TWITCH_CLIENT_ID=... TWITCH_CLIENT_SECRET=... python indexer/list_families.py <games.json> <out.json>
Follows IGDB's version_parent (editions) and parent_game (remasters, ports, expansions, standalone modes such as
GTA Online), except for remakes, which look like a different game. Output: {game id: family root id} for every
game in a family of 2 or more. Days Gone and Days Gone Remastered then take one slot in the top 10, not two.
"""
import json, os, sys, time
import requests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from indexer.list_screenshots import token

REMAKE = 8  # IGDB game_type

def main(src: str, out: str) -> None:
    ids = sorted({g["igdb_id"] for g in json.load(open(src))})
    headers = {"Client-ID": os.environ["TWITCH_CLIENT_ID"], "Authorization": f"Bearer {token()}"}
    parent: dict[int, int] = {}
    for i in range(0, len(ids), 500):
        body = f"fields version_parent,parent_game,game_type; where id=({','.join(map(str, ids[i:i + 500]))}); limit 500;"
        for attempt in range(8):
            try:
                r = requests.post("https://api.igdb.com/v4/games", headers=headers, data=body, timeout=60)
                if r.status_code == 429:
                    time.sleep(2)
                    continue
                r.raise_for_status()
                break
            except requests.RequestException:
                if attempt == 7:
                    raise
                time.sleep(2 ** attempt)
        for g in r.json():
            p = g.get("version_parent") or (g.get("parent_game") if g.get("game_type") != REMAKE else None)
            if p:
                parent[g["id"]] = p
        time.sleep(0.3)
    def root(g: int) -> int:
        seen = set()
        while g in parent and g not in seen:
            seen.add(g)
            g = parent[g]
        return g
    families = {g: root(g) for g in parent}
    families.update({r: r for r in set(families.values())})
    json.dump(families, open(out, "w"))
    print(f"{len(ids)} games, {len(families)} in {len(set(families.values()))} families -> {out}")

if __name__ == "__main__":
    main(*sys.argv[1:3])
