"""Score PE retrieval on the benchmark clips: recall@1/5/10, overall and on the rows the current pipeline missed.

Usage: python benchmark/run_retrieval.py <candidates.tsv> <baseline.jsonl> <index base> <out.jsonl> [games.json] [families.json]
<games.json> holds every indexed game ({igdb_id, title, release_date}); default data/pool-games.json.
With <families.json> a candidate is a game family: a hit when the truth is any of its editions (exact_rank: the shown one).
LENS_SCREEN=1: clips SAM 3.1 finds a filmed screen in are queried on the screen crop (data/sam/<row>.json caches SAM).
Truth cells: "A | B" alternatives, "prefix:X", "Title (YYYY)"; rows with no single game are skipped.
"""
import csv, json, os, re, sys, time
import numpy as np, torch
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from indexer.embed import load_model, encode
from service.search import Index, extract_frames, keep_frames, search_rows, views
from service.screen import crop_to_screen, find_screen

SCREEN = os.environ.get("LENS_SCREEN") == "1"  # SAM 3.1 screen crop for filmed-screen clips (service/screen.py)

def acceptable(truth: str, games: list[dict]) -> set[int]:
    ids = set()
    for a in (x.strip() for x in truth.split("|")):
        for g in games:
            if a.startswith("prefix:"):
                ok = g["title"].startswith(a[7:])
            elif m := re.match(r"^(.*) \((\d{4})\)$", a):
                ok = g["title"] == m[1] and (g["release_date"] or "")[:4] == m[2]
            else:
                ok = g["title"] == a
            if ok:
                ids.add(g["igdb_id"])
    return ids

def main(tsv: str, baseline: str, index_base: str, out: str, games_path: str = "data/pool-games.json",
         families_path: str | None = None) -> None:
    games = json.load(open(games_path))
    title = {g["igdb_id"]: g["title"] for g in games}
    base = {r["row"]: r for r in map(json.loads, open(baseline))}
    index = Index(index_base, games_path, families_path)
    model, preprocess = load_model()
    results = []
    for n, r in enumerate(csv.reader(open(tsv), delimiter="\t", quoting=csv.QUOTE_NONE), 1):
        if r[0] == "source" or (r[7] if len(r) > 7 else "") not in ("yes", "") or r[4].startswith("?"):
            continue
        row, truth = n, r[4]
        if truth in ("NOT A GAME", "(no game)", "MULTIPLE GAMES") or not os.path.exists(f"data/clips/{row}.mp4"):
            continue
        ok = acceptable(truth, games)
        t0 = time.time()
        frames = extract_frames(f"data/clips/{row}.mp4", f"data/frames/{row}")
        if not frames:
            continue
        keep = keep_frames(frames)  # flat frames (outros, fades) out, as the service does
        boxes = find_screen(frames, f"data/sam/{row}.json", keep) if SCREEN else {}
        frames = [frames[i] for i in keep]
        tag = ("screen-boxed" if boxes else "views4") + ("-kept" if len(keep) < 16 else "")
        cached = f"data/query-emb/{row}-{tag}.npy"
        if os.path.exists(cached):
            query = np.load(cached)
        else:
            per_frame = [crop_to_screen(frames, boxes)] if boxes else [views(Image.open(f)) for f in frames]
            query = encode(model, torch.stack([preprocess(v) for vs in per_frame for v in vs]))
            if not boxes:  # as the service: drop webcam bands of split portrait frames
                query = search_rows(query, [len(vs) for vs in per_frame])
            os.makedirs("data/query-emb", exist_ok=True)
            np.save(cached, query)
        ranked = index.rank(query)
        hit = next((i + 1 for i, (g, _) in enumerate(ranked) if ok & set(index.members(g))), None)
        exact = next((i + 1 for i, (g, _) in enumerate(ranked) if g in ok), None)
        b = base.get(row, {})
        rec = {"row": row, "truth": truth, "hit_rank": hit, "exact_rank": exact, "top": [title.get(g, g) for g, _ in ranked],
               "top_ids": [g for g, _ in ranked], "evidence": {str(g): index.evidence(query, g) for g, _ in ranked},
               "editions": {str(g): [m for m in index.members(g) if m != g] for g, _ in ranked},
               "scores": [round(s, 2) for _, s in ranked], "screen": {str(k): v for k, v in boxes.items()}, "baseline": b.get("verdict"),
               "hard": b.get("verdict") != "right", "ms": int((time.time() - t0) * 1000)}
        results.append(rec)
        print(f"{row:>3} {'@'+str(hit) if hit else 'miss':>5} {'HARD' if rec['hard'] else '    '} {rec['top'][0][:30]:30} | {truth[:40]}", flush=True)
    with open(out, "w") as f:
        for rec in results:
            f.write(json.dumps(rec) + "\n")
    summarise(results)

def summarise(results: list[dict], cap: int = 3) -> None:
    """Recall overall and on the hard rows, uncapped and with at most `cap` clips per truth (first rows win).
    `prysm@1` is the current pipeline's top-1 on the same clips."""
    seen: dict[str, int] = {}
    capped = []
    for x in results:
        seen[x["truth"]] = seen.get(x["truth"], 0) + 1
        if seen[x["truth"]] <= cap:
            capped.append(x)
    for label, rows in (("uncapped", results), (f"cap {cap}/game", capped)):
        for name, rs in (("all", rows), ("hard", [x for x in rows if x["hard"]])):
            if rs:
                at = lambda k: sum(1 for x in rs if x["hit_rank"] and x["hit_rank"] <= k)
                prysm = sum(1 for x in rs if x["baseline"] == "right")
                print(f"{label:>12} {name:>4}: n={len(rs):>3}  R@1 {at(1):>3}  R@5 {at(5):>3}  R@10 {at(10):>3}  | prysm@1 {prysm:>3}")

if __name__ == "__main__":
    main(*sys.argv[1:7])
