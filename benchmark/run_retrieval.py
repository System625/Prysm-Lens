"""Score PE retrieval on the benchmark clips: recall@1/5/10, overall and on the rows the current pipeline missed.

Usage: python benchmark/run_retrieval.py <candidates.tsv> <baseline.jsonl> <index base> <out.jsonl> [games.json]
<games.json> holds every indexed game ({igdb_id, title, release_date}); default data/pool-games.json.
Truth cells: "A | B" alternatives, "prefix:X", "Title (YYYY)"; rows with no single game are skipped.
"""
import csv, json, os, re, sys, time
import numpy as np, torch
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from indexer.embed import load_model, encode
from service.search import Index, extract_frames, views

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

def main(tsv: str, baseline: str, index_base: str, out: str, games_path: str = "data/pool-games.json") -> None:
    games = json.load(open(games_path))
    title = {g["igdb_id"]: g["title"] for g in games}
    base = {r["row"]: r for r in map(json.loads, open(baseline))}
    index = Index(index_base)
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
        cached = f"data/query-emb/{row}-views.npy"
        if os.path.exists(cached):
            query = np.load(cached)
        else:
            query = encode(model, torch.stack([preprocess(v) for f in frames for v in views(Image.open(f))]))
            os.makedirs("data/query-emb", exist_ok=True)
            np.save(cached, query)
        ranked = index.rank(query)
        hit = next((i + 1 for i, (g, _) in enumerate(ranked) if g in ok), None)
        b = base.get(row, {})
        rec = {"row": row, "truth": truth, "hit_rank": hit, "top": [title.get(g, g) for g, _ in ranked],
               "top_ids": [g for g, _ in ranked], "evidence": {str(g): index.evidence(query, g) for g, _ in ranked},
               "scores": [round(s, 2) for _, s in ranked], "baseline": b.get("verdict"),
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
    main(*sys.argv[1:6])
