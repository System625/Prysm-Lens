"""Score Muse Spark as the verifier over Lens's top 10, or blind (no candidates), on the benchmark clips.

Usage: python benchmark/run_verify.py <retrieval.jsonl> <shots dir> <out.jsonl> [options]
<retrieval.jsonl> comes from run_retrieval.py (needs its top_ids + evidence fields).
Nothing is sent without --live (and MODEL_API_KEY set): a dry run builds every request and reports its size.
Results append to <out.jsonl>; a re-run skips (row, mode, run) already there, so a stopped run resumes.
--max-usd stops the run once the summed cost reaches it. LLM answers vary: use --runs 5 before trusting a number.
"""
import argparse, json, os, re, sys
from collections import Counter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmark.run_retrieval import acceptable
from service.search import extract_frames
from service.verify import LABELS, build_request, call, prepare_clip

def norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()

def judge(rec: dict, answer: dict, ok: set[int], title: dict[int, str]) -> dict:
    """right = the picked candidate is the truth; own_right = game_title names the truth (counts for none/blind)."""
    pick = answer["pick"]
    gid = rec["top_ids"][LABELS.index(pick)] if pick != "none" and LABELS.index(pick) < len(rec["top_ids"]) else None
    own = norm(answer["game_title"]) in {norm(title[g]) for g in ok if g in title}
    return {"pick_id": gid, "right": gid in ok if gid else False, "own_right": own}

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("retrieval"); p.add_argument("shots"); p.add_argument("out")
    p.add_argument("--mode", choices=["video", "frames", "blind"], default="video")
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--rows", help="comma-separated benchmark rows; default all")
    p.add_argument("--hard", action="store_true", help="only rows the current Prysm pipeline missed")
    p.add_argument("--games", default="data/catalog-games.json")
    p.add_argument("--effort", default="low")
    p.add_argument("--max-usd", type=float, default=2.0)
    p.add_argument("--live", action="store_true")
    a = p.parse_args()
    key = os.environ.get("MODEL_API_KEY")
    if a.live and not key:
        raise SystemExit("--live needs MODEL_API_KEY")

    games = json.load(open(a.games))
    title = {g["igdb_id"]: g["title"] for g in games}
    year = {g["igdb_id"]: (g.get("release_date") or "")[:4] for g in games}
    recs = [json.loads(l) for l in open(a.retrieval)]
    if a.rows:
        keep = {int(x) for x in a.rows.split(",")}
        recs = [r for r in recs if r["row"] in keep]
    if a.hard:
        recs = [r for r in recs if r["hard"]]
    done = {(d["row"], d["mode"], d["run"]) for d in map(json.loads, open(a.out))} if os.path.exists(a.out) else set()
    spent = sum(d.get("usd", 0) for d in map(json.loads, open(a.out))) if os.path.exists(a.out) else 0.0

    for rec in recs:
        row = rec["row"]
        clip = prepare_clip(f"data/clips/{row}.mp4", f"data/verify-clips/{row}.mp4")
        frames = extract_frames(f"data/clips/{row}.mp4", f"data/frames/{row}")[::2]
        cands = [] if a.mode == "blind" else [
            {"title": title.get(g, str(g)), "year": year.get(g), "shots": [f"{a.shots}/{s}.jpg" for s in rec["evidence"][str(g)]
                                                                          if os.path.exists(f"{a.shots}/{s}.jpg")]}
            for g in rec["top_ids"]]
        body = build_request(clip, cands, a.mode, frames, a.effort)
        parts = body["input"][0]["content"]
        size = len(json.dumps(body)) / 1e6
        if not a.live:
            print(f"{row:>3} dry  {size:5.1f} MB  {sum(c['type'] == 'input_image' for c in parts):>2} images  "
                  f"video {os.path.getsize(clip) / 1e6:4.1f} MB  | {rec['truth'][:40]}")
            continue
        ok = acceptable(rec["truth"], games)
        for run in range(a.runs):
            if (row, a.mode, run) in done:
                continue
            if spent >= a.max_usd:
                raise SystemExit(f"stopped: spent ${spent:.2f} of --max-usd {a.max_usd}")
            try:
                res = call(body, key)
            except RuntimeError as e:
                print(f"{row:>3} run {run} ERROR {e}", flush=True)
                continue
            spent += res["usd"]
            out = {"row": row, "mode": a.mode, "run": run, "truth": rec["truth"], "hard": rec["hard"],
                   "lens_hit_rank": rec["hit_rank"], "baseline": rec["baseline"], **res,
                   **(judge(rec, res["answer"], ok, title) if res["answer"] else {"right": False, "own_right": False})}
            with open(a.out, "a") as f:
                f.write(json.dumps(out) + "\n")
            ans = res["answer"] or {}
            print(f"{row:>3} run {run} {'RIGHT' if out['right'] or out['own_right'] else 'wrong':5} {ans.get('pick', '-'):>4} "
                  f"{ans.get('confidence', '-'):6} {ans.get('game_title', res.get('error', ''))[:30]:30} | {rec['truth'][:30]}"
                  f"  ${res['usd']:.4f} {res['ms']} ms", flush=True)
    if a.live:
        summarise([json.loads(l) for l in open(a.out)], a.mode)

def summarise(results: list[dict], mode: str, cap: int = 3) -> None:
    """Per row, the majority answer over its runs (a tie or any wrong majority counts wrong); also the share of
    single runs right, to show how much the answer moves between runs. Same 3-clips-per-game cap as retrieval."""
    by_row: dict[int, list[dict]] = {}
    for d in results:
        if d["mode"] == mode:
            by_row.setdefault(d["row"], []).append(d)
    seen: Counter = Counter()
    rows = []
    for row in sorted(by_row):
        runs = by_row[row]
        seen[runs[0]["truth"]] += 1
        votes = Counter(((d["answer"] or {}).get("game_title") or "", d["right"] or d["own_right"]) for d in runs)
        (_, right), n = votes.most_common(1)[0]
        rows.append({**runs[0], "maj_right": right and n > len(runs) / 2, "runs_right": sum(d["right"] or d["own_right"] for d in runs),
                     "n_runs": len(runs), "confident_wrong": sum(1 for d in runs if not (d["right"] or d["own_right"])
                                                                 and (d["answer"] or {}).get("confidence") == "high"),
                     "capped": seen[runs[0]["truth"]] <= cap})
    usd = sum(d.get("usd", 0) for d in results if d["mode"] == mode)
    for label, rs in (("uncapped", rows), (f"cap {cap}/game", [r for r in rows if r["capped"]])):
        for name, sub in (("all", rs), ("hard", [r for r in rs if r["hard"]])):
            if sub:
                lens1 = sum(1 for r in sub if r["lens_hit_rank"] == 1)
                lens10 = sum(1 for r in sub if r["lens_hit_rank"])
                prysm = sum(1 for r in sub if r["baseline"] == "right")
                runs = sum(r["n_runs"] for r in sub)
                print(f"{label:>12} {name:>4}: n={len(sub):>3}  spark {sum(r['maj_right'] for r in sub):>3} "
                      f"(single runs {sum(r['runs_right'] for r in sub)}/{runs}, confident-wrong {sum(r['confident_wrong'] for r in sub)})"
                      f"  | lens@1 {lens1:>3}  lens@10 {lens10:>3}  prysm@1 {prysm:>3}")
    print(f"{mode}: ${usd:.2f} over {sum(r['n_runs'] for r in rows)} calls")

if __name__ == "__main__":
    main()
