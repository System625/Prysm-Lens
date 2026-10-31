"""Score the cascade: keep Prysm's caption answer when it was confident, ask Muse Spark only when it wasn't,
and show Muse Spark the Lens candidates only when it isn't sure on its own.

  python benchmark/score_combined.py data/baseline-26-oct.jsonl data/verify-all.jsonl data/verify-all-blind.jsonl [first.jsonl]

Per row, Spark is right when most of its runs are, as in run_verify.summarise. A row is confidently wrong when
the final answer is wrong and was given with confidence (Prysm's confident flag, or a high-confidence Spark majority).
Any edition in the truth's family counts as right, for Prysm too (rescore the Spark files with run_verify --rescore).
The optional 4th file adds Spark's "first" mode (own answer before the candidates), scored on the rows it has.
"""
import json, os, sys
from collections import Counter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmark.run_verify import family_ok

def spark_rows(path: str, mode: str | None = None) -> dict[int, dict]:
    by_row: dict[int, list[dict]] = {}
    for d in map(json.loads, open(path)):
        if mode is None or d["mode"] == mode:
            by_row.setdefault(d["row"], []).append(d)
    out = {}
    for row, runs in by_row.items():
        votes = Counter(((d["answer"] or {}).get("game_title") or "", d["right"] or d["own_right"]) for d in runs)
        game = votes.most_common(1)[0][0][0]
        right = sum(d["right"] or d["own_right"] for d in runs) > len(runs) / 2
        high = [(d["answer"] or {}).get("confidence") == "high" for d in runs]
        sure_wrong = sum(1 for d, h in zip(runs, high) if h and not (d["right"] or d["own_right"])) > len(runs) / 2
        out[row] = {"right": right, "confident_wrong": not right and sure_wrong, "sure": sum(high) > len(runs) / 2,
                    "truth": runs[0]["truth"], "hard": runs[0]["hard"], "game": game}
    return out

def main() -> None:
    base_path, lens_path, blind_path = sys.argv[1:4]
    games, families = json.load(open("data/catalog-games.json")), json.load(open("data/catalog-families.json"))
    title = {g["igdb_id"]: g["title"] for g in games}
    base = {r["row"]: r for r in map(json.loads, open(base_path))}
    lens, blind = spark_rows(lens_path), spark_rows(blind_path)
    first = spark_rows(sys.argv[4], "first") if len(sys.argv) > 4 else {}
    seen: Counter = Counter()
    rows = []
    for row in sorted(lens):
        b = base[row]
        seen[lens[row]["truth"]] += 1
        right = b["verdict"] == "right" or b.get("top") in {title[g] for g in family_ok(b["truth"], games, families) if g in title}
        prysm = {"right": right, "confident_wrong": not right and b["confident"]}
        cascade = lambda spark: prysm if b["confident"] else spark[row]
        blind_first = blind[row] if blind[row]["sure"] else lens[row]  # Lens only when Spark alone isn't sure
        rows.append({"hard": lens[row]["hard"], "capped": seen[lens[row]["truth"]] <= 3, "confident": b["confident"],
                     "pipes": {"prysm": prysm, "spark+lens": lens[row], "spark blind": blind[row],
                               "prysm, else spark+lens": cascade(lens), "prysm, else spark blind": cascade(blind),
                               "spark blind if sure, else +lens": blind_first,
                               "prysm, else that": prysm if b["confident"] else blind_first,
                               **({"spark first": first[row], "prysm, else spark first": cascade(first)} if row in first else {})}})
    for label, rs in (("uncapped", rows), ("cap 3/game", [r for r in rows if r["capped"]])):
        for name, sub in (("all", rs), ("hard", [r for r in rs if r["hard"]]), ("unsure", [r for r in rs if not r["confident"]])):
            print(f"{label} {name}: n={len(sub)}")
            for pipe in rows[0]["pipes"]:
                right = sum(r["pipes"][pipe]["right"] for r in sub)
                cw = sum(r["pipes"][pipe]["confident_wrong"] for r in sub)
                print(f"  {pipe:<32} right {right:>3}  confidently wrong {cw:>3}")
            if first:  # "first" only ran on some rows: compare every pipe on just those
                part = [r for r in sub if "spark first" in r["pipes"]]
                print(f"  -- on the {len(part)} rows with a 'first' run:")
                for pipe in [*rows[0]["pipes"], "spark first", "prysm, else spark first"]:
                    print(f"     {pipe:<32} right {sum(r['pipes'][pipe]['right'] for r in part):>3}  "
                          f"confidently wrong {sum(r['pipes'][pipe]['confident_wrong'] for r in part):>3}")

if __name__ == "__main__":
    main()
