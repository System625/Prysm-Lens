// Prysm Lens baseline: the labelled benchmark list through the DEPLOYED share-resolve,
// as a fresh Pro test user. Usage: tsx scripts/measure-lens-baseline.ts <candidates.tsv> <out.jsonl>
//
// share-resolve answers from a user-confirmed game or its 7-day cache before running the
// pipeline, so every row records `source`, and the summary scores both the product
// (shortcuts included) and the pipeline alone (cache/confirmed answers left out).
// Truth cells: "A | B" alternatives, "prefix:X", "Title (YYYY)" for same-named games,
// "NOT A GAME"/"(no game)" and "MULTIPLE GAMES".
import { readFileSync, appendFileSync, writeFileSync } from "node:fs";
import { createClient } from "@supabase/supabase-js";
import { admin, grantTestPro } from "./supabase-admin.ts";
import { required } from "./env.ts";

const [tsvPath, outPath] = process.argv.slice(2);
if (!tsvPath || !outPath) throw new Error("usage: measure-lens-baseline.ts <candidates.tsv> <out.jsonl>");
const U = required("SUPABASE_URL"), ANON = required("SUPABASE_ANON_KEY");

const [, ...lines] = readFileSync(tsvPath, "utf8").trimEnd().split("\n");
const rows = lines
  .map((l, i) => {
    const c = l.split("\t");
    return { row: i + 2, provider: c[1], url: c[3], truth: c[4], include: c[7] ?? "" };
  })
  .filter((r) => (r.include === "yes" || r.include === "" || r.include === "negative") && !r.truth.startsWith("?"));

type Kind = "game" | "none" | "multi";
const kindOf = (t: string): Kind =>
  t === "NOT A GAME" || t === "(no game)" ? "none" : t === "MULTIPLE GAMES" ? "multi" : "game";

function accepts(truth: string, g?: { title: string; releaseDate?: string }): boolean {
  if (!g) return false;
  return truth.split("|").map((a) => a.trim()).some((a) => {
    if (a.startsWith("prefix:")) return g.title.startsWith(a.slice(7));
    const m = a.match(/^(.*) \((\d{4})\)$/);
    if (m) return g.title === m[1] && g.releaseDate?.slice(0, 4) === m[2];
    return g.title === a;
  });
}

const email = `lens-baseline-${Date.now()}@shelf.test`, password = `Lens-${Date.now()}!`;
const { data: u, error: ue } = await admin.auth.admin.createUser({ email, password, email_confirm: true });
if (ue) throw ue;
await grantTestPro(u.user!.id);
writeFileSync(outPath, "");
console.log(`${rows.length} rows -> ${outPath}`);
try {
  const { data: s } = await createClient(U, ANON, { auth: { persistSession: false } }).auth.signInWithPassword({ email, password });
  for (const r of rows) {
    const kind = kindOf(r.truth);
    const t0 = Date.now();
    let b: any = {}, status = 0;
    try {
      const res = await fetch(`${U}/functions/v1/share-resolve`, {
        method: "POST",
        headers: { apikey: ANON, Authorization: `Bearer ${s.session!.access_token}`, "Content-Type": "application/json" },
        body: JSON.stringify({ url: r.url }),
      });
      status = res.status;
      b = await res.json();
    } catch (e) {
      b = { error: String(e) };
    }
    const ms = Date.now() - t0;
    const cands: any[] = b.candidates ?? [];
    const top = cands[0];
    let verdict: string;
    if (status !== 200) verdict = "error";
    else if (kind === "none") verdict = cands.length === 0 ? "right" : b.confident ? "CONFIDENT-WRONG" : "offered-junk";
    else if (kind === "multi") verdict = b.confident ? "CONFIDENT-WRONG" : "right";
    else if (accepts(r.truth, top)) verdict = "right";
    else if (b.confident) verdict = "CONFIDENT-WRONG";
    else if (cands.some((c) => accepts(r.truth, c))) verdict = "offered";
    else verdict = cands.length ? "wrong" : "empty";
    const rec = {
      row: r.row, provider: r.provider, url: r.url, truth: r.truth, kind, verdict, status,
      source: b.source ?? null, confident: !!b.confident, top: top?.title ?? null,
      candidates: cands.map((c) => c.title), ms, error: b.error ?? null,
    };
    appendFileSync(outPath, JSON.stringify(rec) + "\n");
    console.log(`${String(r.row).padStart(3)} ${r.provider.padEnd(9)} ${verdict.padEnd(15)} ${rec.confident ? "C" : "-"} ${String(rec.source).padEnd(12)} ${String(ms).padStart(6)}ms  ${String(rec.top ?? "—").slice(0, 34).padEnd(34)} | ${r.truth.slice(0, 40)}`);
  }
} finally {
  await admin.auth.admin.deleteUser(u.user!.id);
}
