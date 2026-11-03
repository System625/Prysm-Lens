import { Client } from "@gradio/client";

// The ZeroGPU Space bills GPU time to whoever calls it. A browser calling it directly is an anonymous visitor
// (2 GPU minutes a day and a small cap on runs, measured 1 Nov: the demo failed after a few clips), so the GPU call
// goes through here with the Space owner's token instead (Hugging Face PRO: 40 minutes a day, top queue priority).
// The browser still uploads the video straight to the Space (no GPU, and no 4.5 MB request limit here); this route
// only sends the uploaded file's path, or a link, and streams the stages back as NDJSON.
const SPACE = process.env.NEXT_PUBLIC_LENS_SPACE ?? "System625/prysm-lens-gpu";

export const maxDuration = 300;

// Client.connect first asks huggingface.co where the Space lives. That lookup sometimes fails once ("Space metadata
// could not be loaded", seen 1 Nov on a clip the live site then named fine), and one failure used to end the run.
// Nothing has run on the GPU yet at this point, so retrying costs no quota.
async function connectWithRetry(token: `hf_${string}` | undefined, tries = 3) {
  for (let i = 1; ; i++) {
    try {
      return await Client.connect(SPACE, { ...(token ? { token } : {}), events: ["data", "status"] });  // without status, errors end the stream silently
    } catch (err) {
      if (i >= tries || /could not be accessed|401|not authorized/i.test((err as Error).message)) throw err;
      await new Promise((r) => setTimeout(r, 1000 * i));
    }
  }
}

type Body = { path?: unknown; name?: unknown; url?: unknown };

// "Open in Prysm": the app opens a game from prysm://game/<games.id>, but Lens names games by IGDB id. lookup/prysm-ids.json
// maps every game in Lens's catalog (91,976, all of them) to its Prysm id; exported 1 Nov from Shelf's games table, where
// igdb_id is unique. Loaded on the first answer only, so runs that fail early never pay for the 4.4 MB.
let prysmIds: Promise<Record<string, string>> | null = null;
async function withPrysmId(e: unknown) {
  const a = e as { stage?: string; game?: { id?: number } | null; catalog?: { id?: number } | null };
  const igdb = a?.stage === "answer" ? (a.game ?? a.catalog)?.id : undefined;
  if (igdb == null) return e;
  prysmIds ??= import("@/lookup/prysm-ids.json").then((m) => m.default as Record<string, string>);
  const id = (await prysmIds)[String(igdb)];
  return id ? { ...a, prysm_id: id } : e;
}

export async function POST(req: Request) {
  const { path, name, url } = (await req.json().catch(() => ({}))) as Body;
  const file = typeof path === "string" && path && !/^[a-z]+:\/\//i.test(path) ? path : null;  // an upload, not a URL
  const link = typeof url === "string" ? url.trim().slice(0, 2000) : "";
  if (!file && !link) return Response.json({ error: "Send an uploaded file's path or a link." }, { status: 400 });

  const token = process.env.HF_TOKEN as `hf_${string}` | undefined;
  const enc = new TextEncoder();
  const body = new ReadableStream({
    async start(ctrl) {
      const send = (e: unknown) => ctrl.enqueue(enc.encode(JSON.stringify(e) + "\n"));
      try {
        const app = await connectWithRetry(token);
        const video = file ? { path: file, orig_name: typeof name === "string" ? name : undefined, meta: { _type: "gradio.FileData" } } : null;
        const job = app.submit("/identify", { video, url: link });
        let seen = 0;
        let ended = false;  // an answer or an error went out
        for await (const msg of job) {
          if (msg.type === "data") {
            const events = (msg.data as unknown[])[0] as unknown[];  // the list so far; send only new ones
            for (const e of events.slice(seen)) {
              const stage = (e as { stage?: string })?.stage;
              if (stage === "answer" || stage === "error") ended = true;
              send(await withPrysmId(e));
            }
            seen = events.length;
          } else if (msg.type === "status" && msg.stage === "error") {
            send({ stage: "error", message: typeof msg.message === "string" && msg.message ? msg.message : "The Space returned an error." });
            ended = true;
            break;
          } else if (msg.type === "status" && msg.stage === "complete") {
            break;
          }
        }
        if (!ended) send({ stage: "error", message: "Lens stopped before it answered. Try again." });
      } catch (err) {
        send({ stage: "error", message: (err as Error).message || "Couldn't reach the Space." });
      } finally {
        ctrl.close();
      }
    },
  });
  return new Response(body, { headers: { "content-type": "application/x-ndjson", "cache-control": "no-store" } });
}
