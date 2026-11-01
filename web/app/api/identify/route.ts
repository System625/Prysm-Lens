import { Client } from "@gradio/client";

// The ZeroGPU Space bills GPU time to whoever calls it. A browser calling it directly is an anonymous visitor
// (2 GPU minutes a day and a small cap on runs, measured 1 Nov: the demo failed after a few clips), so the GPU call
// goes through here with the Space owner's token instead (Hugging Face PRO: 40 minutes a day, top queue priority).
// The browser still uploads the video straight to the Space (no GPU, and no 4.5 MB request limit here); this route
// only sends the uploaded file's path, or a link, and streams the stages back as NDJSON.
const SPACE = process.env.NEXT_PUBLIC_LENS_SPACE ?? "System625/prysm-lens-gpu";

export const maxDuration = 300;

type Body = { path?: unknown; name?: unknown; url?: unknown };

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
        const app = await Client.connect(SPACE, { ...(token ? { token } : {}), events: ["data", "status"] });  // without status, errors end the stream silently
        const video = file ? { path: file, orig_name: typeof name === "string" ? name : undefined, meta: { _type: "gradio.FileData" } } : null;
        const job = app.submit("/identify", { video, url: link });
        let seen = 0;
        for await (const msg of job) {
          if (msg.type === "data") {
            const events = (msg.data as unknown[])[0] as unknown[];  // the list so far; send only new ones
            for (const e of events.slice(seen)) send(e);
            seen = events.length;
          } else if (msg.type === "status" && msg.stage === "error") {
            send({ stage: "error", message: typeof msg.message === "string" && msg.message ? msg.message : "The Space returned an error." });
            break;
          } else if (msg.type === "status" && msg.stage === "complete") {
            break;
          }
        }
      } catch (err) {
        send({ stage: "error", message: (err as Error).message || "Couldn't reach the Space." });
      } finally {
        ctrl.close();
      }
    },
  });
  return new Response(body, { headers: { "content-type": "application/x-ndjson", "cache-control": "no-store" } });
}
