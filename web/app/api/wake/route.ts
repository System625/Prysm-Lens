import { Client } from "@gradio/client";

// A Space with no visitors for 48 hours goes to sleep (gcTimeout 172800, checked 3 Nov), and the next visitor waits
// while it boots and pulls the index. Judging runs for days after submission, so a daily Vercel cron (vercel.json)
// connects here. Connecting reads the Space's config and wakes it if needed; it runs nothing on the GPU.
const SPACE = process.env.NEXT_PUBLIC_LENS_SPACE ?? "System625/prysm-lens-gpu";

export const maxDuration = 300;

export async function GET() {
  const t = Date.now();
  try {
    const token = process.env.HF_TOKEN as `hf_${string}` | undefined;
    await Client.connect(SPACE, token ? { token } : {});
    return Response.json({ ok: true, ms: Date.now() - t });
  } catch (err) {
    return Response.json({ ok: false, error: (err as Error).message, ms: Date.now() - t }, { status: 502 });
  }
}
