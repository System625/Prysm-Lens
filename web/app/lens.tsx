"use client";

import { useEffect, useRef, useState } from "react";
import { Client } from "@gradio/client";

// Default: the ZeroGPU Space. The browser uploads the clip straight to it, then /api/identify makes the GPU call with
// the Space owner's token (anonymous visitors get almost no ZeroGPU quota). Set NEXT_PUBLIC_LENS_API to use the NDJSON
// service instead (the CPU Docker Space, or `uvicorn service.app:app` on a laptop).
const SPACE = process.env.NEXT_PUBLIC_LENS_SPACE ?? "System625/prysm-lens-gpu";
const API = process.env.NEXT_PUBLIC_LENS_API;

type Candidate = { id: number; title: string; year: string | null; score: number; editions: number[]; shots: string[] };
type Answer = {
  own_guess?: string;
  own_confidence?: string;
  pick?: string;
  game_title?: string;
  confidence?: string;
  evidence?: string;
  game?: { id: number; title: string; year: string | null; shots: string[] } | null;
  catalog?: { id: number; title: string; year: string | null; shots: string[] } | null;  // pick=none: its own answer, looked up
  prysm_id?: string;  // the game's id in the Prysm app, added by /api/identify
  error?: string | null;
};
type Run = {
  frames?: string[];
  skipped?: number;
  screen?: { found: boolean; crops: string[]; error?: string | null };
  candidates?: Candidate[];
  answer?: Answer;
  done?: { ms: Record<string, number>; usd: number };
  error?: string;
  started: number;
  at: Record<string, number>;
};

const LABELS = "ABCDEFGHIJ";
// Each model wears one of the app's status colours, so the four steps read apart at a glance.
const STAGES = [
  { key: "frames", n: "1", name: "Frames", model: "ffmpeg", tone: "dark", what: "16 frames spread across the clip" },
  { key: "screen", n: "2", name: "Find the screen", model: "SAM 3.1", tone: "gold", what: "Is this a phone filming a monitor or TV? Then search the screen, not the room" },
  { key: "candidates", n: "3", name: "Visual search", model: "Meta Perception Encoder", tone: "bone", what: "Nearest of 575,973 screenshots from 87,077 games" },
  { key: "answer", n: "4", name: "Verdict", model: "Muse Spark 1.3", tone: "orange", what: "Watches the clip (and the filmed screen, if SAM found one), names the game itself, then checks the candidates' screenshots" },
] as const;

type Input = { file?: File; url?: string };
type OnEvent = (e: Record<string, unknown>) => void;

// Client.connect first asks huggingface.co where the Space lives, and that lookup sometimes fails once ("Space
// metadata could not be loaded", 1 Nov). Retry before giving up; no GPU has been used yet.
async function connectWithRetry(tries = 3): Promise<Client> {
  for (let i = 1; ; i++) {
    try {
      return await Client.connect(SPACE);
    } catch (e) {
      if (i >= tries) throw e;
      await new Promise((r) => setTimeout(r, 1000 * i));
    }
  }
}
let gradio: Promise<Client> | null = null;
const connect = () => (gradio ??= connectWithRetry().catch((e) => { gradio = null; throw e; }));

// Gradio's and the browser's own error texts mean nothing to a visitor; say what happened and what to do.
function friendly(message: string) {
  if (/quota|ZeroGPU/i.test(message)) return "The demo's free GPU time is used up for now. Try again later.";
  if (/busy/i.test(message)) return "Lens is busy with other clips right now. Try again in a minute.";
  if (/Space metadata|Connection errored|resolve app config|space status|API info|Failed to fetch|fetch failed|NetworkError|Load failed/i.test(message))
    return "Couldn't reach the Lens service. Check your connection and try again.";
  return message;
}

async function streamSpace(input: Input, onEvent: OnEvent) {
  let path: string | undefined;
  if (input.file) {
    const app = await connect();
    const up = await app.upload_files(app.config!.root, [input.file]);
    if (up.error || !up.files?.[0]) throw new Error(`Upload failed: ${up.error ?? "no file path returned"}`);
    path = up.files[0];
  }
  const res = await fetch("/api/identify", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ path, name: input.file?.name, url: input.url }),
  });
  await readNdjson(res, (e) => {
    if (e.stage === "error" && typeof e.message === "string") e = { ...e, message: friendly(e.message) };
    onEvent(e);
  });
}

function stream(input: Input, onEvent: OnEvent) {
  if (!API) return streamSpace(input, onEvent);
  const body = new FormData();
  if (input.file) body.append("file", input.file);
  if (input.url) body.append("url", input.url);
  return fetch(`${API}/identify`, { method: "POST", body }).then((res) => readNdjson(res, onEvent));
}

async function readNdjson(res: Response, onEvent: OnEvent) {
  if (!res.ok || !res.body) {
    const j = await res.json().catch(() => ({ error: `${res.status} ${res.statusText}` }));
    throw new Error(j.error ?? `HTTP ${res.status}`);
  }
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i: number;
    while ((i = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, i).trim();
      buf = buf.slice(i + 1);
      if (line) onEvent(JSON.parse(line));
    }
  }
}

export default function Lens() {
  const [tab, setTab] = useState<"upload" | "link" | "camera">("upload");
  const [run, setRun] = useState<Run | null>(null);
  const [busy, setBusy] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const [now, setNow] = useState(Date.now());
  const [awake, setAwake] = useState<"checking" | "ready" | "waking">("checking");

  useEffect(() => {
    let stop = false;
    const ping = async () => {
      try {
        if (!API) {
          setAwake("waking");
          await connect();
          if (!stop) setAwake("ready");
          return;
        }
        const r = await fetch(`${API}/`, { cache: "no-store" });
        const j = await r.json();
        if (!stop) setAwake(j.loaded ? "ready" : "waking");
        if (!j.loaded && !stop) setTimeout(ping, 5000);
      } catch {
        if (!stop) { setAwake("waking"); setTimeout(ping, 8000); }
      }
    };
    ping();
    return () => { stop = true; };
  }, []);

  useEffect(() => {
    if (!busy) return;
    const t = setInterval(() => setNow(Date.now()), 250);
    return () => clearInterval(t);
  }, [busy]);

  async function go(input: Input, videoUrl: string | null) {
    setPreview(videoUrl);
    const started = Date.now();
    setRun({ started, at: {} });
    setBusy(true);
    try {
      await stream(input, (e) => {
        const stage = e.stage as string;
        setRun((r) => {
          if (!r) return r;
          const at = { ...r.at, [stage]: Date.now() };
          if (stage === "frames") return { ...r, at, frames: e.frames as string[], skipped: (e.skipped as number) ?? 0 };
          if (stage === "screen") return { ...r, at, screen: e as unknown as Run["screen"] };
          if (stage === "candidates") return { ...r, at, candidates: e.items as Candidate[] };
          if (stage === "answer") return { ...r, at, answer: e as Answer };
          if (stage === "done") return { ...r, at, done: e as unknown as Run["done"] };
          if (stage === "error") return { ...r, at, error: e.message as string };
          return r;
        });
      });
    } catch (err) {
      setRun((r) => r && { ...r, error: friendly((err as Error).message) });
    } finally {
      setBusy(false);
    }
  }

  function submitFile(f: File) {
    if (f.size > 200e6) return setRun({ started: Date.now(), at: {}, error: "That video is over 200 MB." });
    go({ file: f }, URL.createObjectURL(f));
  }

  return (
    <div className="glow min-h-dvh">
      <main className="mx-auto max-w-5xl px-4 pb-24 pt-6 sm:px-6">
        <header className="flex items-center gap-3">
          <a href="https://prysmblast.com" className="flex items-center gap-2.5" aria-label="Prysm">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src="/prysm-logo.svg" alt="Prysm" className="h-7 w-auto sm:h-8" />
            <span className="sticker sticker--gold">Lens</span>
          </a>
          <span className="ml-auto flex items-center gap-2 rounded-full bg-pill px-3 py-1.5 text-xs font-bold text-muted backdrop-blur">
            <span className={`h-2 w-2 rounded-full ${awake === "ready" ? "bg-ok" : "bg-warn pulse"}`} />
            {awake === "ready" ? "Ready" : awake === "waking" ? "Waking up (about a minute)" : "Checking"}
          </span>
        </header>

        <section className="pb-8 pt-12 sm:pb-12 sm:pt-16">
          <h1 className="text-[clamp(3rem,1.6rem+5vw,5.75rem)]">
            <span className="display -rotate-2">Which game</span>
            <span className="display mt-2 translate-x-[0.3em] rotate-[1.5deg] sm:mt-3">is this?</span>
          </h1>
          <p className="mt-8 max-w-xl text-lg font-black leading-snug text-gold sm:text-xl">
            Drop in any gameplay clip, even a phone filming a TV.
          </p>
          <p className="mt-3 max-w-xl text-base leading-relaxed text-muted sm:text-lg">
            Lens finds the screen, searches half a million game screenshots, and has Muse Spark name the game, with
            its evidence.
          </p>
        </section>

        <section className="rounded-[28px] border border-line bg-[linear-gradient(160deg,#191311,#0d0a09_70%)] p-3 shadow-[0_30px_80px_-30px_rgba(253,80,33,0.35)] sm:p-5">
          <div className="mb-4 flex gap-1 rounded-full bg-pill p-1 text-sm" role="tablist">
            {(["upload", "link", "camera"] as const).map((t) => (
              <button key={t} role="tab" aria-selected={tab === t} onClick={() => setTab(t)} disabled={busy}
                className={`flex-1 rounded-full px-3 py-2.5 font-extrabold transition disabled:opacity-50 ${tab === t ? "bg-fg text-[#170802]" : "text-muted hover:text-fg"}`}>
                {t === "upload" ? "Upload" : t === "link" ? "Paste a link" : "Camera"}
              </button>
            ))}
          </div>
          {tab === "upload" && <Upload disabled={busy} onFile={submitFile} />}
          {tab === "link" && <Link disabled={busy} onUrl={(u) => go({ url: u }, null)} />}
          {tab === "camera" && <Camera disabled={busy} onClip={submitFile} />}
        </section>

        {run && <Pipeline run={run} now={now} busy={busy} preview={preview} />}

        <About />
      </main>
    </div>
  );
}

function Upload({ disabled, onFile }: { disabled: boolean; onFile: (f: File) => void }) {
  const [over, setOver] = useState(false);
  return (
    <label
      onDragOver={(e) => { e.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); const f = e.dataTransfer.files[0]; if (f && !disabled) onFile(f); }}
      className={`flex cursor-pointer flex-col items-center justify-center rounded-[20px] border-2 border-dashed px-4 py-12 text-center transition ${over ? "border-accent bg-accent-soft" : "border-white/15 hover:border-white/35"} ${disabled ? "pointer-events-none opacity-50" : ""}`}>
      <span className="step-num mb-4 !h-12 !w-12">
        <svg viewBox="0 0 24 24" className="h-6 w-6" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
          <path d="M12 16V4M6 10l6-6 6 6M4 20h16" />
        </svg>
      </span>
      <span className="text-lg font-extrabold">Drop a video here, or tap to choose one</span>
      <span className="mt-1 text-sm text-faint">MP4, MOV or WebM, up to 200 MB. Only the first 3 minutes are used.</span>
      <input type="file" accept="video/*" className="hidden" disabled={disabled}
        onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f); e.target.value = ""; }} />
    </label>
  );
}

function Link({ disabled, onUrl }: { disabled: boolean; onUrl: (u: string) => void }) {
  const [url, setUrl] = useState("");
  return (
    <form className="flex flex-col gap-3 p-1 sm:flex-row" onSubmit={(e) => { e.preventDefault(); if (url.trim()) onUrl(url.trim()); }}>
      <input value={url} onChange={(e) => setUrl(e.target.value)} disabled={disabled} inputMode="url"
        placeholder="https://www.youtube.com/shorts/..."
        className="min-h-[52px] min-w-0 flex-1 rounded-[14px] border-2 border-white/10 bg-black/40 px-4 font-semibold outline-none placeholder:text-white/35 focus:border-accent" />
      <button disabled={disabled || !url.trim()} className="btn">Identify</button>
      <p className="text-xs text-faint sm:hidden">Some sites block servers from fetching their videos. If a link fails, download the clip and upload it.</p>
    </form>
  );
}
function Camera({ disabled, onClip }: { disabled: boolean; onClip: (f: File) => void }) {
  const video = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [state, setState] = useState<"off" | "live" | "rec">("off");
  const [left, setLeft] = useState(0);
  const [err, setErr] = useState<string | null>(null);
  const SECONDS = 8;

  useEffect(() => () => streamRef.current?.getTracks().forEach((t) => t.stop()), []);

  async function start() {
    setErr(null);
    try {
      const s = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", height: { ideal: 720 } }, audio: true });
      streamRef.current = s;
      if (video.current) { video.current.srcObject = s; await video.current.play(); }
      setState("live");
    } catch (e) {
      setErr(`Camera unavailable: ${(e as Error).message}`);
    }
  }

  function record() {
    const s = streamRef.current;
    if (!s) return;
    const type = ["video/mp4", "video/webm;codecs=vp9,opus", "video/webm"].find((t) => MediaRecorder.isTypeSupported(t)) ?? "";
    const rec = new MediaRecorder(s, type ? { mimeType: type } : undefined);
    const parts: Blob[] = [];
    rec.ondataavailable = (e) => e.data.size && parts.push(e.data);
    rec.onstop = () => {
      const blob = new Blob(parts, { type: rec.mimeType || "video/webm" });
      const ext = blob.type.includes("mp4") ? "mp4" : "webm";
      s.getTracks().forEach((t) => t.stop());
      setState("off");
      onClip(new File([blob], `camera.${ext}`, { type: blob.type }));
    };
    rec.start(1000);
    setState("rec");
    setLeft(SECONDS);
    let n = SECONDS;
    const t = setInterval(() => { n -= 1; setLeft(n); if (n <= 0) { clearInterval(t); rec.stop(); } }, 1000);
  }

  return (
    <div className="flex flex-col items-center gap-4">
      <video ref={video} playsInline muted className={`aspect-video w-full rounded-[20px] bg-black object-cover ${state === "off" ? "hidden" : ""}`} />
      {state === "off" && (
        <button onClick={start} disabled={disabled}
          className="flex w-full flex-col items-center rounded-[20px] border-2 border-dashed border-white/15 px-4 py-12 text-center hover:border-white/35 disabled:opacity-50">
          <span className="step-num mb-4 !h-12 !w-12">
            <svg viewBox="0 0 24 24" className="h-6 w-6" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
              <path d="M4 8h3l2-3h6l2 3h3v11H4z" /><circle cx="12" cy="13" r="3.5" />
            </svg>
          </span>
          <span className="text-lg font-extrabold">Start the camera</span>
          <span className="mt-1 text-sm text-faint">Then point it at a TV or monitor showing a game</span>
        </button>
      )}
      {state === "live" && <button onClick={record} className="btn">Record {SECONDS} seconds</button>}
      {state === "rec" && <p className="flex items-center gap-2 font-extrabold"><span className="h-2.5 w-2.5 rounded-full bg-bad pulse" />Recording… {left}s</p>}
      {err && <p className="text-sm font-semibold text-bad">{err}</p>}
    </div>
  );
}

function Pipeline({ run, now, busy, preview }: { run: Run; now: number; busy: boolean; preview: string | null }) {
  const has: Record<string, boolean> = { frames: !!run.frames, screen: !!run.screen, candidates: !!run.candidates, answer: !!run.answer };
  const firstPending = STAGES.find((s) => !has[s.key])?.key;
  const prevAt = (i: number) => (i === 0 ? run.started : run.at[STAGES[i - 1].key] ?? run.started);
  const seconds = (i: number) => (((run.at[STAGES[i].key] ?? now) - prevAt(i)) / 1000).toFixed(1);

  // The answer lands at the top, but people scroll down to watch the frames and never saw it (user, 1 Nov). So the top
  // slot holds a "working" card from the start, the page scrolls to it when a run starts, and scrolls back to it when
  // the answer or an error replaces it, unless it's already on screen.
  const slot = useRef<HTMLDivElement>(null);
  const kind = run.answer ? "answer" : run.error ? "error" : busy ? "working" : "none";
  useEffect(() => {
    const el = slot.current;
    if (!el || kind === "none") return;
    const r = el.getBoundingClientRect();
    if (kind !== "working" && r.top >= 0 && r.top < window.innerHeight * 0.5) return;  // its top is already in view
    const smooth = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    el.scrollIntoView({ behavior: smooth ? "smooth" : "auto", block: "start" });
  }, [kind, run.started]);
  const current = STAGES.findIndex((s) => s.key === firstPending);

  return (
    <section className="mt-10 space-y-4">
      <div ref={slot} className="scroll-mt-4 space-y-4">
        {run.answer && <Verdict a={run.answer} candidates={run.candidates ?? []} />}
        {run.error && <div className="rounded-[20px] border-2 border-bad/50 bg-bad/10 p-4 text-sm font-semibold">{run.error}</div>}
        {kind === "working" && current >= 0 && (
          <div className="rise flex items-center gap-4 rounded-[28px] border-2 border-dashed border-accent/50 bg-accent-soft p-5 sm:p-6">
            <span className="step-num pulse !h-12 !w-12 text-lg">{STAGES[current].n}</span>
            <div className="min-w-0 flex-1">
              <p className="text-xl font-black sm:text-2xl">Working it out…</p>
              <p className="mt-0.5 text-sm text-muted">
                Step {STAGES[current].n} of {STAGES.length}: {STAGES[current].name} ({STAGES[current].model}). The answer appears here.
              </p>
            </div>
            <span className="rounded-full bg-pill px-2.5 py-1 font-mono text-xs text-muted">{((now - run.started) / 1000).toFixed(0)}s</span>
          </div>
        )}
      </div>

      <h2 className="px-1 pt-4 text-2xl font-black tracking-tight sm:text-3xl">How Lens got there</h2>

      {STAGES.map((s, i) => {
        const done = has[s.key];
        const active = busy && !run.error && s.key === firstPending;
        return (
          <div key={s.key}
            className={`rounded-3xl border bg-panel p-4 transition sm:p-5 ${active ? "border-accent/60 shadow-[0_0_0_4px_rgba(253,80,33,0.12)]" : "border-line"} ${!done && !active ? "opacity-50" : ""}`}>
            <div className="flex items-center gap-3">
              <span className={`step-num ${done || active ? "" : "step-num--idle"}`}>{s.n}</span>
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
                  <h3 className="text-lg font-black">{s.name}</h3>
                  <span className={`sticker sticker--${s.tone}`}>{s.model}</span>
                </div>
                <p className="mt-1 text-sm text-faint">{s.what}</p>
              </div>
              <span className="self-start rounded-full bg-pill px-2.5 py-1 font-mono text-xs text-muted">
                {done ? `${seconds(i)}s` : active ? <span className="pulse">{seconds(i)}s</span> : "waiting"}
              </span>
            </div>
            {done && <div className="rise mt-4">{body(s.key, run, preview)}</div>}
          </div>
        );
      })}

      {run.done && (
        <p className="pt-2 text-center text-sm font-bold text-faint">
          {((run.at.done - run.started) / 1000).toFixed(1)}s end to end · about ${run.done.usd.toFixed(3)} of Meta API calls
        </p>
      )}
    </section>
  );
}

function body(key: string, run: Run, preview: string | null) {
  if (key === "frames" && run.frames)
    return (
      <div className="space-y-3">
        {preview && <video src={preview} controls muted playsInline className="max-h-72 rounded-2xl" />}
        <div className="grid grid-cols-4 gap-1.5 sm:grid-cols-8">
          {run.frames.map((f, i) => <img key={i} src={f} alt="" className="aspect-square w-full rounded-lg object-cover" />)}
        </div>
        {!!run.skipped && (
          <p className="text-sm text-muted">
            Skipped {run.skipped} blank {run.skipped === 1 ? "frame" : "frames"} (fades, outros, title cards).
          </p>
        )}
      </div>
    );
  if (key === "screen" && run.screen)
    return run.screen.found ? (
      <div className="space-y-3">
        <p className="text-sm text-muted"><span className="font-extrabold text-gold">Filmed screen found</span> on most frames, so Lens searches the screen and ignores the room around it.</p>
        <div className="grid grid-cols-4 gap-1.5 sm:grid-cols-8">
          {run.screen.crops.map((f, i) => <img key={i} src={f} alt="" className="aspect-video w-full rounded-lg object-cover ring-2 ring-gold/70" />)}
        </div>
      </div>
    ) : (
      <p className="text-sm text-muted">
        {run.screen.error ? `SAM 3.1 didn't answer (${run.screen.error}); searching the full frames.` : "No filmed screen: this is direct gameplay, so Lens searches the full frames."}
      </p>
    );
  if (key === "candidates" && run.candidates) {
    const max = Math.max(...run.candidates.map((c) => c.score), 0.001);
    const picked = run.answer?.game?.id;
    return (
      <ol className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
        {run.candidates.map((c, i) => (
          <li key={c.id} className={`relative rounded-2xl border-2 p-3 ${picked === c.id ? "border-accent bg-accent-soft" : "border-transparent bg-panel-2"}`}>
            {picked === c.id && <span className="sticker sticker--orange absolute -top-3 right-3 rotate-[4deg]">Muse Spark&apos;s pick</span>}
            <div className="flex items-center gap-2.5 text-sm">
              <span className={`grid h-6 w-6 shrink-0 place-items-center rounded-full text-xs font-black ${picked === c.id ? "bg-accent text-white" : "bg-white/10 text-muted"}`}>{LABELS[i]}</span>
              <span className="truncate font-extrabold">{c.title}</span>
              {c.year && <span className="shrink-0 text-faint">{c.year}</span>}
            </div>
            <div className="mt-2 h-1.5 rounded-full bg-black/50"><div className="h-1.5 rounded-full bg-accent" style={{ width: `${(c.score / max) * 100}%` }} /></div>
            <div className="mt-2.5 grid grid-cols-3 gap-1">
              {c.shots.map((s) => <img key={s} src={s} alt="" loading="lazy" className="aspect-video w-full rounded-md object-cover" />)}
            </div>
          </li>
        ))}
      </ol>
    );
  }
  if (key === "answer" && run.answer)
    return (
      <div className="grid gap-3 text-sm sm:grid-cols-2">
        <div className="rounded-2xl bg-panel-2 p-4">
          <p className="text-xs font-extrabold uppercase tracking-wider text-faint">Step 1 · from the clip alone</p>
          <p className="mt-1.5 text-base font-black">{run.answer.own_guess ?? "–"}</p>
          <p className="text-xs text-muted">{run.answer.own_confidence} confidence</p>
        </div>
        <div className="rounded-2xl bg-panel-2 p-4">
          <p className="text-xs font-extrabold uppercase tracking-wider text-faint">Step 2 · after checking the candidates</p>
          <p className="mt-1.5 text-base font-black">{run.answer.game_title ?? "–"}</p>
          <p className="text-xs text-muted">
            {run.answer.pick && run.answer.pick !== "none" ? `Matched candidate ${run.answer.pick}` : "No candidate matched; kept its own answer"}
          </p>
        </div>
      </div>
    );
  return null;
}

function Verdict({ a, candidates }: { a: Answer; candidates: Candidate[] }) {
  if (!a.game_title) return <div className="rounded-[28px] border border-line bg-panel p-6 font-semibold text-muted">Muse Spark couldn&apos;t give an answer{a.error ? `: ${a.error}` : "."}</div>;
  const tone = a.confidence === "high" ? "orange" : a.confidence === "medium" ? "gold" : "rose";
  const fromLens = !!a.game;
  const rank = fromLens ? candidates.findIndex((c) => c.id === a.game!.id) + 1 : 0;
  const cat = fromLens ? null : a.catalog;
  const shots = a.game?.shots ?? cat?.shots ?? [];
  return (
    <div className="pop overflow-hidden rounded-[28px] border-2 border-accent bg-[radial-gradient(80%_120%_at_0%_0%,rgba(253,80,33,0.28),rgba(73,23,10,0.25)_45%,#120e0d_80%)] shadow-[0_30px_80px_-24px_rgba(253,80,33,0.45)]">
      <div className="p-5 sm:p-7">
        {/* A low-confidence own answer is a guess (a filmed Horizon clip read as Ghost of Tsushima, 1 Nov): say so. */}
        <p className="text-base font-black text-gold sm:text-lg">{a.confidence === "high" ? "This is" : a.confidence === "medium" ? "Most likely" : "Best guess, not sure"}</p>
        <h2 className="mt-1 text-4xl font-black leading-[1.02] tracking-tight text-white sm:text-6xl">
          {/* Muse Spark sometimes ends the title with the year ("... Breakpoint (2019)"); the year is shown beside it anyway. */}
          {(a.game ?? cat)?.year ? a.game_title.replace(new RegExp(`\\s*\\(${(a.game ?? cat)!.year}\\)$`), "") : a.game_title}
          {(a.game ?? cat)?.year && <span className="ml-3 text-2xl font-extrabold text-white/40 sm:text-3xl">{(a.game ?? cat)!.year}</span>}
        </h2>
        <div className="mt-4 flex flex-wrap items-center gap-x-3 gap-y-2 text-sm">
          <span className={`sticker sticker--${tone}`}>{(a.confidence ?? "").replace(/^./, (c) => c.toUpperCase())} confidence</span>
          <span className="font-semibold text-muted">{fromLens ? `Confirmed against search result #${rank}` : "Muse Spark's own answer; none of the search results matched"}</span>
        </div>
        {a.evidence && (
          <div className="mt-5 rounded-2xl bg-black/35 p-4">
            <p className="text-xs font-extrabold uppercase tracking-wider text-accent">Muse Spark&apos;s evidence</p>
            <p className="mt-1.5 text-sm leading-relaxed text-fg/85 sm:text-base">{a.evidence}</p>
          </div>
        )}
        {a.prysm_id && <SaveToPrysm id={a.prysm_id} />}
      </div>
      {cat && (
        <p className="border-t border-white/10 px-5 py-3 text-xs font-semibold text-faint sm:px-7">
          {cat.shots.length
            ? `Looked up in the catalog: ${cat.title}'s screenshots closest to your clip. Muse Spark didn't match them to a search result, so compare by eye.`
            : `${cat.title} is in the catalog, but IGDB has no screenshots of it, so visual search could never have found it.`}
        </p>
      )}
      {shots.length > 0 && (
        <div className="grid grid-cols-3 gap-1 px-1 pb-1">
          {shots.map((s) => <img key={s} src={s} alt="" className="aspect-video w-full rounded-[10px] object-cover" />)}
        </div>
      )}
    </div>
  );
}

const APP_STORE = "https://apps.apple.com/us/app/prysm-game-backlog-tracker/id6814823840";
const PLAY_STORE = "https://play.google.com/store/apps/details?id=com.prysmblast.app";

// Saving needs an account, and Prysm already has them: the app opens prysm://game/<id> on that game's page (the link its
// home-screen widgets use), where it goes into the library in one tap. Computers can't open the app, so they get the stores.
function SaveToPrysm({ id }: { id: string }) {
  const phone = typeof navigator !== "undefined" && /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
  const stores = (
    <>
      <a className="font-bold text-fg underline underline-offset-2" href={APP_STORE}>App Store</a>
      {" · "}
      <a className="font-bold text-fg underline underline-offset-2" href={PLAY_STORE}>Google Play</a>
    </>
  );
  return (
    <div className="mt-5 flex flex-col gap-4 rounded-2xl border border-white/10 bg-white/[0.04] p-4 sm:flex-row sm:items-center">
      <div className="min-w-0 flex-1">
        <p className="text-base font-black">Save it to your library</p>
        <p className="mt-0.5 text-sm text-muted">
          {phone ? <>Opens the game in Prysm. Don&apos;t have it? {stores}</> : <>Open this page on your phone to add it in Prysm, or get the app: {stores}</>}
        </p>
      </div>
      {phone && <a href={`prysm://game/${encodeURIComponent(id)}`} className="btn shrink-0">Open in Prysm</a>}
    </div>
  );
}

const STATS = [
  { n: "85", label: "Captions only (Prysm today)" },
  { n: "77", label: "Muse Spark watching the clip, no search" },
  { n: "91", label: "Captions, else Lens + Muse Spark", best: true },
  { n: "1 → 13", label: "On the 19 clips captions can't name", best: true },
];

function About() {
  return (
    <section className="mt-20 border-t border-line pt-12">
      <div className="grid gap-10 sm:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
        <div>
          <h2 className="text-3xl font-black leading-tight tracking-tight sm:text-4xl">Why Lens?</h2>
          <p className="mt-3 leading-relaxed text-muted">
            Gamers share clips constantly, and the caption rarely names the game. Prysm, a game-library app, reads
            captions and names the game in 85 of 104 real shared clips. Lens is for the rest: indie games, filmed TVs,
            junk titles.
          </p>
        </div>
        <div>
          <p className="text-xs font-extrabold uppercase tracking-wider text-faint">Benchmark · 104 real shared clips</p>
          <ul className="mt-3 grid grid-cols-2 gap-2.5">
            {STATS.map((s) => (
              <li key={s.label} className={`rounded-2xl border p-4 ${s.best ? "border-accent/50 bg-accent-soft" : "border-line bg-panel"}`}>
                <p className={`text-3xl font-black tracking-tight sm:text-4xl ${s.best ? "text-accent" : "text-fg"}`}>{s.n}</p>
                <p className="mt-1 text-sm leading-snug text-muted">{s.label}</p>
              </li>
            ))}
          </ul>
        </div>
      </div>
      <footer className="mt-14 flex flex-col gap-4 border-t border-line pt-6 text-xs text-faint sm:flex-row sm:items-center">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src="/prysm-logo.svg" alt="Prysm" className="h-6 w-auto self-start" />
        <p className="leading-relaxed sm:ml-4">
          Built during the Meta Global AI Developer Hackathon (Oct 26 to Nov 4, 2026) with SAM 3.1, Muse Spark 1.3 and
          Meta Perception Encoder. Screenshots and game data from IGDB.com. Code and benchmark:{" "}
          <a className="font-bold text-accent underline underline-offset-2" href="https://github.com/System625/Prysm-Lens">github.com/System625/Prysm-Lens</a>
        </p>
      </footer>
    </section>
  );
}
