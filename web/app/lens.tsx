"use client";

import { useEffect, useRef, useState } from "react";

const API = process.env.NEXT_PUBLIC_LENS_API ?? "https://system625-prysm-lens.hf.space";

type Candidate = { id: number; title: string; year: string | null; score: number; editions: number[]; shots: string[] };
type Answer = {
  own_guess?: string;
  own_confidence?: string;
  pick?: string;
  game_title?: string;
  confidence?: string;
  evidence?: string;
  game?: { id: number; title: string; year: string | null; shots: string[] } | null;
  error?: string | null;
};
type Run = {
  frames?: string[];
  screen?: { found: boolean; crops: string[]; error?: string | null };
  candidates?: Candidate[];
  answer?: Answer;
  done?: { ms: Record<string, number>; usd: number };
  error?: string;
  started: number;
  at: Record<string, number>;
};

const LABELS = "ABCDEFGHIJ";
const STAGES = [
  { key: "frames", n: "01", name: "Frames", model: "ffmpeg", what: "16 frames spread across the clip" },
  { key: "screen", n: "02", name: "Find the screen", model: "SAM 3.1", what: "Is this a phone filming a monitor or TV? Then search the screen, not the room" },
  { key: "candidates", n: "03", name: "Visual search", model: "Meta Perception Encoder", what: "Nearest of 575,973 screenshots from 87,077 games" },
  { key: "answer", n: "04", name: "Verdict", model: "Muse Spark 1.3", what: "Watches the clip, names the game itself, then checks the candidates' screenshots" },
] as const;

async function stream(body: FormData, onEvent: (e: Record<string, unknown>) => void) {
  const res = await fetch(`${API}/identify`, { method: "POST", body });
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

  async function go(body: FormData, videoUrl: string | null) {
    setPreview(videoUrl);
    const started = Date.now();
    setRun({ started, at: {} });
    setBusy(true);
    try {
      await stream(body, (e) => {
        const stage = e.stage as string;
        setRun((r) => {
          if (!r) return r;
          const at = { ...r.at, [stage]: Date.now() };
          if (stage === "frames") return { ...r, at, frames: e.frames as string[] };
          if (stage === "screen") return { ...r, at, screen: e as unknown as Run["screen"] };
          if (stage === "candidates") return { ...r, at, candidates: e.items as Candidate[] };
          if (stage === "answer") return { ...r, at, answer: e as Answer };
          if (stage === "done") return { ...r, at, done: e as unknown as Run["done"] };
          if (stage === "error") return { ...r, at, error: e.message as string };
          return r;
        });
      });
    } catch (err) {
      setRun((r) => r && { ...r, error: (err as Error).message });
    } finally {
      setBusy(false);
    }
  }

  function submitFile(f: File) {
    if (f.size > 200e6) return setRun({ started: Date.now(), at: {}, error: "That video is over 200 MB." });
    const body = new FormData();
    body.append("file", f);
    go(body, URL.createObjectURL(f));
  }

  return (
    <main className="mx-auto max-w-5xl px-4 pb-24 pt-10 sm:px-6">
      <header className="mb-8">
        <div className="flex items-center gap-2 text-sm text-muted">
          <span className="inline-block h-2.5 w-2.5 rounded-full bg-accent" />
          Prysm Lens
          <span className="ml-auto flex items-center gap-1.5 text-xs">
            <span className={`h-1.5 w-1.5 rounded-full ${awake === "ready" ? "bg-ok" : "bg-warn pulse"}`} />
            {awake === "ready" ? "Service ready" : awake === "waking" ? "Service waking up (about a minute)" : "Checking service"}
          </span>
        </div>
        <h1 className="mt-4 text-3xl font-semibold tracking-tight sm:text-5xl">Which game is this?</h1>
        <p className="mt-3 max-w-2xl text-muted sm:text-lg">
          Drop in any gameplay clip, even a phone filming a TV. Lens finds the screen, searches half a million game
          screenshots, and has Muse Spark name the game with its evidence.
        </p>
      </header>

      <section className="rounded-2xl border border-line bg-panel p-4 sm:p-5">
        <div className="mb-4 flex gap-1 rounded-xl bg-panel-2 p-1 text-sm">
          {(["upload", "link", "camera"] as const).map((t) => (
            <button key={t} onClick={() => setTab(t)} disabled={busy}
              className={`flex-1 rounded-lg px-3 py-2 capitalize transition ${tab === t ? "bg-bg text-fg" : "text-muted hover:text-fg"}`}>
              {t === "upload" ? "Upload a clip" : t === "link" ? "Paste a link" : "Point your camera"}
            </button>
          ))}
        </div>
        {tab === "upload" && <Upload disabled={busy} onFile={submitFile} />}
        {tab === "link" && <Link disabled={busy} onUrl={(u) => { const b = new FormData(); b.append("url", u); go(b, null); }} />}
        {tab === "camera" && <Camera disabled={busy} onClip={submitFile} />}
      </section>

      {run && <Pipeline run={run} now={now} busy={busy} preview={preview} />}

      <About />
    </main>
  );
}

function Upload({ disabled, onFile }: { disabled: boolean; onFile: (f: File) => void }) {
  const [over, setOver] = useState(false);
  return (
    <label
      onDragOver={(e) => { e.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); const f = e.dataTransfer.files[0]; if (f && !disabled) onFile(f); }}
      className={`flex cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed px-4 py-10 text-center transition ${over ? "border-accent bg-accent-soft" : "border-line hover:border-muted"} ${disabled ? "pointer-events-none opacity-50" : ""}`}>
      <span className="text-base">Drop a video here, or tap to choose one</span>
      <span className="mt-1 text-sm text-faint">MP4, MOV or WebM, up to 200 MB. Only the first 3 minutes are used.</span>
      <input type="file" accept="video/*" className="hidden" disabled={disabled}
        onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f); e.target.value = ""; }} />
    </label>
  );
}

function Link({ disabled, onUrl }: { disabled: boolean; onUrl: (u: string) => void }) {
  const [url, setUrl] = useState("");
  return (
    <form className="flex flex-col gap-2 sm:flex-row" onSubmit={(e) => { e.preventDefault(); if (url.trim()) onUrl(url.trim()); }}>
      <input value={url} onChange={(e) => setUrl(e.target.value)} disabled={disabled} inputMode="url"
        placeholder="https://www.youtube.com/shorts/..."
        className="min-w-0 flex-1 rounded-xl border border-line bg-bg px-4 py-3 outline-none placeholder:text-faint focus:border-accent" />
      <button disabled={disabled || !url.trim()} className="rounded-xl bg-accent px-5 py-3 font-medium text-white disabled:opacity-40">Identify</button>
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
    <div className="flex flex-col items-center gap-3">
      <video ref={video} playsInline muted className={`aspect-video w-full rounded-xl bg-bg object-cover ${state === "off" ? "hidden" : ""}`} />
      {state === "off" && (
        <button onClick={start} disabled={disabled} className="w-full rounded-xl border border-line py-10 text-muted hover:border-muted disabled:opacity-50">
          Start the camera, then point it at a TV or monitor showing a game
        </button>
      )}
      {state === "live" && <button onClick={record} className="rounded-xl bg-accent px-6 py-3 font-medium text-white">Record {SECONDS} seconds</button>}
      {state === "rec" && <p className="flex items-center gap-2 text-sm"><span className="h-2 w-2 rounded-full bg-bad pulse" />Recording… {left}s</p>}
      {err && <p className="text-sm text-bad">{err}</p>}
    </div>
  );
}

function Pipeline({ run, now, busy, preview }: { run: Run; now: number; busy: boolean; preview: string | null }) {
  const has: Record<string, boolean> = { frames: !!run.frames, screen: !!run.screen, candidates: !!run.candidates, answer: !!run.answer };
  const firstPending = STAGES.find((s) => !has[s.key])?.key;
  const prevAt = (i: number) => (i === 0 ? run.started : run.at[STAGES[i - 1].key] ?? run.started);
  const seconds = (i: number) => (((run.at[STAGES[i].key] ?? now) - prevAt(i)) / 1000).toFixed(1);

  return (
    <section className="mt-8 space-y-4">
      {run.answer && <Verdict a={run.answer} candidates={run.candidates ?? []} />}
      {run.error && <div className="rounded-2xl border border-bad/40 bg-bad/10 p-4 text-sm">{run.error}</div>}

      {STAGES.map((s, i) => {
        const done = has[s.key];
        const active = busy && !run.error && s.key === firstPending;
        return (
          <div key={s.key} className={`rounded-2xl border bg-panel p-4 transition sm:p-5 ${active ? "border-accent/50" : "border-line"} ${!done && !active ? "opacity-50" : ""}`}>
            <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
              <span className="font-mono text-xs text-faint">{s.n}</span>
              <h2 className="font-medium">{s.name}</h2>
              <span className="rounded-full bg-accent-soft px-2 py-0.5 text-xs text-accent">{s.model}</span>
              <span className="ml-auto font-mono text-xs text-muted">
                {done ? `${seconds(i)}s` : active ? <span className="pulse">{seconds(i)}s</span> : "waiting"}
              </span>
            </div>
            <p className="mt-1 text-sm text-faint">{s.what}</p>
            {done && <div className="rise mt-4">{body(s.key, run, preview)}</div>}
          </div>
        );
      })}

      {run.done && (
        <p className="text-center font-mono text-xs text-faint">
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
        {preview && <video src={preview} controls muted playsInline className="max-h-72 rounded-lg" />}
        <div className="grid grid-cols-4 gap-1.5 sm:grid-cols-8">
          {run.frames.map((f, i) => <img key={i} src={f} alt="" className="aspect-square w-full rounded-md object-cover" />)}
        </div>
      </div>
    );
  if (key === "screen" && run.screen)
    return run.screen.found ? (
      <div className="space-y-3">
        <p className="text-sm"><span className="text-ok">Filmed screen found</span> on most frames, so Lens searches the screen and ignores the room around it.</p>
        <div className="grid grid-cols-4 gap-1.5 sm:grid-cols-8">
          {run.screen.crops.map((f, i) => <img key={i} src={f} alt="" className="aspect-video w-full rounded-md object-cover" />)}
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
      <ol className="grid gap-2 sm:grid-cols-2">
        {run.candidates.map((c, i) => (
          <li key={c.id} className={`rounded-xl border p-2.5 ${picked === c.id ? "border-ok bg-ok/10" : "border-line bg-panel-2"}`}>
            <div className="flex items-center gap-2 text-sm">
              <span className="font-mono text-xs text-faint">{LABELS[i]}</span>
              <span className="truncate font-medium">{c.title}</span>
              {c.year && <span className="text-faint">{c.year}</span>}
              {picked === c.id && <span className="ml-auto rounded-full bg-ok/20 px-2 text-xs text-ok">picked</span>}
            </div>
            <div className="mt-1.5 h-1 rounded-full bg-bg"><div className="h-1 rounded-full bg-accent" style={{ width: `${(c.score / max) * 100}%` }} /></div>
            <div className="mt-2 grid grid-cols-3 gap-1">
              {c.shots.map((s) => <img key={s} src={s} alt="" loading="lazy" className="aspect-video w-full rounded object-cover" />)}
            </div>
          </li>
        ))}
      </ol>
    );
  }
  if (key === "answer" && run.answer)
    return (
      <div className="grid gap-3 text-sm sm:grid-cols-2">
        <div className="rounded-xl bg-panel-2 p-3">
          <p className="text-xs text-faint">Step 1 · from the clip alone</p>
          <p className="mt-1 font-medium">{run.answer.own_guess ?? "–"}</p>
          <p className="text-xs text-muted">{run.answer.own_confidence} confidence</p>
        </div>
        <div className="rounded-xl bg-panel-2 p-3">
          <p className="text-xs text-faint">Step 2 · after checking the candidates</p>
          <p className="mt-1 font-medium">{run.answer.game_title ?? "–"}</p>
          <p className="text-xs text-muted">
            {run.answer.pick && run.answer.pick !== "none" ? `Matched candidate ${run.answer.pick}` : "No candidate matched; kept its own answer"}
          </p>
        </div>
      </div>
    );
  return null;
}

function Verdict({ a, candidates }: { a: Answer; candidates: Candidate[] }) {
  if (!a.game_title) return <div className="rounded-2xl border border-line bg-panel p-5 text-muted">Muse Spark couldn&apos;t give an answer{a.error ? `: ${a.error}` : "."}</div>;
  const tone = a.confidence === "high" ? "text-ok" : a.confidence === "medium" ? "text-warn" : "text-bad";
  const shots = a.game?.shots ?? [];
  const fromLens = !!a.game;
  const rank = fromLens ? candidates.findIndex((c) => c.id === a.game!.id) + 1 : 0;
  return (
    <div className="rise overflow-hidden rounded-2xl border border-accent/40 bg-panel">
      <div className="p-5 sm:p-6">
        <p className="text-sm text-muted">This is</p>
        <h2 className="mt-1 text-3xl font-semibold tracking-tight sm:text-4xl">{a.game_title}</h2>
        <p className="mt-2 text-sm">
          <span className={tone}>{a.confidence} confidence</span>
          <span className="text-faint"> · {fromLens ? `confirmed against search result #${rank}` : "Muse Spark's own answer; none of the search results matched"}</span>
        </p>
        {a.evidence && <p className="mt-4 border-l-2 border-accent pl-3 text-sm text-muted">{a.evidence}</p>}
      </div>
      {shots.length > 0 && (
        <div className="grid grid-cols-3 gap-0.5">
          {shots.map((s) => <img key={s} src={s} alt="" className="aspect-video w-full object-cover" />)}
        </div>
      )}
    </div>
  );
}

function About() {
  return (
    <section className="mt-16 grid gap-6 border-t border-line pt-8 text-sm text-muted sm:grid-cols-2">
      <div>
        <h3 className="mb-2 font-medium text-fg">Why</h3>
        <p>
          Gamers share clips constantly, and the caption rarely names the game. Prysm, a game-library app, reads captions and
          names the game in 85 of 104 real shared clips. Lens is for the rest: indie games, filmed TVs, junk titles.
        </p>
      </div>
      <div>
        <h3 className="mb-2 font-medium text-fg">Benchmark: 104 real shared clips</h3>
        <ul className="space-y-1">
          <li>Captions only (Prysm today): <span className="text-fg">85</span></li>
          <li>Muse Spark watching the clip, no search: <span className="text-fg">77</span></li>
          <li>Captions, else Lens + Muse Spark: <span className="text-fg">91</span></li>
          <li>On the 19 clips captions can&apos;t name: <span className="text-fg">1 → 13</span></li>
        </ul>
      </div>
      <p className="text-xs text-faint sm:col-span-2">
        Built during the Meta Global AI Developer Hackathon (Oct 26 to Nov 4, 2026) with SAM 3.1, Muse Spark 1.3 and Meta
        Perception Encoder. Screenshots and game data from IGDB.com. Code and benchmark:{" "}
        <a className="text-accent underline underline-offset-2" href="https://github.com/System625/Prysm-Lens">github.com/System625/Prysm-Lens</a>
      </p>
    </section>
  );
}
