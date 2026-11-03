# Prysm Lens

**Name the game in any gameplay clip, from the pixels alone.**

Prysm Lens is a "Shazam for games". Give it a short gameplay video, such as a TikTok, a Reel or a YouTube Short, and it tells you which game it is and shows the evidence. It does not need the caption to name the game.

**Try it: [prysm-lens.vercel.app](https://prysm-lens.vercel.app)** (upload a clip, paste a link, or point your phone at a screen).

It is built for the [Meta Global AI Developer Hackathon](https://cerebralvalley.ai/e/meta-ai-developer-hackathon) (26 Oct – 4 Nov 2026), Indie track, problem statement **Build for Your World**.

---

## The problem

[Prysm](https://www.prysmblast.com) is our game-tracking app. Players share gaming clips into it, and it adds the game to their shelf. Today it works out the game mostly from the **text** around a clip: the caption, the hashtags, the title.

That works when the poster names the game. It fails when they don't, and those clips are the ones where people most want to know "what game is this?". The clips that defeat it are:

- **YouTube titles that say nothing about the game**, such as "Cherry blossom", "Delete" or "Spin It!" on Brawlhalla clips
- **indie horror** captioned "Scary Game" or "Stream"
- **new and long-tail releases** that language models don't know yet
- **sequels and remasters** that look like their predecessors, such as Modern Warfare 4 vs Modern Warfare (2019), or *Silent Hill: Townfall* vs the original

A language model guessing from memory can't fix this; it confidently names the wrong game. Prysm Lens looks at the video itself.

## How it works

Lens finds candidates first, then checks them.

```
clip ──► frames ──► SAM 3.1 ──► Perception Encoder ──► index search ──► Muse Spark ──► answer + evidence
                    (find the    (fingerprint the       (top 10 games     (watches the clip
                    filmed       frames, or the         from 575,973      next to each
                    screen,      screen crop)           game screenshots) candidate's screenshots
                    crop to it)                                           and picks one, or none)
```

1. **Index.** Fingerprint every game's screenshots with Meta's **Perception Encoder** (PE-Core-L14-336, Apache 2.0). It does not depend on what a model happens to remember, so a game released last week is covered as soon as it has screenshots.
2. **Crop.** Many shared clips are a phone filming a monitor or a TV. The room around the screen (desk, LED strips, the player) matches room-like games better than the game being played. **SAM 3.1** tracks "screen" through the clip's frames in one video call, and when it holds a confident screen on at least half of them, Lens fingerprints the crop instead of the room. (We first expected facecams to be the problem; masking faces measured as noise, so we looked at the misses and found filmed screens.)
3. **Find candidates.** A nearest-neighbour search returns the top 10 games.
4. **Check.** **Muse Spark** watches the clip (plus SAM's crops of the screen, when the clip films one), names the game from the clip alone first, then looks at each candidate's screenshots and keeps its own answer unless one of them clearly shows the same game. It cites what it saw.
5. **Show the evidence.** When Muse Spark keeps its own answer, Lens looks that title up in the 91,976-game catalog and shows the game's screenshots closest to the clip, so the answer can be checked by eye. (On the benchmark's 102 such answers, 52 land on the right game, 28 on the wrong game Spark named, where the screenshots show the mismatch, and 22 name no game.)

### Meta models used

| Model | Role |
|---|---|
| Perception Encoder (open weights) | Visual fingerprint for screenshots and clip frames |
| SAM 3.1 | Finds the filmed monitor / TV / phone in a clip and crops the frames to it |
| Muse Spark 1.3 | Watches the clip (video and audio, plus the screen crops), names the game, checks it against the candidates, with evidence |

## Benchmark

There is no public benchmark for naming a video game from a clip. The closest we found is a 2023 paper that classifies 10 games. So we are building one from real clips people shared into Prysm. Each clip is labelled by hand, and no game contributes more than 3 clips.

**Baseline, measured before the hackathon on Prysm's existing text-based pipeline:** 61 of 75 clips right on the first guess, **81%**. The misses are almost all the clips described above, where the caption doesn't name the game. That "hard subset" is what Lens is for, and we will report every result on it separately.

We publish the **clip links and labels only** ([benchmark/clips.tsv](benchmark/clips.tsv)). The videos belong to their creators and are not redistributed here. Links are the posts' own URLs, never the share links that pointed us to them.

**Index alone, first gate (575,973 screenshots of 87,077 games, 104 distinct clips):** the right game is first for 45 clips and in the top 10 for 72; on the hard subset, in the top 10 for 13 of 19. Muse Spark then picks from those ten.

**With the SAM 3.1 screen crop:** first for 47, top 10 for **75** (top 10 with at most 3 clips per game: 54 → 57). It fires on 7 clips, all of them filmed screens, and changes nothing else: Resident Evil 4 and A Way Out go to first place, S.T.A.L.K.E.R. 2 from a miss to third, Call of Duty: Mobile from a miss to tenth. Games with monitors inside the game don't trigger it, because a one-off box isn't enough. One call per clip: the 16 sampled frames go up as a 16-second video, billed at SAM's 50-frame minimum, $0.01 a clip.

**Muse Spark, 5 runs per clip, same 104 clips** ([benchmark/score_combined.py](benchmark/score_combined.py)). A clip counts as right when most of its runs are, and any edition of the right game counts (Days Gone Remastered for Days Gone), for every row:

| | All 104 | Hard 19 | Confidently wrong |
|---|---|---|---|
| Prysm today (captions) | 85 | 1 | 5 |
| Muse Spark, video only | 77 | 12 | 7 |
| Muse Spark + Lens top 10 | 74 | **13** | 15 |
| Muse Spark + Lens (with SAM crop), own answer first | 77 | **13** | 13 |
| **Prysm when confident, else Muse Spark + Lens (with SAM crop), own answer first** | **91** | 11 | 8 |

Lens helps where it was meant to: on clips the caption pipeline misses, Muse Spark names 13 of 19 with the candidates and 12 without. Shown a candidate list straight away, it sometimes trades a right answer of its own for a wrong candidate (15 confidently wrong vs 7). So the shipped prompt has it name the game from the clip first and keep that answer unless a candidate's screenshots clearly show the same game: one call, hard 13 of 19, and only one confident mistake there. Four hard clips are ones Prysm answers confidently and wrongly, so the cascade never reaches them; with no game counted more than 3 times (86 clips), the cascade gets 74 to Prysm's 73. Those, and the 19-clip sample, are the open problems. About $27 for all 1,560 Muse Spark calls, plus $1.09 for SAM on every clip.

Before the SAM crop, own-answer-first scored 76 and the cascade 90 (73 with the cap). The crop's end-to-end gain is small because Muse Spark already knows the big games in those filmed clips from the video; it matters in retrieval, which is what covers the long tail Spark doesn't know.

**Showing Muse Spark the screen crops too (1 Nov).** Until then the crops only went to the search, and Spark judged the room: on a phone filming Horizon Forbidden West it said "gameplay is only seen on a distant room monitor" and named Ghost of Tsushima. Now, when SAM finds a screen, Spark also gets up to 8 crops after the clip. On the 8 filmed-screen clips (the 7 above plus that one, row 124), 5 runs each: single runs right **26 of 35 → 33 of 34** (Assassin's Creed Valhalla 4 → 5, Call of Duty: Mobile 3 → 5, Resident Evil 4 4 → 5, no clip worse), majority-right clips unchanged at 6 of 7. Still wrong: row 98 (the monitor is ~160 px wide) and row 124, which moved from Ghost of Tsushima at low confidence (4 of 5) to Horizon Zero Dawn Remastered, right series but wrong game, in 2 of 5. Clips with no filmed screen get exactly the request they got before, so the table above stands for them.

## What is new in this repo

The hackathon only counts work built during the event. Here is the split:

| Built during the hackathon (this repo) | Existed before |
|---|---|
| Screenshot index and the indexing pipeline | The Prysm app (iOS / Android) |
| SAM 3.1 screen crop | Prysm's text-based share pipeline (the 81% baseline) |
| Search service (Hugging Face ZeroGPU Space) | The game catalog Lens indexes against |
| Muse Spark verification step and the catalog lookup | |
| The benchmark harness and results | |
| The web demo | |

Prysm's backend stays private and Lens is not wired into it yet. Everything built for Lens is in this repo.

## Repo layout

```
Prysm-Lens/
  indexer/       Python, run locally: fingerprints game screenshots into the index
  service/       the pipeline (frames, SAM 3.1, search, Muse Spark) as a ZeroGPU Gradio Space, or a FastAPI NDJSON service
  web/           Next.js demo site (Vercel): paste a link, upload a clip, or use your camera. YouTube blocks servers
                 from downloading, so a YouTube link is recorded in the visitor's browser (or screen-recorded on a phone)
  benchmark/     clip links + labels, the scoring script, results
```

The web demo shows every stage, not only the answer: the frames it pulled out, the screen SAM 3.1 found and the crop Lens searched with, the top candidates with their matching screenshots, and Muse Spark's pick with its reasoning. On a phone, the camera mode lets you point it at a TV.

## Status

- [x] Benchmark clips collected and hand-labelled
- [x] Baseline measured on the existing pipeline (81%)
- [x] Muse Spark on the video alone (no index), same clips
- [x] Perception Encoder index plus a recall@10 check (72 of 104 in the top 10)
- [x] SAM 3.1 screen crop (top 10: 72 → 75 of 104)
- [x] Full pipeline vs baseline, on all clips and on the hard subset
- [x] Search service deployed (Hugging Face ZeroGPU Space)
- [x] Web demo live: [prysm-lens.vercel.app](https://prysm-lens.vercel.app)
- [ ] Demo video

Not built, and next: Muse Voice Transcribe for streamers who say the game's name, wiring Lens into Prysm's share pipeline, adding confirmed clips to the index, and testing Muse Glimmer's vision encoder as the fingerprinter.

## Team

Lens is built by **Tunde-Ajayi Olamiposi**: the recognition pipeline, the search service, the web demo and the benchmark.

The Prysm app it plugs into is the existing work of the wider Prysm team.

## Licence

The code in this repo is open source under the [MIT licence](LICENSE).

The licence covers the code and our benchmark labels only. It does not cover the Prysm name and logo, the game screenshots the indexer downloads (they belong to the games' publishers, served via IGDB), or the clips the benchmark links to (they belong to their creators). The screenshot index itself is not published.
