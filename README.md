# Prysm Lens

**Name the game in any gameplay clip, from the pixels alone.**

Prysm Lens is a "Shazam for games". Give it a short gameplay video, such as a TikTok, a Reel or a YouTube Short, and it tells you which game it is and shows the evidence. It does not need the caption to name the game.

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
                    (mask       (fingerprint the       (top 10 games     (watches the clip
                    facecams,    cleaned frames)        from ~450k        next to each
                    subtitles,                          game screenshots) candidate's screenshots
                    overlays)                                             and picks one, or none)
```

1. **Index.** Fingerprint every game's screenshots with Meta's **Perception Encoder** (PE-Core-L14-336, Apache 2.0). It does not depend on what a model happens to remember, so a game released last week is covered as soon as it has screenshots.
2. **Clean.** Streamers' facecams, subtitles and on-screen UI overlays get in the way of the gameplay. **SAM 3.1** masks them out of the frames before they are fingerprinted.
3. **Find candidates.** A nearest-neighbour search returns the top 10 games.
4. **Check.** **Muse Spark** watches the clip alongside each candidate's screenshots and picks the right game, citing what it saw. It can also decide that none of them match.
5. **Listen.** **Muse Voice Transcribe**, biased toward the candidate titles, catches a streamer saying the game's name out loud.
6. **Learn.** Every clip a user confirms adds its frames to the index.

### Meta models used

| Model | Role |
|---|---|
| Perception Encoder (open weights) | Visual fingerprint for screenshots and clip frames |
| SAM 3.1 | Masks facecams, subtitles and overlays out of clip frames |
| Muse Spark 1.3 | Watches the clip next to the candidates and picks one, with evidence |
| Muse Voice Transcribe | Catches spoken game names, with keyword biasing |
| Muse Glimmer (to be tested) | Its vision encoder as an alternative fingerprinter |

## Benchmark

There is no public benchmark for naming a video game from a clip. The closest we found is a 2023 paper that classifies 10 games. So we are building one from real clips people shared into Prysm. Each clip is labelled by hand, and no game contributes more than 3 clips.

**Baseline, measured before the hackathon on Prysm's existing text-based pipeline:** 61 of 75 clips right on the first guess, **81%**. The misses are almost all the clips described above, where the caption doesn't name the game. That "hard subset" is what Lens is for, and we will report every result on it separately.

We publish the **clip links and labels only** ([benchmark/clips.tsv](benchmark/clips.tsv)). The videos belong to their creators and are not redistributed here. Links are the posts' own URLs, never the share links that pointed us to them.

**Index alone, first gate (575,973 screenshots of 87,077 games, 104 distinct clips):** the right game is first for 45 clips and in the top 10 for 72; on the hard subset, in the top 10 for 13 of 19. Muse Spark then picks from those ten.

**Muse Spark, 5 runs per clip, same 104 clips** ([benchmark/score_combined.py](benchmark/score_combined.py)). A clip counts as right when most of its runs are, and any edition of the right game counts (Days Gone Remastered for Days Gone), for every row:

| | All 104 | Hard 19 | Confidently wrong |
|---|---|---|---|
| Prysm today (captions) | 85 | 1 | 5 |
| Muse Spark, video only | 77 | 12 | 7 |
| Muse Spark + Lens top 10 | 74 | **13** | 15 |
| Muse Spark + Lens, own answer first | 76 | **13** | 13 |
| **Prysm when confident, else Muse Spark + Lens, own answer first** | **90** | 11 | 8 |

Lens helps where it was meant to: on clips the caption pipeline misses, Muse Spark names 13 of 19 with the candidates and 12 without. Shown a candidate list straight away, it sometimes trades a right answer of its own for a wrong candidate (15 confidently wrong vs 7). So the shipped prompt has it name the game from the clip first and keep that answer unless a candidate's screenshots clearly show the same game: one call, hard 13 of 19, and only one confident mistake there. Four hard clips are ones Prysm answers confidently and wrongly, so the cascade never reaches them; with no game counted more than 3 times (86 clips), the cascade ties Prysm at 73. Those, and the 19-clip sample, are the open problems. About $27 for all 1,560 calls.

## What is new in this repo

The hackathon only counts work built during the event. Here is the split:

| Built during the hackathon (this repo) | Existed before |
|---|---|
| Screenshot index and the indexing pipeline | The Prysm app (iOS / Android) |
| SAM 3.1 frame cleaning | Prysm's text-based share pipeline (the 81% baseline) |
| Search service | The game catalog Lens indexes against |
| Muse Spark verification step | |
| Voice step | |
| The benchmark harness and results | |
| The web demo | |
| The hook that lets Prysm call Lens | |

Prysm's backend stays private. Everything Lens adds, including the code that connects it to Prysm, is in this repo.

## Repo layout

```
Prysm-Lens/
  indexer/       Python, run locally: fingerprints game screenshots into the index
  service/       Modal app: the search API, plus the SAM 3.1 and Muse Spark calls
  web/           Next.js demo site (Vercel): paste a link, upload a clip, or use your camera
  benchmark/     clip links + labels, the scoring script, results
  integrations/  the call Prysm's share pipeline makes to Lens
```

The web demo shows every stage, not only the answer: the frames it pulled out, the frames after SAM 3.1 masks the overlays, the top candidates with their matching screenshots, and Muse Spark's pick with its reasoning. On a phone, the camera mode lets you point it at a TV.

## Status

- [x] Benchmark clips collected and hand-labelled
- [x] Baseline measured on the existing pipeline (81%)
- [x] Muse Spark on the video alone (no index), same clips
- [x] Perception Encoder index plus a recall@10 check (72 of 104 in the top 10)
- [ ] SAM 3.1 cleaning
- [ ] Full pipeline vs baseline, on all clips and on the hard subset
- [ ] Search service deployed
- [ ] Web demo live
- [ ] Prysm integration and demo

## Team

Lens is built by **Tunde-Ajayi Olamiposi**: the recognition pipeline, the search service, the web demo, the benchmark and the Prysm integration.

The Prysm app it plugs into is the existing work of the wider Prysm team.
