---
title: Prysm Lens
emoji: 🎮
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Name the video game in any gameplay clip
---

# Prysm Lens: search service

Name the video game in a gameplay clip. Built for the Meta Global AI Developer Hackathon (Oct 26 to Nov 4, 2026).
Code and benchmark: https://github.com/System625/Prysm-Lens

`POST /identify` with a video `file` (or a `url`) streams NDJSON events, one per stage:
16 frames → **SAM 3.1** finds a filmed screen and crops to it → **Meta Perception Encoder** fingerprints the frames →
nearest of 575,973 game screenshots, top 10 games → **Muse Spark 1.3** names the game from the clip first, then checks
the candidates' screenshots.

Screenshots are shown from IGDB's own image links. Game data from IGDB.com.
