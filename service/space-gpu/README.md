---
title: Prysm Lens
emoji: 🎮
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Name the video game in any gameplay clip
---

# Prysm Lens (GPU)

Name the video game in a gameplay clip. Built for the Meta Global AI Developer Hackathon (Oct 26 to Nov 4, 2026).
Code and benchmark: https://github.com/System625/Prysm-Lens

16 frames → **SAM 3.1** finds a filmed screen and crops to it → **Meta Perception Encoder** fingerprints the frames on a
ZeroGPU slot → nearest of 575,973 game screenshots, top 10 games → **Muse Spark 1.3** names the game from the clip first,
then checks the candidates' screenshots.

API: `/identify(video, url)` returns the list of stage events. Screenshots are shown from IGDB's own image links. Game data from IGDB.com.
