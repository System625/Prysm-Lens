"""Deploy the Lens service to Hugging Face.

  python service/space/deploy.py [--secrets]         # Docker CPU Space System625/prysm-lens (NDJSON /identify)
  python service/space/deploy.py --gpu [--secrets]   # ZeroGPU Gradio Space System625/prysm-lens-gpu (/identify API)
  python service/space/deploy.py --cpu [--secrets]   # the same Gradio Space on free cpu-basic hardware (from 11 Nov 2026)

--secrets (re)sets MODEL_API_KEY, HF_TOKEN and LENS_SECRET (the key prysmblast.com signs run tickets with) from the
environment. The index lives in the private dataset
System625/prysm-lens-index (data/upload_index.py); both Spaces pull it at boot. ZeroGPU Spaces need PRO.
"""
import os, shutil, sys, tempfile
from huggingface_hub import HfApi

CPU = "--cpu" in sys.argv
GPU = "--gpu" in sys.argv or CPU  # the Gradio Space; CPU only changes its hardware
HARDWARE = "cpu-basic" if CPU else "zero-a10g"
SPACE = "System625/prysm-lens-gpu" if GPU else "System625/prysm-lens"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
FILES = ["service/__init__.py", "service/app.py", "service/search.py", "service/screen.py", "service/verify.py", "service/adult-ids.json", "service/webcam-prompts.json",
         "indexer/__init__.py", "indexer/embed.py"] + (["service/gradio_app.py"] if GPU else [])
EXTRA_DIR = os.path.join(HERE, "..", "space-gpu") if GPU else HERE
EXTRA = ["app.py", "requirements.txt", "packages.txt", "README.md"] if GPU else ["Dockerfile", "requirements.txt", "start.sh", "README.md"]

api = HfApi()
api.create_repo(SPACE, repo_type="space", space_sdk="gradio" if GPU else "docker", private=False, exist_ok=True)
if "--secrets" in sys.argv:
    for k in ("MODEL_API_KEY", "HF_TOKEN", "LENS_SECRET"):
        api.add_space_secret(SPACE, k, os.environ[k])
with tempfile.TemporaryDirectory() as d:
    for f in FILES:
        os.makedirs(os.path.dirname(f"{d}/{f}"), exist_ok=True)
        shutil.copy(f"{ROOT}/{f}", f"{d}/{f}")
    for f in EXTRA:
        shutil.copy(f"{EXTRA_DIR}/{f}", f"{d}/{f}")
    api.upload_folder(folder_path=d, repo_id=SPACE, repo_type="space", commit_message="Deploy Lens service")
if GPU and api.get_space_runtime(SPACE).requested_hardware != HARDWARE:
    api.request_space_hardware(SPACE, HARDWARE)
print(f"https://huggingface.co/spaces/{SPACE}")
