"""Deploy the Lens service to the Hugging Face Space System625/prysm-lens.

  python service/space/deploy.py            # code only
  python service/space/deploy.py --secrets  # also (re)set MODEL_API_KEY and HF_TOKEN from the environment

The index lives in the private dataset System625/prysm-lens-index (data/upload_index.py); start.sh pulls it at boot.
"""
import os, shutil, sys, tempfile
from huggingface_hub import HfApi

SPACE = "System625/prysm-lens"
ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
FILES = ["service/__init__.py", "service/app.py", "service/search.py", "service/screen.py", "service/verify.py",
         "indexer/__init__.py", "indexer/embed.py"]

api = HfApi()
api.create_repo(SPACE, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
if "--secrets" in sys.argv:
    for k in ("MODEL_API_KEY", "HF_TOKEN"):
        api.add_space_secret(SPACE, k, os.environ[k])
with tempfile.TemporaryDirectory() as d:
    for f in FILES:
        os.makedirs(os.path.dirname(f"{d}/{f}"), exist_ok=True)
        shutil.copy(f"{ROOT}/{f}", f"{d}/{f}")
    for f in ("Dockerfile", "requirements.txt", "start.sh", "README.md"):
        shutil.copy(f"{os.path.dirname(__file__)}/{f}", f"{d}/{f}")
    api.upload_folder(folder_path=d, repo_id=SPACE, repo_type="space", commit_message="Deploy Lens service")
print(f"https://huggingface.co/spaces/{SPACE}")
