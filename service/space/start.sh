#!/bin/sh
# Pull the index + catalog from the private dataset (HF_TOKEN secret), then serve.
set -e
python -c "
import os
from huggingface_hub import snapshot_download
snapshot_download('System625/prysm-lens-index', repo_type='dataset', local_dir=os.environ['LENS_DATA'], token=os.environ['HF_TOKEN'])
"
exec python -m uvicorn service.app:app --host 0.0.0.0 --port 7860
