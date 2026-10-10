"""Deploy the full FlowSight backend (+ console) to a Hugging Face Docker Space.

    python scripts/deploy_hf.py [--space <user>/flowsight]

Needs a Hugging Face login with write access (`hf auth login`). Uploads the tracked repository files (minus docs
screenshots, which the Space does not need), the Space Dockerfile and a Space README with the required YAML header.
The Space builds itself on Hugging Face (CPU torch, YOLO11n, CAVIAR clips) and serves the API + console on port 7860.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_PREFIXES = ("docs/screenshots/", "runs/eval/", "deploy/")

SPACE_README = """---
title: FlowSight
emoji: 🛒
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: Retail dwell-time & bottleneck intelligence from CCTV
---

# FlowSight: full backend

This Space runs the complete FlowSight backend: FastAPI, YOLO11n + ByteTrack on CPU, floor-plan analytics, action
engine, simulator. It also serves the Store Ops Console. On start it launches a rush-hour simulation and the real
CAVIAR CCTV clip; use the top bar to start any other scenario, camera or multi-camera fusion.

* Source code and full documentation: https://github.com/amartya1056/flowsight
* API docs: `/docs`
* Free CPU hardware: video runs process at a few frames per second; the simulator runs in real time or faster.
"""


def main():
    from huggingface_hub import HfApi

    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default=None)
    a = ap.parse_args()
    api = HfApi()
    user = api.whoami()["name"]
    space = a.space or f"{user}/flowsight"
    api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)

    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        for f in files:
            if not f or f.startswith(SKIP_PREFIXES) or not (ROOT / f).is_file():
                continue
            (out / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / f, out / f)
        shutil.copy2(ROOT / "deploy" / "huggingface" / "Dockerfile", out / "Dockerfile")
        (out / "README.md").write_text(SPACE_README, encoding="utf-8")
        (out / ".dockerignore").write_text("web/node_modules\nweb/dist\nruns/*\nmodels/\ndata/videos/*\ndata/raw/\n")
        api.upload_folder(folder_path=str(out), repo_id=space, repo_type="space",
                          commit_message="Deploy FlowSight backend", delete_patterns=["*"])
    host = space.replace("/", "-").lower().replace("_", "-")
    print(f"Space: https://huggingface.co/spaces/{space}")
    print(f"App:   https://{host}.hf.space")


if __name__ == "__main__":
    main()
