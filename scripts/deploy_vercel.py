"""Deploy the replay console to Vercel (production).

    python scripts/deploy_vercel.py [--record]

Builds the console with `vite build --mode vercel` (static replay mode, served from the domain root), adds the
recorded `site/demo-data/`, and runs `vercel deploy --prod` on the folder. Needs `npx vercel login` once.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    npm = shutil.which("npm") or shutil.which("npm.cmd") or "npm"
    npx = shutil.which("npx") or shutil.which("npx.cmd") or "npx"
    if a.record or not (ROOT / "site" / "demo-data" / "runs.json").exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "export_static_demo.py")], check=True)
    subprocess.run([npm, "run", "build:vercel"], cwd=ROOT / "web", check=True)
    out = ROOT / "site" / "vercel"
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(ROOT / "web" / "dist-vercel", out)
    shutil.copytree(ROOT / "site" / "demo-data", out / "demo-data")
    (out / "vercel.json").write_text('{\n  "$schema": "https://openapi.vercel.sh/vercel.json",\n  "framework": null\n}\n')
    subprocess.run([npx, "--yes", "vercel@latest", "deploy", "--prod", "--yes", "--name", "flowsight"], cwd=out, check=True)


if __name__ == "__main__":
    main()
