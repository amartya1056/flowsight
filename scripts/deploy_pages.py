"""Build and publish the static replay demo to GitHub Pages (gh-pages branch).

    python scripts/deploy_pages.py [--record] [--remote origin]

  --record   re-record the demo runs first (scripts/export_static_demo.py; needs the models and `make data`)

The site is the console built with `vite build --mode static` plus `site/demo-data/`; it is force-pushed as a single
commit to the `gh-pages` branch of the given remote (GitHub Pages serves it at https://<user>.github.io/<repo>/).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sh(cmd, cwd=ROOT):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--remote", default="origin")
    a = ap.parse_args()
    if a.record or not (ROOT / "site" / "demo-data" / "runs.json").exists():
        sh([sys.executable, ROOT / "scripts" / "export_static_demo.py"])
    npm = shutil.which("npm") or shutil.which("npm.cmd") or "npm"
    sh([npm, "run", "build:static"], cwd=ROOT / "web")
    url = subprocess.run(["git", "remote", "get-url", a.remote], cwd=ROOT, capture_output=True, text=True,
                         check=True).stdout.strip()
    with tempfile.TemporaryDirectory() as td:
        site = Path(td) / "site"
        shutil.copytree(ROOT / "web" / "dist-static", site)
        shutil.copytree(ROOT / "site" / "demo-data", site / "demo-data")
        (site / ".nojekyll").write_text("")
        (site / "404.html").write_text((site / "index.html").read_text(encoding="utf-8"), encoding="utf-8")
        sh(["git", "init", "-q", "-b", "gh-pages"], cwd=site)
        sh(["git", "add", "-A"], cwd=site)
        sh(["git", "commit", "-qm", "Deploy FlowSight static replay demo"], cwd=site)
        sh(["git", "push", "-f", url, "gh-pages"], cwd=site)
    print("pushed gh-pages ->", url)


if __name__ == "__main__":
    main()
