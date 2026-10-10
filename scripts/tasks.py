"""Cross-platform task runner (the Makefile delegates here so Windows without `make` works too).

    python scripts/tasks.py setup|data|process|sim|eval|test|api|web|build|demo|bench
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"


def venv_dir() -> Path:
    if os.environ.get("FLOWSIGHT_VENV"):
        return Path(os.environ["FLOWSIGHT_VENV"])
    # keep multi-GB venvs out of cloud-synced folders (OneDrive/Dropbox), they break installs
    if any(s in str(ROOT) for s in ("OneDrive", "Dropbox", "iCloud")):
        return Path.home() / ".venvs" / "flowsight"
    return ROOT / ".venv"


def vpy() -> str:
    v = venv_dir()
    p = v / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return str(p) if p.exists() else sys.executable


def sh(cmd, cwd=ROOT, env=None, check=True):
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run([str(c) for c in cmd], cwd=cwd, env={**os.environ, **(env or {})}, check=check)


def npm() -> str:
    return shutil.which("npm") or shutil.which("npm.cmd") or "npm"


def setup():
    v = venv_dir()
    uv = shutil.which("uv")
    if not (v / ("Scripts" if os.name == "nt" else "bin")).exists():
        sh([uv, "venv", "--python", "3.11", v] if uv else [sys.executable, "-m", "venv", v])
    py = vpy()
    pip = [uv, "pip", "install", "--python", py] if uv else [py, "-m", "pip", "install"]
    has_gpu = shutil.which("nvidia-smi") is not None
    torch_idx = "https://download.pytorch.org/whl/cu126" if has_gpu else "https://download.pytorch.org/whl/cpu"
    sh(pip + ["torch", "torchvision", "--index-url", torch_idx])
    sh(pip + ["-r", ROOT / "requirements.txt"])
    sh([py, "-c", "from flowsight.config import load_config; from flowsight.perception.detector_tracker import "
                  "ensure_weights, select_device; c=load_config().runtime; ensure_weights(c.model_gpu, c.models_dir); "
                  "ensure_weights(c.model_cpu, c.models_dir); print('device:', select_device(c.device))"])
    sh([npm(), "install"], cwd=WEB)
    print(f"\nsetup OK. venv: {v}")


def data():
    sh([vpy(), ROOT / "scripts" / "download_data.py"])


def process(store="caviar_corridor", camera="cam1", *extra):
    sh([vpy(), "-m", "flowsight.cli", "process", "--store", store, "--camera", camera, *extra])


def sim(scenario="rush_hour", *extra):
    sh([vpy(), "-m", "flowsight.cli", "sim", "--scenario", scenario, *extra])


def eval_(*extra):
    sh([vpy(), "-m", "flowsight.cli", "eval", *extra])


def test(*extra):
    sh([vpy(), "-m", "pytest", "-q", *extra])


def api():
    sh([vpy(), "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000"])


def web():
    sh([npm(), "run", "dev"], cwd=WEB)


def build():
    sh([npm(), "run", "build"], cwd=WEB)


def demo():
    if not (WEB / "dist" / "index.html").exists():
        build()
    print("\nFlowSight demo -> http://127.0.0.1:8000  (Ctrl+C to stop)\n", flush=True)
    sh([vpy(), "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000"],
       env={"FLOWSIGHT_DEMO": "1", "HF_HUB_OFFLINE": "1", "YOLO_OFFLINE": "1"})


def bench(*extra):
    sh([vpy(), ROOT / "scripts" / "bench.py", *extra])


TASKS = {"setup": setup, "data": data, "process": process, "sim": sim, "eval": eval_, "test": test, "api": api,
         "web": web, "build": build, "demo": demo, "bench": bench}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in TASKS:
        print("usage: python scripts/tasks.py " + "|".join(TASKS))
        sys.exit(2)
    TASKS[sys.argv[1]](*sys.argv[2:])
