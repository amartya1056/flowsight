"""Fetch real test footage (best effort) and generate floor-plan PNGs.

Order of preference (keep whatever succeeds):
  1. CAVIAR "Shopping Centre in Portugal" corridor clips (real shopping-centre CCTV with shop fronts)
  2. MOT17 / MOT20 raw sequence videos (crowded scenes for bottleneck testing)
  3. anything the user already put in data/videos/

If the network blocks downloads we log it and continue: the simulator guarantees a working demo.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flowsight.config import list_stores, load_store, resolve_path  # noqa: E402
from flowsight.render.annotate import make_floorplan_image  # noqa: E402

VIDEOS = ROOT / "data" / "videos"
RAW = ROOT / "data" / "raw"
CAVIAR = "https://groups.inf.ed.ac.uk/vision/DATASETS/CAVIAR/CAVIARDATA2"
CAVIAR_CLIPS = ["OneShopOneWait1cor", "ThreePastShop1cor", "TwoEnterShop1cor", "EnterExitCrossingPaths1cor"]
CAVIAR_FRONT = ["ThreePastShop1front", "OneShopOneWait1front"]          # synchronised 2nd view (multi-camera)
CAVIAR_GT = {"ThreePastShop1cor": "c3ps1gt.xml", "OneShopOneWait1cor": "cosow1gt.xml", "TwoEnterShop1cor": "c2es1gt.xml",
             "ThreePastShop1front": "f3ps1gt.xml", "OneShopOneWait1front": "fosow1gt.xml"}
MOT = {
    "mot17_09.webm": "https://motchallenge.net/sequenceVideos/MOT17-09-FRCNN-raw.webm",
    "mot20_01.webm": "https://motchallenge.net/sequenceVideos/MOT20-01-raw.webm",
}


def fetch(url: str, dst: Path, timeout: int = 60) -> bool:
    if dst.exists() and dst.stat().st_size > 10_000:
        print(f"  have {dst.name}")
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "flowsight-hackathon/0.1"})
        with urllib.request.urlopen(req, timeout=timeout) as r, open(dst.with_suffix(".part"), "wb") as f:
            shutil.copyfileobj(r, f)
        dst.with_suffix(".part").replace(dst)
        print(f"  downloaded {dst.name} ({dst.stat().st_size / 1e6:.1f} MB)")
        return True
    except Exception as e:
        print(f"  FAILED {url}: {e}")
        return False


def to_mp4(src: Path, dst: Path) -> bool:
    ff = shutil.which("ffmpeg")
    if not ff:
        print("  ffmpeg not found; cannot convert", src.name)
        return False
    if dst.exists() and dst.stat().st_size > 10_000:
        return True
    r = subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-preset", "veryfast", "-crf", "20", "-an", str(dst)], capture_output=True, text=True)
    if r.returncode != 0:
        print("  ffmpeg failed:", r.stderr[-300:])
        return False
    return True


def concat(parts: list[Path], dst: Path) -> bool:
    ff = shutil.which("ffmpeg")
    if not ff or not parts:
        return False
    lst = dst.with_suffix(".txt")
    lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
    r = subprocess.run([ff, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                        str(dst)], capture_output=True, text=True)
    lst.unlink(missing_ok=True)
    return r.returncode == 0


def main() -> int:
    VIDEOS.mkdir(parents=True, exist_ok=True)
    got = []
    print("[1/3] CAVIAR shopping-centre corridor clips")
    parts = []
    for clip in CAVIAR_CLIPS:
        mpg = RAW / f"{clip}.mpg"
        if fetch(f"{CAVIAR}/{clip}/{clip}.mpg", mpg):
            mp4 = VIDEOS / f"caviar_{clip}.mp4"
            if to_mp4(mpg, mp4):
                parts.append(mp4)
                got.append(mp4.name)
    if parts:
        combo = VIDEOS / "caviar_corridor.mp4"
        if concat(parts, combo):
            got.append(combo.name)
            print(f"  concatenated {len(parts)} clips -> {combo.name}")

    print("[1b] CAVIAR front views + ground truth (multi-camera, tracking metrics)")
    for clip in CAVIAR_FRONT:
        mpg = RAW / f"{clip}.mpg"
        if fetch(f"{CAVIAR}/{clip}/{clip}.mpg", mpg):
            to_mp4(mpg, VIDEOS / f"caviar_{clip}.mp4")
    for clip, xml in CAVIAR_GT.items():
        fetch(f"{CAVIAR}/{clip}/{xml}", ROOT / "data" / "gt" / xml)
    pair = {"caviar_pair_cor.mp4": ["ThreePastShop1cor", "OneShopOneWait1cor"],
            "caviar_pair_front.mp4": ["ThreePastShop1front", "OneShopOneWait1front"]}
    for out, clips in pair.items():
        src = [VIDEOS / f"caviar_{c}.mp4" for c in clips]
        if all(p.exists() for p in src) and concat(src, VIDEOS / out):
            got.append(out)

    print("[2/3] MOT sequences")
    for name, url in MOT.items():
        src = RAW / name
        if fetch(url, src, timeout=120):
            mp4 = VIDEOS / name.replace(".webm", ".mp4")
            if to_mp4(src, mp4):
                got.append(mp4.name)

    print("[3/3] user videos in data/videos/")
    for p in sorted(VIDEOS.glob("*.mp4")):
        if p.name not in got:
            got.append(p.name)
            print("  found", p.name)

    print("[floor plans]")
    for sid in list_stores():
        st = load_store(sid)
        out = resolve_path(st.floorplan.image)
        out.parent.mkdir(parents=True, exist_ok=True)
        import cv2

        cv2.imwrite(str(out), make_floorplan_image(st, 50))
        print("  wrote", out.relative_to(ROOT))

    if not got:
        print("\nNO VIDEO available (network blocked?). Running in SIMULATOR MODE.\n"
              "Drop retail CCTV MP4s into data/videos/ and add a camera to config/stores/<store>.yaml to use them.")
    else:
        print("\nvideos available:", ", ".join(got))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
