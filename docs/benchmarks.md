# FlowSight benchmarks

Host: Intel64 Family 6 Model 191 Stepping 2, GenuineIntel · Windows 10 · GPU: NVIDIA GeForce RTX 4060 Laptop GPU · torch 2.14.1+cu126. Analytics target rate: 10 FPS per camera.

## Detector + ByteTrack — CAVIAR corridor (384×288, sparse)

| Model | imgsz | Device | FP16 | FPS | ms/frame | people/frame |
|---|---|---|---|---|---|---|
| yolo11n.pt | 640 | cuda:0 | True | 84.0 | 11.9 | 1.9 |
| yolo11n.pt | 960 | cuda:0 | True | 79.2 | 12.6 | 2.0 |
| yolo11s.pt | 640 | cuda:0 | True | 79.4 | 12.6 | 2.9 |
| yolo11s.pt | 960 | cuda:0 | True | 73.5 | 13.6 | 2.6 |

## Detector + ByteTrack — MOT20-01 (1920×1080, dense crowd)

| Model | imgsz | Device | FP16 | FPS | ms/frame | people/frame |
|---|---|---|---|---|---|---|
| yolo11n.pt | 640 | cuda:0 | True | 59.8 | 16.7 | 9.7 |
| yolo11n.pt | 960 | cuda:0 | True | 72.7 | 13.8 | 16.3 |
| yolo11s.pt | 640 | cuda:0 | True | 70.7 | 14.1 | 10.9 |
| yolo11s.pt | 960 | cuda:0 | True | 66.7 | 15.0 | 16.5 |

## Analytics only (TrackStore + all analytics + action engine, no detector)

| People in frame | Frames/s | ms/frame | Cameras per core @10 FPS |
|---|---|---|---|
| 5 | 2544 | 0.39 | 254 |
| 20 | 1363 | 0.73 | 136 |
| 50 | 1049 | 0.95 | 105 |
| 100 | 658 | 1.52 | 66 |
| 200 | 394 | 2.54 | 39 |

## Cameras per edge box (estimate)

* Detector (yolo11s.pt @ 960, cuda:0): 73 FPS → **~7 cameras** at 10 FPS each (sequential, single stream; batching multiple cameras raises this further).
* Analytics (~20 people/frame): 1363 FPS on one CPU core → ~136 cameras per core.
* The detector is the bottleneck: one edge box with this GPU ≈ **7 cameras** at 10 FPS analytics. Running at 5 FPS (enough for dwell/flow) doubles that.

Analytics cost is independent of image resolution — it only sees floor-metre points.

Notes: per-call latency (~12 ms) dominates at these resolutions, so n vs s and 640 vs 960 differ little and
the GPU is under-utilised; batching frames from several cameras per call is the obvious next step. The full
`flowsight process` pipeline on the CAVIAR clip (detector + tracker + analytics + head blur + rendering + MP4
encode) measured 46 FPS. ONNX export is benchmarked in the CPU section below (OpenVINO/TensorRT not tested).

<!-- cpu-onnx:start -->
## CPU-only edge box: PyTorch vs ONNX Runtime (detector only, imgsz 640, CAVIAR)

| Model | Runtime | FPS | ms/frame | people/frame | same #detections as PyTorch |
|---|---|---|---|---|---|
| yolo11n | pytorch | 33.7 | 29.7 | 0.64 | 100% |
| yolo11n | onnx | 34.9 | 28.6 | 0.67 | 90% |
| yolo11s | pytorch | 15.2 | 65.9 | 1.45 | 100% |
| yolo11s | onnx | 16.4 | 61.1 | 1.45 | 85% |

ONNX Runtime speed-up over PyTorch on CPU — yolo11n: ×1.04, yolo11s: ×1.08. At 5 FPS analytics (enough for dwell and flow) this CPU (all cores) with yolo11n ONNX serves ~7 cameras' worth of detection. Export: `python scripts/export_onnx.py`; use via `runtime.model_cpu: yolo11n_640.onnx`.

**End to end on CPU only:** `flowsight process` on the CAVIAR clip with `yolo11n_640.onnx` (detector + ByteTrack + foot
points + homography + analytics + action engine + head blur) runs at **24.6 FPS on the i7-13700HX with no GPU**,
2.5× the 10 FPS analytics rate. A GPU-less edge box can therefore serve ~2 cameras at 10 FPS, or ~5 at 5 FPS.
<!-- cpu-onnx:end -->
