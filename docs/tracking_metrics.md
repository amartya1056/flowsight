# Tracking metrics on CAVIAR ground truth

Detector `yolo11s.pt` @ imgsz 960, conf 0.35, Ultralytics ByteTrack; every frame (25 FPS, 384×288), IoU ≥ 0.5 matching (`motmetrics`). *FlowSight* = the same boxes with FlowSight's stable IDs (floor projection + EMA + ID stitching).

| Sequence | IDs | MOTA | IDF1 | IDP | IDR | ID switches | Recall | Precision | MT / ML | Stitches |
|---|---|---|---|---|---|---|---|---|---|---|
| ThreePastShop1cor | ByteTrack | 0.760 | 0.685 | 0.755 | 0.627 | 14 | 0.796 | 0.959 | 5 / 1 of 14 | – |
| ThreePastShop1cor | FlowSight | 0.761 | 0.746 | 0.823 | 0.683 | 6 | 0.796 | 0.959 | 5 / 1 of 14 | 11 |
| OneShopOneWait1cor | ByteTrack | 0.613 | 0.746 | 0.674 | 0.836 | 6 | 0.927 | 0.747 | 7 / 0 of 7 | – |
| OneShopOneWait1cor | FlowSight | 0.613 | 0.746 | 0.674 | 0.836 | 6 | 0.927 | 0.747 | 7 / 0 of 7 | 3 |
| TwoEnterShop1cor | ByteTrack | 0.727 | 0.572 | 0.593 | 0.553 | 19 | 0.832 | 0.891 | 9 / 1 of 12 | – |
| TwoEnterShop1cor | FlowSight | 0.728 | 0.605 | 0.626 | 0.584 | 15 | 0.832 | 0.891 | 9 / 1 of 12 | 11 |

| Mean | MOTA | IDF1 | ID switches (total) |
|---|---|---|---|
| ByteTrack | 0.700 | 0.668 | 39 |
| FlowSight | 0.700 | 0.699 | 27 |

Notes: CAVIAR annotates people as small as ~10 px wide at the far end of the corridor (and partially visible people at the frame edge); these dominate the misses. MOTA/precision/recall are identical for both rows because the boxes are the same — only the identities differ, so IDF1 and ID switches measure the stitching layer. FlowSight runs analytics at 10 FPS in production; this evaluation uses every frame.
