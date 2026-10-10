# FlowSight — analytics accuracy vs simulator ground truth

30 runs = 6 scenarios × 5 seeds × 30 sim-minutes (10 Hz). Measured pipeline sees σ=0.15 m noise, 2% dropouts and random ID breaks; it goes through the real TrackStore (EMA + ID stitching). Ground truth = the same metric definitions on perfect data. Wall time 3.6 min.

| Scenario | Visit P | Visit R | Visit F1 | Dwell MAE (s) | Engaged MAE (s) | BN recall | BN recall ≥60 s | BN recall 1:1 | BN precision | BN latency (s) | Zone class. acc | Queue wait MAE (s) | Track frag. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.979 | 0.952 | 0.966 | 0.21 | 2.08 | 0.44 | – | 0.44 | 0.80 | 0.4 | 100% | 7.9 | 1.08 |
| rush_hour | 0.921 | 0.935 | 0.928 | 0.32 | 2.09 | 0.82 | 1.00 | 0.81 | 0.94 | 4.2 | 100% | 10.8 | 1.08 |
| blocked_aisle | 0.935 | 0.938 | 0.936 | 0.28 | 2.25 | 0.75 | 1.00 | 0.75 | 0.99 | 0.8 | 100% | 12.9 | 1.04 |
| ignored_promo | 0.983 | 0.956 | 0.969 | 0.19 | 2.05 | 0.72 | – | 0.72 | 0.50 | 1.7 | 100% | 8.7 | 1.06 |
| dead_aisle | 0.984 | 0.956 | 0.970 | 0.22 | 1.95 | 0.60 | 1.00 | 0.60 | 0.60 | 0.0 | 100% | 8.6 | 1.08 |
| with_staff | 0.958 | 0.953 | 0.956 | 0.22 | 1.98 | 0.53 | 1.00 | 0.53 | 0.78 | -1.8 | 100% | 7.2 | 1.08 |
| **all** | 0.960 | 0.948 | **0.954** | **0.24** | 2.07 | 0.64 | 1.00 | 0.64 | 0.78 | 0.9 | 100% | 9.3 | 1.07 |

## Targets

| Target | Measured | Pass |
|---|---|---|
| Dwell MAE < 1.5 s | 0.24 s | ✅ |
| Visit F1 > 0.9 | 0.954 | ✅ |
| Bottleneck recall ≥ 0.9 on rush_hour (all episodes, any-overlap) | 0.82 | ❌ |
| Bottleneck recall ≥ 0.9 on rush_hour (sustained ≥ 60 s) | 1.00 | ✅ |
| ignored_promo: z_promo1 classified IGNORED | 100% of seeds | ✅ |
| dead_aisle: z_aisle7 classified UNVISITED | 100% of seeds | ✅ |

## Staff filtering (with_staff scenario)

2 simulated employees restock zones for the whole run (45 min). Ground truth contains shoppers only.

| Seed | Staff tracks flagged | Staff visits purged | Staff visits left in analytics | Visit F1 |
|---|---|---|---|---|
| 0 | 2 | 108 | 1 | 0.969 |
| 1 | 2 | 124 | 0 | 0.964 |
| 2 | 3 | 150 | 0 | 0.969 |
| 3 | 2 | 71 | 127 | 0.919 |
| 4 | 4 | 147 | 0 | 0.957 |

## Per-run zone status (measured)

| Scenario | Seed | z_aisle1 | z_aisle2 | z_aisle3 | z_aisle5 | z_aisle6 | z_aisle7 | z_electro | z_pharmacy | z_promo1 |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0 | NORMAL | NORMAL | HOT | NORMAL | NORMAL | NORMAL | NORMAL | NORMAL | NORMAL |
| baseline | 1 | NORMAL | NORMAL | HOT | NORMAL | NORMAL | NORMAL | NORMAL | NORMAL | NORMAL |
| baseline | 2 | NORMAL | NORMAL | HOT | NORMAL | NORMAL | NORMAL | NORMAL | NORMAL | NORMAL |
| baseline | 3 | NORMAL | NORMAL | HOT | NORMAL | NORMAL | NORMAL | NORMAL | NORMAL | NORMAL |
| baseline | 4 | NORMAL | NORMAL | HOT | NORMAL | NORMAL | NORMAL | NORMAL | NORMAL | NORMAL |
| rush_hour | 0 | NORMAL | NORMAL | HOT | NORMAL | NORMAL | NORMAL | NORMAL | NORMAL | NORMAL |
| rush_hour | 1 | NORMAL | NORMAL | HOT | NORMAL | NORMAL | NORMAL | NORMAL | NORMAL | NORMAL |
| rush_hour | 2 | NORMAL | NORMAL | HOT | NORMAL | NORMAL | NORMAL | NORMAL | NORMAL | NORMAL |
| rush_hour | 3 | NORMAL | NORMAL | HOT | NORMAL | NORMAL | NORMAL | NORMAL | NORMAL | NORMAL |
| rush_hour | 4 | NORMAL | NORMAL | NORMAL | HOT | NORMAL | NORMAL | NORMAL | NORMAL | NORMAL |
| blocked_aisle | 0 | NORMAL | NORMAL | NORMAL | HOT | NORMAL | NORMAL | NORMAL | NORMAL | NORMAL |
| blocked_aisle | 1 | NORMAL | NORMAL | HOT | NORMAL | NORMAL | NORMAL | NORMAL | NORMAL | NORMAL |
| blocked_aisle | 2 | NORMAL | NORMAL | HOT | NORMAL | NORMAL | NORMAL | NORMAL | NORMAL | NORMAL |
| blocked_aisle | 3 | NORMAL | NORMAL | HOT | NORMAL | NORMAL | NORMAL | NORMAL | NORMAL | NORMAL |
| blocked_aisle | 4 | NORMAL | NORMAL | HOT | NORMAL | NORMAL | NORMAL | NORMAL | NORMAL | NORMAL |
| ignored_promo | 0 | NORMAL | NORMAL | HOT | NORMAL | NORMAL | NORMAL | NORMAL | NORMAL | IGNORED |
| ignored_promo | 1 | NORMAL | NORMAL | HOT | NORMAL | NORMAL | NORMAL | NORMAL | NORMAL | IGNORED |
| ignored_promo | 2 | NORMAL | NORMAL | HOT | NORMAL | NORMAL | NORMAL | NORMAL | NORMAL | IGNORED |
| ignored_promo | 3 | NORMAL | NORMAL | HOT | NORMAL | NORMAL | NORMAL | NORMAL | NORMAL | IGNORED |
| ignored_promo | 4 | NORMAL | NORMAL | HOT | NORMAL | NORMAL | NORMAL | NORMAL | NORMAL | IGNORED |
| dead_aisle | 0 | NORMAL | NORMAL | HOT | NORMAL | NORMAL | UNVISITED | NORMAL | NORMAL | NORMAL |
| dead_aisle | 1 | NORMAL | HOT | NORMAL | NORMAL | NORMAL | UNVISITED | NORMAL | NORMAL | NORMAL |
| dead_aisle | 2 | NORMAL | NORMAL | HOT | NORMAL | NORMAL | UNVISITED | NORMAL | NORMAL | NORMAL |
| dead_aisle | 3 | NORMAL | NORMAL | HOT | NORMAL | NORMAL | UNVISITED | NORMAL | NORMAL | NORMAL |
| dead_aisle | 4 | NORMAL | NORMAL | HOT | NORMAL | NORMAL | UNVISITED | NORMAL | NORMAL | NORMAL |
| with_staff | 0 | NORMAL | NORMAL | HOT | NORMAL | NORMAL | NORMAL | NORMAL | NORMAL | NORMAL |
| with_staff | 1 | NORMAL | NORMAL | HOT | NORMAL | NORMAL | NORMAL | NORMAL | NORMAL | NORMAL |
| with_staff | 2 | NORMAL | NORMAL | HOT | NORMAL | NORMAL | NORMAL | NORMAL | NORMAL | NORMAL |
| with_staff | 3 | NORMAL | NORMAL | HOT | NORMAL | NORMAL | NORMAL | NORMAL | NORMAL | NORMAL |
| with_staff | 4 | NORMAL | NORMAL | HOT | NORMAL | NORMAL | NORMAL | NORMAL | NORMAL | NORMAL |

## Notes

* **Visit matching**: a measured visit matches a ground-truth visit when it is in the same zone, its track maps (majority vote over observations) to the same true shopper and the intervals overlap.
* **Bottleneck matching**: same zone and overlapping interval (±10 s), *any-overlap* (a ground-truth jam counts as detected if any measured episode overlaps it, even when the two sides split it differently). `BN recall 1:1` is the stricter one-to-one matching. `≥60 s` = recall on sustained congestion only. Latency = measured confirmation time − ground-truth confirmation time (positive = later).
* **Why short episodes are missed**: ground truth flags every ≥20 s window run at LoS D or speed < 50% free-flow, so 20–40 s aisle crowds sitting exactly on the threshold are coin-flips once 0.15 m noise is added. All sustained jams (queue spill-back, blocked walkway) are detected with a few seconds' latency.
* **Queue wait**: per 5-minute block, mean Little's-law prediction W = L/λ vs the mean true time in the checkout zone of shoppers who entered in that block.
* **Track fragmentation** = stable tracks created / true shoppers (1.0 = every ID break was stitched).
