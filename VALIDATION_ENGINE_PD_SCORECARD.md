# PD-Focused Validation Scorecard (Engine)

## Snapshot and Version
- Package folders covered: `/Users/stevenchung/Desktop/P12B_File/Macro risk v3` and `/Users/stevenchung/Desktop/P12B_File/Macro risk v4`
- Validation contract version: `v4.1`
- Objective mode: `pd_levels_primary`
- Incumbent / challenger: `champion_a` / `champion_b`
- Snapshot timestamp (local file mtime): `2026-02-17T19:35:55`
- `validation_summary.json` SHA256: `e403740ef035ea0c8363dc94ea57d51ac07b972143c31a00388c7b30f210a670`
- `backtest_metrics.csv` SHA256: `26854caeda6356cb54df7f49e9bf7d44de41f8f341c3285c72d91ee21c02b1f7`
- Note: v3 and v4 validation artifacts are identical hashes at this snapshot.

## Primary PD Scope
- Primary gated targets: `unemployment_rate`, `ust10_rate`, `hpi_yoy` (mapped from `hpi_growth_yoy`)
- Gated horizons: `h1..h12` (36 required cells)
- Power rule: minimum required-cell `n_oos >= 40`
- Current minimum required-cell `n_oos`: `44`

## Release Gate Scorecard (Incumbent)
| Metric | Threshold | Actual | Status |
|---|---:|---:|---|
| Power (min n_oos) | >= 40 | 44 | PASS |
| Median rRMSE h1..2 | <= 1.00 | 0.9744 | PASS |
| Median rRMSE h3..4 | < 0.98 | 0.9208 | PASS |
| Mean CRPS gain vs RW h5..12 | >= 3.0% | 15.13% | PASS |
| Coverage90 pass-rate | >= 0.75 | 0.8611 | PASS |
| Width ratio mean | <= 1.35 | 1.3069 | PASS |
| Width ratio per-var max | <= 1.60 | 1.4015 | PASS |
| Boundary median Z | <= 1.50 | 0.0012 | PASS |
| Boundary max Z | <= 2.50 | 0.0350 | PASS |
| Scenario timing+ordering | pass | timing=True ordering=True | PASS |
| Release gate overall | pass | True | PASS |

## Promotion Gate Scorecard (Challenger vs Incumbent)
| Check | Threshold | Actual | Status |
|---|---:|---:|---|
| Incumbent release pass | required | True | PASS |
| Challenger release pass | required | False | FAIL |
| Promotion power subset min n_oos (h9..12) | >= 40 | 40 | PASS |
| CRPS gain challenger vs incumbent h9..12 | >= 5.0% | 1.64% | FAIL |
| Short-horizon CRPS worsen h1..4 | <= 1.0% | 0.42% | PASS |
| Boundary comparator | challenger <= incumbent + 0.15 | incumbent=0.0012, challenger=0.0015 | PASS |
| Promotion overall | pass | False | FAIL |

## Per-Target Diagnostics (Incumbent Champion A)
| Target | Bucket 1..4 champion | Bucket 5..12 champion | Release width ratio (mean over h1..12) | Calibration note |
|---|---|---|---:|---|
| `unemployment_rate` | BVAR | BVAR | 1.2934 | scale=1.205, cov90 0.8722 -> 0.8864, samples=352, width_cap=1.60 |
| `ust10_rate` | AR | AR | 1.2258 | scale=1.55, cov90 0.7216 -> 0.9006, samples=352, width_cap=1.70 |
| `hpi_yoy` | BVAR | BVAR | 1.4015 | scale=0.9, cov90 0.8210 -> 0.8097, samples=352, width_cap=1.45 |

### Coverage fail cells (diagnostic)
| Variable | Horizon | Coverage90 | Width ratio vs RW | Bucket |
|---|---:|---:|---:|---|
| `hpi_growth_yoy` | 5 | 0.7955 | 1.2663 | bucket_5_12 |
| `hpi_growth_yoy` | 6 | 0.7727 | 1.3126 | bucket_5_12 |
| `hpi_growth_yoy` | 11 | 0.7955 | 1.5911 | bucket_5_12 |
| `ust10_rate` | 3 | 0.7955 | 0.9575 | bucket_1_4 |
| `ust10_rate` | 4 | 0.7727 | 0.9339 | bucket_1_4 |

## Trade-off Readout (What to improve next)
- Incumbent already clears release gates with strong CRPS gain; the primary bottleneck is promotion.
- Challenger has better coverage pass-rate but fails release due to width inflation and weaker short-horizon quality.
- Largest challenger width issue is `ust10_rate` (`width_ratio_per_var` ~ 1.885 > 1.60 threshold).
- HPI path still shows mid-horizon coverage weak spots; width cap is binding, so pure scaling is not enough.

## One-Change Next Experiment (Recommended)
1. Improve challenger `ust10_rate` bucket `5..12` density shape before any regime promotion attempt.
Reason: promotion currently fails mainly on challenger quality/width, not power or boundary continuity.
Success criteria for this one change:
- challenger release `width_ratio_per_var(ust10_rate) <= 1.60`
- challenger release `mean_crps_gain_h5..12 >= 3.0%`
- promotion `CRPS gain h9..12 >= 5.0%` or materially closer to threshold.

## Ablation Snapshot (PD-target evaluation)
| Regime | Scope | min n_oos | median rRMSE h1..2 | median rRMSE h3..4 | mean CRPS gain h5..12 | coverage pass-rate | width mean | release-like pass |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `champion_a` | `full_28` | 44 | 1.0266 | 0.9155 | 9.46% | 0.8056 | 1.4955 | False |
| `single_stage_h12_no_bridge` | `full_28` | 44 | 1.0409 | 0.9200 | 9.40% | 0.8333 | 1.5097 | False |
| `champion_b` | `full_28` | 40 | 1.0113 | 0.9823 | 2.35% | 0.9722 | 1.5442 | False |
| `single_stage_h12_no_bridge` | `pd_core_6` | 44 | 0.9795 | 0.9536 | -7.49% | 0.1667 | 0.8987 | False |
| `champion_a` | `pd_core_6` | 44 | 0.9860 | 0.9461 | -7.72% | 0.1667 | 0.9045 | False |
| `champion_b` | `pd_core_6` | 40 | 1.0017 | 1.0188 | -10.97% | 0.2778 | 0.9068 | False |

## Artifacts used
- `/Users/stevenchung/Desktop/P12B_File/Macro risk v3/outputs/macro_engine/validation/validation_summary.json`
- `/Users/stevenchung/Desktop/P12B_File/Macro risk v3/outputs/macro_engine/validation/backtest_metrics.csv`
- `/Users/stevenchung/Desktop/P12B_File/Macro risk v3/outputs/macro_engine/validation/calibration_factors.json`
- `/Users/stevenchung/Desktop/P12B_File/Macro risk v3/outputs/macro_engine/validation/coverage_fail_cells.csv`
- `/Users/stevenchung/Desktop/P12B_File/Macro risk v3/outputs/macro_engine/validation/pd_ablation_results.csv`
- `/Users/stevenchung/Desktop/P12B_File/Macro risk v3/outputs/macro_engine/champion_map.json`
