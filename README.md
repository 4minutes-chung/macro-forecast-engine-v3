# Macro Risk V3

## 1) What this package is
Macro Risk V3 is a frozen macro-forecast package for PD input workflows.
It produces quarterly macro paths and validation evidence.

Validation contract in this frozen package: `v3.freeze.1`.

## 2) Status
Current status from `outputs/macro_engine/validation/validation_summary.json`:
- Release gate: `PASS`
- Promotion gate: `FAIL` (challenger not promoted)
- Diagnostic-only mode: `false` (power gate satisfied)

Interpretation:
- V3 is approved as the fallback baseline.
- Challenger regime is retained as evidence but not promoted.

## 3) Canonical PD handoff
Use these three files for PD ingestion:
- `outputs/macro_engine/pd_regressors_forecast_levels.csv` (canonical)
- `outputs/macro_engine/pd_regressors_forecast_derived.csv` (convenience)
- `outputs/macro_engine/pd_regressors_metadata.json` (contract metadata)

Primary level targets in canonical handoff: (Due to request from the PD user)
- `unemployment_rate`
- `ust10_rate`
- `hpi_yoy` (mapped from `hpi_growth_yoy`)

## 4) Repository map 
Top-level:
- `macro_engine_config.json`: full assumptions, thresholds, and contract config.
- `macro_engine_plain_report_v3.tex` / `macro_engine_plain_report_v3.pdf`: full technical report and freeze record.
- `macro_engine_schema.md`: output schemas and validation artifact map.
- `run_all.sh`: end-to-end pipeline.

Data:
- `data/macro_panel_quarterly_raw.csv`: transformed quarterly panel before modeling subset filtering.
- `data/macro_panel_quarterly_model.csv`: modeling panel used by forecast/validation scripts.
- `data/macro_panel_metadata.json`: source and transformation metadata.

Scripts:
- `scripts/fetch_macro_panel_fred.py`: data refresh and transformation.
- `scripts/run_macro_forecast_engine.py`: forecast generation.
- `scripts/run_macro_validation.py`: validation gates, challenger tests, artifacts.
- `scripts/backtest_bvar_oos.py`: OOS backtest mechanics.
- `scripts/export_pd_macro_subset.py`: PD export generation.
- `scripts/macro_model_core.py`: reusable model/scoring utilities.

Forecast outputs:
- `outputs/macro_engine/macro_forecast_paths.csv`
- `outputs/macro_engine/macro_forecast_short_horizon_intervals.csv`
- `outputs/macro_engine/macro_model_diagnostics.json`
- `outputs/macro_engine/macro_anchor_assumptions.json`
- `outputs/macro_engine/macro_impulse_responses.csv`
- `outputs/macro_engine/champion_map.json`
- `outputs/macro_engine/bvar_oos_backtest_table.csv` (legacy baseline table)

Validation outputs:
- Core final-result files:
  - `outputs/macro_engine/validation/validation_summary.json`
  - `outputs/macro_engine/validation/backtest_metrics.csv`
  - `outputs/macro_engine/validation/calibration_factors.json`
  - `outputs/macro_engine/validation/pd_ablation_results.csv`
- Extended audit diagnostics:
  - `outputs/macro_engine/validation/boundary_culprits.csv`
  - `outputs/macro_engine/validation/coverage_fail_cells.csv`
  - `outputs/macro_engine/validation/dm_results.csv`
  - `outputs/macro_engine/validation/seed_registry.csv`
  - `outputs/macro_engine/validation/scenario_timing_checks.csv`
  - `outputs/macro_engine/validation/special_series_sweep.csv`
  - `outputs/macro_engine/validation/anchor_sensitivity.csv`
  - `outputs/macro_engine/validation/regime_split_scores.csv`
  - `outputs/macro_engine/validation/assumption_*` and `regime_*` subfolders

## 5) Model and governance summary
Model structure:
- Horizon: 80 quarters.
- Incumbent regime (`champion_a`): Q1--Q12 short model, Q13--Q24 bridge, Q25--Q80 long-run/scenario.
- Challenger regime (`champion_b`): Q1--Q16 short model, Q17--Q28 bridge, Q29--Q80 long-run/scenario.

Short-horizon championing:
- Candidate models: BVAR, AR, RW.
- Selection by variable and bucket (`Q1..Q4`, `Q5..Q12`).
- Tie-break order: CRPS, then coverage closeness to 0.90, then width-ratio closeness to 1.0.

Validation gates:
- Power gate: minimum required-cell `n_oos >= 40`.
- Release and promotion profiles are gating.
- Operational profile is diagnostic-only.
- Hard invariants: boundary smoothness, scenario timing/order, reproducibility.

## 6) Repro commands
```bash
python3 scripts/fetch_macro_panel_fred.py \
  --raw-output data/macro_panel_quarterly_raw.csv \
  --model-output data/macro_panel_quarterly_model.csv \
  --metadata-output data/macro_panel_metadata.json

python3 scripts/run_macro_forecast_engine.py \
  --config macro_engine_config.json \
  --output-dir outputs/macro_engine

python3 scripts/run_macro_validation.py \
  --config macro_engine_config.json \
  --input data/macro_panel_quarterly_model.csv \
  --output-dir outputs/macro_engine/validation \
  --champion-map-output outputs/macro_engine/champion_map.json

python3 scripts/export_pd_macro_subset.py \
  --config macro_engine_config.json \
  --input outputs/macro_engine/macro_forecast_paths.csv \
  --model-panel data/macro_panel_quarterly_model.csv \
  --levels-output outputs/macro_engine/pd_regressors_forecast_levels.csv \
  --derived-output outputs/macro_engine/pd_regressors_forecast_derived.csv \
  --metadata-output outputs/macro_engine/pd_regressors_metadata.json
```

Or run end-to-end:
```bash
bash run_all.sh
```

## 7) Additional references
- Full technical narrative and limitations: `macro_engine_plain_report_v3.pdf`
- Schema and artifact definitions: `macro_engine_schema.md`
- Compact scorecard snapshot: `VALIDATION_ENGINE_PD_SCORECARD.md`
