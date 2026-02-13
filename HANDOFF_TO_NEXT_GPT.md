# Handoff: Macro Risk Forecast Engine (Stage 1 Complete)

This folder already contains a runnable macro forecasting engine. Do NOT rebuild from scratch.

## Goal (what we built)
- A U.S. macro forecasting package that produces **20-year quarterly** macro paths for mortgage risk models (PD/LGD).
- It does **not** estimate PD models; it produces macro inputs that PD models consume later.
- It supports 3 horizon regimes:
  - **Q1–Q8 (1–2y):** statistical forecast with VAR/BVAR (BVAR used for production path by default).
  - **Q9–Q20 (2–5y):** "anchor bridge" that steers paths toward long-run economic anchors.
  - **Q21–Q80 (5–20y):** scenario envelopes layered on top of anchor path.

## What you can run (one command)
From the project root (`Macro risk`):
```bash
./run_all.sh
```
This runs:
1) `scripts/fetch_macro_panel_fred.py` (builds quarterly panel)
2) `scripts/run_macro_forecast_engine.py` (fits models + generates 20y forecasts)
3) `scripts/export_pd_macro_subset.py` (exports a PD-ready subset sample)

## Core files (the only ones you need to care about)

### Config (assumptions + scenarios)
- `macro_engine_config.json`
  - Variable list (28 vars) + VAR benchmark subset.
  - BVAR hyperparams (Minnesota prior).
  - Anchors: NAIRU, inflation target, neutral rate, productivity/demographics, mortgage spread.
  - Scenario shock schedules (start/peak/end quarters + magnitudes) for Mild/Severe/Demographic.

### Code
- `scripts/fetch_macro_panel_fred.py`
  - Pulls 28 U.S. series from FRED.
  - Aggregates to quarterly.
  - Applies transforms: YoY for growth/inflation; levels for rates/spreads/sentiment.
  - Writes `data/macro_panel_quarterly_model.csv` (complete-case).
- `scripts/run_macro_forecast_engine.py`
  - Fits VAR on stable benchmark subset for diagnostics/IRFs.
  - Fits Minnesota-style BVAR on full panel for production path.
  - Generates short-horizon intervals by simulation.
  - Bridges to anchors (Q9–Q20) then mean-reverts to anchors.
  - Applies scenario envelopes starting Q21.
  - Writes the final outputs in `outputs/macro_engine/`.
- `scripts/export_pd_macro_subset.py`
  - Takes the full forecast and exports a smaller PD-ready subset with units/transform metadata.

### Data (panel used by models)
- `data/macro_panel_quarterly_model.csv` (model-ready; complete-case)
- `data/macro_panel_quarterly_raw.csv` (raw quarterly levels; aligned to model rows)
- `data/macro_panel_metadata.json` (FRED IDs + transforms used)

### Outputs (deliverables for PD/LGD)
- `outputs/macro_engine/macro_forecast_paths.csv`
  - Full 20-year quarterly forecast.
  - Columns include: `scenario`, `forecast_q`, `quarter_end`, then all 28 macro variables.
  - Scenarios: `Baseline`, `Mild_Adverse`, `Severe_Adverse`, `Demographic_LowGrowth`.
- `outputs/macro_engine/macro_forecast_short_horizon_intervals.csv`
  - Short-horizon density/intervals (quantiles 0.05/0.5/0.95) for Q1–Q8.
  - VAR provides values only for benchmark vars; others are blank/NaN. BVAR covers all 28.
- `outputs/macro_engine/macro_anchor_assumptions.json`
  - Anchor targets and assumption values used in the bridge.
- `outputs/macro_engine/macro_model_diagnostics.json`
  - Lag order, stability flags, hyperparameters, scenario list.
- `outputs/macro_engine/macro_impulse_responses.csv`
  - VAR impulse responses for configured shocks (mortgage rate, HPI growth, unemployment).
- `outputs/macro_engine/pd_macro_subset_sample.csv`
- `outputs/macro_engine/pd_macro_subset_sample.json`
  - PD-ready sample subset + units/transform metadata.

### Documentation (human + AI)
- `README_MACRO_ENGINE.md` (run guide + auto-synced schema)
- `macro_engine_schema.md` (full column schema)
- `macro_engine_plain_report.tex` (economist-friendly explanation; can compile to PDF)

## Assumptions (must know)

### Anchor bridge (Q9–Q20; 2–5y)
- Uses explicit anchors from `macro_engine_config.json`:
  - NAIRU ~ 4.2%
  - inflation target ~ 2.1%
  - neutral real rate ~ 0.8%
  - 10y term premium ~ 1.2%
  - mortgage spread ~ 1.7%
  - productivity ~ 1.2%
  - working-age growth ~ 0.4%
  - population growth ~ 0.6%
  - housing supply drag ~ 0.5 pp
- The bridge gradually blends model output toward these values between Q9 and Q20.

### Scenario envelopes (Q21–Q80; 5–20y)
- Mild vs Severe vs Demographic are defined in `macro_engine_config.json` with triangular shocks:
  - each shock has `start_q`, `peak_q`, `end_q`, `peak_delta`.
- Persistent anchor shifts exist for some scenarios (demographics, HPI drift).

## Environment note (don’t get stuck again)
- Requires Python with `numpy` and `pandas`.
- Using conda is fine. The key is: run from the project root so relative paths resolve.

## “Stage 2” suggestions for next GPT (optional)
- Calibrate macro assumptions to your house view or to regulator scenarios.
- Add additional scenario templates (stagflation, housing-only shock, rates-down refi boom).
- Produce a minimal PD ingestion contract file (schema + units) pinned to versioned outputs.
- Add VS Code `tasks.json` to run `run_all.sh` from the UI (no terminal typing).

