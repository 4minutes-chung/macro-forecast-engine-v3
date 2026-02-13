#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

python scripts/fetch_macro_panel_fred.py \
  --raw-output data/macro_panel_quarterly_raw.csv \
  --model-output data/macro_panel_quarterly_model.csv \
  --metadata-output data/macro_panel_metadata.json

python scripts/run_macro_forecast_engine.py \
  --config macro_engine_config.json \
  --output-dir outputs/macro_engine

python scripts/export_pd_macro_subset.py \
  --input outputs/macro_engine/macro_forecast_paths.csv \
  --output-csv outputs/macro_engine/pd_macro_subset_sample.csv \
  --output-json outputs/macro_engine/pd_macro_subset_sample.json
