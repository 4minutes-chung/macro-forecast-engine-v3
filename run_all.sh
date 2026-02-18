#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

STRICT_FLAG=""
VERBOSE_FLAG=""
if [[ "${STRICT_VALIDATION:-0}" == "1" ]]; then
  STRICT_FLAG="--strict-validation"
fi
if [[ "${VERBOSE_VALIDATION:-0}" == "1" ]]; then
  VERBOSE_FLAG="--verbose-validation"
fi

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
  --champion-map-output outputs/macro_engine/champion_map.json \
  ${STRICT_FLAG} \
  ${VERBOSE_FLAG}

python3 scripts/export_pd_macro_subset.py \
  --config macro_engine_config.json \
  --input outputs/macro_engine/macro_forecast_paths.csv \
  --model-panel data/macro_panel_quarterly_model.csv \
  --levels-output outputs/macro_engine/pd_regressors_forecast_levels.csv \
  --derived-output outputs/macro_engine/pd_regressors_forecast_derived.csv \
  --metadata-output outputs/macro_engine/pd_regressors_metadata.json
