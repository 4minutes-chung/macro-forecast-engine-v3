#!/usr/bin/env python3
"""V3 freeze PD exporter: canonical 3-level file + convenience derived file + metadata."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export V3 freeze PD regressors.")
    parser.add_argument("--config", default="macro_engine_config.json", help="Config JSON path.")
    parser.add_argument("--input", default="outputs/macro_engine/macro_forecast_paths.csv", help="Forecast paths CSV.")
    parser.add_argument(
        "--model-panel",
        default="data/macro_panel_quarterly_model.csv",
        help="Model panel for lag-history fallback.",
    )
    parser.add_argument(
        "--levels-output",
        default="outputs/macro_engine/pd_regressors_forecast_levels.csv",
        help="Canonical levels output CSV.",
    )
    parser.add_argument(
        "--derived-output",
        default="outputs/macro_engine/pd_regressors_forecast_derived.csv",
        help="Convenience derived output CSV.",
    )
    parser.add_argument(
        "--metadata-output",
        default="outputs/macro_engine/pd_regressors_metadata.json",
        help="Metadata output JSON.",
    )
    parser.add_argument(
        "--champion-map",
        default="outputs/macro_engine/champion_map.json",
        help="Champion map JSON for calibration references.",
    )
    parser.add_argument(
        "--legacy-output-csv",
        default="outputs/macro_engine/pd_macro_subset_sample.csv",
        help="Legacy sample CSV output path.",
    )
    parser.add_argument(
        "--legacy-output-json",
        default="outputs/macro_engine/pd_macro_subset_sample.json",
        help="Legacy sample JSON output path.",
    )
    parser.add_argument(
        "--write-legacy-sample",
        action="store_true",
        help="Also emit legacy sample files for backward compatibility.",
    )
    return parser.parse_args()


def _resolve_lag(validation: dict, target_name: str, default_lag: int = 2) -> int:
    defaults = validation.get("pd_delta_defaults", {})
    overrides = validation.get("pd_delta_overrides", {})
    if target_name in overrides and isinstance(overrides[target_name], dict):
        return int(overrides[target_name].get("lag_quarters", default_lag))
    if target_name in defaults and isinstance(defaults[target_name], dict):
        return int(defaults[target_name].get("lag_quarters", default_lag))
    return int(default_lag)


def _lag_value(series_fc: np.ndarray, history: np.ndarray, h: int, lag_q: int) -> float:
    back = h - lag_q
    if back >= 1:
        return float(series_fc[back - 1])
    if history.shape[0] >= lag_q:
        hist_idx = back + lag_q - 1
        if 0 <= hist_idx < history.shape[0]:
            return float(history[hist_idx])
    return float("nan")


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    validation = config.get("validation", {})
    df = pd.read_csv(args.input)
    required_cols = {"scenario", "forecast_q", "quarter_end", "unemployment_rate", "ust10_rate", "hpi_growth_yoy"}
    missing = sorted(c for c in required_cols if c not in df.columns)
    if missing:
        raise RuntimeError(f"Missing columns in forecast paths: {missing}")

    levels = df[["scenario", "forecast_q", "quarter_end", "unemployment_rate", "ust10_rate", "hpi_growth_yoy"]].copy()
    levels = levels.rename(columns={"hpi_growth_yoy": "hpi_yoy"})

    u_hist = np.array([], dtype=float)
    r_hist = np.array([], dtype=float)
    data_vintage_date = ""
    if os.path.exists(args.model_panel):
        panel = pd.read_csv(args.model_panel, parse_dates=["quarter_end"]).sort_values("quarter_end")
        if {"unemployment_rate", "ust10_rate", "quarter_end"}.issubset(panel.columns):
            lag_max = max(_resolve_lag(validation, "du_6m"), _resolve_lag(validation, "d10y_6m"))
            u_hist = panel["unemployment_rate"].to_numpy(dtype=float)[-lag_max:]
            r_hist = panel["ust10_rate"].to_numpy(dtype=float)[-lag_max:]
            data_vintage_date = str(panel["quarter_end"].iloc[-1].date())

    du_lag = _resolve_lag(validation, "du_6m", default_lag=2)
    d10_lag = _resolve_lag(validation, "d10y_6m", default_lag=2)
    derived_rows = []
    lag_nan_count = 0
    for scenario, g in levels.groupby("scenario", sort=False):
        g = g.sort_values("forecast_q").reset_index(drop=True)
        u_fc = g["unemployment_rate"].to_numpy(dtype=float)
        r_fc = g["ust10_rate"].to_numpy(dtype=float)
        for _, row in g.iterrows():
            h = int(row["forecast_q"])
            u_lag = _lag_value(u_fc, u_hist, h, du_lag)
            r_lag = _lag_value(r_fc, r_hist, h, d10_lag)
            du = float(row["unemployment_rate"] - u_lag) if np.isfinite(u_lag) else np.nan
            d10 = float(row["ust10_rate"] - r_lag) if np.isfinite(r_lag) else np.nan
            if not np.isfinite(du) or not np.isfinite(d10):
                lag_nan_count += 1
            derived_rows.append(
                {
                    "scenario": scenario,
                    "forecast_q": h,
                    "quarter_end": row["quarter_end"],
                    "du_6m": du,
                    "d10y_6m": d10,
                    "hpi_yoy": float(row["hpi_yoy"]),
                }
            )
    derived = pd.DataFrame(derived_rows)

    for path in [args.levels_output, args.derived_output]:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    levels.to_csv(args.levels_output, index=False, float_format="%.6f")
    derived.to_csv(args.derived_output, index=False, float_format="%.6f")

    metadata = {
        "contract_version": str(validation.get("contract_version", "v3.freeze.1")),
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data_vintage_date": data_vintage_date,
        "scenario_set": levels["scenario"].drop_duplicates().tolist(),
        "canonical_file": "levels",
        "units": {
            "unemployment_rate": "percent",
            "ust10_rate": "percent",
            "hpi_yoy": "percent_yoy",
            "du_6m": "percentage_points_6m",
            "d10y_6m": "percentage_points_6m",
        },
        "hpi_definition": "yoy growth",
        "derived_formulas": {
            "du_6m": f"unemployment_rate_t - unemployment_rate_(t-{du_lag})",
            "d10y_6m": f"ust10_rate_t - ust10_rate_(t-{d10_lag})",
        },
        "lag_overrides_used": {"du_6m": du_lag, "d10y_6m": d10_lag},
        "derived_missing_lag_rows": int(lag_nan_count),
        "calibration_applied": False,
        "calibration_scale_references": {},
    }
    if os.path.exists(args.champion_map):
        try:
            champion_map = json.load(open(args.champion_map, "r", encoding="utf-8"))
            refs = {}
            for src_var in ["unemployment_rate", "ust10_rate", "hpi_growth_yoy"]:
                spec = champion_map.get("variables", {}).get(src_var, {})
                if not spec:
                    continue
                refs[src_var] = {
                    "bucket_1_4": float(spec.get("bucket_1_4", {}).get("calibration_scale", 1.0)),
                    "bucket_5_12": float(spec.get("bucket_5_12", {}).get("calibration_scale", 1.0)),
                }
            metadata["calibration_scale_references"] = refs
            metadata["calibration_applied"] = bool(any(abs(v - 1.0) > 1e-9 for m in refs.values() for v in m.values()))
        except Exception:
            metadata["calibration_scale_references"] = {"error": "failed_to_parse_champion_map"}
    os.makedirs(os.path.dirname(args.metadata_output) or ".", exist_ok=True)
    with open(args.metadata_output, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if args.write_legacy_sample:
        legacy_fields = [
            "unemployment_rate",
            "hpi_yoy",
            "ust10_rate",
            "du_6m",
            "d10y_6m",
        ]
        legacy_df = levels.merge(
            derived[["scenario", "forecast_q", "du_6m", "d10y_6m"]],
            on=["scenario", "forecast_q"],
            how="left",
        )[["scenario", "forecast_q", "quarter_end"] + legacy_fields]
        os.makedirs(os.path.dirname(args.legacy_output_csv) or ".", exist_ok=True)
        legacy_df.to_csv(args.legacy_output_csv, index=False, float_format="%.6f")
        with open(args.legacy_output_json, "w", encoding="utf-8") as f:
            json.dump({"source_file": args.input, "fields": legacy_fields}, f, indent=2)


if __name__ == "__main__":
    main()
