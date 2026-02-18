#!/usr/bin/env python3
"""Run V4.1 macro forecast engine with PD-regressor exports and champion short-horizon blending."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from macro_model_core import (
    apply_transform,
    ar_point_forecast,
    deterministic_forecast,
    fit_bvar_minnesota,
    inverse_transform_draws,
    quantiles_from_draws,
    rw_point_forecast,
    select_ar_order,
    select_var_lag,
    simulate_ar_draws,
    simulate_rw_draws,
    simulate_var_draws,
    stable_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run macro forecast engine.")
    parser.add_argument("--config", default="macro_engine_config.json", help="Path to config JSON.")
    parser.add_argument("--output-dir", default="outputs/macro_engine", help="Output directory.")
    parser.add_argument(
        "--assumption-set",
        default="base",
        choices=["low", "base", "high"],
        help="Structural assumption set for long-run anchor construction.",
    )
    parser.add_argument(
        "--regime-id",
        default="",
        help="Regime candidate id (default: model.active_regime or champion_a).",
    )
    parser.add_argument(
        "--champion-map",
        default="outputs/macro_engine/champion_map.json",
        help="Champion map JSON path. Defaults to output champion map location.",
    )
    parser.add_argument(
        "--disable-champion",
        action="store_true",
        help="Disable champion blending and use BVAR-only short horizon.",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resolve_regime(config: Dict, regime_id_arg: str) -> Tuple[str, Dict]:
    regime_candidates = config.get("regime_candidates", {})
    chosen = regime_id_arg or config.get("model", {}).get("active_regime", "champion_a")
    if chosen in regime_candidates:
        return chosen, regime_candidates[chosen]

    if "horizons" in config:
        return "legacy_horizons", config["horizons"]

    if regime_candidates:
        first_key = sorted(regime_candidates.keys())[0]
        return first_key, regime_candidates[first_key]

    raise RuntimeError("No regime candidate or legacy horizons found in config.")


def make_forecast_dates(last_date: pd.Timestamp, horizon: int) -> List[pd.Timestamp]:
    d = last_date
    dates: List[pd.Timestamp] = []
    for _ in range(horizon):
        d = (d + pd.offsets.QuarterEnd(1)).normalize()
        dates.append(d)
    return dates


def bridge_to_anchors(
    model_path: np.ndarray,
    anchor: np.ndarray,
    short_horizon: int,
    medium_horizon: int,
    force_continuity_short_to_bridge: bool = True,
) -> np.ndarray:
    out = model_path.copy()
    if force_continuity_short_to_bridge and short_horizon < out.shape[0] and short_horizon > 0:
        out[short_horizon, :] = out[short_horizon - 1, :]
    start_h = short_horizon + 1 if force_continuity_short_to_bridge else short_horizon
    for h in range(start_h, medium_horizon):
        w = float((h - short_horizon) / max(1, (medium_horizon - short_horizon)))
        out[h, :] = (1.0 - w) * out[h, :] + w * anchor
    kappa = 0.08
    for h in range(medium_horizon, out.shape[0]):
        out[h, :] = out[h - 1, :] + kappa * (anchor - out[h - 1, :])
    return out


def triangular_shock(h: int, start: int, peak: int, end: int, peak_delta: float) -> float:
    # h is 1-based quarter index. First non-zero effect starts at h == start.
    if h < start:
        return 0.0
    if h <= peak:
        if peak == start:
            return peak_delta
        return peak_delta * ((h - start + 1) / float(peak - start + 1))
    if h <= end:
        if end == peak:
            return 0.0
        return peak_delta * ((end - h) / float(end - peak))
    return 0.0


def apply_scenario_envelope(
    baseline: np.ndarray,
    variables: List[str],
    scenario_cfg: Dict,
    long_start_quarter: int,
    anchor: np.ndarray,
) -> np.ndarray:
    idx = {v: i for i, v in enumerate(variables)}
    out = baseline.copy()
    horizon = out.shape[0]

    for h in range(long_start_quarter, horizon + 1):
        for var, spec in scenario_cfg.get("shock_profiles", {}).items():
            if var not in idx:
                continue
            delta = triangular_shock(
                h,
                int(spec["start_q"]),
                int(spec["peak_q"]),
                int(spec["end_q"]),
                float(spec["peak_delta"]),
            )
            out[h - 1, idx[var]] = baseline[h - 1, idx[var]] + delta

    for var, shift in scenario_cfg.get("persistent_anchor_shift", {}).items():
        if var not in idx:
            continue
        shifted_anchor = anchor[idx[var]] + float(shift)
        for h in range(long_start_quarter, horizon):
            out[h, idx[var]] = out[h - 1, idx[var]] + 0.10 * (shifted_anchor - out[h - 1, idx[var]])

    return out


def build_anchor_vector(
    baseline_path: np.ndarray,
    variables: List[str],
    structural_assumptions: Dict,
    long_run_regime_params: Dict,
) -> np.ndarray:
    idx = {v: i for i, v in enumerate(variables)}
    last40_mean = np.nanmean(baseline_path[-40:, :], axis=0)
    anchor = last40_mean.copy()

    working_age_growth = float(structural_assumptions["working_age_pop_growth_yoy"])
    productivity = float(structural_assumptions["labor_productivity_trend_yoy"])
    inflation_target = float(structural_assumptions["inflation_target"])
    neutral_real = float(structural_assumptions["neutral_real_rate"])
    term_premium = float(structural_assumptions["term_premium_10y"])
    nairu = float(structural_assumptions["nairu"])
    pop_growth = float(structural_assumptions["population_growth_yoy"])

    housing_supply_drag = float(long_run_regime_params.get("housing_supply_elasticity_drag", 0.5))
    mortgage_spread = float(long_run_regime_params.get("mortgage_spread_30y", 1.7))
    hy_spread_median = float(long_run_regime_params.get("hy_spread_long_run_median", 3.8))

    anchor[idx["working_age_pop_growth_yoy"]] = working_age_growth
    anchor[idx["population_growth_yoy"]] = pop_growth
    anchor[idx["real_gdp_growth_yoy"]] = productivity + working_age_growth
    anchor[idx["headline_cpi_yoy"]] = inflation_target
    anchor[idx["core_cpi_yoy"]] = inflation_target
    anchor[idx["pce_inflation_yoy"]] = inflation_target
    anchor[idx["oer_inflation_yoy"]] = inflation_target + 0.2
    anchor[idx["medical_cpi_yoy"]] = inflation_target + 0.6
    anchor[idx["unemployment_rate"]] = nairu

    ust10_anchor = neutral_real + inflation_target + term_premium
    anchor[idx["ust10_rate"]] = ust10_anchor
    anchor[idx["mortgage30_rate"]] = ust10_anchor + mortgage_spread
    anchor[idx["fed_funds_rate"]] = neutral_real + inflation_target
    anchor[idx["prime_rate"]] = anchor[idx["fed_funds_rate"]] + 3.0
    anchor[idx["high_yield_spread"]] = hy_spread_median

    wage_anchor = float(np.clip(last40_mean[idx["wage_growth_yoy"]], 1.5, 5.0))
    hpi_anchor = wage_anchor + pop_growth - housing_supply_drag
    anchor[idx["hpi_growth_yoy"]] = hpi_anchor
    anchor[idx["rent_inflation_yoy"]] = inflation_target + 0.5
    anchor[idx["retail_sales_yoy"]] = anchor[idx["real_gdp_growth_yoy"]] + inflation_target
    anchor[idx["industrial_production_yoy"]] = anchor[idx["real_gdp_growth_yoy"]] - 0.2

    return anchor


def default_champion_map(variables: List[str]) -> Dict:
    return {
        "contract_version": "v4.1",
        "variables": {
            v: {
                "bucket_1_4": {"kind": "single", "model": "BVAR", "transform_id": "level", "calibration_scale": 1.0},
                "bucket_5_12": {"kind": "single", "model": "BVAR", "transform_id": "level", "calibration_scale": 1.0},
            }
            for v in variables
        },
    }


def load_champion_map(path: str, variables: List[str]) -> Dict:
    if not path or not os.path.exists(path):
        return default_champion_map(variables)
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if "variables" not in payload:
        return default_champion_map(variables)
    out = default_champion_map(variables)
    for v in variables:
        if v in payload["variables"]:
            out["variables"][v].update(payload["variables"][v])
    out["contract_version"] = payload.get("contract_version", "v4.1")
    return out


def _bucket_for_h(h: int) -> str:
    if 1 <= h <= 4:
        return "bucket_1_4"
    if 5 <= h <= 12:
        return "bucket_5_12"
    return "bucket_5_12"


def _univariate_forecast_with_transform(
    series: np.ndarray,
    transform_id: str,
    model_name: str,
    horizon: int,
    n_sims: int,
    ar_p_candidates: List[int],
    seed_base: int,
) -> Tuple[np.ndarray, np.ndarray]:
    transformed, meta = apply_transform(series, transform_id)

    if model_name == "RW":
        point_t = rw_point_forecast(transformed[-1], horizon)
        diff = np.diff(transformed)
        sigma = float(np.std(diff, ddof=1)) if diff.shape[0] > 1 else 1e-6
        draws_t = simulate_rw_draws(transformed[-1], sigma, horizon, n_sims, seed_base)
    elif model_name == "AR":
        ar_model = select_ar_order(transformed, ar_p_candidates)
        point_t = ar_point_forecast(ar_model, transformed, horizon)
        draws_t = simulate_ar_draws(ar_model, transformed, horizon, n_sims, seed_base)
    else:
        raise RuntimeError(f"Unsupported transformed univariate model: {model_name}")

    draws_level, _ = inverse_transform_draws(draws_t, float(series[-1]), transform_id, meta)
    point_level = np.mean(draws_level, axis=0)
    return point_level, draws_level


def combine_short_horizon(
    variables: List[str],
    short_horizon: int,
    champion_map: Dict,
    bvar_point: np.ndarray,
    bvar_draws: np.ndarray,
    ar_point: np.ndarray,
    ar_draws: np.ndarray,
    rw_point: np.ndarray,
    rw_draws: np.ndarray,
    full_history: np.ndarray,
    ar_p_candidates: List[int],
    ensemble_temp: float,
    seed_namespace: str,
) -> Tuple[np.ndarray, np.ndarray]:
    n_sims = bvar_draws.shape[0]
    n_vars = len(variables)
    chosen_point = bvar_point[:short_horizon, :].copy()
    chosen_draws = bvar_draws[:, :short_horizon, :].copy()
    var_to_idx = {v: i for i, v in enumerate(variables)}

    for var in variables:
        j = var_to_idx[var]
        var_cfg = champion_map.get("variables", {}).get(var, {})

        for h in range(1, short_horizon + 1):
            bucket = _bucket_for_h(h)
            spec = var_cfg.get(bucket, {"kind": "single", "model": "BVAR", "transform_id": "level"})

            model_points = {
                "BVAR": bvar_point[h - 1, j],
                "AR": ar_point[h - 1, j],
                "RW": rw_point[h - 1, j],
            }
            model_draws = {
                "BVAR": bvar_draws[:, h - 1, j],
                "AR": ar_draws[:, h - 1, j],
                "RW": rw_draws[:, h - 1, j],
            }

            if spec.get("kind", "single") == "single":
                model_name = spec.get("model", "BVAR").upper()
                transform_id = spec.get("transform_id", "level")
                if model_name in {"AR", "RW"} and transform_id != "level":
                    seed = stable_seed(seed_namespace, "champion", var, bucket, model_name, transform_id)
                    point_level, draws_level = _univariate_forecast_with_transform(
                        series=full_history[:, j],
                        transform_id=transform_id,
                        model_name=model_name,
                        horizon=short_horizon,
                        n_sims=n_sims,
                        ar_p_candidates=ar_p_candidates,
                        seed_base=seed,
                    )
                    chosen_point[h - 1, j] = point_level[h - 1]
                    chosen_draws[:, h - 1, j] = draws_level[:, h - 1]
                else:
                    chosen_point[h - 1, j] = model_points.get(model_name, model_points["BVAR"])
                    chosen_draws[:, h - 1, j] = model_draws.get(model_name, model_draws["BVAR"])
                calib_scale = float(spec.get("calibration_scale", 1.0))
                if calib_scale > 0 and abs(calib_scale - 1.0) > 1e-9:
                    d = chosen_draws[:, h - 1, j]
                    d_mean = float(np.mean(d))
                    chosen_draws[:, h - 1, j] = d_mean + calib_scale * (d - d_mean)
                continue

            if spec.get("kind") == "ensemble":
                raw_weights = spec.get("weights", {"BVAR": 1.0})
                models = [m.upper() for m in raw_weights.keys() if m.upper() in {"BVAR", "AR", "RW"}]
                if not models:
                    models = ["BVAR"]
                    raw_weights = {"BVAR": 1.0}
                w_vec = np.array([max(0.0, float(raw_weights[m])) for m in models], dtype=float)
                if np.sum(w_vec) <= 0:
                    w_vec = np.ones_like(w_vec)
                w_vec = w_vec / np.sum(w_vec)
                chosen_point[h - 1, j] = float(np.sum([w * model_points[m] for w, m in zip(w_vec, models)]))

                counts = np.floor(w_vec * n_sims).astype(int)
                while np.sum(counts) < n_sims:
                    counts[np.argmax(w_vec)] += 1
                mix = []
                for count, model_name in zip(counts.tolist(), models):
                    if count <= 0:
                        continue
                    arr = model_draws[model_name]
                    seed = stable_seed(seed_namespace, "mix", var, bucket, h, model_name)
                    rng = np.random.default_rng(seed)
                    idx = rng.choice(arr.shape[0], size=count, replace=True)
                    mix.append(arr[idx])
                if mix:
                    mixed = np.concatenate(mix)[:n_sims]
                    chosen_draws[:, h - 1, j] = mixed
                calib_scale = float(spec.get("calibration_scale", 1.0))
                if calib_scale > 0 and abs(calib_scale - 1.0) > 1e-9:
                    d = chosen_draws[:, h - 1, j]
                    d_mean = float(np.mean(d))
                    chosen_draws[:, h - 1, j] = d_mean + calib_scale * (d - d_mean)
                continue

            # Fallback
            chosen_point[h - 1, j] = model_points["BVAR"]
            chosen_draws[:, h - 1, j] = model_draws["BVAR"]

    return chosen_point, chosen_draws


def _hash_json(payload: Dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _resolve_pd_output_path(output_dir: str, configured_path: str, fallback_name: str) -> str:
    if not configured_path:
        return os.path.join(output_dir, fallback_name)
    if os.path.isabs(configured_path):
        return configured_path
    # Keep pd artifacts colocated with the current output dir.
    return os.path.join(output_dir, os.path.basename(configured_path))


def _pd_lag_quarters(validation_cfg: Dict, target_name: str, default_lag: int = 2) -> int:
    defaults = validation_cfg.get("pd_delta_defaults", {})
    overrides = validation_cfg.get("pd_delta_overrides", {})
    if target_name in overrides and isinstance(overrides[target_name], dict):
        return int(overrides[target_name].get("lag_quarters", default_lag))
    if target_name in defaults and isinstance(defaults[target_name], dict):
        return int(defaults[target_name].get("lag_quarters", default_lag))
    return int(default_lag)


def _lag_value(series_forecast: np.ndarray, history: np.ndarray, h: int, lag_q: int) -> float:
    # h is 1-based forecast horizon. We need value at (h - lag_q).
    back = h - lag_q
    if back >= 1:
        return float(series_forecast[back - 1])
    if history.shape[0] >= lag_q:
        hist_idx = back + lag_q - 1
        if 0 <= hist_idx < history.shape[0]:
            return float(history[hist_idx])
    return float("nan")


def write_pd_regressor_exports(
    output_dir: str,
    pd_export_cfg: Dict,
    validation_cfg: Dict,
    scenarios: Dict[str, np.ndarray],
    variables: List[str],
    forecast_dates: List[pd.Timestamp],
    last_observed_date: pd.Timestamp,
    history_levels: np.ndarray,
    champion_map: Dict,
) -> None:
    idx = {v: i for i, v in enumerate(variables)}
    needed = {"unemployment_rate", "ust10_rate", "hpi_growth_yoy"}
    if not needed.issubset(idx.keys()):
        return

    levels_path = _resolve_pd_output_path(
        output_dir,
        str(pd_export_cfg.get("levels_file", "")),
        "pd_regressors_forecast_levels.csv",
    )
    derived_path = _resolve_pd_output_path(
        output_dir,
        str(pd_export_cfg.get("derived_file", "")),
        "pd_regressors_forecast_derived.csv",
    )
    metadata_path = _resolve_pd_output_path(
        output_dir,
        str(pd_export_cfg.get("metadata_file", "")),
        "pd_regressors_metadata.json",
    )

    lag_du = _pd_lag_quarters(validation_cfg, "du_6m", default_lag=2)
    lag_d10 = _pd_lag_quarters(validation_cfg, "d10y_6m", default_lag=2)
    u_hist = history_levels[:, idx["unemployment_rate"]]
    r_hist = history_levels[:, idx["ust10_rate"]]

    levels_rows: List[Dict] = []
    derived_rows: List[Dict] = []
    for sc_name, mat in scenarios.items():
        u_fc = mat[:, idx["unemployment_rate"]]
        r_fc = mat[:, idx["ust10_rate"]]
        hpi_fc = mat[:, idx["hpi_growth_yoy"]]
        for h in range(1, mat.shape[0] + 1):
            q_end = forecast_dates[h - 1].date().isoformat()
            u_now = float(u_fc[h - 1])
            r_now = float(r_fc[h - 1])
            hpi_now = float(hpi_fc[h - 1])
            levels_rows.append(
                {
                    "scenario": sc_name,
                    "forecast_q": h,
                    "quarter_end": q_end,
                    "unemployment_rate": u_now,
                    "ust10_rate": r_now,
                    "hpi_yoy": hpi_now,
                }
            )

            u_lag = _lag_value(u_fc, u_hist, h, lag_du)
            r_lag = _lag_value(r_fc, r_hist, h, lag_d10)
            derived_rows.append(
                {
                    "scenario": sc_name,
                    "forecast_q": h,
                    "quarter_end": q_end,
                    "du_6m": float(u_now - u_lag) if np.isfinite(u_lag) else np.nan,
                    "d10y_6m": float(r_now - r_lag) if np.isfinite(r_lag) else np.nan,
                    "hpi_yoy": hpi_now,
                }
            )

    os.makedirs(os.path.dirname(levels_path) or ".", exist_ok=True)
    pd.DataFrame(levels_rows).to_csv(levels_path, index=False, float_format="%.6f")
    os.makedirs(os.path.dirname(derived_path) or ".", exist_ok=True)
    pd.DataFrame(derived_rows).to_csv(derived_path, index=False, float_format="%.6f")

    calib_refs = {}
    for source_var in ["unemployment_rate", "ust10_rate", "hpi_growth_yoy"]:
        spec = champion_map.get("variables", {}).get(source_var, {})
        if not spec:
            continue
        calib_refs[source_var] = {
            "bucket_1_4": float(spec.get("bucket_1_4", {}).get("calibration_scale", 1.0)),
            "bucket_5_12": float(spec.get("bucket_5_12", {}).get("calibration_scale", 1.0)),
        }

    metadata = {
        "contract_version": str(validation_cfg.get("contract_version", "v4.1")),
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data_vintage_date": str(last_observed_date.date()) if pd.notna(last_observed_date) else "",
        "scenario_set": list(scenarios.keys()),
        "canonical_file": str(pd_export_cfg.get("canonical_file", "levels")),
        "levels_file": os.path.basename(levels_path),
        "derived_file": os.path.basename(derived_path),
        "units": {
            "unemployment_rate": "percent",
            "ust10_rate": "percent",
            "hpi_yoy": "percent_yoy",
            "du_6m": "percentage_points_6m",
            "d10y_6m": "percentage_points_6m",
        },
        "definitions": {
            "hpi_yoy": "hpi_growth_yoy",
            "du_6m": "unemployment_rate_t - unemployment_rate_(t-lag_q)",
            "d10y_6m": "ust10_rate_t - ust10_rate_(t-lag_q)",
            "hpi_measure": "yoy growth",
        },
        "lag_overrides_used": {"du_6m": lag_du, "d10y_6m": lag_d10},
        "calibration_applied": bool(any(abs(vv - 1.0) > 1e-9 for v in calib_refs.values() for vv in v.values())),
        "calibration_scale_references": calib_refs,
    }
    os.makedirs(os.path.dirname(metadata_path) or ".", exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def build_impulse_responses(model, variables: List[str], horizon: int, shocks: Dict[str, float]) -> pd.DataFrame:
    n = len(variables)
    p = model.lag_order
    rows = []
    for shock_var, shock_size in shocks.items():
        if shock_var not in variables:
            continue
        shock_vec = np.zeros(n, dtype=float)
        shock_vec[variables.index(shock_var)] = float(shock_size)
        irf_hist = np.zeros((p, n), dtype=float)
        irf_hist[-1, :] = shock_vec
        for h in range(1, horizon + 1):
            response = np.zeros(n, dtype=float)
            for lag in range(1, p + 1):
                response += model.coefs[lag - 1, :, :] @ irf_hist[-lag, :]
            rows.append(
                {
                    "shock_variable": shock_var,
                    "horizon_q": h,
                    **{v: response[i] for i, v in enumerate(variables)},
                }
            )
            irf_hist = np.vstack([irf_hist, response])
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_dir(args.output_dir)

    regime_id, regime = resolve_regime(config, args.regime_id)
    short_horizon = int(regime["short_quarters"])
    medium_horizon = int(regime["medium_quarters"])
    long_start = int(regime["long_start_quarter"])
    total_horizon = int(config.get("horizons", {}).get("total_quarters", 80))

    data_path = config["inputs"]["model_panel_csv"]
    panel = pd.read_csv(data_path, parse_dates=["quarter_end"]).sort_values("quarter_end")

    variables = config["model"]["variables"]
    var_benchmark_vars = config["model"].get("var_benchmark_variables", variables)
    panel = panel[["quarter_end"] + variables].dropna().copy()
    y_full = panel[variables].to_numpy(dtype=float)
    y_bench = panel[var_benchmark_vars].to_numpy(dtype=float)
    last_date = panel["quarter_end"].iloc[-1]

    lag_candidates = [int(x) for x in config["model"]["lag_candidates"]]
    lag_model = select_var_lag(y_bench, var_benchmark_vars, lag_candidates)

    level_vars = [bool(config["model"]["level_variable_flags"].get(v, False)) for v in variables]
    bvar_cfg = config["model"]["bvar_hyperparams"]
    bvar_model, bvar_cov, _ = fit_bvar_minnesota(
        y=y_full,
        p=lag_model.lag_order,
        variables=variables,
        level_vars=level_vars,
        lambda1=float(bvar_cfg["lambda1"]),
        lambda2=float(bvar_cfg["lambda2"]),
        lambda3=float(bvar_cfg["lambda3"]),
        lambda4=float(bvar_cfg["lambda4"]),
    )

    history_bvar = y_full[-bvar_model.lag_order :, :]
    bvar_point = deterministic_forecast(bvar_model, history_bvar, total_horizon)

    validation_cfg = config.get("validation", {})
    scoring_draws = int(validation_cfg.get("scoring_draws", config["model"].get("interval_simulations", 600)))
    ar_p_candidates = [int(x) for x in validation_cfg.get("ar_p_candidates", [1, 2, 3, 4, 5, 6, 7, 8])]

    bvar_seed = stable_seed("forecast", "BVAR", regime_id, scoring_draws)
    bvar_draws_short = simulate_var_draws(
        model=bvar_model,
        history=history_bvar,
        horizon=short_horizon,
        n_sims=scoring_draws,
        rng_seed=bvar_seed,
    )

    # Diagnostics-only VAR intervals on benchmark subset.
    var_model = lag_model
    var_hist = y_bench[-var_model.lag_order :, :]
    var_seed = stable_seed("forecast", "VAR", regime_id, scoring_draws)
    var_draws_short = simulate_var_draws(
        model=var_model,
        history=var_hist,
        horizon=short_horizon,
        n_sims=scoring_draws,
        rng_seed=var_seed,
    )

    n_vars = len(variables)
    ar_point = np.zeros((short_horizon, n_vars), dtype=float)
    ar_draws = np.zeros((scoring_draws, short_horizon, n_vars), dtype=float)
    rw_point = np.zeros((short_horizon, n_vars), dtype=float)
    rw_draws = np.zeros((scoring_draws, short_horizon, n_vars), dtype=float)

    for j, var in enumerate(variables):
        series = y_full[:, j]
        ar_model = select_ar_order(series, ar_p_candidates)
        ar_point[:, j] = ar_point_forecast(ar_model, series, short_horizon)
        ar_seed = stable_seed("forecast", "AR", var, regime_id, ar_model.lag_order, scoring_draws)
        ar_draws[:, :, j] = simulate_ar_draws(ar_model, series, short_horizon, scoring_draws, ar_seed)

        rw_point[:, j] = rw_point_forecast(series[-1], short_horizon)
        diff = np.diff(series)
        rw_sigma = float(np.std(diff, ddof=1)) if diff.shape[0] > 1 else 1e-6
        rw_seed = stable_seed("forecast", "RW", var, regime_id, scoring_draws)
        rw_draws[:, :, j] = simulate_rw_draws(series[-1], rw_sigma, short_horizon, scoring_draws, rw_seed)

    champion_map = load_champion_map(args.champion_map, variables)
    if args.disable_champion:
        champion_map = default_champion_map(variables)

    ensemble_temp = float(validation_cfg.get("ensemble_temperature", 0.20))
    champion_point, champion_draws = combine_short_horizon(
        variables=variables,
        short_horizon=short_horizon,
        champion_map=champion_map,
        bvar_point=bvar_point,
        bvar_draws=bvar_draws_short,
        ar_point=ar_point,
        ar_draws=ar_draws,
        rw_point=rw_point,
        rw_draws=rw_draws,
        full_history=y_full,
        ar_p_candidates=ar_p_candidates,
        ensemble_temp=ensemble_temp,
        seed_namespace=f"forecast:{regime_id}",
    )

    chosen_path = bvar_point.copy()
    chosen_path[:short_horizon, :] = champion_point

    if "assumption_sets" in config:
        if args.assumption_set not in config["assumption_sets"]:
            raise RuntimeError(f"assumption_set '{args.assumption_set}' missing in config.assumption_sets")
        structural_assumptions = config["assumption_sets"][args.assumption_set]
    else:
        structural_assumptions = config.get("assumptions", {})

    long_run_regime_params = config.get("long_run_regime_params", {})
    if not long_run_regime_params:
        # Legacy compatibility fallback.
        legacy_assump = config.get("assumptions", {})
        long_run_regime_params = {
            "hy_spread_long_run_median": float(legacy_assump.get("hy_spread_anchor", 3.8)),
            "hy_spread_tail_q90": float(legacy_assump.get("hy_spread_anchor", 3.8) + 2.0),
            "housing_supply_elasticity_drag": float(legacy_assump.get("housing_supply_elasticity_drag", 0.5)),
            "mortgage_spread_30y": float(legacy_assump.get("mortgage_spread_30y", 1.7)),
        }

    anchor = build_anchor_vector(
        baseline_path=chosen_path,
        variables=variables,
        structural_assumptions=structural_assumptions,
        long_run_regime_params=long_run_regime_params,
    )
    if bool(regime.get("no_bridge", False)):
        baseline = chosen_path.copy()
    else:
        baseline = bridge_to_anchors(
            model_path=chosen_path.copy(),
            anchor=anchor,
            short_horizon=short_horizon,
            medium_horizon=medium_horizon,
            force_continuity_short_to_bridge=bool(
                validation_cfg.get("boundary", {}).get("force_continuity_short_to_bridge", True)
            ),
        )

    scenarios = {"Baseline": baseline}
    for sc_name, sc_cfg in config.get("scenarios", {}).items():
        scenarios[sc_name] = apply_scenario_envelope(
            baseline=baseline,
            variables=variables,
            scenario_cfg=sc_cfg,
            long_start_quarter=long_start,
            anchor=anchor,
        )

    forecast_dates = make_forecast_dates(last_date, total_horizon)

    rows = []
    for sc_name, mat in scenarios.items():
        for h in range(total_horizon):
            rows.append(
                {
                    "scenario": sc_name,
                    "forecast_q": h + 1,
                    "quarter_end": forecast_dates[h].date().isoformat(),
                    **{v: float(mat[h, i]) for i, v in enumerate(variables)},
                }
            )
    pd.DataFrame(rows).to_csv(
        os.path.join(args.output_dir, "macro_forecast_paths.csv"),
        index=False,
        float_format="%.6f",
    )
    write_pd_regressor_exports(
        output_dir=args.output_dir,
        pd_export_cfg=config.get("outputs", {}).get("pd_export", {}),
        validation_cfg=validation_cfg,
        scenarios=scenarios,
        variables=variables,
        forecast_dates=forecast_dates,
        last_observed_date=last_date,
        history_levels=y_full,
        champion_map=champion_map,
    )

    quantiles = [0.05, 0.5, 0.95]
    var_q = quantiles_from_draws(var_draws_short, quantiles)
    bvar_q = quantiles_from_draws(bvar_draws_short, quantiles)
    champion_q = quantiles_from_draws(champion_draws, quantiles)

    int_rows = []
    var_idx = {v: i for i, v in enumerate(var_benchmark_vars)}
    full_idx = {v: i for i, v in enumerate(variables)}

    for model_name, qmap in [("VAR", var_q), ("BVAR", bvar_q), ("CHAMPION", champion_q)]:
        for h in range(short_horizon):
            for q in quantiles:
                payload = {v: np.nan for v in variables}
                if model_name == "VAR":
                    for v in var_benchmark_vars:
                        payload[v] = float(qmap[q][h, var_idx[v]])
                else:
                    for v in variables:
                        payload[v] = float(qmap[q][h, full_idx[v]])
                int_rows.append(
                    {
                        "model": model_name,
                        "forecast_q": h + 1,
                        "quarter_end": forecast_dates[h].date().isoformat(),
                        "quantile": q,
                        **payload,
                    }
                )

    pd.DataFrame(int_rows).to_csv(
        os.path.join(args.output_dir, "macro_forecast_short_horizon_intervals.csv"),
        index=False,
        float_format="%.6f",
    )

    anchors_payload = {
        "assumption_set": args.assumption_set,
        "variables": variables,
        "anchor_values": {v: float(anchor[i]) for i, v in enumerate(variables)},
        "structural_assumptions": structural_assumptions,
        "long_run_regime_params": long_run_regime_params,
    }
    with open(os.path.join(args.output_dir, "macro_anchor_assumptions.json"), "w", encoding="utf-8") as f:
        json.dump(anchors_payload, f, indent=2)

    diagnostics = {
        "contract_version": str(validation_cfg.get("contract_version", "v4.1")),
        "regime_id": regime_id,
        "assumption_set": args.assumption_set,
        "data_rows_used": int(panel.shape[0]),
        "var_benchmark": {
            "lag_order": int(var_model.lag_order),
            "bic": float(var_model.bic),
            "stable": bool(var_model.stable),
            "n_variables": int(len(var_benchmark_vars)),
            "variables": var_benchmark_vars,
        },
        "bvar": {
            "lag_order": int(bvar_model.lag_order),
            "bic_like": float(bvar_model.bic),
            "stable": bool(bvar_model.stable),
            "hyperparams": bvar_cfg,
        },
        "regime": {
            "short_quarters": short_horizon,
            "medium_quarters": medium_horizon,
            "long_start_quarter": long_start,
            "total_quarters": total_horizon,
        },
        "validation": {
            "scoring_draws": int(scoring_draws),
            "ensemble_temperature": float(ensemble_temp),
            "ensemble_gap_pct": float(validation_cfg.get("ensemble_gap_pct", 1.5)),
        },
        "champion_map": {
            "path": args.champion_map,
            "hash": _hash_json(champion_map),
        },
        "generated_scenarios": ["Baseline"] + list(config.get("scenarios", {}).keys()),
    }

    with open(os.path.join(args.output_dir, "macro_model_diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    irf_h = int(config.get("outputs", {}).get("irf_horizon_quarters", 20))
    irf_shocks = config.get("outputs", {}).get(
        "irf_shocks",
        {"mortgage30_rate": 1.0, "hpi_growth_yoy": -5.0, "unemployment_rate": 2.0},
    )
    irf_df = build_impulse_responses(var_model, var_benchmark_vars, irf_h, irf_shocks)
    irf_df.to_csv(
        os.path.join(args.output_dir, "macro_impulse_responses.csv"),
        index=False,
        float_format="%.6f",
    )


if __name__ == "__main__":
    main()
