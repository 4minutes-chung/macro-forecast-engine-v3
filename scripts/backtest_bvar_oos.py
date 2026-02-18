#!/usr/bin/env python3
"""Rolling OOS backtest with BVAR, RW, and AR baselines under shared lag logic (V4.1-ready)."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from macro_model_core import (
    apply_transform,
    ar_point_forecast,
    crps_from_draws,
    dm_test_hac_hln,
    empirical_coverage,
    fit_bvar_minnesota,
    interval_width,
    inverse_transform_draws,
    rw_point_forecast,
    select_ar_order,
    select_var_lag,
    simulate_ar_draws,
    simulate_rw_draws,
    simulate_var_draws,
    stable_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest macro models over rolling OOS origins.")
    parser.add_argument("--config", default="macro_engine_config.json", help="Config JSON path.")
    parser.add_argument(
        "--input",
        default="data/macro_panel_quarterly_model.csv",
        help="Model panel CSV path.",
    )
    parser.add_argument(
        "--output",
        default="outputs/macro_engine/validation/backtest_metrics.csv",
        help="Output aggregate metrics CSV path.",
    )
    parser.add_argument(
        "--raw-output",
        default="",
        help="Optional raw per-origin metric output CSV path.",
    )
    parser.add_argument(
        "--seed-output",
        default="",
        help="Optional seed registry output CSV path.",
    )
    parser.add_argument("--regime-id", default="champion_a", help="Regime candidate ID.")
    parser.add_argument(
        "--variable-set",
        choices=["benchmark", "full", "pd_core"],
        default="full",
        help="Variable set to evaluate.",
    )
    parser.add_argument("--min-train", type=int, default=-1, help="Minimum training rows before first origin.")
    parser.add_argument("--max-origins", type=int, default=-1, help="Maximum OOS origins (latest kept).")
    parser.add_argument("--origin-stride", type=int, default=-1, help="Evaluate every kth origin.")
    parser.add_argument("--horizon", type=int, default=-1, help="Override forecast horizon.")
    parser.add_argument("--scoring-draws", type=int, default=-1, help="Draws for CRPS/coverage/width.")
    parser.add_argument("--namespace", default="backtest", help="Seed namespace.")
    return parser.parse_args()


def _resolve_regime(config: Dict, regime_id: str) -> Dict:
    regime_cfg = config.get("regime_candidates", {})
    if regime_id in regime_cfg:
        return regime_cfg[regime_id]
    horizons = config.get("horizons", {})
    if not horizons:
        raise RuntimeError(f"Regime '{regime_id}' not found and no legacy horizons in config.")
    return {
        "short_quarters": int(horizons["short_quarters"]),
        "medium_quarters": int(horizons["medium_quarters"]),
        "long_start_quarter": int(horizons["long_start_quarter"]),
    }


def _series_draw_metrics(draws: np.ndarray, actual: float) -> Tuple[float, float, float, float]:
    return (
        crps_from_draws(draws, actual),
        empirical_coverage(draws, actual, 0.50),
        empirical_coverage(draws, actual, 0.90),
        interval_width(draws, 0.90),
    )


def _pd_lag_quarters(validation_cfg: Dict, target_name: str, default_lag: int = 2) -> int:
    defaults = validation_cfg.get("pd_delta_defaults", {})
    overrides = validation_cfg.get("pd_delta_overrides", {})
    if target_name in overrides and isinstance(overrides[target_name], dict):
        return int(overrides[target_name].get("lag_quarters", default_lag))
    if target_name in defaults and isinstance(defaults[target_name], dict):
        return int(defaults[target_name].get("lag_quarters", default_lag))
    return int(default_lag)


def _lag_value_from_forecast(series_forecast: np.ndarray, history_levels: np.ndarray, h: int, lag_q: int) -> float:
    back = h - lag_q
    if back >= 1:
        return float(series_forecast[back - 1])
    if history_levels.shape[0] >= lag_q:
        hist_idx = back + lag_q - 1
        if 0 <= hist_idx < history_levels.shape[0]:
            return float(history_levels[hist_idx])
    return float("nan")


def run_backtest(
    config: Dict,
    input_path: str,
    regime_id: str,
    variable_set: str,
    min_train: Optional[int] = None,
    max_origins: Optional[int] = None,
    origin_stride: Optional[int] = None,
    horizon: Optional[int] = None,
    scoring_draws: Optional[int] = None,
    namespace: str = "backtest",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = pd.read_csv(input_path, parse_dates=["quarter_end"]).sort_values("quarter_end")
    full_variables = config["model"]["variables"]
    benchmark_variables = config["model"].get("var_benchmark_variables", full_variables)
    pd_core_variables = config["model"].get(
        "pd_core_variables",
        ["unemployment_rate", "ust10_rate", "hpi_growth_yoy", "fed_funds_rate", "core_cpi_yoy", "high_yield_spread"],
    )
    if variable_set == "benchmark":
        variables = benchmark_variables
    elif variable_set == "pd_core":
        variables = [v for v in pd_core_variables if v in full_variables]
    else:
        variables = full_variables
    panel = panel[["quarter_end"] + variables].dropna().copy()

    y_all = panel[variables].to_numpy(dtype=float)
    dates = panel["quarter_end"].to_numpy()
    var_benchmark_vars = [v for v in benchmark_variables if v in variables]

    lag_candidates = [int(x) for x in config["model"]["lag_candidates"]]
    bvar_cfg = config["model"]["bvar_hyperparams"]
    level_flags = [bool(config["model"]["level_variable_flags"].get(v, False)) for v in variables]

    validation_cfg = config.get("validation", {})
    ar_p_candidates = [int(x) for x in validation_cfg.get("ar_p_candidates", [1, 2, 3, 4, 5, 6, 7, 8])]
    draws_n = int(scoring_draws or validation_cfg.get("scoring_draws", 200))

    regime = _resolve_regime(config, regime_id)
    h_max = int(horizon) if horizon and horizon > 0 else int(regime["short_quarters"])

    p_max = max(max(lag_candidates), max(ar_p_candidates))
    min_train_eff = int(min_train) if min_train and min_train > 0 else int(validation_cfg.get("min_train", 80))
    min_train_eff = max(min_train_eff, p_max + 12)
    if y_all.shape[0] < min_train_eff + h_max + 4:
        raise RuntimeError("Not enough rows for requested min_train/horizon.")

    stride = int(origin_stride) if origin_stride and origin_stride > 0 else int(validation_cfg.get("origin_stride", 1))
    stride = max(1, stride)
    origin_start = min_train_eff
    origin_end = y_all.shape[0] - h_max
    origins = list(range(origin_start, origin_end + 1, stride))
    max_orig_eff = int(max_origins) if max_origins and max_origins > 0 else int(validation_cfg.get("max_origins", 0))
    if max_orig_eff > 0 and len(origins) > max_orig_eff:
        origins = origins[-max_orig_eff:]

    raw_rows: List[Dict] = []
    seed_rows: List[Dict] = []

    for idx_origin, origin in enumerate(origins, start=1):
        if idx_origin % 5 == 0:
            print(f"Backtest {regime_id}: {idx_origin}/{len(origins)} origins", flush=True)

        y_train = y_all[:origin, :]
        y_train_bench = panel[var_benchmark_vars].to_numpy(dtype=float)[:origin, :]

        lag_model = select_var_lag(y_train_bench, var_benchmark_vars, lag_candidates)
        p = lag_model.lag_order

        bvar_model, bvar_cov, _ = fit_bvar_minnesota(
            y=y_train,
            p=p,
            variables=variables,
            level_vars=level_flags,
            lambda1=float(bvar_cfg["lambda1"]),
            lambda2=float(bvar_cfg["lambda2"]),
            lambda3=float(bvar_cfg["lambda3"]),
            lambda4=float(bvar_cfg["lambda4"]),
        )
        hist_bvar = y_train[-p:, :]
        bvar_seed = stable_seed(namespace, "BVAR", regime_id, str(pd.to_datetime(dates[origin - 1]).date()), draws_n)
        bvar_draws = simulate_var_draws(
            model=bvar_model,
            history=hist_bvar,
            horizon=h_max,
            n_sims=draws_n,
            rng_seed=bvar_seed,
        )
        bvar_point = np.mean(bvar_draws, axis=0)
        seed_rows.append(
            {
                "regime_id": regime_id,
                "origin_idx": int(origin),
                "origin_date": pd.to_datetime(dates[origin - 1]).date().isoformat(),
                "model": "BVAR",
                "seed": int(bvar_seed),
                "draws": int(draws_n),
            }
        )

        ar_points = np.zeros((h_max, len(variables)), dtype=float)
        ar_draws = np.zeros((draws_n, h_max, len(variables)), dtype=float)
        rw_points = np.zeros((h_max, len(variables)), dtype=float)
        rw_draws = np.zeros((draws_n, h_max, len(variables)), dtype=float)

        for j, var in enumerate(variables):
            series = y_train[:, j]
            ar_model = select_ar_order(series, ar_p_candidates)
            ar_points[:, j] = ar_point_forecast(ar_model, series, h_max)
            ar_seed = stable_seed(
                namespace,
                "AR",
                var,
                regime_id,
                str(pd.to_datetime(dates[origin - 1]).date()),
                ar_model.lag_order,
                draws_n,
            )
            ar_draws[:, :, j] = simulate_ar_draws(ar_model, series, h_max, draws_n, ar_seed)
            seed_rows.append(
                {
                    "regime_id": regime_id,
                    "origin_idx": int(origin),
                    "origin_date": pd.to_datetime(dates[origin - 1]).date().isoformat(),
                    "model": "AR",
                    "variable": var,
                    "seed": int(ar_seed),
                    "draws": int(draws_n),
                    "ar_lag": int(ar_model.lag_order),
                }
            )

            rw_points[:, j] = rw_point_forecast(series[-1], h_max)
            diff = np.diff(series)
            rw_sigma = float(np.std(diff, ddof=1)) if diff.shape[0] > 1 else 1e-6
            rw_seed = stable_seed(
                namespace,
                "RW",
                var,
                regime_id,
                str(pd.to_datetime(dates[origin - 1]).date()),
                draws_n,
            )
            rw_draws[:, :, j] = simulate_rw_draws(series[-1], rw_sigma, h_max, draws_n, rw_seed)
            seed_rows.append(
                {
                    "regime_id": regime_id,
                    "origin_idx": int(origin),
                    "origin_date": pd.to_datetime(dates[origin - 1]).date().isoformat(),
                    "model": "RW",
                    "variable": var,
                    "seed": int(rw_seed),
                    "draws": int(draws_n),
                }
            )

        actual_mat = y_all[origin : origin + h_max, :]
        models = {
            "BVAR": (bvar_point, bvar_draws),
            "AR": (ar_points, ar_draws),
            "RW": (rw_points, rw_draws),
        }

        for model_name, (point_mat, draw_mat) in models.items():
            for h in range(1, h_max + 1):
                for j, var in enumerate(variables):
                    actual = float(actual_mat[h - 1, j])
                    pred = float(point_mat[h - 1, j])
                    err = actual - pred
                    draws = draw_mat[:, h - 1, j]
                    crps, cov50, cov90, width90 = _series_draw_metrics(draws, actual)
                    raw_rows.append(
                        {
                            "regime_id": regime_id,
                            "origin_idx": int(origin),
                            "origin_date": pd.to_datetime(dates[origin - 1]).date().isoformat(),
                            "forecast_date": pd.to_datetime(dates[origin + h - 1]).date().isoformat(),
                            "model": model_name,
                            "variable": var,
                            "horizon_q": int(h),
                            "pred": float(pred),
                            "actual": float(actual),
                            "se": float(err**2),
                            "ae": float(abs(err)),
                            "crps": float(crps),
                            "coverage_50": float(cov50),
                            "coverage_90": float(cov90),
                            "width90": float(width90),
                            "scoring_domain": "level",
                            "transform_id": "level",
                        }
                    )

            if bool(validation_cfg.get("pd_convenience_targets_enabled", True)):
                var_to_idx = {v: i for i, v in enumerate(variables)}
                u_idx = var_to_idx.get("unemployment_rate")
                r_idx = var_to_idx.get("ust10_rate")
                hpi_idx = var_to_idx.get("hpi_growth_yoy")

                # Alias HPI target name used by PD-facing outputs.
                if hpi_idx is not None:
                    for h in range(1, h_max + 1):
                        hpi_actual = float(actual_mat[h - 1, hpi_idx])
                        hpi_pred = float(point_mat[h - 1, hpi_idx])
                        hpi_err = hpi_actual - hpi_pred
                        hpi_draws = draw_mat[:, h - 1, hpi_idx]
                        crps, cov50, cov90, width90 = _series_draw_metrics(hpi_draws, hpi_actual)
                        raw_rows.append(
                            {
                                "regime_id": regime_id,
                                "origin_idx": int(origin),
                                "origin_date": pd.to_datetime(dates[origin - 1]).date().isoformat(),
                                "forecast_date": pd.to_datetime(dates[origin + h - 1]).date().isoformat(),
                                "model": model_name,
                                "variable": "hpi_yoy",
                                "horizon_q": int(h),
                                "pred": float(hpi_pred),
                                "actual": float(hpi_actual),
                                "se": float(hpi_err**2),
                                "ae": float(abs(hpi_err)),
                                "crps": float(crps),
                                "coverage_50": float(cov50),
                                "coverage_90": float(cov90),
                                "width90": float(width90),
                                "scoring_domain": "level",
                                "transform_id": "level",
                            }
                        )

                if u_idx is not None:
                    lag_u = _pd_lag_quarters(validation_cfg, "du_6m", default_lag=2)
                    u_hist = y_train[:, u_idx][-lag_u:]
                    for h in range(1, h_max + 1):
                        u_now_actual = float(actual_mat[h - 1, u_idx])
                        u_now_pred = float(point_mat[h - 1, u_idx])
                        u_now_draws = draw_mat[:, h - 1, u_idx]
                        u_lag_actual = _lag_value_from_forecast(actual_mat[:, u_idx], u_hist, h, lag_u)
                        u_lag_pred = _lag_value_from_forecast(point_mat[:, u_idx], u_hist, h, lag_u)
                        if np.isfinite(u_lag_actual) and np.isfinite(u_lag_pred):
                            u_lag_draws = np.full_like(u_now_draws, u_lag_pred)
                            if h - lag_u >= 1:
                                u_lag_draws = draw_mat[:, h - lag_u - 1, u_idx]
                            du_draws = u_now_draws - u_lag_draws
                            du_actual = float(u_now_actual - u_lag_actual)
                            du_pred = float(u_now_pred - u_lag_pred)
                            du_err = du_actual - du_pred
                            crps, cov50, cov90, width90 = _series_draw_metrics(du_draws, du_actual)
                            raw_rows.append(
                                {
                                    "regime_id": regime_id,
                                    "origin_idx": int(origin),
                                    "origin_date": pd.to_datetime(dates[origin - 1]).date().isoformat(),
                                    "forecast_date": pd.to_datetime(dates[origin + h - 1]).date().isoformat(),
                                    "model": model_name,
                                    "variable": "du_6m",
                                    "horizon_q": int(h),
                                    "pred": float(du_pred),
                                    "actual": float(du_actual),
                                    "se": float(du_err**2),
                                    "ae": float(abs(du_err)),
                                    "crps": float(crps),
                                    "coverage_50": float(cov50),
                                    "coverage_90": float(cov90),
                                    "width90": float(width90),
                                    "scoring_domain": "level",
                                    "transform_id": "derived",
                                }
                            )

                if r_idx is not None:
                    lag_r = _pd_lag_quarters(validation_cfg, "d10y_6m", default_lag=2)
                    r_hist = y_train[:, r_idx][-lag_r:]
                    for h in range(1, h_max + 1):
                        r_now_actual = float(actual_mat[h - 1, r_idx])
                        r_now_pred = float(point_mat[h - 1, r_idx])
                        r_now_draws = draw_mat[:, h - 1, r_idx]
                        r_lag_actual = _lag_value_from_forecast(actual_mat[:, r_idx], r_hist, h, lag_r)
                        r_lag_pred = _lag_value_from_forecast(point_mat[:, r_idx], r_hist, h, lag_r)
                        if np.isfinite(r_lag_actual) and np.isfinite(r_lag_pred):
                            r_lag_draws = np.full_like(r_now_draws, r_lag_pred)
                            if h - lag_r >= 1:
                                r_lag_draws = draw_mat[:, h - lag_r - 1, r_idx]
                            d10_draws = r_now_draws - r_lag_draws
                            d10_actual = float(r_now_actual - r_lag_actual)
                            d10_pred = float(r_now_pred - r_lag_pred)
                            d10_err = d10_actual - d10_pred
                            crps, cov50, cov90, width90 = _series_draw_metrics(d10_draws, d10_actual)
                            raw_rows.append(
                                {
                                    "regime_id": regime_id,
                                    "origin_idx": int(origin),
                                    "origin_date": pd.to_datetime(dates[origin - 1]).date().isoformat(),
                                    "forecast_date": pd.to_datetime(dates[origin + h - 1]).date().isoformat(),
                                    "model": model_name,
                                    "variable": "d10y_6m",
                                    "horizon_q": int(h),
                                    "pred": float(d10_pred),
                                    "actual": float(d10_actual),
                                    "se": float(d10_err**2),
                                    "ae": float(abs(d10_err)),
                                    "crps": float(crps),
                                    "coverage_50": float(cov50),
                                    "coverage_90": float(cov90),
                                    "width90": float(width90),
                                    "scoring_domain": "level",
                                    "transform_id": "derived",
                                }
                            )

    raw_df = pd.DataFrame(raw_rows)
    seed_df = pd.DataFrame(seed_rows)

    agg = (
        raw_df.groupby(["regime_id", "model", "variable", "horizon_q"], as_index=False)
        .agg(
            n_oos=("se", "size"),
            mse=("se", "mean"),
            mae=("ae", "mean"),
            crps=("crps", "mean"),
            coverage_50=("coverage_50", "mean"),
            coverage_90=("coverage_90", "mean"),
            width90=("width90", "mean"),
        )
    )
    agg["rmse"] = np.sqrt(agg["mse"])

    rw_base = agg[agg["model"] == "RW"][["regime_id", "variable", "horizon_q", "rmse", "crps", "width90"]].rename(
        columns={"rmse": "rmse_rw", "crps": "crps_rw", "width90": "width90_rw"}
    )
    agg = agg.merge(rw_base, on=["regime_id", "variable", "horizon_q"], how="left")

    eps = float(config.get("validation", {}).get("crps_rw_epsilon", 1e-8))
    agg["rrmse_vs_rw"] = agg["rmse"] / np.maximum(agg["rmse_rw"], eps)
    agg["crps_gain_pct_vs_rw"] = 100.0 * (agg["crps_rw"] - agg["crps"]) / np.maximum(agg["crps_rw"], eps)
    agg["width_ratio_vs_rw"] = agg["width90"] / np.maximum(agg["width90_rw"], eps)

    dm_rows: List[Dict] = []
    for var in sorted(raw_df["variable"].unique()):
        for h in sorted(raw_df["horizon_q"].unique()):
            base = raw_df[(raw_df["model"] == "RW") & (raw_df["variable"] == var) & (raw_df["horizon_q"] == h)]
            base = base.sort_values("origin_idx")
            if base.empty:
                continue
            for model_name in ["BVAR", "AR"]:
                cur = raw_df[(raw_df["model"] == model_name) & (raw_df["variable"] == var) & (raw_df["horizon_q"] == h)]
                cur = cur.sort_values("origin_idx")
                merged = cur[["origin_idx", "se", "crps"]].merge(
                    base[["origin_idx", "se", "crps"]],
                    on="origin_idx",
                    suffixes=("_model", "_rw"),
                )
                if merged.empty:
                    continue
                dm_mse_stat, dm_mse_p, hac_lags = dm_test_hac_hln(
                    merged["se_model"].to_numpy(), merged["se_rw"].to_numpy(), int(h)
                )
                dm_crps_stat, dm_crps_p, _ = dm_test_hac_hln(
                    merged["crps_model"].to_numpy(), merged["crps_rw"].to_numpy(), int(h)
                )
                dm_rows.append(
                    {
                        "regime_id": regime_id,
                        "model": model_name,
                        "variable": var,
                        "horizon_q": int(h),
                        "dm_stat_mse_vs_rw": dm_mse_stat,
                        "dm_pvalue_mse_vs_rw": dm_mse_p,
                        "dm_stat_crps_vs_rw": dm_crps_stat,
                        "dm_pvalue_crps_vs_rw": dm_crps_p,
                        "hac_lags": int(hac_lags),
                    }
                )

    dm_df = pd.DataFrame(dm_rows)
    if not dm_df.empty:
        agg = agg.merge(dm_df, on=["regime_id", "model", "variable", "horizon_q"], how="left")
    else:
        agg["dm_stat_mse_vs_rw"] = np.nan
        agg["dm_pvalue_mse_vs_rw"] = np.nan
        agg["dm_stat_crps_vs_rw"] = np.nan
        agg["dm_pvalue_crps_vs_rw"] = np.nan
        agg["hac_lags"] = np.nan

    keep_cols = [
        "regime_id",
        "model",
        "variable",
        "horizon_q",
        "n_oos",
        "rmse",
        "mae",
        "crps",
        "coverage_50",
        "coverage_90",
        "width90",
        "rrmse_vs_rw",
        "crps_gain_pct_vs_rw",
        "width_ratio_vs_rw",
        "dm_stat_mse_vs_rw",
        "dm_pvalue_mse_vs_rw",
        "dm_stat_crps_vs_rw",
        "dm_pvalue_crps_vs_rw",
        "hac_lags",
    ]
    agg = agg[keep_cols].sort_values(["regime_id", "model", "variable", "horizon_q"]).reset_index(drop=True)
    raw_df = raw_df.sort_values(["regime_id", "model", "variable", "horizon_q", "origin_idx"]).reset_index(drop=True)
    seed_df = seed_df.sort_values(["regime_id", "origin_idx", "model", "variable"], na_position="last").reset_index(drop=True)

    return agg, raw_df, seed_df


def _mkdir_for(path: str):
    if not path:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    agg, raw_df, seed_df = run_backtest(
        config=config,
        input_path=args.input,
        regime_id=args.regime_id,
        variable_set=args.variable_set,
        min_train=(args.min_train if args.min_train > 0 else None),
        max_origins=(args.max_origins if args.max_origins > 0 else None),
        origin_stride=(args.origin_stride if args.origin_stride > 0 else None),
        horizon=(args.horizon if args.horizon > 0 else None),
        scoring_draws=(args.scoring_draws if args.scoring_draws > 0 else None),
        namespace=args.namespace,
    )

    _mkdir_for(args.output)
    agg.to_csv(args.output, index=False)

    if args.raw_output:
        _mkdir_for(args.raw_output)
        raw_df.to_csv(args.raw_output, index=False)

    if args.seed_output:
        _mkdir_for(args.seed_output)
        seed_df.to_csv(args.seed_output, index=False)

    n_origins = int(raw_df["origin_idx"].nunique()) if not raw_df.empty else 0
    print(f"Wrote backtest metrics: {args.output}")
    print(f"Regime={args.regime_id}, origins={n_origins}, models={sorted(agg['model'].unique().tolist())}")


if __name__ == "__main__":
    main()
