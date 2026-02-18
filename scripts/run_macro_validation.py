#!/usr/bin/env python3
"""V3 freeze validation orchestrator: PD-level-target gating, calibration, boundary diagnostics, promotion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from backtest_bvar_oos import run_backtest
from macro_model_core import (
    apply_transform,
    ar_point_forecast,
    crps_from_draws,
    empirical_coverage,
    interval_width,
    inverse_transform_draws,
    rw_point_forecast,
    select_ar_order,
    simulate_ar_draws,
    simulate_rw_draws,
    stable_seed,
    transform_candidates_for_variable,
)


@dataclass
class GateResult:
    passed: bool
    details: Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run macro validation and tiered gate checks.")
    parser.add_argument("--config", default="macro_engine_config.json", help="Config JSON path.")
    parser.add_argument(
        "--input",
        default="data/macro_panel_quarterly_model.csv",
        help="Model panel CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/macro_engine/validation",
        help="Validation output directory.",
    )
    parser.add_argument(
        "--verbose-validation",
        action="store_true",
        help="Emit verbose artifacts.",
    )
    parser.add_argument(
        "--strict-validation",
        action="store_true",
        help="Exit non-zero on strict profile fail.",
    )
    parser.add_argument(
        "--champion-map-output",
        default="outputs/macro_engine/champion_map.json",
        help="Champion map output path.",
    )
    parser.add_argument("--regime-incumbent", default="champion_a", help="Incumbent regime id.")
    parser.add_argument("--regime-challenger", default="champion_b", help="Challenger regime id.")
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resolve_regime(config: Dict, regime_id: str) -> Dict:
    candidates = config.get("regime_candidates", {})
    if regime_id in candidates:
        return candidates[regime_id]
    if "horizons" in config:
        return config["horizons"]
    raise RuntimeError(f"Regime id not found: {regime_id}")


def bucket_for_h(h: int) -> str:
    return "bucket_1_4" if h <= 4 else "bucket_5_12"


def bucket_horizons(bucket_name: str) -> Tuple[int, int]:
    return (1, 4) if bucket_name == "bucket_1_4" else (5, 12)


def summarize_seed_hash(seed_df: pd.DataFrame) -> str:
    if seed_df.empty:
        return ""
    cols = [c for c in ["regime_id", "origin_idx", "model", "variable", "seed", "draws"] if c in seed_df.columns]
    payload = seed_df[cols].fillna("").astype(str).agg("|".join, axis=1).str.cat(sep="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _aggregate_from_raw(raw_df: pd.DataFrame, regime_id: str) -> pd.DataFrame:
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
    rw = agg[agg["model"] == "RW"]["variable horizon_q rmse crps width90".split()].rename(
        columns={"rmse": "rmse_rw", "crps": "crps_rw", "width90": "width90_rw"}
    )
    agg = agg.merge(rw, on=["variable", "horizon_q"], how="left")
    agg["regime_id"] = regime_id
    agg["rrmse_vs_rw"] = agg["rmse"] / np.maximum(agg["rmse_rw"], 1e-8)
    agg["crps_gain_pct_vs_rw"] = 100.0 * (agg["crps_rw"] - agg["crps"]) / np.maximum(agg["crps_rw"], 1e-8)
    agg["width_ratio_vs_rw"] = agg["width90"] / np.maximum(agg["width90_rw"], 1e-8)
    return agg[
        [
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
        ]
    ]


def get_threshold_profiles(validation: Dict) -> Dict:
    defaults = {
        "operational": {
            "coverage90_passrate_min": 0.65,
            "rrmse_h1_h2_max": 1.02,
            "rrmse_h3_h4_max": 1.00,
            "crps_gain_h5_h12_min": 1.5,
            "width_ratio_mean_max": 1.50,
            "width_ratio_per_var_max": 1.80,
            "boundary_median_z_max": 2.0,
            "boundary_max_z_max": 4.0,
        },
        "release": {
            "coverage90_passrate_min": 0.75,
            "rrmse_h1_h2_max": 1.00,
            "rrmse_h3_h4_max": 0.98,
            "crps_gain_h5_h12_min": 3.0,
            "width_ratio_mean_max": 1.35,
            "width_ratio_per_var_max": 1.60,
            "boundary_median_z_max": 1.5,
            "boundary_max_z_max": 2.5,
        },
        "promotion": {
            "crps_gain_h9_h12_min": 5.0,
            "short_horizon_crps_worsen_h1_h4_max": 1.0,
            "boundary_median_delta_max": 0.15,
        },
    }
    user = validation.get("threshold_profiles", {})
    out = {}
    for k, d in defaults.items():
        merged = d.copy()
        merged.update(user.get(k, {}))
        out[k] = merged
    return out


def get_pd_target_mappings(validation: Dict) -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    level_targets = validation.get("pd_level_targets", ["unemployment_rate", "ust10_rate", "hpi_yoy"])
    src_map = validation.get("pd_source_mapping", {"hpi_yoy": "hpi_growth_yoy"})
    canonical_to_source = {str(t): str(src_map.get(t, t)) for t in level_targets}
    source_to_canonical = {src: can for can, src in canonical_to_source.items()}
    return [str(t) for t in level_targets], canonical_to_source, source_to_canonical


def pd_primary_source_targets(validation: Dict) -> List[str]:
    _, canonical_to_source, _ = get_pd_target_mappings(validation)
    return list(canonical_to_source.values())


def canonical_var_name(variable: str, source_to_canonical: Dict[str, str]) -> str:
    return source_to_canonical.get(variable, variable)


def build_champion_map(agg_df: pd.DataFrame, config: Dict, regime_id: str) -> Dict:
    validation = config.get("validation", {})
    gap_pct = float(validation.get("ensemble_gap_pct", 1.5))
    target_cov = float(
        validation.get("pd_calibration", {}).get(
            "target_coverage90",
            validation.get("calibration", {}).get("target_coverage90", 0.90),
        )
    )

    variables = config["model"]["variables"]
    champion = {"contract_version": validation.get("contract_version", "v3.freeze.1"), "regime_id": regime_id, "variables": {}}

    sub = agg_df[agg_df["regime_id"] == regime_id].copy()
    sub = sub[sub["model"].isin(["BVAR", "AR", "RW"]) & sub["horizon_q"].between(1, 12)]

    for var in variables:
        champion["variables"][var] = {}
        var_df = sub[sub["variable"] == var]
        for bucket_name, hr in [("bucket_1_4", (1, 4)), ("bucket_5_12", (5, 12))]:
            bdf = var_df[var_df["horizon_q"].between(hr[0], hr[1])]
            if bdf.empty:
                champion["variables"][var][bucket_name] = {
                    "kind": "single",
                    "model": "BVAR",
                    "transform_id": "level",
                    "scoring_domain": "level",
                    "calibration_scale": 1.0,
                    "coverage90_before": None,
                    "coverage90_after_est": None,
                    "calibration_samples": 0,
                }
                continue

            m = bdf.groupby("model", as_index=False).agg(
                crps=("crps", "mean"),
                coverage_90=("coverage_90", "mean"),
                width_ratio_vs_rw=("width_ratio_vs_rw", "mean"),
            )
            m = m.sort_values("crps").reset_index(drop=True)
            top = m.iloc[0]
            chosen = top
            gap = 999.0
            if m.shape[0] >= 2:
                runner = m.iloc[1]
                gap = 100.0 * (float(runner["crps"]) - float(top["crps"])) / max(float(runner["crps"]), 1e-8)
                if gap < gap_pct:
                    top_cov_dist = abs(float(top["coverage_90"]) - target_cov)
                    run_cov_dist = abs(float(runner["coverage_90"]) - target_cov)
                    if run_cov_dist + 1e-9 < top_cov_dist:
                        chosen = runner
                    elif abs(run_cov_dist - top_cov_dist) <= 1e-9:
                        top_w_dist = abs(float(top["width_ratio_vs_rw"]) - 1.0)
                        run_w_dist = abs(float(runner["width_ratio_vs_rw"]) - 1.0)
                        if run_w_dist + 1e-9 < top_w_dist:
                            chosen = runner

            champion["variables"][var][bucket_name] = {
                "kind": "single",
                "model": str(chosen["model"]),
                "transform_id": "level",
                "scoring_domain": "level",
                "top2_gap_pct": float(gap),
                "coverage90_bucket": float(chosen["coverage_90"]),
                "width_ratio_bucket": float(chosen["width_ratio_vs_rw"]),
                "calibration_scale": 1.0,
                "coverage90_before": None,
                "coverage90_after_est": None,
                "calibration_samples": 0,
            }

    return champion


def apply_special_series_sweep(
    config: Dict,
    panel: pd.DataFrame,
    regime_id: str,
    origin_indices: List[int],
    h_max: int,
    draws_n: int,
) -> pd.DataFrame:
    validation = config.get("validation", {})
    ar_p_candidates = [int(x) for x in validation.get("ar_p_candidates", [1, 2, 3, 4, 5, 6, 7, 8])]
    specials = validation.get("key_vars_special", ["high_yield_spread", "housing_starts_yoy"])

    rows: List[Dict] = []
    for var in specials:
        if var not in panel.columns:
            continue
        y_all = panel[var].to_numpy(dtype=float)
        for transform_id in transform_candidates_for_variable(var):
            for model_name in ["RW", "AR"]:
                for origin in origin_indices:
                    train = y_all[:origin]
                    transformed, meta = apply_transform(train, transform_id)
                    if model_name == "RW":
                        diff = np.diff(transformed)
                        sigma = float(np.std(diff, ddof=1)) if diff.shape[0] > 1 else 1e-6
                        seed = stable_seed("special_sweep", regime_id, var, transform_id, model_name, origin, draws_n)
                        draws_t = simulate_rw_draws(transformed[-1], sigma, h_max, draws_n, seed)
                    else:
                        ar_model = select_ar_order(transformed, ar_p_candidates)
                        seed = stable_seed(
                            "special_sweep",
                            regime_id,
                            var,
                            transform_id,
                            model_name,
                            origin,
                            ar_model.lag_order,
                            draws_n,
                        )
                        draws_t = simulate_ar_draws(ar_model, transformed, h_max, draws_n, seed)

                    draws_level, scoring_domain = inverse_transform_draws(draws_t, float(train[-1]), transform_id, meta)
                    point_level = np.mean(draws_level, axis=0)
                    actual = y_all[origin : origin + h_max]

                    for h in range(1, h_max + 1):
                        e = float(actual[h - 1] - point_level[h - 1])
                        rows.append(
                            {
                                "regime_id": regime_id,
                                "model": model_name,
                                "variable": var,
                                "transform_id": transform_id,
                                "scoring_domain": scoring_domain,
                                "origin_idx": int(origin),
                                "horizon_q": int(h),
                                "se": e**2,
                                "ae": abs(e),
                                "crps": crps_from_draws(draws_level[:, h - 1], float(actual[h - 1])),
                                "coverage_50": empirical_coverage(draws_level[:, h - 1], float(actual[h - 1]), 0.50),
                                "coverage_90": empirical_coverage(draws_level[:, h - 1], float(actual[h - 1]), 0.90),
                                "width90": interval_width(draws_level[:, h - 1], 0.90),
                            }
                        )

    if not rows:
        return pd.DataFrame()

    raw = pd.DataFrame(rows)
    agg = (
        raw.groupby(["regime_id", "model", "variable", "transform_id", "scoring_domain", "horizon_q"], as_index=False)
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
    return agg


def merge_special_into_champion(
    champion_map: Dict,
    special_sweep_df: pd.DataFrame,
    bvar_agg_df: pd.DataFrame,
    config: Dict,
    regime_id: str,
) -> Dict:
    if special_sweep_df.empty:
        return champion_map

    validation = config.get("validation", {})
    gap_pct = float(validation.get("ensemble_gap_pct", 1.5))
    target_cov = float(
        validation.get("pd_calibration", {}).get(
            "target_coverage90",
            validation.get("calibration", {}).get("target_coverage90", 0.90),
        )
    )
    key_special = validation.get("key_vars_special", ["high_yield_spread", "housing_starts_yoy"])

    for var in key_special:
        bvar_var = bvar_agg_df[
            (bvar_agg_df["regime_id"] == regime_id)
            & (bvar_agg_df["model"] == "BVAR")
            & (bvar_agg_df["variable"] == var)
            & (bvar_agg_df["horizon_q"].between(1, 12))
        ].copy()
        if bvar_var.empty:
            continue
        bvar_var["transform_id"] = "level"
        bvar_var["scoring_domain"] = "level"

        sp_var = special_sweep_df[(special_sweep_df["regime_id"] == regime_id) & (special_sweep_df["variable"] == var)].copy()
        all_candidates = pd.concat(
            [
                sp_var[["model", "transform_id", "scoring_domain", "horizon_q", "crps", "coverage_90"]],
                bvar_var[["model", "transform_id", "scoring_domain", "horizon_q", "crps", "coverage_90"]],
            ],
            ignore_index=True,
        )

        for bucket_name, hr in [("bucket_1_4", (1, 4)), ("bucket_5_12", (5, 12))]:
            bdf = all_candidates[all_candidates["horizon_q"].between(hr[0], hr[1])]
            if bdf.empty:
                continue
            g = bdf.groupby(["model", "transform_id", "scoring_domain"], as_index=False).agg(
                crps=("crps", "mean"), coverage_90=("coverage_90", "mean")
            )
            g = g.sort_values("crps").reset_index(drop=True)
            top = g.iloc[0]
            chosen = top
            gap = 999.0
            if g.shape[0] >= 2:
                runner = g.iloc[1]
                gap = 100.0 * (float(runner["crps"]) - float(top["crps"])) / max(float(runner["crps"]), 1e-8)
                if gap < gap_pct:
                    top_cov_dist = abs(float(top["coverage_90"]) - target_cov)
                    run_cov_dist = abs(float(runner["coverage_90"]) - target_cov)
                    if run_cov_dist + 1e-9 < top_cov_dist:
                        chosen = runner

            champion_map["variables"][var][bucket_name] = {
                "kind": "single",
                "model": str(chosen["model"]),
                "transform_id": str(chosen["transform_id"]),
                "scoring_domain": str(chosen["scoring_domain"]),
                "top2_gap_pct": float(gap),
                "coverage90_bucket": float(chosen["coverage_90"]),
                "calibration_scale": 1.0,
                "coverage90_before": None,
                "coverage90_after_est": None,
                "calibration_samples": 0,
            }

    return champion_map


def build_champion_raw(raw_df: pd.DataFrame, champion_map: Dict) -> pd.DataFrame:
    rows: List[Dict] = []
    keys = ["regime_id", "origin_idx", "variable", "horizon_q"]

    for (_, _, variable, horizon_q), grp in raw_df.groupby(keys):
        bucket = bucket_for_h(int(horizon_q))
        spec = (
            champion_map.get("variables", {})
            .get(variable, {})
            .get(bucket, {"kind": "single", "model": "BVAR", "transform_id": "level"})
        )
        grp_std = grp[grp["model"].isin(["BVAR", "AR", "RW"])].copy()
        if grp_std.empty:
            continue

        model_name = str(spec.get("model", "BVAR")).split("::")[0].upper()
        pick = grp_std[grp_std["model"] == model_name]
        if pick.empty:
            pick = grp_std[grp_std["model"] == "BVAR"]
        row = pick.iloc[0].to_dict()
        row["model"] = "CHAMPION"
        row["bucket"] = bucket
        row["champion_spec"] = json.dumps(spec, sort_keys=True)
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=list(raw_df.columns) + ["bucket", "champion_spec"])
    return pd.DataFrame(rows)


def fit_scale_objective_grid(
    abs_err: np.ndarray,
    half_width: np.ndarray,
    target_coverage: float,
    scale_min: float,
    scale_max: float,
    base_width_ratio: float,
    width_cap: float,
    lambda_penalty: float,
    grid_points: int,
) -> Tuple[float, float, float]:
    if abs_err.size == 0 or half_width.size == 0:
        return 1.0, float("nan"), float("nan")

    s_min = float(scale_min)
    s_max = float(scale_max)
    n_grid = max(5, int(grid_points))
    best_scale = 1.0
    best_loss = float("inf")
    best_cov = float("nan")
    best_width_ratio = float("nan")
    hw = np.maximum(half_width, 1e-8)
    for s in np.linspace(s_min, s_max, n_grid):
        cov = float(np.mean(abs_err <= hw * s))
        width_ratio = float(base_width_ratio * s)
        penalty = max(0.0, width_ratio - width_cap) if np.isfinite(width_cap) else 0.0
        loss = abs(cov - target_coverage) + float(lambda_penalty) * (penalty**2)
        if loss + 1e-12 < best_loss:
            best_loss = loss
            best_scale = float(s)
            best_cov = cov
            best_width_ratio = width_ratio
    return best_scale, best_cov, best_width_ratio


def fit_calibration_factors_from_raw(
    raw_df: pd.DataFrame,
    model_name: str,
    config: Dict,
    regime_id: str,
    rw_reference_df: pd.DataFrame | None = None,
) -> Dict:
    validation = config.get("validation", {})
    objective_mode = str(validation.get("objective_mode_default", "macro"))
    cal_cfg = (
        validation.get("pd_calibration", {})
        if objective_mode == "pd_levels_primary"
        else validation.get("calibration", {})
    )
    enabled = bool(cal_cfg.get("enabled", True))
    target_cov = float(cal_cfg.get("target_coverage90", validation.get("calibration", {}).get("target_coverage90", 0.90)))
    min_samples = int(cal_cfg.get("min_samples_per_bucket", 100))
    apply_buckets = set(cal_cfg.get("apply_to_buckets", ["bucket_5_12"]))
    low_sample_fallback = str(cal_cfg.get("low_sample_fallback", "target_global_shrink_to_1"))
    low_sample_shrink = float(cal_cfg.get("low_sample_shrinkage_weight", 0.5))
    lambda_penalty = float(cal_cfg.get("objective_lambda_width_penalty", 2.0))
    grid_points = int(cal_cfg.get("scale_grid_points", 61))
    per_target_cfg = cal_cfg.get("per_target", {})
    _, _, src_to_can = get_pd_target_mappings(validation)

    factors: Dict[str, Dict[str, Dict]] = {}
    if not enabled:
        return factors

    sub = raw_df[(raw_df["regime_id"] == regime_id) & (raw_df["model"] == model_name) & (raw_df["horizon_q"].between(1, 12))].copy()
    if sub.empty:
        return factors

    if objective_mode == "pd_levels_primary":
        target_sources = set(pd_primary_source_targets(validation))
        sub = sub[sub["variable"].isin(target_sources)].copy()
        if sub.empty:
            return factors

    rw_ref = pd.DataFrame()
    if rw_reference_df is not None and not rw_reference_df.empty:
        rw_ref = rw_reference_df[
            (rw_reference_df["regime_id"] == regime_id)
            & (rw_reference_df["model"] == "RW")
            & (rw_reference_df["horizon_q"].between(1, 12))
        ][["origin_idx", "variable", "horizon_q", "width90"]].rename(columns={"width90": "width90_rw"})

    sub["bucket"] = sub["horizon_q"].map(bucket_for_h)
    for (var, bucket), g in sub.groupby(["variable", "bucket"]):
        if bucket not in apply_buckets:
            continue
        can_var = canonical_var_name(str(var), src_to_can)
        target_cfg = per_target_cfg.get(can_var, per_target_cfg.get(var, {}))
        s_min = float(target_cfg.get("scale_min", cal_cfg.get("scale_min", 0.7)))
        s_max = float(target_cfg.get("scale_max", cal_cfg.get("scale_max", 2.0)))
        width_cap = float(target_cfg.get("width_cap", np.inf))

        abs_err = np.abs(g["actual"].to_numpy(dtype=float) - g["pred"].to_numpy(dtype=float))
        half_width = 0.5 * np.maximum(g["width90"].to_numpy(dtype=float), 1e-8)
        n = int(abs_err.shape[0])
        cov_before = float(np.mean(abs_err <= half_width)) if n > 0 else np.nan
        method = "bucket_objective"

        base_width_ratio = 1.0
        if not rw_ref.empty:
            g_rw = g[["origin_idx", "variable", "horizon_q", "width90"]].merge(
                rw_ref,
                on=["origin_idx", "variable", "horizon_q"],
                how="left",
            )
            if not g_rw.empty:
                ratios = g_rw["width90"].to_numpy(dtype=float) / np.maximum(g_rw["width90_rw"].to_numpy(dtype=float), 1e-8)
                ratios = ratios[np.isfinite(ratios)]
                if ratios.size > 0:
                    base_width_ratio = float(np.mean(ratios))

        if n < min_samples:
            if low_sample_fallback == "target_global_shrink_to_1":
                g_all = sub[sub["variable"] == var].copy()
                abs_all = np.abs(g_all["actual"].to_numpy(dtype=float) - g_all["pred"].to_numpy(dtype=float))
                half_all = 0.5 * np.maximum(g_all["width90"].to_numpy(dtype=float), 1e-8)
                base_all = base_width_ratio
                if not rw_ref.empty and not g_all.empty:
                    g_all_rw = g_all[["origin_idx", "variable", "horizon_q", "width90"]].merge(
                        rw_ref,
                        on=["origin_idx", "variable", "horizon_q"],
                        how="left",
                    )
                    ratios_all = g_all_rw["width90"].to_numpy(dtype=float) / np.maximum(
                        g_all_rw["width90_rw"].to_numpy(dtype=float), 1e-8
                    )
                    ratios_all = ratios_all[np.isfinite(ratios_all)]
                    if ratios_all.size > 0:
                        base_all = float(np.mean(ratios_all))
                global_scale, _, _ = fit_scale_objective_grid(
                    abs_err=abs_all,
                    half_width=half_all,
                    target_coverage=target_cov,
                    scale_min=s_min,
                    scale_max=s_max,
                    base_width_ratio=base_all,
                    width_cap=width_cap,
                    lambda_penalty=lambda_penalty,
                    grid_points=grid_points,
                )
                scale = float(1.0 + low_sample_shrink * (global_scale - 1.0))
                method = "low_sample_global_shrink_to_1"
            else:
                scale = 1.0
                method = "insufficient_samples_no_calibration"
            cov_after = float(np.mean(abs_err <= half_width * scale)) if n > 0 else np.nan
            width_after = float(base_width_ratio * scale)
        else:
            scale, cov_after, width_after = fit_scale_objective_grid(
                abs_err=abs_err,
                half_width=half_width,
                target_coverage=target_cov,
                scale_min=s_min,
                scale_max=s_max,
                base_width_ratio=base_width_ratio,
                width_cap=width_cap,
                lambda_penalty=lambda_penalty,
                grid_points=grid_points,
            )

        factors.setdefault(var, {})[bucket] = {
            "calibration_scale": float(scale),
            "coverage90_before": cov_before,
            "coverage90_after_est": cov_after,
            "calibration_samples": n,
            "target_coverage90": target_cov,
            "base_width_ratio": float(base_width_ratio),
            "projected_width_ratio_after": float(width_after),
            "width_cap": float(width_cap) if np.isfinite(width_cap) else None,
            "calibration_method": method,
        }

    return factors


def apply_calibration_to_raw(raw_df: pd.DataFrame, model_name: str, factors: Dict, regime_id: str) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df
    out = raw_df.copy()
    mask = (out["regime_id"] == regime_id) & (out["model"] == model_name) & (out["horizon_q"].between(1, 12))
    if not mask.any():
        return out

    idxs = out[mask].index
    for idx in idxs:
        var = str(out.at[idx, "variable"])
        h = int(out.at[idx, "horizon_q"])
        bucket = bucket_for_h(h)
        scale = float(factors.get(var, {}).get(bucket, {}).get("calibration_scale", 1.0))
        if abs(scale - 1.0) <= 1e-9:
            continue
        pred = float(out.at[idx, "pred"])
        actual = float(out.at[idx, "actual"])
        width = float(out.at[idx, "width90"])
        half = 0.5 * max(width, 1e-8)
        out.at[idx, "width90"] = width * scale
        out.at[idx, "coverage_90"] = float(abs(actual - pred) <= half * scale)

    return out


def attach_calibration_to_champion_map(champion_map: Dict, factors: Dict) -> Dict:
    out = json.loads(json.dumps(champion_map))
    for var, buckets in out.get("variables", {}).items():
        for bucket_name in ["bucket_1_4", "bucket_5_12"]:
            spec = buckets.get(bucket_name)
            if spec is None:
                continue
            f = factors.get(var, {}).get(bucket_name, {})
            spec["calibration_scale"] = float(f.get("calibration_scale", 1.0))
            spec["coverage90_before"] = f.get("coverage90_before")
            spec["coverage90_after_est"] = f.get("coverage90_after_est")
            spec["calibration_samples"] = int(f.get("calibration_samples", 0))
            spec["calibration_method"] = f.get("calibration_method", "none")
    return out


def run_engine_for_regime(config_path: str, regime_id: str, assumption_set: str, champion_map_path: str, output_dir: str) -> str:
    ensure_dir(output_dir)
    cmd = [
        sys.executable,
        os.path.join("scripts", "run_macro_forecast_engine.py"),
        "--config",
        config_path,
        "--regime-id",
        regime_id,
        "--assumption-set",
        assumption_set,
        "--champion-map",
        champion_map_path,
        "--output-dir",
        output_dir,
    ]
    subprocess.run(cmd, check=True)
    return os.path.join(output_dir, "macro_forecast_paths.csv")


def compute_boundary_stats(
    config: Dict,
    panel: pd.DataFrame,
    forecast_csv: str,
    regime_id: str,
    raw_df: pd.DataFrame,
    model_name: str = "CHAMPION",
) -> Tuple[Dict, pd.DataFrame]:
    regime = resolve_regime(config, regime_id)
    short_b = int(regime["short_quarters"])
    medium_b = int(regime["medium_quarters"])
    validation = config.get("validation", {})
    objective_mode = str(validation.get("objective_mode_default", "macro"))
    if objective_mode == "pd_levels_primary":
        key_vars = pd_primary_source_targets(validation)
    else:
        key_vars = validation.get("key_vars_primary", [])

    culprit_rows: List[Dict] = []
    z_short: List[float] = []
    z_medium: List[float] = []

    paths = pd.read_csv(forecast_csv)
    base = paths[paths["scenario"] == "Baseline"].set_index("forecast_q")
    # Gate boundary uses stitched baseline path.
    for v in key_vars:
        if v not in base.columns or v not in panel.columns:
            continue
        if (short_b not in base.index) or ((short_b + 1) not in base.index):
            continue
        y_b = float(base.loc[short_b, v])
        y_b1 = float(base.loc[short_b + 1, v])
        jump = abs(y_b1 - y_b)
        sigma = float(np.std(np.diff(panel[v].to_numpy(dtype=float)), ddof=1))
        sigma = max(sigma, 1e-6)
        z = jump / sigma
        z_short.append(z)
        culprit_rows.append(
            {
                "regime_id": regime_id,
                "origin_date": "",
                "origin_idx": -1,
                "variable": v,
                "boundary": f"Q{short_b}->Q{short_b+1}",
                "yhat_b": y_b,
                "yhat_b1": y_b1,
                "jump": jump,
                "sigma_v": sigma,
                "z": z,
                "boundary_type": "short_to_bridge",
            }
        )

    # Per-origin short boundary diagnostics from rolling raw forecasts.
    for v in key_vars:
        sub = raw_df[
            (raw_df["model"] == model_name)
            & (raw_df["variable"] == v)
            & (raw_df["horizon_q"].isin([short_b, short_b + 1]))
        ][["origin_idx", "origin_date", "horizon_q", "pred"]].copy()
        if sub.empty:
            continue
        for origin_idx, grp in sub.groupby("origin_idx"):
            if grp["horizon_q"].nunique() < 2:
                continue
            y_b = float(grp.loc[grp["horizon_q"] == short_b, "pred"].iloc[0])
            y_b1 = float(grp.loc[grp["horizon_q"] == short_b + 1, "pred"].iloc[0])
            jump = abs(y_b1 - y_b)
            series = panel[v].to_numpy(dtype=float)[: int(origin_idx)]
            if series.shape[0] < 4:
                continue
            sigma = float(np.std(np.diff(series), ddof=1))
            sigma = max(sigma, 1e-6)
            z = jump / sigma
            origin_date = ""
            if "origin_date" in grp.columns and grp["origin_date"].notna().any():
                origin_date = str(grp["origin_date"].dropna().iloc[0])
            culprit_rows.append(
                {
                    "regime_id": regime_id,
                    "origin_date": origin_date,
                    "origin_idx": int(origin_idx),
                    "variable": v,
                    "boundary": f"Q{short_b}->Q{short_b+1}",
                    "yhat_b": y_b,
                    "yhat_b1": y_b1,
                    "jump": jump,
                    "sigma_v": sigma,
                    "z": z,
                    "boundary_type": "short_to_bridge_origin_diagnostic",
                }
            )

    for v in key_vars:
        if v not in base.columns or v not in panel.columns:
            continue
        if (medium_b not in base.index) or ((medium_b + 1) not in base.index):
            continue
        y_b = float(base.loc[medium_b, v])
        y_b1 = float(base.loc[medium_b + 1, v])
        jump = abs(y_b1 - y_b)
        sigma = float(np.std(np.diff(panel[v].to_numpy(dtype=float)), ddof=1))
        sigma = max(sigma, 1e-6)
        z = jump / sigma
        z_medium.append(z)
        culprit_rows.append(
            {
                "regime_id": regime_id,
                "origin_date": "",
                "origin_idx": -1,
                "variable": v,
                "boundary": f"Q{medium_b}->Q{medium_b+1}",
                "yhat_b": y_b,
                "yhat_b1": y_b1,
                "jump": jump,
                "sigma_v": sigma,
                "z": z,
                "boundary_type": "medium_to_long",
            }
        )

    all_z = np.array(z_short + z_medium, dtype=float)
    if all_z.size == 0:
        stats = {
            "combined": {"max_z": np.nan, "median_z": np.nan, "n": 0},
            "short": {"max_z": np.nan, "median_z": np.nan, "n": 0},
            "medium": {"max_z": np.nan, "median_z": np.nan, "n": 0},
        }
    else:
        short_arr = np.array(z_short, dtype=float) if z_short else np.array([])
        med_arr = np.array(z_medium, dtype=float) if z_medium else np.array([])
        stats = {
            "combined": {"max_z": float(np.max(all_z)), "median_z": float(np.median(all_z)), "n": int(all_z.shape[0])},
            "short": {
                "max_z": float(np.max(short_arr)) if short_arr.size else np.nan,
                "median_z": float(np.median(short_arr)) if short_arr.size else np.nan,
                "n": int(short_arr.shape[0]),
            },
            "medium": {
                "max_z": float(np.max(med_arr)) if med_arr.size else np.nan,
                "median_z": float(np.median(med_arr)) if med_arr.size else np.nan,
                "n": int(med_arr.shape[0]),
            },
        }

    culprits_df = pd.DataFrame(culprit_rows)
    if not culprits_df.empty:
        culprits_df = culprits_df.sort_values("z", ascending=False).reset_index(drop=True)
    return stats, culprits_df


def compute_scenario_checks(config: Dict, forecast_csv: str) -> Dict:
    df = pd.read_csv(forecast_csv)
    base = df[df["scenario"] == "Baseline"].set_index("forecast_q")

    timing_pass = True
    timing_violations = []
    for sc_name, sc_cfg in config.get("scenarios", {}).items():
        if sc_name not in df["scenario"].unique():
            continue
        sc = df[df["scenario"] == sc_name].set_index("forecast_q")
        for var, spec in sc_cfg.get("shock_profiles", {}).items():
            if var not in sc.columns or var not in base.columns:
                continue
            diff = (sc[var] - base[var]).abs()
            nz = diff[diff > 1e-9]
            if nz.empty:
                continue
            first_q = int(nz.index.min())
            exp_q = int(spec["start_q"])
            if first_q != exp_q:
                timing_pass = False
                timing_violations.append({"scenario": sc_name, "variable": var, "expected": exp_q, "actual": first_q})

    ordering_pass = True
    ordering_violations = []
    if "Mild_Adverse" in df["scenario"].unique() and "Severe_Adverse" in df["scenario"].unique():
        mild = df[df["scenario"] == "Mild_Adverse"].set_index("forecast_q")
        severe = df[df["scenario"] == "Severe_Adverse"].set_index("forecast_q")
        positive_stress = {"unemployment_rate", "high_yield_spread", "mortgage30_rate", "consumer_delinquency_rate"}
        negative_stress = {"hpi_growth_yoy", "real_gdp_growth_yoy"}
        stress_vars = positive_stress | negative_stress
        for v in stress_vars:
            if v not in base.columns or v not in mild.columns or v not in severe.columns:
                continue
            ms = config.get("scenarios", {}).get("Mild_Adverse", {}).get("shock_profiles", {}).get(v, {})
            ss = config.get("scenarios", {}).get("Severe_Adverse", {}).get("shock_profiles", {}).get(v, {})
            q_peak = max(int(ms.get("peak_q", 1)), int(ss.get("peak_q", 1))) if (ms or ss) else 32
            if q_peak not in base.index or q_peak not in mild.index or q_peak not in severe.index:
                continue
            dm = float(mild.loc[q_peak, v] - base.loc[q_peak, v])
            ds = float(severe.loc[q_peak, v] - base.loc[q_peak, v])
            if v in positive_stress and ds + 1e-9 < dm:
                ordering_pass = False
                ordering_violations.append({"variable": v, "q": int(q_peak), "mild_delta": dm, "severe_delta": ds})
            if v in negative_stress and ds - 1e-9 > dm:
                ordering_pass = False
                ordering_violations.append({"variable": v, "q": int(q_peak), "mild_delta": dm, "severe_delta": ds})

    return {
        "timing_pass": timing_pass,
        "timing_violations": timing_violations,
        "ordering_pass": ordering_pass,
        "ordering_violations": ordering_violations,
    }


def evaluate_validation_gate(
    champion_agg: pd.DataFrame,
    config: Dict,
    regime_id: str,
    boundary_stats: Dict,
    scenario_checks: Dict,
    profile_name: str,
    profile_cfg: Dict,
) -> GateResult:
    validation = config.get("validation", {})
    objective_mode = str(validation.get("objective_mode_default", "macro"))
    gate_horizons = [int(h) for h in validation.get("pd_gate_horizons", list(range(1, 13)))]
    h_min = min(gate_horizons) if gate_horizons else 1
    h_max = max(gate_horizons) if gate_horizons else 12
    _, _, source_to_canonical = get_pd_target_mappings(validation)
    if objective_mode == "pd_levels_primary":
        key_vars = pd_primary_source_targets(validation)
        required_cells_definition = validation.get("pd_required_cells_definition", "pd_level_targets_x_h1_12")
    else:
        key_vars = validation.get("key_vars_primary", [])
        required_cells_definition = validation.get("required_cells_definition", "key_vars_primary_x_h1_12")

    c = champion_agg[
        (champion_agg["regime_id"] == regime_id)
        & (champion_agg["model"] == "CHAMPION")
        & (champion_agg["variable"].isin(key_vars))
        & (champion_agg["horizon_q"].between(h_min, h_max))
    ].copy()
    if c.empty:
        return GateResult(False, {"profile": profile_name, "reason": "No champion metrics for required cells."})

    c["variable_canonical"] = c["variable"].map(lambda v: canonical_var_name(str(v), source_to_canonical))
    min_n = int(c["n_oos"].min())
    power_pass = min_n >= int(validation.get("min_oos_origins_required", 40))

    h12 = c[c["horizon_q"].between(1, 2)]
    h34 = c[c["horizon_q"].between(3, 4)]
    h512 = c[c["horizon_q"].between(5, 12)]

    med_rrmse_h12 = float(h12["rrmse_vs_rw"].median()) if not h12.empty else np.nan
    med_rrmse_h34 = float(h34["rrmse_vs_rw"].median()) if not h34.empty else np.nan
    mean_crps_gain_h512 = float(h512["crps_gain_pct_vs_rw"].mean()) if not h512.empty else np.nan

    coverage_lo, coverage_hi = validation.get("pd_coverage90_band", validation.get("coverage90_band", [0.80, 0.98]))
    coverage_passrate = float(np.mean((c["coverage_90"] >= coverage_lo) & (c["coverage_90"] <= coverage_hi)))
    coverage_pass = coverage_passrate >= float(profile_cfg.get("coverage90_passrate_min", 0.75))

    width_ratio_mean = float(c["width_ratio_vs_rw"].mean())
    per_var_width = c.groupby("variable_canonical")["width_ratio_vs_rw"].mean()
    width_mean_pass = width_ratio_mean <= float(profile_cfg.get("width_ratio_mean_max", 1.35))
    width_per_var_pass = bool((per_var_width <= float(profile_cfg.get("width_ratio_per_var_max", 1.60))).all())

    h12_pass = bool(med_rrmse_h12 <= float(profile_cfg.get("rrmse_h1_h2_max", 1.00)))
    h34_pass = bool(med_rrmse_h34 <= float(profile_cfg.get("rrmse_h3_h4_max", 0.98)))
    crps_pass = bool(mean_crps_gain_h512 >= float(profile_cfg.get("crps_gain_h5_h12_min", 3.0)))

    b_comb = boundary_stats.get("combined", {})
    boundary_pass = bool(
        float(b_comb.get("max_z", np.inf)) <= float(profile_cfg.get("boundary_max_z_max", 2.5))
        and float(b_comb.get("median_z", np.inf)) <= float(profile_cfg.get("boundary_median_z_max", 1.5))
    )

    scenario_pass = bool(scenario_checks.get("timing_pass", False) and scenario_checks.get("ordering_pass", False))

    passed = (
        power_pass
        and h12_pass
        and h34_pass
        and crps_pass
        and coverage_pass
        and width_mean_pass
        and width_per_var_pass
        and boundary_pass
        and scenario_pass
    )

    return GateResult(
        passed=passed,
        details={
            "profile": profile_name,
            "thresholds": profile_cfg,
            "required_cells_definition": required_cells_definition,
            "required_cells_count": int(c.shape[0]),
            "min_n_oos_required_cells": min_n,
            "power_pass": bool(power_pass),
            "median_rrmse_h1_h2": med_rrmse_h12,
            "median_rrmse_h3_h4": med_rrmse_h34,
            "mean_crps_gain_pct_h5_h12": mean_crps_gain_h512,
            "h1_h2_pass": h12_pass,
            "h3_h4_pass": h34_pass,
            "crps_gain_pass": crps_pass,
            "coverage90_band": [coverage_lo, coverage_hi],
            "coverage90_passrate": coverage_passrate,
            "coverage90_pass": coverage_pass,
            "width_ratio_mean": width_ratio_mean,
            "width_ratio_mean_pass": width_mean_pass,
            "width_ratio_per_var": {k: float(v) for k, v in per_var_width.to_dict().items()},
            "width_ratio_per_var_pass": width_per_var_pass,
            "boundary": boundary_stats,
            "boundary_pass": boundary_pass,
            "scenario": scenario_checks,
            "scenario_pass": scenario_pass,
        },
    )


def evaluate_promotion(
    incumbent_agg: pd.DataFrame,
    challenger_agg: pd.DataFrame,
    incumbent_release_gate: GateResult,
    challenger_release_gate: GateResult,
    config: Dict,
    incumbent_boundary: Dict,
    challenger_boundary: Dict,
    promotion_profile: Dict,
) -> Dict:
    validation = config.get("validation", {})
    objective_mode = str(validation.get("objective_mode_default", "macro"))
    if objective_mode == "pd_levels_primary":
        key_vars = pd_primary_source_targets(validation)
    else:
        key_vars = validation.get("key_vars_primary", [])

    inc = incumbent_agg[
        (incumbent_agg["model"] == "CHAMPION")
        & (incumbent_agg["variable"].isin(key_vars))
        & (incumbent_agg["horizon_q"].between(1, 12))
    ]
    ch = challenger_agg[
        (challenger_agg["model"] == "CHAMPION")
        & (challenger_agg["variable"].isin(key_vars))
        & (challenger_agg["horizon_q"].between(1, 12))
    ]

    merged = inc[["variable", "horizon_q", "n_oos", "crps"]].merge(
        ch[["variable", "horizon_q", "n_oos", "crps"]],
        on=["variable", "horizon_q"],
        suffixes=("_inc", "_chal"),
    )
    if merged.empty:
        return {"promotion_pass": False, "reason": "No overlap between incumbent/challenger metrics."}

    sub_912 = merged[merged["horizon_q"].between(9, 12)]
    sub_14 = merged[merged["horizon_q"].between(1, 4)]

    power_912 = int(sub_912[["n_oos_inc", "n_oos_chal"]].min().min()) if not sub_912.empty else 0
    power_pass = power_912 >= int(validation.get("min_oos_origins_required", 40))

    gain_912 = (
        100.0 * (sub_912["crps_inc"].mean() - sub_912["crps_chal"].mean()) / max(float(sub_912["crps_inc"].mean()), 1e-8)
        if not sub_912.empty
        else np.nan
    )
    worsen_14 = (
        100.0 * (sub_14["crps_chal"].mean() - sub_14["crps_inc"].mean()) / max(float(sub_14["crps_inc"].mean()), 1e-8)
        if not sub_14.empty
        else np.nan
    )

    inc_med = float(incumbent_boundary.get("combined", {}).get("median_z", np.nan))
    chal_med = float(challenger_boundary.get("combined", {}).get("median_z", np.nan))
    boundary_cmp = bool(chal_med <= inc_med + float(promotion_profile.get("boundary_median_delta_max", 0.15)))

    pass_flag = (
        incumbent_release_gate.passed
        and challenger_release_gate.passed
        and power_pass
        and (gain_912 >= float(promotion_profile.get("crps_gain_h9_h12_min", 5.0)))
        and (worsen_14 <= float(promotion_profile.get("short_horizon_crps_worsen_h1_h4_max", 1.0)))
        and boundary_cmp
    )

    return {
        "promotion_pass": bool(pass_flag),
        "thresholds": promotion_profile,
        "incumbent_release_pass": bool(incumbent_release_gate.passed),
        "challenger_release_pass": bool(challenger_release_gate.passed),
        "power_h9_h12_min_n_oos": int(power_912),
        "power_pass": bool(power_pass),
        "crps_gain_pct_h9_h12_challenger_vs_incumbent": float(gain_912),
        "short_horizon_crps_worsen_pct_h1_h4": float(worsen_14),
        "boundary_comparator_pass": bool(boundary_cmp),
        "incumbent_median_z": inc_med,
        "challenger_median_z": chal_med,
    }


def run_pd_ablation(config: Dict, input_path: str, draws_n: int, profiles: Dict) -> pd.DataFrame:
    validation = config.get("validation", {})
    objective_mode = str(validation.get("objective_mode_default", "macro"))
    if objective_mode == "pd_levels_primary":
        target_sources = pd_primary_source_targets(validation)
    else:
        target_sources = validation.get("key_vars_primary", [])
    coverage_lo, coverage_hi = validation.get("pd_coverage90_band", validation.get("coverage90_band", [0.80, 0.98]))
    release_profile = profiles.get("release", {})
    regimes = list(config.get("regime_candidates", {}).keys())
    scope_candidates = config.get("model", {}).get("bvar_scope_candidates", ["full_28"])
    rows: List[Dict] = []

    for regime_id in regimes:
        regime = resolve_regime(config, regime_id)
        horizon = max(12, int(regime.get("short_quarters", 12)) + 1)
        for scope in scope_candidates:
            variable_set = "pd_core" if scope == "pd_core_6" else "full"
            try:
                agg, _, _ = run_backtest(
                    config=config,
                    input_path=input_path,
                    regime_id=regime_id,
                    variable_set=variable_set,
                    horizon=horizon,
                    scoring_draws=draws_n,
                    namespace=f"pd_ablation:{regime_id}:{scope}",
                )
            except Exception as exc:  # pragma: no cover - diagnostic path
                rows.append(
                    {
                        "regime_id": regime_id,
                        "scope": scope,
                        "status": "error",
                        "release_like_pass": np.nan,
                        "error": str(exc),
                    }
                )
                continue

            sub = agg[
                (agg["model"] == "BVAR")
                & (agg["variable"].isin(target_sources))
                & (agg["horizon_q"].between(1, 12))
            ].copy()
            if sub.empty:
                rows.append(
                    {
                        "regime_id": regime_id,
                        "scope": scope,
                        "status": "no_data",
                        "release_like_pass": np.nan,
                    }
                )
                continue
            min_n = int(sub["n_oos"].min())
            h12 = sub[sub["horizon_q"].between(1, 2)]
            h34 = sub[sub["horizon_q"].between(3, 4)]
            h512 = sub[sub["horizon_q"].between(5, 12)]
            med_rrmse_h12 = float(h12["rrmse_vs_rw"].median()) if not h12.empty else np.nan
            med_rrmse_h34 = float(h34["rrmse_vs_rw"].median()) if not h34.empty else np.nan
            mean_crps_gain_h512 = float(h512["crps_gain_pct_vs_rw"].mean()) if not h512.empty else np.nan
            coverage_passrate = float(np.mean((sub["coverage_90"] >= coverage_lo) & (sub["coverage_90"] <= coverage_hi)))
            width_ratio_mean = float(sub["width_ratio_vs_rw"].mean())
            width_ratio_per_var_max = float(sub.groupby("variable")["width_ratio_vs_rw"].mean().max())

            release_like_pass = bool(
                min_n >= int(validation.get("min_oos_origins_required", 40))
                and med_rrmse_h12 <= float(release_profile.get("rrmse_h1_h2_max", 1.0))
                and med_rrmse_h34 <= float(release_profile.get("rrmse_h3_h4_max", 0.98))
                and mean_crps_gain_h512 >= float(release_profile.get("crps_gain_h5_h12_min", 3.0))
                and coverage_passrate >= float(release_profile.get("coverage90_passrate_min", 0.75))
                and width_ratio_mean <= float(release_profile.get("width_ratio_mean_max", 1.35))
                and width_ratio_per_var_max <= float(release_profile.get("width_ratio_per_var_max", 1.60))
            )

            rows.append(
                {
                    "regime_id": regime_id,
                    "scope": scope,
                    "status": "ok",
                    "n_cells": int(sub.shape[0]),
                    "min_n_oos": min_n,
                    "median_rrmse_h1_h2": med_rrmse_h12,
                    "median_rrmse_h3_h4": med_rrmse_h34,
                    "mean_crps_gain_pct_h5_h12": mean_crps_gain_h512,
                    "coverage90_passrate": coverage_passrate,
                    "width_ratio_mean": width_ratio_mean,
                    "width_ratio_per_var_max": width_ratio_per_var_max,
                    "release_like_pass": release_like_pass,
                }
            )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["status", "release_like_pass", "mean_crps_gain_pct_h5_h12"], ascending=[True, False, False])


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    validation = config.get("validation", {})
    profiles = get_threshold_profiles(validation)
    active_profile = str(validation.get("active_profile_default", "release"))
    strict_profile = str(validation.get("strict_profile_default", "release"))
    strict_mode = bool(args.strict_validation or validation.get("strict_mode_default", False))

    regime_inc = args.regime_incumbent
    regime_chal = args.regime_challenger
    reg_inc = resolve_regime(config, regime_inc)
    reg_chal = resolve_regime(config, regime_chal)

    draws_n = int(validation.get("scoring_draws", 200))
    inc_h = max(12, int(reg_inc["short_quarters"]) + 1)
    chal_h = max(12, int(reg_chal["short_quarters"]) + 1)

    inc_agg, inc_raw, inc_seed = run_backtest(
        config=config,
        input_path=args.input,
        regime_id=regime_inc,
        variable_set="full",
        horizon=inc_h,
        scoring_draws=draws_n,
        namespace="validation_incumbent",
    )
    chal_agg, chal_raw, chal_seed = run_backtest(
        config=config,
        input_path=args.input,
        regime_id=regime_chal,
        variable_set="full",
        horizon=chal_h,
        scoring_draws=draws_n,
        namespace="validation_challenger",
    )

    champion_map = build_champion_map(inc_agg, config, regime_inc)

    panel = pd.read_csv(args.input, parse_dates=["quarter_end"]).sort_values("quarter_end")
    full_panel = panel[["quarter_end"] + config["model"]["variables"]].dropna().copy()

    origin_indices = sorted(inc_raw["origin_idx"].unique().tolist()) if not inc_raw.empty else []
    special_sweep = apply_special_series_sweep(
        config=config,
        panel=full_panel,
        regime_id=regime_inc,
        origin_indices=origin_indices,
        h_max=12,
        draws_n=draws_n,
    )
    champion_map = merge_special_into_champion(champion_map, special_sweep, inc_agg, config, regime_inc)

    champion_raw = build_champion_raw(inc_raw, champion_map)

    inc_cal_factors = fit_calibration_factors_from_raw(
        champion_raw,
        "CHAMPION",
        config,
        regime_inc,
        rw_reference_df=inc_raw[inc_raw["model"] == "RW"].copy(),
    )
    champion_map = attach_calibration_to_champion_map(champion_map, inc_cal_factors)
    champion_raw = apply_calibration_to_raw(champion_raw, "CHAMPION", inc_cal_factors, regime_inc)

    champion_eval_raw = pd.concat([champion_raw, inc_raw[inc_raw["model"] == "RW"].copy()], ignore_index=True)
    champion_agg_full = _aggregate_from_raw(champion_eval_raw, regime_inc)
    champion_agg = champion_agg_full[champion_agg_full["model"] == "CHAMPION"].copy()

    chal_proxy_raw = chal_raw[chal_raw["model"].isin(["BVAR", "RW"])].copy()
    chal_proxy_raw.loc[chal_proxy_raw["model"] == "BVAR", "model"] = "CHAMPION"
    chal_cal_factors = fit_calibration_factors_from_raw(
        chal_proxy_raw,
        "CHAMPION",
        config,
        regime_chal,
        rw_reference_df=chal_proxy_raw[chal_proxy_raw["model"] == "RW"].copy(),
    )
    chal_proxy_raw = apply_calibration_to_raw(chal_proxy_raw, "CHAMPION", chal_cal_factors, regime_chal)
    chal_proxy_agg_full = _aggregate_from_raw(chal_proxy_raw, regime_chal)
    chal_proxy_agg = chal_proxy_agg_full[chal_proxy_agg_full["model"] == "CHAMPION"].copy()

    all_metrics = pd.concat([inc_agg, chal_agg, champion_agg, chal_proxy_agg], ignore_index=True)
    all_metrics = all_metrics.sort_values(["regime_id", "model", "variable", "horizon_q"]).reset_index(drop=True)
    pd_ablation_df = run_pd_ablation(config=config, input_path=args.input, draws_n=draws_n, profiles=profiles)
    pd_ablation_df.to_csv(os.path.join(args.output_dir, "pd_ablation_results.csv"), index=False)

    ensure_dir(os.path.dirname(args.champion_map_output) or ".")
    with open(args.champion_map_output, "w", encoding="utf-8") as f:
        json.dump(champion_map, f, indent=2)

    calibration_payload = {
        "contract_version": validation.get("contract_version", "v3.freeze.1"),
        "incumbent_regime": regime_inc,
        "challenger_regime": regime_chal,
        "incumbent_factors": inc_cal_factors,
        "challenger_proxy_factors": chal_cal_factors,
    }
    with open(os.path.join(args.output_dir, "calibration_factors.json"), "w", encoding="utf-8") as f:
        json.dump(calibration_payload, f, indent=2)

    inc_forecast = run_engine_for_regime(
        config_path=args.config,
        regime_id=regime_inc,
        assumption_set="base",
        champion_map_path=args.champion_map_output,
        output_dir=os.path.join(args.output_dir, "regime_incumbent"),
    )
    chal_forecast = run_engine_for_regime(
        config_path=args.config,
        regime_id=regime_chal,
        assumption_set="base",
        champion_map_path=args.champion_map_output,
        output_dir=os.path.join(args.output_dir, "regime_challenger"),
    )

    inc_boundary, inc_boundary_culprits = compute_boundary_stats(
        config=config,
        panel=full_panel,
        forecast_csv=inc_forecast,
        regime_id=regime_inc,
        raw_df=champion_raw,
        model_name="CHAMPION",
    )
    chal_boundary, chal_boundary_culprits = compute_boundary_stats(
        config=config,
        panel=full_panel,
        forecast_csv=chal_forecast,
        regime_id=regime_chal,
        raw_df=chal_proxy_raw,
        model_name="CHAMPION",
    )

    inc_scenario = compute_scenario_checks(config, inc_forecast)

    inc_profile_results: Dict[str, Dict] = {}
    chal_profile_results: Dict[str, Dict] = {}
    for pname in ["operational", "release"]:
        inc_gate = evaluate_validation_gate(champion_agg, config, regime_inc, inc_boundary, inc_scenario, pname, profiles[pname])
        chal_gate = evaluate_validation_gate(chal_proxy_agg, config, regime_chal, chal_boundary, inc_scenario, pname, profiles[pname])
        inc_profile_results[pname] = {"pass": bool(inc_gate.passed), **inc_gate.details}
        chal_profile_results[pname] = {"pass": bool(chal_gate.passed), **chal_gate.details}

    inc_release_gate = GateResult(bool(inc_profile_results["release"]["pass"]), inc_profile_results["release"])
    chal_release_gate = GateResult(bool(chal_profile_results["release"]["pass"]), chal_profile_results["release"])

    promotion = evaluate_promotion(
        incumbent_agg=champion_agg,
        challenger_agg=chal_proxy_agg,
        incumbent_release_gate=inc_release_gate,
        challenger_release_gate=chal_release_gate,
        config=config,
        incumbent_boundary=inc_boundary,
        challenger_boundary=chal_boundary,
        promotion_profile=profiles["promotion"],
    )

    objective_mode = str(validation.get("objective_mode_default", "macro"))
    if objective_mode == "pd_levels_primary":
        required_targets = pd_primary_source_targets(validation)
    else:
        required_targets = validation.get("key_vars_primary", [])
    required_cells = champion_agg[
        (champion_agg["regime_id"] == regime_inc)
        & (champion_agg["model"] == "CHAMPION")
        & (champion_agg["variable"].isin(required_targets))
        & (champion_agg["horizon_q"].between(1, 12))
    ]
    diagnostic_only = bool(
        required_cells.empty or int(required_cells["n_oos"].min()) < int(validation.get("min_oos_origins_required", 40))
    )
    ablation_winner = {}
    if not pd_ablation_df.empty:
        ok_ablation = pd_ablation_df[pd_ablation_df["status"] == "ok"].copy()
        if not ok_ablation.empty:
            ok_ablation = ok_ablation.sort_values(
                ["release_like_pass", "mean_crps_gain_pct_h5_h12", "width_ratio_mean"],
                ascending=[False, False, True],
            )
            ablation_winner = ok_ablation.iloc[0].to_dict()

    summary = {
        "contract_version": validation.get("contract_version", "v3.freeze.1"),
        "objective_mode": validation.get("objective_mode_default", "macro"),
        "validation_mode": "strict" if strict_mode else str(validation.get("gate_mode_default", "warn")),
        "active_profile": "release",
        "strict_profile": strict_profile,
        "diagnostic_only": diagnostic_only,
        "regimes": {"incumbent": regime_inc, "challenger": regime_chal},
        "pd_ablation_winner": ablation_winner,
        "seed_registry_hash": summarize_seed_hash(pd.concat([inc_seed, chal_seed], ignore_index=True)),
        "validation_incumbent": {
            "pass_active_profile": bool(inc_profile_results.get("release", {}).get("pass", False)),
            "profile_results": inc_profile_results,
            "boundary_short_stats": inc_boundary.get("short", {}),
            "boundary_medium_stats": inc_boundary.get("medium", {}),
            "boundary_combined_stats": inc_boundary.get("combined", {}),
            "scenario": inc_scenario,
        },
        "validation_challenger_proxy": {
            "pass_active_profile": bool(chal_profile_results.get("release", {}).get("pass", False)),
            "profile_results": chal_profile_results,
            "boundary_short_stats": chal_boundary.get("short", {}),
            "boundary_medium_stats": chal_boundary.get("medium", {}),
            "boundary_combined_stats": chal_boundary.get("combined", {}),
        },
        "profile_results": {
            "operational": inc_profile_results.get("operational", {}),
            "release": inc_profile_results.get("release", {}),
            "promotion": promotion,
        },
        "promotion": promotion,
        "notes": [
            "DM statistics are diagnostic-only and excluded from pass/fail.",
            "Special-series variables are excluded from promotion gate by contract.",
            "Operational profile is reporting-only in V3 freeze; release and promotion are gating profiles.",
        ],
    }

    all_metrics.to_csv(os.path.join(args.output_dir, "backtest_metrics.csv"), index=False)
    with open(os.path.join(args.output_dir, "validation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if args.verbose_validation:
        dm_cols = [
            "regime_id",
            "model",
            "variable",
            "horizon_q",
            "dm_stat_mse_vs_rw",
            "dm_pvalue_mse_vs_rw",
            "dm_stat_crps_vs_rw",
            "dm_pvalue_crps_vs_rw",
            "hac_lags",
        ]
        dm_df = all_metrics[[c for c in dm_cols if c in all_metrics.columns]].dropna(how="all")
        dm_df.to_csv(os.path.join(args.output_dir, "dm_results.csv"), index=False)

        regime_scores = pd.DataFrame(
            [
                {
                    "regime_id": regime_inc,
                    "score_boundary_median_z": float(inc_boundary.get("combined", {}).get("median_z", np.nan)),
                    "score_boundary_max_z": float(inc_boundary.get("combined", {}).get("max_z", np.nan)),
                    "release_pass": bool(inc_profile_results.get("release", {}).get("pass", False)),
                    "operational_pass": bool(inc_profile_results.get("operational", {}).get("pass", False)),
                },
                {
                    "regime_id": regime_chal,
                    "score_boundary_median_z": float(chal_boundary.get("combined", {}).get("median_z", np.nan)),
                    "score_boundary_max_z": float(chal_boundary.get("combined", {}).get("max_z", np.nan)),
                    "release_pass": bool(chal_profile_results.get("release", {}).get("pass", False)),
                    "operational_pass": bool(chal_profile_results.get("operational", {}).get("pass", False)),
                },
            ]
        )
        regime_scores.to_csv(os.path.join(args.output_dir, "regime_split_scores.csv"), index=False)

        if not special_sweep.empty:
            special_sweep.to_csv(os.path.join(args.output_dir, "special_series_sweep.csv"), index=False)

        pd.concat([inc_seed, chal_seed], ignore_index=True).to_csv(
            os.path.join(args.output_dir, "seed_registry.csv"), index=False
        )

        culprits = pd.concat([inc_boundary_culprits, chal_boundary_culprits], ignore_index=True)
        culprits.to_csv(os.path.join(args.output_dir, "boundary_culprits.csv"), index=False)

        coverage_lo, coverage_hi = validation.get("pd_coverage90_band", validation.get("coverage90_band", [0.80, 0.98]))
        if str(validation.get("objective_mode_default", "macro")) == "pd_levels_primary":
            cfail_targets = set(pd_primary_source_targets(validation))
        else:
            cfail_targets = set(validation.get("key_vars_primary", []))
        cfail = champion_agg[
            (champion_agg["regime_id"] == regime_inc)
            & (champion_agg["model"] == "CHAMPION")
            & (champion_agg["variable"].isin(cfail_targets))
            & (champion_agg["horizon_q"].between(1, 12))
            & ((champion_agg["coverage_90"] < coverage_lo) | (champion_agg["coverage_90"] > coverage_hi))
        ][["variable", "horizon_q", "coverage_90", "width_ratio_vs_rw", "model"]].copy()
        if not cfail.empty:
            cfail["bucket"] = cfail["horizon_q"].map(bucket_for_h)
        cfail.to_csv(os.path.join(args.output_dir, "coverage_fail_cells.csv"), index=False)

        sensitivity_rows = []
        if str(validation.get("objective_mode_default", "macro")) == "pd_levels_primary":
            sensitivity_vars = pd_primary_source_targets(validation)
        else:
            sensitivity_vars = validation.get("key_vars_primary", [])
        for a_set in ["low", "base", "high"]:
            out_dir = os.path.join(args.output_dir, f"assumption_{a_set}")
            fcst = run_engine_for_regime(args.config, regime_inc, a_set, args.champion_map_output, out_dir)
            p = pd.read_csv(fcst)
            b = p[p["scenario"] == "Baseline"]
            for var in sensitivity_vars:
                if var not in b.columns:
                    continue
                for q in [12, 24, 40]:
                    row = b[b["forecast_q"] == q]
                    if row.empty:
                        continue
                    sensitivity_rows.append(
                        {
                            "assumption_set": a_set,
                            "variable": var,
                            "forecast_q": int(q),
                            "value": float(row.iloc[0][var]),
                        }
                    )
        pd.DataFrame(sensitivity_rows).to_csv(os.path.join(args.output_dir, "anchor_sensitivity.csv"), index=False)

        pd.DataFrame(inc_scenario.get("timing_violations", []) + inc_scenario.get("ordering_violations", [])).to_csv(
            os.path.join(args.output_dir, "scenario_timing_checks.csv"), index=False
        )

    active_pass = bool(inc_profile_results.get("release", {}).get("pass", False))
    strict_pass = bool(inc_profile_results.get(strict_profile, {}).get("pass", False))

    print(f"Wrote validation summary: {os.path.join(args.output_dir, 'validation_summary.json')}")
    print(f"Active profile ({active_profile}) pass: {active_pass}; strict profile ({strict_profile}) pass: {strict_pass}")
    print(f"Diagnostic only: {diagnostic_only}; Promotion pass: {promotion.get('promotion_pass', False)}")

    hard_fail = diagnostic_only or (not strict_pass)
    if strict_mode and hard_fail:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
