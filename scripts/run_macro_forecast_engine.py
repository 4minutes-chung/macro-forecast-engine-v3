#!/usr/bin/env python3
"""
Run a rigorous macro forecasting engine:
- 1-2 years: VAR and Minnesota-style BVAR forecasts (point + intervals)
- 2-5 years: bridge from model path to structural anchors
- 5-20 years: scenario envelopes with demographic variants

Inputs:
- data/macro_panel_quarterly_model.csv (transformed quarterly panel)

Outputs (under outputs/macro_engine by default):
- macro_forecast_paths.csv
- macro_forecast_short_horizon_intervals.csv
- macro_model_diagnostics.json
- macro_anchor_assumptions.json
- macro_impulse_responses.csv
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class VarModel:
    intercept: np.ndarray  # (n,)
    coefs: np.ndarray      # (p, n, n) where y_t = c + sum_l coefs[l-1] @ y_{t-l}
    sigma: np.ndarray      # residual covariance (n,n)
    lag_order: int
    variables: List[str]
    bic: float
    stable: bool


def parse_args():
    parser = argparse.ArgumentParser(description="Run macro forecast engine.")
    parser.add_argument(
        "--config",
        default="macro_engine_config.json",
        help="Path to macro engine config JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/macro_engine",
        help="Output directory for forecast artifacts.",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def make_lagged_xy(y: np.ndarray, p: int) -> Tuple[np.ndarray, np.ndarray]:
    t, n = y.shape
    rows = t - p
    x = np.ones((rows, 1 + n * p), dtype=float)
    y_target = y[p:, :]
    for lag in range(1, p + 1):
        x[:, 1 + (lag - 1) * n: 1 + lag * n] = y[p - lag: t - lag, :]
    return x, y_target


def companion_matrix(coefs: np.ndarray) -> np.ndarray:
    # coefs shape (p,n,n) using left multiply on y_{t-l}
    p, n, _ = coefs.shape
    top = np.hstack([coefs[i, :, :] for i in range(p)])
    if p == 1:
        return top
    lower = np.hstack(
        [np.eye(n * (p - 1)), np.zeros((n * (p - 1), n))]
    )
    return np.vstack([top, lower])


def is_stable(coefs: np.ndarray, tol: float = 0.999) -> bool:
    comp = companion_matrix(coefs)
    eigvals = np.linalg.eigvals(comp)
    return np.max(np.abs(eigvals)) < tol


def fit_var_ols(y: np.ndarray, p: int, variables: List[str]) -> VarModel:
    x, y_target = make_lagged_xy(y, p)
    xtx = x.T @ x
    xty = x.T @ y_target
    beta = np.linalg.solve(xtx, xty)  # (k,n)
    resid = y_target - x @ beta
    dof = max(1, x.shape[0] - x.shape[1])
    sigma = (resid.T @ resid) / dof

    n = y.shape[1]
    intercept = beta[0, :]
    coefs = np.zeros((p, n, n), dtype=float)
    for lag in range(1, p + 1):
        block = beta[1 + (lag - 1) * n: 1 + lag * n, :]  # (n,n) rows=features var, cols=eq
        coefs[lag - 1, :, :] = block.T

    t_eff = x.shape[0]
    k_params = n * (1 + n * p)
    sign, logdet = np.linalg.slogdet(sigma)
    if sign <= 0:
        logdet = np.log(np.abs(np.linalg.det(sigma + 1e-8 * np.eye(n))))
    bic = logdet + (np.log(t_eff) * k_params) / t_eff

    return VarModel(
        intercept=intercept,
        coefs=coefs,
        sigma=sigma,
        lag_order=p,
        variables=variables,
        bic=float(bic),
        stable=is_stable(coefs),
    )


def select_var_model(y: np.ndarray, variables: List[str], lag_candidates: List[int]) -> VarModel:
    fitted = []
    for p in lag_candidates:
        if p >= y.shape[0] - 5:
            continue
        model = fit_var_ols(y, p, variables)
        fitted.append(model)
    if not fitted:
        raise RuntimeError("No feasible VAR lag candidates.")

    stable_models = [m for m in fitted if m.stable]
    if stable_models:
        return min(stable_models, key=lambda m: m.bic)
    return min(fitted, key=lambda m: m.bic)


def fit_bvar_minnesota(
    y: np.ndarray,
    p: int,
    variables: List[str],
    level_vars: List[bool],
    lambda1: float,
    lambda2: float,
    lambda3: float,
    lambda4: float,
) -> Tuple[VarModel, List[np.ndarray]]:
    x, y_target = make_lagged_xy(y, p)
    t_eff, k = x.shape
    n = y.shape[1]

    # Scale factors from univariate variance (fallback for Minnesota prior scaling).
    sigma_scale = np.var(y_target, axis=0, ddof=1)
    sigma_scale = np.where(sigma_scale <= 1e-8, 1.0, sigma_scale)

    beta_mean = np.zeros((k, n), dtype=float)
    beta_cov_blocks = []
    sigma_resid = np.zeros((n, n), dtype=float)

    xtx = x.T @ x

    for j in range(n):
        b0 = np.zeros(k, dtype=float)
        # Prior mean on own first lag: 1 for persistent level vars, else 0.
        if level_vars[j]:
            b0[1 + j] = 1.0

        prior_var = np.zeros(k, dtype=float)
        prior_var[0] = (lambda4 ** 2) * sigma_scale[j]
        for lag in range(1, p + 1):
            for m in range(n):
                idx = 1 + (lag - 1) * n + m
                tight = (lambda1 ** 2) / (lag ** (2.0 * lambda3))
                if m != j:
                    tight *= (lambda2 ** 2)
                tight *= sigma_scale[j] / sigma_scale[m]
                prior_var[idx] = tight

        # Avoid singular prior precision.
        prior_var = np.where(prior_var <= 1e-10, 1e-10, prior_var)
        prior_prec = np.diag(1.0 / prior_var)
        post_prec = xtx + prior_prec
        post_cov_core = np.linalg.inv(post_prec)
        rhs = x.T @ y_target[:, j] + prior_prec @ b0
        beta_j = post_cov_core @ rhs

        resid_j = y_target[:, j] - x @ beta_j
        sig_j = float((resid_j @ resid_j) / max(1, t_eff - k))
        sigma_resid[j, j] = max(sig_j, 1e-8)
        beta_mean[:, j] = beta_j
        beta_cov_blocks.append(post_cov_core * sigma_resid[j, j])

    # Build coefficient matrices.
    intercept = beta_mean[0, :]
    coefs = np.zeros((p, n, n), dtype=float)
    for lag in range(1, p + 1):
        block = beta_mean[1 + (lag - 1) * n: 1 + lag * n, :]  # (n,n)
        coefs[lag - 1, :, :] = block.T

    # BIC-like score from diagonal residual covariance for diagnostics.
    sigma_diag = np.diag(np.diag(sigma_resid))
    sign, logdet = np.linalg.slogdet(sigma_diag)
    if sign <= 0:
        logdet = np.log(np.abs(np.linalg.det(sigma_diag + 1e-8 * np.eye(n))))
    k_params = n * (1 + n * p)
    bic = logdet + (np.log(t_eff) * k_params) / t_eff

    model = VarModel(
        intercept=intercept,
        coefs=coefs,
        sigma=sigma_diag,
        lag_order=p,
        variables=variables,
        bic=float(bic),
        stable=is_stable(coefs),
    )
    return model, beta_cov_blocks


def deterministic_forecast(model: VarModel, history: np.ndarray, horizon: int) -> np.ndarray:
    p = model.lag_order
    n = history.shape[1]
    hist = history.copy()
    out = np.zeros((horizon, n), dtype=float)
    for h in range(horizon):
        y_next = model.intercept.copy()
        for lag in range(1, p + 1):
            y_next += model.coefs[lag - 1, :, :] @ hist[-lag, :]
        out[h, :] = y_next
        hist = np.vstack([hist, y_next])
    return out


def simulate_forecast_intervals(
    model: VarModel,
    history: np.ndarray,
    horizon: int,
    n_sims: int,
    quantiles: List[float],
    rng_seed: int,
    beta_cov_blocks: List[np.ndarray] = None,
    beta_mean_matrix: np.ndarray = None,
) -> Dict[float, np.ndarray]:
    rng = np.random.default_rng(rng_seed)
    p = model.lag_order
    n = history.shape[1]
    sims = np.zeros((n_sims, horizon, n), dtype=float)

    # Convert model coefs to stacked beta mean if posterior sampling is requested.
    if beta_cov_blocks is not None and beta_mean_matrix is None:
        k = 1 + n * p
        beta_mean_matrix = np.zeros((k, n), dtype=float)
        beta_mean_matrix[0, :] = model.intercept
        for lag in range(1, p + 1):
            block = model.coefs[lag - 1, :, :].T  # (n,n) feature x eq
            beta_mean_matrix[1 + (lag - 1) * n: 1 + lag * n, :] = block

    for s in range(n_sims):
        hist = history.copy()
        if beta_cov_blocks is not None:
            sampled_beta = np.zeros_like(beta_mean_matrix)
            for j in range(n):
                sampled_beta[:, j] = rng.multivariate_normal(
                    beta_mean_matrix[:, j], beta_cov_blocks[j]
                )
            sampled_intercept = sampled_beta[0, :]
            sampled_coefs = np.zeros((p, n, n), dtype=float)
            for lag in range(1, p + 1):
                block = sampled_beta[1 + (lag - 1) * n: 1 + lag * n, :]
                sampled_coefs[lag - 1, :, :] = block.T
        else:
            sampled_intercept = model.intercept
            sampled_coefs = model.coefs

        for h in range(horizon):
            eps = rng.multivariate_normal(np.zeros(n), model.sigma)
            y_next = sampled_intercept.copy()
            for lag in range(1, p + 1):
                y_next += sampled_coefs[lag - 1, :, :] @ hist[-lag, :]
            y_next = y_next + eps
            sims[s, h, :] = y_next
            hist = np.vstack([hist, y_next])

    out = {}
    for q in quantiles:
        out[q] = np.quantile(sims, q, axis=0)
    return out


def build_anchor_vector(
    baseline_path: np.ndarray,
    variables: List[str],
    assumptions: Dict,
) -> np.ndarray:
    idx = {v: i for i, v in enumerate(variables)}
    last40_mean = np.nanmean(baseline_path[-40:, :], axis=0)
    anchor = last40_mean.copy()

    # Core structural assumptions
    working_age_growth = float(assumptions["working_age_pop_growth_yoy"])
    productivity = float(assumptions["labor_productivity_trend_yoy"])
    inflation_target = float(assumptions["inflation_target"])
    neutral_real = float(assumptions["neutral_real_rate"])
    term_premium = float(assumptions["term_premium_10y"])
    mortgage_spread = float(assumptions["mortgage_spread_30y"])
    nairu = float(assumptions["nairu"])
    pop_growth = float(assumptions["population_growth_yoy"])
    housing_supply_drag = float(assumptions["housing_supply_elasticity_drag"])

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
    anchor[idx["high_yield_spread"]] = float(assumptions["hy_spread_anchor"])

    # Housing and demand related anchors
    wage_anchor = float(np.clip(last40_mean[idx["wage_growth_yoy"]], 1.5, 5.0))
    hpi_anchor = wage_anchor + pop_growth - housing_supply_drag
    anchor[idx["hpi_growth_yoy"]] = hpi_anchor
    anchor[idx["rent_inflation_yoy"]] = inflation_target + 0.5
    anchor[idx["retail_sales_yoy"]] = anchor[idx["real_gdp_growth_yoy"]] + inflation_target
    anchor[idx["industrial_production_yoy"]] = anchor[idx["real_gdp_growth_yoy"]] - 0.2
    return anchor


def bridge_to_anchors(
    model_path: np.ndarray,
    anchor: np.ndarray,
    short_horizon: int,
    medium_horizon: int,
) -> np.ndarray:
    out = model_path.copy()
    for h in range(short_horizon, medium_horizon):
        # h is zero-based index. Bridge starts after short horizon.
        w = float((h + 1 - short_horizon) / max(1, (medium_horizon - short_horizon)))
        out[h, :] = (1.0 - w) * out[h, :] + w * anchor
    # Beyond medium horizon, smooth mean-reversion toward anchor.
    kappa = 0.08
    for h in range(medium_horizon, out.shape[0]):
        out[h, :] = out[h - 1, :] + kappa * (anchor - out[h - 1, :])
    return out


def triangular_shock(h: int, start: int, peak: int, end: int, peak_delta: float) -> float:
    # h is 1-based forecast quarter index
    if h < start:
        return 0.0
    if h <= peak and peak > start:
        return peak_delta * (h - start) / float(peak - start)
    if h <= peak and peak == start:
        return peak_delta
    if h <= end and end > peak:
        return peak_delta * (1.0 - (h - peak) / float(end - peak))
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

    # Apply shocks only from long horizon onward (5+ years).
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

    # Apply persistent demographic drift modifier where requested.
    for var, shift in scenario_cfg.get("persistent_anchor_shift", {}).items():
        if var not in idx:
            continue
        shifted_anchor = anchor[idx[var]] + float(shift)
        for h in range(long_start_quarter, horizon):
            out[h, idx[var]] = out[h - 1, idx[var]] + 0.10 * (shifted_anchor - out[h - 1, idx[var]])
    return out


def make_forecast_dates(last_date: pd.Timestamp, horizon: int) -> List[pd.Timestamp]:
    dates = []
    d = last_date
    for _ in range(horizon):
        d = (d + pd.offsets.QuarterEnd(1)).normalize()
        dates.append(d)
    return dates


def build_impulse_responses(model: VarModel, variables: List[str], horizon: int, shocks: Dict[str, float]) -> pd.DataFrame:
    n = len(variables)
    p = model.lag_order
    rows = []
    zero_hist = np.zeros((p, n), dtype=float)
    for shock_var, shock_size in shocks.items():
        if shock_var not in variables:
            continue
        shock_vec = np.zeros(n, dtype=float)
        shock_vec[variables.index(shock_var)] = float(shock_size)

        # IRF recursion: x_t = sum A_i x_{t-i}, with x_0 = shock.
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


def main():
    args = parse_args()
    config = load_config(args.config)
    ensure_dir(args.output_dir)

    data_path = config["inputs"]["model_panel_csv"]
    panel = pd.read_csv(data_path, parse_dates=["quarter_end"])
    panel = panel.sort_values("quarter_end")

    variables = config["model"]["variables"]
    var_benchmark_vars = config["model"].get("var_benchmark_variables", variables)

    missing = [v for v in variables if v not in panel.columns]
    if missing:
        raise RuntimeError(f"Missing variables in model panel: {missing}")
    missing_var = [v for v in var_benchmark_vars if v not in panel.columns]
    if missing_var:
        raise RuntimeError(f"Missing VAR benchmark variables in model panel: {missing_var}")

    panel = panel[["quarter_end"] + variables].dropna().copy()
    y = panel[variables].to_numpy(dtype=float)
    y_var = panel[var_benchmark_vars].to_numpy(dtype=float)
    last_date = panel["quarter_end"].iloc[-1]

    lag_candidates = [int(x) for x in config["model"]["lag_candidates"]]
    var_model = select_var_model(y_var, var_benchmark_vars, lag_candidates)

    level_vars = [bool(config["model"]["level_variable_flags"].get(v, False)) for v in variables]
    bvar_cfg = config["model"]["bvar_hyperparams"]
    bvar_model, bvar_beta_cov = fit_bvar_minnesota(
        y=y,
        p=var_model.lag_order,
        variables=variables,
        level_vars=level_vars,
        lambda1=float(bvar_cfg["lambda1"]),
        lambda2=float(bvar_cfg["lambda2"]),
        lambda3=float(bvar_cfg["lambda3"]),
        lambda4=float(bvar_cfg["lambda4"]),
    )

    horizon_total = int(config["horizons"]["total_quarters"])
    short_horizon = int(config["horizons"]["short_quarters"])
    medium_horizon = int(config["horizons"]["medium_quarters"])
    long_start = int(config["horizons"]["long_start_quarter"])

    p_var = var_model.lag_order
    p_bvar = bvar_model.lag_order
    history_var = y_var[-p_var:, :]
    history_bvar = y[-p_bvar:, :]

    var_point = deterministic_forecast(var_model, history_var, horizon_total)
    bvar_point = deterministic_forecast(bvar_model, history_bvar, horizon_total)

    quantiles = [0.05, 0.5, 0.95]
    n_sims = int(config["model"]["interval_simulations"])
    seed = int(config["model"]["random_seed"])
    var_int = simulate_forecast_intervals(
        model=var_model,
        history=history_var,
        horizon=short_horizon,
        n_sims=n_sims,
        quantiles=quantiles,
        rng_seed=seed,
    )
    # Approximate Bayesian density: sample BVAR coefficients + residual noise.
    n = len(variables)
    k = 1 + n * p_bvar
    beta_mean_matrix = np.zeros((k, n), dtype=float)
    beta_mean_matrix[0, :] = bvar_model.intercept
    for lag in range(1, p_bvar + 1):
        beta_mean_matrix[1 + (lag - 1) * n: 1 + lag * n, :] = bvar_model.coefs[lag - 1, :, :].T

    bvar_int = simulate_forecast_intervals(
        model=bvar_model,
        history=history_bvar,
        horizon=short_horizon,
        n_sims=n_sims,
        quantiles=quantiles,
        rng_seed=seed + 17,
        beta_cov_blocks=bvar_beta_cov,
        beta_mean_matrix=beta_mean_matrix,
    )

    short_model_choice = config["model"]["short_horizon_model"].upper()
    if short_model_choice not in {"VAR", "BVAR"}:
        raise RuntimeError("short_horizon_model must be VAR or BVAR")
    chosen_path = var_point if short_model_choice == "VAR" else bvar_point

    anchor = build_anchor_vector(
        baseline_path=chosen_path,
        variables=variables,
        assumptions=config["assumptions"],
    )
    baseline = bridge_to_anchors(
        model_path=chosen_path.copy(),
        anchor=anchor,
        short_horizon=short_horizon,
        medium_horizon=medium_horizon,
    )

    scenarios = {"Baseline": baseline}
    for sc_name, sc_cfg in config["scenarios"].items():
        scenarios[sc_name] = apply_scenario_envelope(
            baseline=baseline,
            variables=variables,
            scenario_cfg=sc_cfg,
            long_start_quarter=long_start,
            anchor=anchor,
        )

    forecast_dates = make_forecast_dates(last_date, horizon_total)

    # Wide forecast output.
    rows = []
    for sc_name, mat in scenarios.items():
        for h in range(horizon_total):
            rows.append(
                {
                    "scenario": sc_name,
                    "forecast_q": h + 1,
                    "quarter_end": forecast_dates[h].date().isoformat(),
                    **{v: float(mat[h, i]) for i, v in enumerate(variables)},
                }
            )
    paths_df = pd.DataFrame(rows)
    paths_df.to_csv(
        os.path.join(args.output_dir, "macro_forecast_paths.csv"),
        index=False,
        float_format="%.6f",
    )

    # Short-horizon intervals for VAR and BVAR.
    int_rows = []
    var_idx = {v: i for i, v in enumerate(var_benchmark_vars)}
    full_idx = {v: i for i, v in enumerate(variables)}
    for model_name, qmap in [("VAR", var_int), ("BVAR", bvar_int)]:
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
    intervals_df = pd.DataFrame(int_rows)
    intervals_df.to_csv(
        os.path.join(args.output_dir, "macro_forecast_short_horizon_intervals.csv"),
        index=False,
        float_format="%.6f",
    )

    irf_df = build_impulse_responses(
        model=var_model,
        variables=var_model.variables,
        horizon=int(config["outputs"]["irf_horizon_quarters"]),
        shocks=config["outputs"]["irf_shocks"],
    )
    irf_df.to_csv(
        os.path.join(args.output_dir, "macro_impulse_responses.csv"),
        index=False,
        float_format="%.6f",
    )

    anchors_payload = {
        "variables": variables,
        "anchor_values": {v: float(anchor[i]) for i, v in enumerate(variables)},
        "assumptions": config["assumptions"],
    }
    with open(os.path.join(args.output_dir, "macro_anchor_assumptions.json"), "w", encoding="utf-8") as f:
        json.dump(anchors_payload, f, indent=2)

    diagnostics = {
        "data_rows_used": int(panel.shape[0]),
        "variables": variables,
        "var": {
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
        "horizons": config["horizons"],
        "short_horizon_model_selected": short_model_choice,
        "generated_scenarios": ["Baseline"] + list(config["scenarios"].keys()),
    }
    with open(os.path.join(args.output_dir, "macro_model_diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)


if __name__ == "__main__":
    main()
