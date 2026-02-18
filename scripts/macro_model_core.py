#!/usr/bin/env python3
"""Shared macro modeling primitives for engine/backtest/validation."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class VarModel:
    intercept: np.ndarray
    coefs: np.ndarray
    sigma: np.ndarray
    lag_order: int
    variables: List[str]
    bic: float
    stable: bool


@dataclass
class ArModel:
    intercept: float
    phi: np.ndarray
    sigma: float
    lag_order: int
    bic: float


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def make_lagged_xy(y: np.ndarray, p: int) -> Tuple[np.ndarray, np.ndarray]:
    t, n = y.shape
    rows = t - p
    x = np.ones((rows, 1 + n * p), dtype=float)
    y_target = y[p:, :]
    for lag in range(1, p + 1):
        x[:, 1 + (lag - 1) * n : 1 + lag * n] = y[p - lag : t - lag, :]
    return x, y_target


def companion_matrix(coefs: np.ndarray) -> np.ndarray:
    p, n, _ = coefs.shape
    top = np.hstack([coefs[i, :, :] for i in range(p)])
    if p == 1:
        return top
    lower = np.hstack([np.eye(n * (p - 1)), np.zeros((n * (p - 1), n))])
    return np.vstack([top, lower])


def is_stable(coefs: np.ndarray, tol: float = 0.999) -> bool:
    comp = companion_matrix(coefs)
    eigvals = np.linalg.eigvals(comp)
    return np.max(np.abs(eigvals)) < tol


def fit_var_ols(y: np.ndarray, p: int, variables: List[str]) -> VarModel:
    x, y_target = make_lagged_xy(y, p)
    xtx = x.T @ x
    xty = x.T @ y_target
    beta = np.linalg.solve(xtx, xty)
    resid = y_target - x @ beta
    dof = max(1, x.shape[0] - x.shape[1])
    sigma = (resid.T @ resid) / dof

    n = y.shape[1]
    intercept = beta[0, :]
    coefs = np.zeros((p, n, n), dtype=float)
    for lag in range(1, p + 1):
        block = beta[1 + (lag - 1) * n : 1 + lag * n, :]
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


def select_var_lag(y: np.ndarray, variables: List[str], lag_candidates: List[int]) -> VarModel:
    fitted: List[VarModel] = []
    for p in lag_candidates:
        if p >= y.shape[0] - 5:
            continue
        fitted.append(fit_var_ols(y, p, variables))
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
) -> Tuple[VarModel, List[np.ndarray], np.ndarray]:
    x, y_target = make_lagged_xy(y, p)
    t_eff, k = x.shape
    n = y.shape[1]

    sigma_scale = np.var(y_target, axis=0, ddof=1)
    sigma_scale = np.where(sigma_scale <= 1e-8, 1.0, sigma_scale)

    beta_mean = np.zeros((k, n), dtype=float)
    beta_cov_blocks = []
    sigma_resid = np.zeros((n, n), dtype=float)
    xtx = x.T @ x

    for j in range(n):
        b0 = np.zeros(k, dtype=float)
        if level_vars[j]:
            b0[1 + j] = 1.0

        prior_var = np.zeros(k, dtype=float)
        prior_var[0] = (lambda4**2) * sigma_scale[j]
        for lag in range(1, p + 1):
            for m in range(n):
                idx = 1 + (lag - 1) * n + m
                tight = (lambda1**2) / (lag ** (2.0 * lambda3))
                if m != j:
                    tight *= lambda2**2
                tight *= sigma_scale[j] / sigma_scale[m]
                prior_var[idx] = tight

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

    intercept = beta_mean[0, :]
    coefs = np.zeros((p, n, n), dtype=float)
    for lag in range(1, p + 1):
        block = beta_mean[1 + (lag - 1) * n : 1 + lag * n, :]
        coefs[lag - 1, :, :] = block.T

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
    return model, beta_cov_blocks, beta_mean


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


def simulate_var_draws(
    model: VarModel,
    history: np.ndarray,
    horizon: int,
    n_sims: int,
    rng_seed: int,
    beta_cov_blocks: Optional[List[np.ndarray]] = None,
    beta_mean_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)
    p = model.lag_order
    n = history.shape[1]
    sims = np.zeros((n_sims, horizon, n), dtype=float)

    if beta_cov_blocks is not None and beta_mean_matrix is None:
        k = 1 + n * p
        beta_mean_matrix = np.zeros((k, n), dtype=float)
        beta_mean_matrix[0, :] = model.intercept
        for lag in range(1, p + 1):
            block = model.coefs[lag - 1, :, :].T
            beta_mean_matrix[1 + (lag - 1) * n : 1 + lag * n, :] = block

    for s in range(n_sims):
        hist = history.copy()
        if beta_cov_blocks is not None:
            sampled_beta = np.zeros_like(beta_mean_matrix)
            for j in range(n):
                sampled_beta[:, j] = rng.multivariate_normal(beta_mean_matrix[:, j], beta_cov_blocks[j])
            sampled_intercept = sampled_beta[0, :]
            sampled_coefs = np.zeros((p, n, n), dtype=float)
            for lag in range(1, p + 1):
                block = sampled_beta[1 + (lag - 1) * n : 1 + lag * n, :]
                sampled_coefs[lag - 1, :, :] = block.T
        else:
            sampled_intercept = model.intercept
            sampled_coefs = model.coefs

        for h in range(horizon):
            eps = rng.multivariate_normal(np.zeros(n), model.sigma)
            y_next = sampled_intercept.copy()
            for lag in range(1, p + 1):
                y_next += sampled_coefs[lag - 1, :, :] @ hist[-lag, :]
            y_next += eps
            sims[s, h, :] = y_next
            hist = np.vstack([hist, y_next])

    return sims


def quantiles_from_draws(draws: np.ndarray, quantiles: Iterable[float]) -> Dict[float, np.ndarray]:
    return {q: np.quantile(draws, q, axis=0) for q in quantiles}


def fit_ar_ols(y: np.ndarray, p: int) -> ArModel:
    if y.shape[0] <= p + 2:
        raise RuntimeError("Insufficient rows to fit AR model.")
    x = np.ones((y.shape[0] - p, 1 + p), dtype=float)
    target = y[p:]
    for lag in range(1, p + 1):
        x[:, lag] = y[p - lag : -lag]
    beta = np.linalg.lstsq(x, target, rcond=None)[0]
    resid = target - x @ beta
    sigma = float(np.std(resid, ddof=max(1, x.shape[1])))
    sigma = max(sigma, 1e-8)
    t_eff = x.shape[0]
    k_params = x.shape[1]
    rss = float(np.sum(resid**2))
    bic = np.log(max(rss / t_eff, 1e-12)) + (np.log(t_eff) * k_params) / t_eff
    return ArModel(intercept=float(beta[0]), phi=beta[1:], sigma=sigma, lag_order=p, bic=float(bic))


def select_ar_order(y: np.ndarray, p_candidates: List[int]) -> ArModel:
    models: List[ArModel] = []
    for p in p_candidates:
        if p < 1 or y.shape[0] <= p + 2:
            continue
        try:
            models.append(fit_ar_ols(y, p))
        except Exception:
            continue
    if not models:
        fallback_p = 1 if y.shape[0] > 4 else max(1, y.shape[0] - 3)
        return fit_ar_ols(y, fallback_p)
    return min(models, key=lambda m: m.bic)


def ar_point_forecast(model: ArModel, history: np.ndarray, horizon: int) -> np.ndarray:
    hist = history.astype(float).copy()
    out = np.zeros(horizon, dtype=float)
    p = model.lag_order
    for h in range(horizon):
        lags = hist[-p:][::-1]
        y_next = model.intercept + float(np.dot(model.phi, lags))
        out[h] = y_next
        hist = np.append(hist, y_next)
    return out


def simulate_ar_draws(model: ArModel, history: np.ndarray, horizon: int, n_sims: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draws = np.zeros((n_sims, horizon), dtype=float)
    p = model.lag_order
    for s in range(n_sims):
        hist = history.astype(float).copy()
        for h in range(horizon):
            lags = hist[-p:][::-1]
            y_next = model.intercept + float(np.dot(model.phi, lags)) + float(rng.normal(0.0, model.sigma))
            draws[s, h] = y_next
            hist = np.append(hist, y_next)
    return draws


def rw_point_forecast(last_value: float, horizon: int) -> np.ndarray:
    return np.full(horizon, float(last_value), dtype=float)


def simulate_rw_draws(last_value: float, innovation_sigma: float, horizon: int, n_sims: int, seed: int) -> np.ndarray:
    sigma = max(float(innovation_sigma), 1e-8)
    rng = np.random.default_rng(seed)
    shocks = rng.normal(0.0, sigma, size=(n_sims, horizon))
    return float(last_value) + np.cumsum(shocks, axis=1)


def crps_from_draws(draws: np.ndarray, y_true: float) -> float:
    x = np.asarray(draws, dtype=float)
    term1 = float(np.mean(np.abs(x - y_true)))
    term2 = 0.5 * float(np.mean(np.abs(x[:, None] - x[None, :])))
    return term1 - term2


def empirical_coverage(draws: np.ndarray, y_true: float, level: float) -> float:
    alpha = (1.0 - level) / 2.0
    lo = float(np.quantile(draws, alpha))
    hi = float(np.quantile(draws, 1.0 - alpha))
    return float(lo <= y_true <= hi)


def interval_width(draws: np.ndarray, level: float) -> float:
    alpha = (1.0 - level) / 2.0
    lo = float(np.quantile(draws, alpha))
    hi = float(np.quantile(draws, 1.0 - alpha))
    return hi - lo


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def dm_test_hac_hln(loss_model: np.ndarray, loss_benchmark: np.ndarray, horizon: int) -> Tuple[float, float, int]:
    d = np.asarray(loss_model, dtype=float) - np.asarray(loss_benchmark, dtype=float)
    d = d[np.isfinite(d)]
    t = d.shape[0]
    if t < max(8, horizon + 3):
        return float("nan"), float("nan"), max(0, horizon - 1)

    d_bar = float(np.mean(d))
    d_center = d - d_bar
    lags = max(0, horizon - 1)

    gamma0 = float(np.mean(d_center * d_center))
    long_var = gamma0
    for ell in range(1, lags + 1):
        w = 1.0 - ell / (lags + 1.0)
        cov = float(np.mean(d_center[ell:] * d_center[:-ell]))
        long_var += 2.0 * w * cov

    long_var = max(long_var, 1e-12)
    dm_stat = d_bar / math.sqrt(long_var / t)

    h = max(1, horizon)
    hln_inner = (t + 1.0 - 2.0 * h + (h * (h - 1.0) / t)) / t
    if hln_inner > 0:
        dm_stat *= math.sqrt(hln_inner)

    pvalue = 2.0 * (1.0 - _norm_cdf(abs(dm_stat)))
    return float(dm_stat), float(pvalue), lags


def transform_candidates_for_variable(variable: str) -> List[str]:
    if variable == "high_yield_spread":
        return ["level", "diff1", "demean", "log_floor_eps"]
    if variable == "housing_starts_yoy":
        return ["level", "diff1", "log_floor_eps", "diff_yoy"]
    return ["level"]


def apply_transform(series: np.ndarray, transform_id: str, eps: float = 1e-4) -> Tuple[np.ndarray, Dict[str, float]]:
    y = np.asarray(series, dtype=float)
    meta: Dict[str, float] = {"eps": float(eps), "scoring_domain": "level"}
    if transform_id == "level":
        return y.copy(), meta
    if transform_id == "diff1":
        out = np.empty_like(y)
        out[0] = 0.0
        out[1:] = y[1:] - y[:-1]
        return out, meta
    if transform_id == "diff_yoy":
        out = np.empty_like(y)
        out[0] = 0.0
        out[1:] = y[1:] - y[:-1]
        return out, meta
    if transform_id == "demean":
        mu = float(np.mean(y))
        meta["mean"] = mu
        return y - mu, meta
    if transform_id == "log_floor_eps":
        y_clip = np.maximum(y, eps)
        return np.log(y_clip), meta
    raise RuntimeError(f"Unknown transform_id: {transform_id}")


def inverse_transform_draws(draws: np.ndarray, last_level: float, transform_id: str, meta: Dict[str, float]) -> Tuple[np.ndarray, str]:
    x = np.asarray(draws, dtype=float)
    if transform_id == "level":
        return x, "level"
    if transform_id in {"diff1", "diff_yoy"}:
        return last_level + np.cumsum(x, axis=1), "level"
    if transform_id == "demean":
        return x + float(meta.get("mean", 0.0)), "level"
    if transform_id == "log_floor_eps":
        return np.exp(x), "level"
    return x, "transformed"


def build_beta_mean_matrix(model: VarModel) -> np.ndarray:
    p = model.lag_order
    n = len(model.variables)
    k = 1 + n * p
    beta_mean = np.zeros((k, n), dtype=float)
    beta_mean[0, :] = model.intercept
    for lag in range(1, p + 1):
        beta_mean[1 + (lag - 1) * n : 1 + lag * n, :] = model.coefs[lag - 1, :, :].T
    return beta_mean
