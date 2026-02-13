#!/usr/bin/env python3
"""
Fetch a multi-variable U.S. macro panel from FRED and build quarterly datasets.

Outputs:
- data/macro_panel_quarterly_raw.csv
- data/macro_panel_quarterly_model.csv
- data/macro_panel_metadata.json
"""

import argparse
import json
import os
import urllib.parse

import numpy as np
import pandas as pd


SERIES_SPECS = [
    # Labor
    {"var": "unemployment_rate", "fred": "UNRATE", "agg": "mean", "transform": "level"},
    {"var": "labor_force_participation", "fred": "CIVPART", "agg": "mean", "transform": "level"},
    {"var": "payroll_growth_yoy", "fred": "PAYEMS", "agg": "mean", "transform": "yoy"},
    {"var": "wage_growth_yoy", "fred": "CES3000000008", "agg": "mean", "transform": "yoy"},
    # Prices
    {"var": "headline_cpi_yoy", "fred": "CPIAUCSL", "agg": "mean", "transform": "yoy"},
    {"var": "core_cpi_yoy", "fred": "CPILFESL", "agg": "mean", "transform": "yoy"},
    {"var": "pce_inflation_yoy", "fred": "PCEPI", "agg": "mean", "transform": "yoy"},
    {"var": "oer_inflation_yoy", "fred": "CUSR0000SEHC", "agg": "mean", "transform": "yoy"},
    {"var": "medical_cpi_yoy", "fred": "CPIMEDSL", "agg": "mean", "transform": "yoy"},
    # Activity
    {"var": "real_gdp_growth_yoy", "fred": "GDPC1", "agg": "last", "transform": "yoy"},
    {"var": "industrial_production_yoy", "fred": "INDPRO", "agg": "mean", "transform": "yoy"},
    {"var": "consumer_sentiment", "fred": "UMCSENT", "agg": "mean", "transform": "level"},
    {"var": "retail_sales_yoy", "fred": "RSAFS", "agg": "mean", "transform": "yoy"},
    # Housing
    {"var": "hpi_growth_yoy", "fred": "USSTHPI", "agg": "last", "transform": "yoy"},
    {"var": "housing_starts_yoy", "fred": "HOUST", "agg": "mean", "transform": "yoy"},
    {"var": "building_permits_yoy", "fred": "PERMIT", "agg": "mean", "transform": "yoy"},
    {"var": "months_supply_homes", "fred": "MSACSR", "agg": "mean", "transform": "level"},
    {"var": "rent_inflation_yoy", "fred": "CUSR0000SEHA", "agg": "mean", "transform": "yoy"},
    # Rates / credit
    {"var": "mortgage30_rate", "fred": "MORTGAGE30US", "agg": "mean", "transform": "level"},
    {"var": "ust10_rate", "fred": "GS10", "agg": "mean", "transform": "level"},
    {"var": "high_yield_spread", "fred": "BAMLH0A0HYM2", "agg": "mean", "transform": "level"},
    {"var": "prime_rate", "fred": "DPRIME", "agg": "mean", "transform": "level"},
    {"var": "fed_funds_rate", "fred": "FEDFUNDS", "agg": "mean", "transform": "level"},
    # Household balance sheet
    {"var": "household_credit_growth_yoy", "fred": "CMDEBT", "agg": "mean", "transform": "yoy"},
    {"var": "household_networth_growth_yoy", "fred": "BOGZ1FL192090005Q", "agg": "last", "transform": "yoy"},
    {"var": "consumer_delinquency_rate", "fred": "DRCCLACBS", "agg": "last", "transform": "level"},
    # Demographics
    {"var": "working_age_pop_growth_yoy", "fred": "LFWA64TTUSM647S", "agg": "mean", "transform": "yoy"},
    {"var": "population_growth_yoy", "fred": "POPTHM", "agg": "mean", "transform": "yoy"},
]


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch macro panel from FRED.")
    parser.add_argument(
        "--raw-output",
        default="data/macro_panel_quarterly_raw.csv",
        help="Path for raw quarterly panel output.",
    )
    parser.add_argument(
        "--model-output",
        default="data/macro_panel_quarterly_model.csv",
        help="Path for transformed model panel output.",
    )
    parser.add_argument(
        "--metadata-output",
        default="data/macro_panel_metadata.json",
        help="Path for metadata output.",
    )
    return parser.parse_args()


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def fetch_fred_series(series_id: str) -> pd.DataFrame:
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=" + urllib.parse.quote(series_id)
    df = pd.read_csv(url)
    if "observation_date" not in df.columns or series_id not in df.columns:
        raise RuntimeError(f"Unexpected FRED payload for series {series_id}")
    out = df.rename(columns={"observation_date": "date", series_id: "value"}).copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["date", "value"])
    out["quarter_end"] = out["date"].dt.to_period("Q").dt.to_timestamp("Q")
    return out[["quarter_end", "value"]]


def aggregate_quarterly(df: pd.DataFrame, agg: str) -> pd.Series:
    grp = df.groupby("quarter_end")["value"]
    if agg == "mean":
        return grp.mean()
    if agg == "last":
        return grp.last()
    raise ValueError(f"Unsupported aggregation method: {agg}")


def transform_series(level_series: pd.Series, transform: str) -> pd.Series:
    if transform == "level":
        return level_series
    if transform == "yoy":
        return level_series.pct_change(4) * 100.0
    raise ValueError(f"Unsupported transform: {transform}")


def build_panels():
    raw_cols = {}
    model_cols = {}
    meta = {"series": []}

    for spec in SERIES_SPECS:
        var_name = spec["var"]
        fred_id = spec["fred"]
        agg = spec["agg"]
        transform = spec["transform"]

        series_df = fetch_fred_series(fred_id)
        q_series = aggregate_quarterly(series_df, agg).sort_index()
        raw_cols[var_name] = q_series
        model_cols[var_name] = transform_series(q_series, transform)

        meta["series"].append(
            {
                "var": var_name,
                "fred_id": fred_id,
                "aggregation": agg,
                "transform": transform,
            }
        )

    raw_panel = pd.DataFrame(raw_cols).sort_index()
    model_panel = pd.DataFrame(model_cols).sort_index()

    # Keep complete-case model rows to make estimation deterministic/reproducible.
    model_panel = model_panel.dropna(axis=0, how="any").copy()
    raw_panel = raw_panel.loc[model_panel.index].copy()

    meta["n_variables"] = len(SERIES_SPECS)
    meta["n_rows_model"] = int(model_panel.shape[0])
    meta["start_quarter"] = str(model_panel.index.min().date()) if len(model_panel) else None
    meta["end_quarter"] = str(model_panel.index.max().date()) if len(model_panel) else None
    return raw_panel, model_panel, meta


def main():
    args = parse_args()
    raw_panel, model_panel, meta = build_panels()
    if model_panel.shape[0] < 80:
        raise RuntimeError(
            f"Model panel too short after alignment: {model_panel.shape[0]} rows (need >= 80)."
        )

    ensure_parent(args.raw_output)
    ensure_parent(args.model_output)
    ensure_parent(args.metadata_output)

    raw_panel.to_csv(args.raw_output, index_label="quarter_end", float_format="%.6f")
    model_panel.to_csv(args.model_output, index_label="quarter_end", float_format="%.6f")
    with open(args.metadata_output, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()

