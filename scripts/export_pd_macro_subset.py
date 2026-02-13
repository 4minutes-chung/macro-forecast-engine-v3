#!/usr/bin/env python3
"""
Export a PD-ready macro subset from macro_forecast_paths.csv with units/transform labels.
"""

import argparse
import json
import os

import pandas as pd


DEFAULT_FIELDS = [
    "unemployment_rate",
    "hpi_growth_yoy",
    "mortgage30_rate",
    "ust10_rate",
    "headline_cpi_yoy",
    "real_gdp_growth_yoy",
    "high_yield_spread",
    "consumer_delinquency_rate",
    "housing_starts_yoy",
    "retail_sales_yoy",
]

FIELD_META = {
    "unemployment_rate": {"units": "percent", "transform": "level"},
    "hpi_growth_yoy": {"units": "percent", "transform": "yoy"},
    "mortgage30_rate": {"units": "percent", "transform": "level"},
    "ust10_rate": {"units": "percent", "transform": "level"},
    "headline_cpi_yoy": {"units": "percent", "transform": "yoy"},
    "real_gdp_growth_yoy": {"units": "percent", "transform": "yoy"},
    "high_yield_spread": {"units": "percent", "transform": "level"},
    "consumer_delinquency_rate": {"units": "percent", "transform": "level"},
    "housing_starts_yoy": {"units": "percent", "transform": "yoy"},
    "retail_sales_yoy": {"units": "percent", "transform": "yoy"},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Export PD-ready macro subset.")
    parser.add_argument(
        "--input",
        default="outputs/macro_engine/macro_forecast_paths.csv",
        help="Path to macro_forecast_paths.csv",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/macro_engine/pd_macro_subset_sample.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/macro_engine/pd_macro_subset_sample.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--fields",
        default=",".join(DEFAULT_FIELDS),
        help="Comma-separated list of fields to include.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    df = pd.read_csv(args.input)
    cols = ["scenario", "forecast_q", "quarter_end"] + fields
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")

    out_df = df[cols].copy()
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    out_df.to_csv(args.output_csv, index=False, float_format="%.6f")

    meta = {
        "source_file": args.input,
        "fields": fields,
        "field_metadata": {f: FIELD_META.get(f, {}) for f in fields},
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()

