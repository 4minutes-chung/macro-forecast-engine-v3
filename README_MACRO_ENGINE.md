# Macro Engine Run Guide

## Purpose
- Forecast a 20-year quarterly U.S. macro path for PD/LGD inputs.
- Use VAR and Minnesota-style BVAR for 1-2 years.
- Use anchor-bridge for years 2-5.
- Use scenario envelopes for years 5-20.

## Inputs
- Config: `macro_engine_config.json`
- Data output path for panel build:
  - `data/macro_panel_quarterly_raw.csv`
  - `data/macro_panel_quarterly_model.csv`
  - `data/macro_panel_metadata.json`

## Run commands
```bash
python3 scripts/fetch_macro_panel_fred.py \
  --raw-output data/macro_panel_quarterly_raw.csv \
  --model-output data/macro_panel_quarterly_model.csv \
  --metadata-output data/macro_panel_metadata.json

python3 scripts/run_macro_forecast_engine.py \
  --config macro_engine_config.json \
  --output-dir outputs/macro_engine
```

## Outputs
- `outputs/macro_engine/macro_forecast_paths.csv`
- `outputs/macro_engine/macro_forecast_short_horizon_intervals.csv`
- `outputs/macro_engine/macro_model_diagnostics.json`
- `outputs/macro_engine/macro_anchor_assumptions.json`
- `outputs/macro_engine/macro_impulse_responses.csv`

## Schema (Auto-synced)
<!-- SCHEMA:START -->

| Column | Type | Units | Description | Source/Transform |
|---|---|---|---|---|
| `scenario` | categorical | n/a | Forecast scenario label (Baseline, Mild_Adverse, Severe_Adverse, Demographic_LowGrowth). | scenario envelope code |
| `forecast_q` | integer | count | Quarter index in the 80-quarter horizon (1=next quarter). | generated |
| `quarter_end` | date | YYYY-MM-DD | Quarter-end date of the forecast row. | `forecast_dates` |
| `unemployment_rate` | numeric | percent | Quarterly unemployment rate (level). | FRED `UNRATE` |
| `labor_force_participation` | numeric | percent | Labor-force participation rate (level). | FRED `CIVPART` |
| `payroll_growth_yoy` | numeric | percent | YoY payroll change based on `PAYEMS`. | YoY pct |
| `wage_growth_yoy` | numeric | percent | YoY hourly wage change (`CES3000000008`). | YoY pct |
| `headline_cpi_yoy` | numeric | percent | Headline CPI YoY (`CPIAUCSL`). | YoY pct |
| `core_cpi_yoy` | numeric | percent | Core CPI YoY (`CPILFESL`). | YoY pct |
| `pce_inflation_yoy` | numeric | percent | PCE index YoY (`PCEPI`). | YoY pct |
| `oer_inflation_yoy` | numeric | percent | Owners’ equivalent rent YoY (`CUSR0000SEHC`). | YoY pct |
| `medical_cpi_yoy` | numeric | percent | Medical CPI YoY (`CPIMEDSL`). | YoY pct |
| `real_gdp_growth_yoy` | numeric | percent | Real GDP YoY (`GDPC1`). | YoY pct |
| `industrial_production_yoy` | numeric | percent | Industrial production YoY (`INDPRO`). | YoY pct |
| `consumer_sentiment` | numeric | index | Michigan Consumer Sentiment (`UMCSENT`). | level |
| `retail_sales_yoy` | numeric | percent | Retail sales YoY (`RSAFS`). | YoY pct |
| `hpi_growth_yoy` | numeric | percent | FHFA HPI YoY (`USSTHPI`). | YoY pct |
| `housing_starts_yoy` | numeric | percent | Housing starts YoY (`HOUST`). | YoY pct |
| `building_permits_yoy` | numeric | percent | Building permits YoY (`PERMIT`). | YoY pct |
| `months_supply_homes` | numeric | months | Months’ supply of homes (`MSACSR`). | level |
| `rent_inflation_yoy` | numeric | percent | Rent inflation YoY (`CUSR0000SEHA`). | YoY pct |
| `mortgage30_rate` | numeric | percent | 30-year mortgage rate (`MORTGAGE30US`). | level |
| `ust10_rate` | numeric | percent | 10-year UST yield (`GS10`). | level |
| `high_yield_spread` | numeric | percent | BAML HY spread (`BAMLH0A0HYM2`). | level |
| `prime_rate` | numeric | percent | Prime bank loan rate (`DPRIME`). | level |
| `fed_funds_rate` | numeric | percent | Fed funds effective rate (`FEDFUNDS`). | level |
| `household_credit_growth_yoy` | numeric | percent | Household credit card/auto YoY (`CMDEBT`). | YoY pct |
| `household_networth_growth_yoy` | numeric | percent | Household net worth YoY (`BOGZ1FL192090005Q`). | YoY pct |
| `consumer_delinquency_rate` | numeric | percent | Consumer delinquency rate (`DRCCLACBS`). | level |
| `working_age_pop_growth_yoy` | numeric | percent | Working-age population growth (`LFWA64TTUSM647S`). | YoY pct |
| `population_growth_yoy` | numeric | percent | Total population growth (`POPTHM`). | YoY pct |

<!-- SCHEMA:END -->

## Sync instructions (for you)
```bash
python3 scripts/sync_schema_to_readme.py --schema macro_engine_schema.md --readme README_MACRO_ENGINE.md
```

## Notes
- `macro_forecast_paths.csv` includes scenarios:
  - `Baseline`
  - `Mild_Adverse`
  - `Severe_Adverse`
  - `Demographic_LowGrowth`
- VAR is estimated on a stable benchmark subset.
- BVAR runs on full variable set and is used for short-horizon production path when configured.
