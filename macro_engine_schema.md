# Macro Engine Schema (V3 Freeze)

## 1) Core Forecast Table
File: `outputs/macro_engine/macro_forecast_paths.csv`

| Column | Type | Units | Description | Source/Transform |
|---|---|---|---|---|
| `scenario` | categorical | n/a | Forecast scenario label (`Baseline`, `Mild_Adverse`, `Severe_Adverse`, `Demographic_LowGrowth`). | scenario envelope code |
| `forecast_q` | integer | count | Quarter index in the 80-quarter horizon (1=next quarter). | generated |
| `quarter_end` | date | YYYY-MM-DD | Quarter-end date of the forecast row. | generated |
| `unemployment_rate` | numeric | percent | Quarterly unemployment rate (level). | FRED `UNRATE` |
| `labor_force_participation` | numeric | percent | Labor-force participation rate (level). | FRED `CIVPART` |
| `payroll_growth_yoy` | numeric | percent | YoY payroll change from `PAYEMS`. | YoY pct |
| `wage_growth_yoy` | numeric | percent | YoY hourly wage change (`CES3000000008`). | YoY pct |
| `headline_cpi_yoy` | numeric | percent | Headline CPI YoY (`CPIAUCSL`). | YoY pct |
| `core_cpi_yoy` | numeric | percent | Core CPI YoY (`CPILFESL`). | YoY pct |
| `pce_inflation_yoy` | numeric | percent | PCE index YoY (`PCEPI`). | YoY pct |
| `oer_inflation_yoy` | numeric | percent | Owners’ equivalent rent YoY (`CUSR0000SEHC`). | YoY pct |
| `medical_cpi_yoy` | numeric | percent | Medical CPI YoY (`CPIMEDSL`). | YoY pct |
| `real_gdp_growth_yoy` | numeric | percent | Real GDP YoY (`GDPC1`). | YoY pct |
| `industrial_production_yoy` | numeric | percent | Industrial production YoY (`INDPRO`). | YoY pct |
| `consumer_sentiment` | numeric | index | Michigan Consumer Sentiment. | level |
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
| `household_credit_growth_yoy` | numeric | percent | Household credit YoY (`CMDEBT`). | YoY pct |
| `household_networth_growth_yoy` | numeric | percent | Household net worth YoY (`BOGZ1FL192090005Q`). | YoY pct |
| `consumer_delinquency_rate` | numeric | percent | Consumer delinquency rate (`DRCCLACBS`). | level |
| `working_age_pop_growth_yoy` | numeric | percent | Working-age population growth (`LFWA64TTUSM647S`). | YoY pct |
| `population_growth_yoy` | numeric | percent | Total population growth (`POPTHM`). | YoY pct |

## 2) Canonical PD Handoff Schema
File: `outputs/macro_engine/pd_regressors_forecast_levels.csv`

| Column | Type | Units | Description |
|---|---|---|---|
| `scenario` | categorical | n/a | Scenario label. |
| `forecast_q` | integer | count | Forecast quarter index. |
| `quarter_end` | date | YYYY-MM-DD | Quarter-end timestamp. |
| `unemployment_rate` | numeric | percent | Canonical PD input level path. |
| `ust10_rate` | numeric | percent | Canonical PD input level path. |
| `hpi_yoy` | numeric | percent | Canonical PD input level path (mapped from `hpi_growth_yoy`). |

## 3) Convenience PD Derived Schema
File: `outputs/macro_engine/pd_regressors_forecast_derived.csv`

| Column | Type | Units | Description |
|---|---|---|---|
| `scenario` | categorical | n/a | Scenario label. |
| `forecast_q` | integer | count | Forecast quarter index. |
| `quarter_end` | date | YYYY-MM-DD | Quarter-end timestamp. |
| `du_6m` | numeric | percentage points | Convenience delta (default lag 2 quarters). |
| `d10y_6m` | numeric | percentage points | Convenience delta (default lag 2 quarters). |
| `hpi_yoy` | numeric | percent | Included for downstream convenience. |

## 4) Validation Evidence Files Kept In Frozen V3
Directory: `outputs/macro_engine/validation/`

Core final-result files:
- `validation_summary.json` (final pass/fail and gate details)
- `backtest_metrics.csv` (cell-level performance metrics)
- `calibration_factors.json` (applied uncertainty scaling factors)
- `pd_ablation_results.csv` (regime/scope ablation comparison)

Extended diagnostics retained for audit:
- `boundary_culprits.csv`, `coverage_fail_cells.csv`, `dm_results.csv`, `seed_registry.csv`
- `scenario_timing_checks.csv`, `special_series_sweep.csv`, `anchor_sensitivity.csv`, `regime_split_scores.csv`
- nested subfolders for assumption-set and regime-specific reruns

## 5) Notes
- Full model and governance narrative is in `macro_engine_plain_report_v3.pdf`.
- Transform definitions and validation thresholds are in `macro_engine_config.json`.
