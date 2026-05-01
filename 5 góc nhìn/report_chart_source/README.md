# Report Chart Source

This folder contains reproducible source code for the two final EDA charts used in `report draft 304.docx`.

## Why this exists

The report charts should not be one-off PNGs. This script rebuilds them from:

- `data/clean/*.parquet`
- `outputs/capital_reallocation_priority_candidates.csv`
- `outputs/sku_quarantine_candidates.csv`

It condenses the stronger notebook visuals into final-report figures:

- `góc nhìn 3/promotion_effectiveness.ipynb`: promo margin gap, promo/revenue correlation, recent-campaign customer mix
- `góc nhìn 4/product_portfolio.ipynb`: SKU GP concentration, return reason Pareto, size diagnostic
- `góc nhìn 6/seasonal_capital_misallocation.ipynb`: peak/off-peak promo timing, stockout/capital signals

## Run

From the workspace root:

```powershell
python "The_4_Outliers_VinDatathon\5 góc nhìn\report_chart_source\make_report_charts.py"
```

Outputs:

- `outputs/fig_report_insight_a_promo_capital_trap.png`
- `outputs/fig_report_insight_b_portfolio_drag.png`
- `outputs/report_chart_metric_snapshot.csv`

To also replace the PNGs used by the current docx rewrite helper:

```powershell
python "The_4_Outliers_VinDatathon\5 góc nhìn\report_chart_source\make_report_charts.py" --copy-to-report-revision
```

## Chart intent

Figure A tells the combined Promo Capital Trap story:

- Promo has negative unit economics.
- Recent promo sales mostly go to existing customers.
- Promo budget is deployed off-peak while Apr-Jun demand is naturally stronger.

Figure B tells the Portfolio Drag story:

- Net GP is concentrated in the top SKU set.
- wrong_size is the largest refund pool.
- Size-level return quantity rates are almost flat, so the fix is PDP/size-guide quality rather than one specific size.
