# Tableau Offline Build Guide

## 1) Generate CSVs from pipeline

```bash
cd /Users/vivekhanumanthu/Desktop/metro_big_data/project
python3 scripts/run_pipeline.py --data-path 'data/raw/MetroPT3(AirCompressor).csv' --run-scalability
```

All dashboard CSVs are generated in `project/data/tableau_exports/`.

## 2) CSVs to add in Tableau

Add these files as Text sources:

- `etl_stage_timings.csv`
- `data_quality_report.csv`
- `pipeline_lineage.csv`
- `class_distribution.csv`
- `split_summary.csv`
- `model_metrics.csv`
- `cv_results.csv`
- `feature_importance.csv`
- `family_risk_distribution.csv`
- `business_insights.csv`
- `business_metric_alignment.csv`
- `bootstrap_confidence_intervals.csv`
- `resource_allocation.csv`
- `bottleneck_analysis.csv`
- `scalability_strong.csv`
- `scalability_weak.csv`
- `cost_performance.csv`

## 3) Connect all CSVs step-by-step

1. Open Tableau Desktop.
2. Click `Connect` -> `Text file`.
3. Select one CSV from `project/data/tableau_exports/`.
4. Repeat until all listed CSVs are added.
5. For each data source:
   - switch to `Extract`
   - click `Refresh`
6. Verify field roles:
   - dimensions: `algorithm`, `split`, `stage`, `class`, `family_risk`, `app_family`, `experiment`
   - measures: `row_count`, `duration_seconds`, `accuracy`, `f1`, `auc`, `estimated_cost_usd`, `expected_profit_usd`

## 4) Build simple dashboard 1: Data quality + pipeline monitoring

Keep it basic with 3 sheets:

1. Bar chart: `stage` vs `duration_seconds` (`etl_stage_timings.csv`)
2. Table: `metric`, `value` (`data_quality_report.csv`)
3. Bar chart: `class` vs `row_count` (`class_distribution.csv`)

Optional filter: `class`

## 5) Build simple dashboard 2: Model performance + feature importance

Keep it basic with 3 sheets:

1. Bar chart: `algorithm` vs `accuracy` (`model_metrics.csv`, `split = test`)
2. Bar chart: `algorithm` vs `cv_metric_auc` (`cv_results.csv`)
3. Bar chart: top 10 `feature` vs `importance` (`feature_importance.csv`)

## 6) Build simple dashboard 3: Business insights

Keep it basic with 3 sheets:

1. Stacked bar: `app_family` with color `family_risk` (`family_risk_distribution.csv`)
2. Table: `insight`, `value`, `recommendation` (`business_insights.csv`)
3. KPI text: `expected_profit_usd`, `tp`, `fp`, `fn` (`business_metric_alignment.csv`, use portfolio row)

## 7) Build simple dashboard 4: Scalability + cost

Keep it basic with 3 sheets:

1. Line: `partitions` vs `duration_seconds` (`scalability_strong.csv`)
2. Line: `rows` vs `throughput_rows_per_sec` (`scalability_weak.csv`)
3. Line: `partitions` vs `estimated_cost_usd` (`cost_performance.csv`)

## 8) Keep interactions minimal

1. Use only one global filter if needed (`algorithm`).
2. Skip advanced actions unless required.
3. LOD fields are optional for this simplified version.

## 9) Export for submission

1. `Dashboard` -> `Export Image`
2. `Worksheet` -> `Export` -> `Data`
3. `File` -> `Export as PDF`

## 10) Final checks (simple)

1. Refresh extracts after each pipeline run.
2. Ensure no broken fields (red `!` icons).
3. Check each dashboard has readable titles and labels.
