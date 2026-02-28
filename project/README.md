# Metro Air Compressor Pipeline - PySpark ETL, MLlib, and Tableau Exports

This repository implements a Spark-based pipeline for processing metro system air compressor telemetry (default dataset: `data/raw/MetroPT3(AirCompressor).csv`). The workflow ingests raw CSV sensor data, performs validation and enrichment, engineers features, trains anomaly detection and classification models, and exports Tableau-ready CSVs for visualization.

## Quick Start

```bash
cd project
bash scripts/setup_environment.sh
python3 scripts/run_pipeline.py --run-scalability
```

## What the pipeline does

- Ingestion and validation with required-column checks and data quality profiling.
- PySpark DataFrame ETL with broadcast lookup enrichment for operating modes.
- Custom domain transformer (`SensorRiskScoreTransformer`) that computes a composite risk score.
- Distributed model training with MLlib algorithms:
  - Logistic Regression
  - Random Forest
  - Gradient Boosted Trees (GBT)
- Hyperparameter tuning via `CrossValidator` + `ParamGridBuilder`.
- Single-node scikit-learn baselines for comparison (IsolationForest, etc.).
- Model serialization:
  - Spark model persistence in `data/models/*_spark_model`
  - Pickled scikit-learn models in `data/models/`
- Scalability experiments (strong and weak scaling) and cost-performance analysis.

## Tableau-ready outputs

Generated CSVs are written to `data/tableau_exports/` for dashboarding, including `etl_stage_timings.csv`, `data_quality_report.csv`, `model_metrics.csv`, `feature_importance.csv`, `scalability_strong.csv`, and more.

## Core directories

```text
project/
├── scripts/
│   ├── run_pipeline.py
│   └── performance_profiler.py
├── config/
│   ├── spark_config.yaml
│   └── tableau_config.json
├── data/
│   ├── raw/
│   ├── curated_parquet/
│   ├── models/
│   └── tableau_exports/
├── notebooks/
├── tableau/
└── tests/
```

## Notes

- Default input is `data/raw/MetroPT3(AirCompressor).csv`. Use `--data-path` to run on a different CSV.
- Processed data are written in Parquet and partitioned by `class` for fast queries.
- Use the CSVs in `data/tableau_exports/` to build dashboards described in `tableau/README_tableau.md`.

## License & Attribution

This project bundles processing code and example exports; raw datasets may be subject to third-party licensing. Include original dataset citations when publishing results.
# Metro Air Compressor Pipeline - PySpark ETL, MLlib, and Tableau Exports

This repository implements a Spark-based pipeline for processing metro system air compressor telemetry (default dataset: `data/raw/MetroPT3(AirCompressor).csv`). The workflow ingests raw CSV sensor data, performs validation and enrichment, engineers features, trains anomaly detection and classification models, and exports Tableau-ready CSVs for visualization.

## Quick Start

```bash
cd project
bash scripts/setup_environment.sh
python3 scripts/run_pipeline.py --run-scalability
```

## What the pipeline does

- Ingestion and validation with required-column checks and data quality profiling.
- PySpark DataFrame ETL with broadcast lookup enrichment for operating modes.
- Custom domain transformer (`SensorRiskScoreTransformer`) that computes a composite risk score.
- Distributed model training with MLlib algorithms:
  - Logistic Regression
  - Random Forest
  - Gradient Boosted Trees (GBT)
# Metro Air Compressor Pipeline - PySpark ETL, MLlib, and Tableau Exports

This repository implements a Spark-based pipeline for processing metro system air compressor telemetry (default dataset: `MetroPT3(AirCompressor).csv`). The workflow ingests raw CSV sensor data, performs validation and enrichment, engineers features, trains anomaly detection and classification models, and exports Tableau-ready CSVs for visualization.

- Model serialization:
  - Spark model persistence in `data/models/*_spark_model`
  - Pickled sklearn model in `data/models/sklearn_baseline.pkl`
- Scalability experiments:
  - Strong scaling (fixed workload, varied partitions)
  - Weak scaling (scaled workload with partitions)
- Cost-performance export and bottleneck analysis export.
- Curated Parquet write partitioned by `class`.

## Tableau-ready offline CSV outputs
- `data_quality_report.csv`
- `pipeline_lineage.csv`
- `class_distribution.csv`
- `feature_importance.csv`
- `family_risk_distribution.csv`
- `resource_allocation.csv`
- `bottleneck_analysis.csv`
- `scalability_strong.csv`
- `scalability_weak.csv`
- `cost_performance.csv`

See `tableau/README_tableau.md` for dashboard mapping.

## Core directories

```text
project/
├── scripts/
│   ├── run_pipeline.py
│   └── performance_profiler.py
├── config/
│   ├── spark_config.yaml
│   └── tableau_config.json
├── data/
│   ├── samples/
│   ├── schemas/
│   ├── checkpoints/
│   ├── curated_parquet/
│   ├── models/
│   ├── spark_ui_jobs/
│   └── tableau_exports/
├── notebooks/

## Notes

- Default input is `data/samples/sample_permissions.csv` for offline operation.
- You can pass a larger local CSV with `--data-path`.
- Spark UI screenshots are environment-dependent and should be captured during your local run.

## Simple dashboards you can build from these CSVs

Use these files from `data/tableau_exports/`:

- `etl_stage_timings.csv`
- `data_quality_report.csv`
- `pipeline_lineage.csv`
- `class_distribution.csv`
- `model_metrics.csv`
- `cv_results.csv`
- `feature_importance.csv`
- `business_insights.csv`
- `business_metric_alignment.csv`
- `scalability_strong.csv`
- `scalability_weak.csv`
- `cost_performance.csv`
- `resource_allocation.csv`
- `bottleneck_analysis.csv`

Suggested simple dashboards:

1. Pipeline Health Dashboard
- Charts: ETL stage runtime, data quality summary table, lineage/status table.
- CSVs: `etl_stage_timings.csv`, `data_quality_report.csv`, `pipeline_lineage.csv`.

2. Data Balance Dashboard
- Charts: class distribution bars, train/validation/test split summary.
- CSVs: `class_distribution.csv`, `split_summary.csv`.

3. Model Performance Dashboard
- Charts: model comparison (Accuracy/F1/AUC), CV comparison, top feature importances.
- CSVs: `model_metrics.csv`, `cv_results.csv`, `feature_importance.csv`.

4. Scalability + Cost Dashboard
- Charts: strong scaling line, weak scaling line, cost-vs-performance line, bottleneck notes.
- CSVs: `scalability_strong.csv`, `scalability_weak.csv`, `cost_performance.csv`, `resource_allocation.csv`, `bottleneck_analysis.csv`.

## How to create dashboards (Tableau quick steps)

1. Open Tableau Desktop and connect each CSV from `data/tableau_exports/` using `Connect -> Text file`.
2. For each source, create an Extract and verify data types (dimensions vs measures).
3. Build one worksheet per chart:
- `stage` vs `duration_seconds` for ETL runtime
- `class` vs `row_count` for class balance
- `algorithm` vs metrics (`accuracy`, `f1`, `auc`) for model comparison
- `partitions`/`rows` vs time/throughput/cost for scalability
4. Create 4 dashboards (Pipeline Health, Data Balance, Model Performance, Scalability + Cost) and drag relevant worksheets to each.
5. Add common filters/controls:
- `algorithm`
- `stage`
- `experiment`
- metric selector parameter (`accuracy` / `f1` / `auc`)
6. Add dashboard actions so clicking one chart filters related charts.
7. Publish or export:
- `File -> Export as PDF` for report sharing
- `Dashboard -> Export Image` for slides
