# Tableau Dashboard Flow (Step-by-Step) + Requirements Check

## Part A: Tableau dashboard flow (simple and submission-ready)

### 1) Generate fresh Tableau CSVs
1. Open terminal.
2. Run:
   ```bash
   cd /Users/vivekhanumanthu/Desktop/metro_big_data/project
   python3 scripts/run_pipeline.py --data-path 'data/raw/MetroPT3(AirCompressor).csv' --run-scalability
   ```
3. Confirm CSVs exist in `data/tableau_exports/`.

### 2) Connect all CSVs in Tableau
1. Open Tableau Desktop.
2. For each CSV in `project/data/tableau_exports/`, use `Connect -> Text file`.
3. For each source, create an `Extract` (not live text).
4. Validate key field types:
   - Dimensions: `algorithm`, `split`, `stage`, `class`, `experiment`, `app_family`, `family_risk`
   - Measures: `duration_seconds`, `row_count`, `anomaly_rate`, `score_p95`, `score_p99`, `estimated_cost_usd`, `expected_profit_usd`

### 3) Build Dashboard 1: Data Quality + Pipeline Monitoring
1. Sheet `ETL Stage Runtime`: bar chart (`stage` vs `duration_seconds`) from `etl_stage_timings.csv`.
2. Sheet `Data Quality`: table (`metric`, `value`) from `data_quality_report.csv`.
3. Sheet `Pipeline Lineage`: table (`stage`, `status`, `notes`) from `pipeline_lineage.csv`.
4. Sheet `Class Distribution`: bar (`class`, `row_count`) from `class_distribution.csv`.
5. Add filter controls: `stage`, `class`.

### 4) Build Dashboard 2: Anomaly Monitoring + Feature Importance
1. Sheet `Model Metrics`: bar by `algorithm`; metric switch for `anomaly_rate`, `score_p95`, `score_p99`.
2. Sheet `Anomaly Score Trend`: line/histogram on `anomaly_score` from `anomaly_scores.csv` by `algorithm`.
3. Sheet `Top Features`: sorted bar (`feature`, `importance`) from `feature_importance.csv`.
4. Add parameter `Metric Selector` with values: `anomaly_rate`, `score_p95`, `score_p99`.

### 5) Build Dashboard 3: Business Insights + Recommendations
1. Sheet `Risk Distribution`: stacked bar using `app_family`, `family_risk`, `row_count`.
2. Sheet `Recommendations`: table from `business_insights.csv`.
3. Sheet `Business Metrics`: KPI cards from `business_metric_alignment.csv` (`expected_profit_usd`).
4. Add dashboard action: selecting a risk segment filters recommendations and KPI cards.

### 6) Build Dashboard 4: Scalability + Cost Analysis
1. Sheet `Strong Scaling`: line (`partitions` vs `duration_seconds`) from `scalability_strong.csv`.
2. Sheet `Weak Scaling`: line (`rows` vs `throughput_rows_per_sec`) from `scalability_weak.csv`.
3. Sheet `Cost Curve`: line (`partitions` vs `estimated_cost_usd`) from `cost_performance.csv`.
4. Sheet `Resource Allocation`: table from `resource_allocation.csv`.
5. Sheet `Bottlenecks`: table/cards from `bottleneck_analysis.csv`.
6. Add parameter `Experiment Toggle`: `strong_scaling`, `weak_scaling`.

### 7) Add best-practice interactions
1. Use extracts for all sources.
2. Add LOD fields:
   - `{ FIXED [algorithm] : AVG([anomaly_rate]) }`
   - `{ FIXED [stage] : SUM([duration_seconds]) }`
   - `{ FIXED [experiment] : AVG([estimated_cost_usd]) }`
3. Add filter actions and highlight actions across all 4 dashboards.
4. Add short annotations for key insights in each dashboard.
5. Add mobile layout in `Device Preview` (2-3 key visuals per dashboard).

### 8) Final export for submission
1. `Dashboard -> Export Image` for slides.
2. `Worksheet -> Export Data` for raw evidence.
3. `File -> Export as PDF` for full story.

---

## Part B: Requirements coverage check (current repo status)

Legend: `PASS` = implemented, `PARTIAL` = present but not fully explicit, `GAP` = missing

### 1) PySpark Data Engineering
- 1a) Data ingestion/storage design
  - SparkSession config: `PASS`
  - Partitioning strategy: `PASS`
  - Storage format + justification: `PASS`
  - Data validation at ingestion: `PASS`
- 1b) Distributed processing pipeline
  - Broadcast join: `PASS`
  - Persist/unpersist memory strategy: `PASS`
  - Error handling + lineage: `PASS`
- 1c) Performance optimization
  - DataFrame vs RDD usage justification: `PARTIAL` (DataFrame-first is implemented; explicit write-up is minimal)
  - Caching strategy documentation: `PARTIAL` (implemented in code, limited narrative)
  - Spark UI screenshot evidence: `PASS`
  - Shuffle + partition tuning: `PASS`

### 2) Scalability and Distributed ML
- 2a) MLlib implementation
  - 3 MLlib algorithms: `PASS`
  - sklearn single-node baseline: `PASS`
  - Custom transformer: `PASS`
  - Model serialization: `PASS`
- 2b) Distributed training/tuning
  - CrossValidator + parallelism: `PASS`
  - Hyperparameter grid constraints: `PASS`
  - Model checkpointing: `PASS` (feature dataset checkpoint before training)
  - Resource allocation justification: `PASS`
- 2c) Scalability analysis
  - Strong scaling: `PASS`
  - Weak scaling: `PASS`
  - Bottleneck identification: `PASS`
  - Cost-performance tradeoff: `PASS`

### 3) Tableau Visualization
- 3a) 4 required dashboards: `PASS`
- 3b) Best practices
  - Extracts: `PASS`
  - LOD expressions: `PASS`
  - Parameters: `PASS`
  - Mobile design: `PASS`
- 3c) Storytelling
  - Narrative flow: `PASS`
  - Annotations: `PASS`
  - Actions/filters: `PASS`
  - Export for stakeholders: `PASS`

### 4) Model Evaluation
- 4a) Distributed evaluation metrics
  - Train/validation/test + temporal handling: `PASS`
  - Cross-validation with stratification for imbalanced data: `PARTIAL` (stratified train/val/test exists; CV fold stratification is not explicit)
  - Bootstrap confidence intervals: `PASS`
  - Business metric alignment: `PASS`

## Quick recommendation before final submission
1. Add 5-8 lines in `README.md` explicitly justifying:
   - why DataFrame API over RDD,
   - caching/persist strategy,
   - CV stratification limitation + mitigation.
2. Keep `data/raw/` and `data/checkpoints/` untracked in Git (already configured in `project/.gitignore`).
