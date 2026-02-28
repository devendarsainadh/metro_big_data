from __future__ import annotations

import argparse
import json
import pickle
import re
import shutil
import time
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from pyspark import StorageLevel
from pyspark.ml import Transformer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.param.shared import Param, Params
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier as SkRandomForestClassifier
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from performance_profiler import PerformanceProfiler


@dataclass
class StageLineage:
    stage: str
    status: str
    input_rows: int
    output_rows: int
    notes: str


class SensorRiskScoreTransformer(Transformer):
    """Domain transformer that scores rows using selected sensor signals."""

    inputCols = Param(Params._dummy(), "inputCols", "Permission columns used for risk score")
    outputCol = Param(Params._dummy(), "outputCol", "Risk score output column")

    def __init__(self, inputCols: list[str] | None = None, outputCol: str = "permission_risk_score"):
        super().__init__()
        self._setDefault(inputCols=[], outputCol="permission_risk_score")
        if inputCols is not None:
            self._set(inputCols=inputCols)
        self._set(outputCol=outputCol)

    def _transform(self, dataset: DataFrame) -> DataFrame:
        cols = [F.col(c).cast("double") for c in self.getOrDefault(self.inputCols)]
        if not cols:
            return dataset.withColumn(self.getOrDefault(self.outputCol), F.lit(0.0))
        expr = reduce(lambda x, y: x + y, cols)
        return dataset.withColumn(self.getOrDefault(self.outputCol), expr.cast(DoubleType()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PySpark ETL + MLlib training pipeline")
    parser.add_argument(
        "--data-path",
        default="data/raw/MetroPT3(AirCompressor).csv",
        help="Input CSV path",
    )
    parser.add_argument(
        "--output-dir",
        default="data/tableau_exports",
        help="Directory for Tableau-ready CSV outputs",
    )
    parser.add_argument(
        "--run-scalability",
        action="store_true",
        help="Run strong/weak scaling experiments",
    )
    return parser.parse_args()


def load_config(path: str = "config/spark_config.yaml") -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def build_spark(config: dict[str, Any], root: Path) -> SparkSession:
    builder = (
        SparkSession.builder.appName(config.get("app_name", "NATICUSdroidPipeline"))
        .master(config.get("master", "local[*]"))
        .config("spark.sql.shuffle.partitions", str(config.get("shuffle_partitions", 8)))
        .config("spark.default.parallelism", str(config.get("default_parallelism", 8)))
        .config("spark.sql.adaptive.enabled", str(config.get("adaptive_query_execution", True)).lower())
        .config("spark.eventLog.enabled", str(config.get("event_log_enabled", False)).lower())
    )

    for key in (
        "spark.executor.instances",
        "spark.executor.cores",
        "spark.executor.memory",
        "spark.driver.memory",
    ):
        if key in config:
            builder = builder.config(key, str(config[key]))

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel(config.get("log_level", "WARN"))
    checkpoint_dir = root / "data" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    spark.sparkContext.setCheckpointDir(str(checkpoint_dir))
    return spark


def _ensure_lowercase_columns(df: DataFrame) -> DataFrame:
    for old in df.columns:
        new = old.strip().lower()
        if old != new:
            df = df.withColumnRenamed(old, new)
    return df


def _sanitize_columns(df: DataFrame) -> DataFrame:
    used: dict[str, int] = {}
    for old in df.columns:
        base = re.sub(r"[^a-z0-9_]", "_", old.strip().lower())
        base = re.sub(r"_+", "_", base).strip("_")
        if not base:
            base = "col"
        idx = used.get(base, 0)
        if idx == 0:
            new = base
        else:
            new = f"{base}_{idx + 1}"
        used[base] = idx + 1
        if old != new:
            df = df.withColumnRenamed(old, new)
    return df


def ingest_and_validate(spark: SparkSession, csv_path: Path) -> tuple[DataFrame, list[dict[str, Any]], list[str]]:
    raw = spark.read.option("header", True).option("inferSchema", True).csv(str(csv_path))
    raw = _ensure_lowercase_columns(raw)
    raw = _sanitize_columns(raw)

    # Drop autogenerated index-like columns from CSV exports.
    for idx_col in ("unnamed_0",):
        if idx_col in raw.columns:
            raw = raw.drop(idx_col)

    if "timestamp" in raw.columns:
        raw = raw.withColumn("timestamp", F.to_timestamp("timestamp"))

    label_candidates = ["class", "result", "label", "target"]
    label_col = next((c for c in label_candidates if c in raw.columns), None)
    if label_col is not None and label_col != "class":
        raw = raw.withColumnRenamed(label_col, "class")

    excluded = {"class", "timestamp"}
    numeric_types = {"int", "bigint", "double", "float", "smallint", "tinyint", "boolean", "long"}
    permission_cols = [name for name, dtype in raw.dtypes if name not in excluded and any(t in dtype.lower() for t in numeric_types)]
    if not permission_cols:
        raise ValueError("No usable feature columns found for permissions")

    if "class" in raw.columns:
        # Normalize labels to benign/malware where possible.
        raw = raw.withColumn(
            "class",
            F.when(F.lower(F.col("class").cast("string")).isin("1", "true", "malware", "anomaly", "fault"), F.lit("malware"))
            .when(F.lower(F.col("class").cast("string")).isin("0", "false", "benign", "normal"), F.lit("benign"))
            .otherwise(F.lower(F.col("class").cast("string"))),
        )
        metrics = [{"metric": "ground_truth_label_available", "value": 1}]
    else:
        raw = raw.withColumn("class", F.lit("unknown"))
        metrics = [{"metric": "ground_truth_label_available", "value": 0}]

    total_rows = raw.count()
    dedup_rows = raw.dropDuplicates().count()
    duplicate_rows = total_rows - dedup_rows
    metrics.append({"metric": "total_rows", "value": total_rows})
    metrics.append({"metric": "duplicate_rows", "value": duplicate_rows})

    for c in ["class", *permission_cols[:10]]:
        metrics.append({"metric": f"null_{c}", "value": raw.filter(F.col(c).isNull()).count()})

    return raw, metrics, permission_cols


def basic_etl(df: DataFrame) -> DataFrame:
    return (
        df.dropDuplicates()
        .withColumn("class", F.lower(F.trim(F.col("class"))))
    )


def broadcast_enrichment(df: DataFrame) -> DataFrame:
    spark = df.sparkSession
    out = df
    mode_expr = F.lit("unknown")
    if {"comp", "towers", "lps"}.issubset(set(df.columns)):
        mode_expr = (
            F.when((F.col("comp") == 1) & (F.col("towers") == 1) & (F.col("lps") == 0), F.lit("production"))
            .when((F.col("comp") == 0) & (F.col("towers") == 0), F.lit("idle"))
            .when((F.col("comp") == 1) & (F.col("lps") == 1), F.lit("pressure_event"))
            .otherwise(F.lit("transition"))
        )
    out = out.withColumn("operating_mode", mode_expr)

    lookup = spark.createDataFrame(
        [
            ("production", "medium", "observe"),
            ("idle", "low", "normal"),
            ("pressure_event", "high", "urgent"),
            ("transition", "medium", "observe"),
            ("unknown", "unknown", "investigate"),
        ],
        ["operating_mode", "mode_risk", "maintenance_priority"],
    )
    return (
        out.join(F.broadcast(lookup), on="operating_mode", how="left")
        .withColumn("mode_risk", F.coalesce(F.col("mode_risk"), F.lit("unknown")))
        .withColumn("maintenance_priority", F.coalesce(F.col("maintenance_priority"), F.lit("investigate")))
    )


def _split_data(df: DataFrame) -> tuple[DataFrame, DataFrame, DataFrame]:
    if "timestamp" in df.columns:
        ts_df = df.select(F.col("timestamp").cast("long").alias("ts")).where(F.col("ts").isNotNull())
        qs = ts_df.approxQuantile("ts", [0.6, 0.8], 0.001)
        if len(qs) == 2:
            q60, q80 = qs
            ts_col = F.col("timestamp").cast("long")
            train = df.filter(ts_col <= q60)
            valid = df.filter((ts_col > q60) & (ts_col <= q80))
            test = df.filter(ts_col > q80)
            if test.count() > 0 and train.count() > 0 and valid.count() > 0:
                return train, valid, test

    # Stratified random split by class to preserve class balance.
    labels = [r["class"] for r in df.select("class").distinct().orderBy("class").collect()]
    if len(labels) >= 2:
        left = df.filter(F.col("class") == labels[0])
        right = df.filter(F.col("class") == labels[1])

        l_train, l_valid, l_test = left.randomSplit([0.6, 0.2, 0.2], seed=42)
        r_train, r_valid, r_test = right.randomSplit([0.6, 0.2, 0.2], seed=42)

        train = l_train.unionByName(r_train)
        valid = l_valid.unionByName(r_valid)
        test = l_test.unionByName(r_test)
    else:
        train, valid, test = df.randomSplit([0.6, 0.2, 0.2], seed=42)

    # Fallback for tiny samples to ensure test set is populated.
    if test.count() == 0:
        train, test = df.randomSplit([0.8, 0.2], seed=42)
        valid = train.limit(max(1, int(train.count() * 0.2)))

    return train, valid, test


def _upsample_for_cv(df: DataFrame, target_rows: int = 120) -> DataFrame:
    rows = df.count()
    if rows >= target_rows:
        return df

    multiplier = max(2, int(target_rows / max(rows, 1)) + 1)
    out = df
    for _ in range(multiplier - 1):
        out = out.unionByName(df)
    return out


def _prepare_features(
    df: DataFrame,
    permission_cols: list[str],
) -> tuple[DataFrame, list[str]]:
    risk_cols = [c for c in ["dv_pressure", "reservoirs", "oil_temperature", "motor_current", "tp2", "tp3", "h1"] if c in permission_cols]
    if not risk_cols:
        risk_cols = permission_cols[: min(6, len(permission_cols))]
    risk_transformer = SensorRiskScoreTransformer(inputCols=risk_cols, outputCol="sensor_risk_score")
    out = risk_transformer.transform(df)
    out = out.withColumn(
        "mode_risk_score",
        F.when(F.col("mode_risk") == "high", F.lit(3.0))
        .when(F.col("mode_risk") == "medium", F.lit(2.0))
        .when(F.col("mode_risk") == "low", F.lit(1.0))
        .otherwise(F.lit(0.0)),
    )

    feature_cols = permission_cols + ["sensor_risk_score", "mode_risk_score"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")
    out = assembler.transform(out)
    return out, feature_cols


def _to_pandas_cap(df: DataFrame, columns: list[str], cap_rows: int = 150000) -> pd.DataFrame:
    row_count = df.count()
    if row_count <= cap_rows:
        return df.select(*columns).toPandas()
    frac = cap_rows / float(max(row_count, 1))
    return df.select(*columns).sample(False, frac, seed=42).limit(cap_rows).toPandas()


def _anomaly_summary(y_true: np.ndarray | None, anomaly_flags: np.ndarray, anomaly_scores: np.ndarray) -> dict[str, Any]:
    row: dict[str, Any] = {
        "anomaly_rate": round(float(anomaly_flags.mean()), 6),
        "score_mean": round(float(np.mean(anomaly_scores)), 6),
        "score_std": round(float(np.std(anomaly_scores)), 6),
        "score_p95": round(float(np.percentile(anomaly_scores, 95)), 6),
        "score_p99": round(float(np.percentile(anomaly_scores, 99)), 6),
        "accuracy": None,
        "f1": None,
        "weightedPrecision": None,
        "weightedRecall": None,
        "auc": None,
    }
    if y_true is None or len(np.unique(y_true)) < 2:
        return row

    anomaly_k = max(1, int(y_true.sum()))
    topk_idx = np.argsort(anomaly_scores)[-anomaly_k:]
    topk_flags = np.zeros_like(y_true, dtype=int)
    topk_flags[topk_idx] = 1
    tp = int(((topk_flags == 1) & (y_true == 1)).sum())
    fp = int(((topk_flags == 1) & (y_true == 0)).sum())
    fn = int(((topk_flags == 0) & (y_true == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-12)

    try:
        auc = roc_auc_score(y_true, anomaly_scores)
    except Exception:
        auc = float("nan")

    row.update(
        {
            "weightedPrecision": round(float(precision), 6),
            "weightedRecall": round(float(recall), 6),
            "f1": round(float(f1), 6),
            "auc": round(float(auc), 6) if auc == auc else None,
            "accuracy": round(float((topk_flags == y_true).mean()), 6),
        }
    )
    return row


def _run_unsupervised_models(
    train_pdf: pd.DataFrame,
    eval_splits: dict[str, pd.DataFrame],
    feature_cols: list[str],
    has_ground_truth: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, bytes]]:
    train_X = train_pdf[feature_cols].fillna(0.0).to_numpy(dtype=float)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_X)

    model_specs: list[tuple[str, Any, float]] = [
        ("isolation_forest", IsolationForest(n_estimators=300, contamination="auto", random_state=42, n_jobs=-1), 0.95),
        ("isolation_forest_strict", IsolationForest(n_estimators=300, contamination="auto", random_state=42, n_jobs=-1), 0.99),
    ]

    metric_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    cv_rows: list[dict[str, Any]] = []
    fi_rows: list[dict[str, Any]] = []
    model_payloads: dict[str, bytes] = {}

    for algo, model, threshold_quantile in model_specs:
        start = time.perf_counter()
        model.fit(train_scaled)
        runtime = time.perf_counter() - start
        train_scores = (-model.decision_function(train_scaled)).astype(float)
        threshold = float(np.quantile(train_scores, threshold_quantile))
        model_payloads[algo] = pickle.dumps(
            {
                "scaler": scaler,
                "model": model,
                "feature_cols": feature_cols,
                "anomaly_threshold": threshold,
            }
        )
        cv_rows.append(
            {
                "algorithm": algo,
                "params": json.dumps({"threshold_quantile": threshold_quantile, "threshold_value": threshold}, sort_keys=True),
                "cv_metric_auc": None,
                "selection_metric": "validation_score_p99",
                "selection_value": None,
            }
        )

        for split_name, split_pdf in eval_splits.items():
            X = split_pdf[feature_cols].fillna(0.0).to_numpy(dtype=float)
            X_scaled = scaler.transform(X)
            anomaly_scores = (-model.decision_function(X_scaled)).astype(float)
            anomaly_flags = (anomaly_scores >= threshold).astype(int)

            y_true = None
            if has_ground_truth and "class" in split_pdf.columns:
                y_true = (split_pdf["class"] == "malware").astype(int).to_numpy()

            summary = _anomaly_summary(y_true, anomaly_flags, anomaly_scores)
            summary.update(
                {
                    "algorithm": algo,
                    "engine": "sklearn_unsupervised",
                    "split": split_name,
                    "runtime_seconds": round(runtime, 4),
                    "anomaly_threshold": round(float(threshold), 6),
                }
            )
            metric_rows.append(summary)
            if split_name == "validation":
                for i, row in enumerate(cv_rows):
                    if row["algorithm"] == algo:
                        cv_rows[i]["selection_value"] = round(float(summary["score_p99"]), 6)
                        break

            split_scores = pd.DataFrame(
                {
                    "algorithm": algo,
                    "split": split_name,
                    "anomaly_flag": anomaly_flags,
                    "anomaly_score": anomaly_scores,
                }
            )
            if "timestamp" in split_pdf.columns:
                split_scores["timestamp"] = split_pdf["timestamp"]
            if "class" in split_pdf.columns:
                split_scores["class"] = split_pdf["class"]
            score_rows.extend(split_scores.to_dict(orient="records"))

        # Feature influence proxy: absolute correlation between feature and anomaly score on validation split.
        valid_pdf = eval_splits.get("validation", train_pdf)
        val_X = valid_pdf[feature_cols].fillna(0.0).to_numpy(dtype=float)
        val_scores = (-model.decision_function(scaler.transform(val_X))).astype(float)
        for idx, feat in enumerate(feature_cols):
            col = val_X[:, idx]
            if np.std(col) <= 1e-12:
                corr = 0.0
            else:
                corr = float(np.corrcoef(col, val_scores)[0, 1])
                if np.isnan(corr):
                    corr = 0.0
            fi_rows.append({"algorithm": algo, "feature": feat, "importance": round(abs(corr), 8)})

    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(score_rows),
        pd.DataFrame(cv_rows),
        pd.DataFrame(fi_rows),
        model_payloads,
    )


def _sklearn_baseline(pdf: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, bytes]:
    if len(pdf) < 4:
        return pd.DataFrame(
            [
                {
                    "algorithm": "sklearn_logistic_regression",
                    "engine": "sklearn_single_node",
                    "split": "test",
                    "runtime_seconds": 0.0,
                    "accuracy": None,
                    "f1": None,
                    "weightedPrecision": None,
                    "weightedRecall": None,
                    "auc": None,
                }
            ]
        ), pickle.dumps({"note": "insufficient_rows_for_sklearn_baseline"})

    X = pdf[feature_cols].fillna(0.0)
    y = (pdf["class"] == "malware").astype(int)

    stratify = y if y.nunique() > 1 else None
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=stratify,
    )

    model = SkLogisticRegression(max_iter=400)
    start = time.perf_counter()
    model.fit(x_train, y_train)
    runtime_lr = time.perf_counter() - start

    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1] if len(model.classes_) == 2 else None

    rows = [
        {
            "algorithm": "sklearn_logistic_regression",
            "engine": "sklearn_single_node",
            "split": "test",
            "runtime_seconds": round(runtime_lr, 4),
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "weightedPrecision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "weightedRecall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "auc": round(float(roc_auc_score(y_test, y_prob)), 4) if y_prob is not None and len(set(y_test)) > 1 else None,
        }
    ]

    # Second sklearn baseline to strengthen comparison.
    rf = SkRandomForestClassifier(n_estimators=100, random_state=42)
    start = time.perf_counter()
    rf.fit(x_train, y_train)
    runtime_rf = time.perf_counter() - start
    rf_pred = rf.predict(x_test)
    rf_prob = rf.predict_proba(x_test)[:, 1] if len(rf.classes_) == 2 else None
    rows.append(
        {
            "algorithm": "sklearn_random_forest",
            "engine": "sklearn_single_node",
            "split": "test",
            "runtime_seconds": round(runtime_rf, 4),
            "accuracy": round(float(accuracy_score(y_test, rf_pred)), 4),
            "f1": round(float(f1_score(y_test, rf_pred, zero_division=0)), 4),
            "weightedPrecision": round(float(precision_score(y_test, rf_pred, zero_division=0)), 4),
            "weightedRecall": round(float(recall_score(y_test, rf_pred, zero_division=0)), 4),
            "auc": round(float(roc_auc_score(y_test, rf_prob)), 4) if rf_prob is not None and len(set(y_test)) > 1 else None,
        }
    )
    return pd.DataFrame(rows), pickle.dumps(model)


def _run_scalability(
    df: DataFrame,
    featured_df: DataFrame,
    executors: int,
    cost_per_executor_hour: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strong_rows: list[dict[str, Any]] = []
    weak_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []

    df_rows = df.count()
    for partitions in [2, 4, 8, 16]:
        start = time.perf_counter()
        _ = df.repartition(partitions).groupBy("class").count().collect()
        duration = time.perf_counter() - start
        strong_rows.append(
            {
                "experiment": "strong_scaling",
                "partitions": partitions,
                "rows": df_rows,
                "duration_seconds": round(duration, 4),
                "throughput_rows_per_sec": round(df_rows / max(duration, 1e-6), 2),
            }
        )
        est_cost = (duration / 3600.0) * executors * cost_per_executor_hour
        cost_rows.append(
            {
                "experiment": "strong_scaling",
                "partitions": partitions,
                "duration_seconds": round(duration, 4),
                "estimated_cost_usd": round(est_cost, 6),
            }
        )

    weak_base = featured_df
    weak_base_rows = weak_base.count()
    weak_base_cap = 250000
    if weak_base_rows > weak_base_cap:
        frac = weak_base_cap / float(weak_base_rows)
        weak_base = weak_base.sample(False, frac, seed=42)
        weak_base_rows = weak_base.count()
    weak_base = weak_base.persist(StorageLevel.MEMORY_AND_DISK)

    for multiplier, partitions in [(1, 2), (2, 4), (3, 8)]:
        scaled = weak_base
        for _ in range(multiplier - 1):
            scaled = scaled.unionByName(weak_base)

        n_rows = scaled.count()
        start = time.perf_counter()
        _ = scaled.repartition(partitions).groupBy("class").count().collect()
        duration = time.perf_counter() - start

        weak_rows.append(
            {
                "experiment": "weak_scaling",
                "partitions": partitions,
                "multiplier": multiplier,
                "rows": n_rows,
                "duration_seconds": round(duration, 4),
                "throughput_rows_per_sec": round(n_rows / max(duration, 1e-6), 2),
            }
        )
        est_cost = (duration / 3600.0) * executors * cost_per_executor_hour
        cost_rows.append(
            {
                "experiment": "weak_scaling",
                "partitions": partitions,
                "duration_seconds": round(duration, 4),
                "estimated_cost_usd": round(est_cost, 6),
            }
        )

    weak_base.unpersist(blocking=False)
    return pd.DataFrame(strong_rows), pd.DataFrame(weak_rows), pd.DataFrame(cost_rows)


def run(data_path: str, output_dir: str, run_scalability: bool) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(str(root / "config" / "spark_config.yaml"))
    spark = build_spark(config, root)

    profiler = PerformanceProfiler()
    lineage: list[StageLineage] = []
    outputs = root / output_dir
    outputs.mkdir(parents=True, exist_ok=True)

    models_dir = root / "data" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    try:
        source = root / data_path
        if not source.exists():
            candidates = sorted((root / "data" / "raw").glob("*MetroPT3*AirCompressor*.csv"))
            if candidates:
                source = candidates[0]
            else:
                raise FileNotFoundError(f"Input CSV not found: {source}")
        profiler.start("ingestion_validation")
        raw_df, dq_metrics, permission_cols = ingest_and_validate(spark, source)
        profiler.stop("ingestion_validation", rows=raw_df.count())
        lineage.append(
            StageLineage(
                stage="ingestion_validation",
                status="success",
                input_rows=raw_df.count(),
                output_rows=raw_df.count(),
                notes="Schema checks + required columns + null/duplicate metrics",
            )
        )

        profiler.start("basic_etl")
        clean_df = basic_etl(raw_df)
        clean_df = clean_df.repartition(int(config.get("shuffle_partitions", 8)), "class")
        clean_df.persist(StorageLevel.MEMORY_AND_DISK)
        clean_count = clean_df.count()
        profiler.stop("basic_etl", rows=clean_count)
        lineage.append(
            StageLineage(
                stage="basic_etl",
                status="success",
                input_rows=raw_df.count(),
                output_rows=clean_count,
                notes="Dedup + normalization + partition by class",
            )
        )

        profiler.start("broadcast_etl")
        enriched_df = broadcast_enrichment(clean_df)
        enriched_df.persist(StorageLevel.MEMORY_AND_DISK)
        enriched_count = enriched_df.count()
        profiler.stop("broadcast_etl", rows=enriched_count)
        lineage.append(
            StageLineage(
                stage="broadcast_etl",
                status="success",
                input_rows=clean_count,
                output_rows=enriched_count,
                notes="Broadcast join with operating mode risk lookup",
            )
        )

        profiler.start("feature_engineering")
        featured_df, feature_cols = _prepare_features(enriched_df, permission_cols)
        featured_df = featured_df.checkpoint(eager=True)
        featured_df.persist(StorageLevel.MEMORY_AND_DISK)
        feat_count = featured_df.count()
        profiler.stop("feature_engineering", rows=feat_count)
        lineage.append(
            StageLineage(
                stage="feature_engineering",
                status="success",
                input_rows=enriched_count,
                output_rows=feat_count,
                notes="Custom transformer + vector assembler + checkpoint",
            )
        )

        profiler.start("data_split")
        train_df, valid_df, test_df = _split_data(featured_df)
        train_df = train_df.checkpoint(eager=True)
        train_df.persist(StorageLevel.MEMORY_AND_DISK)
        valid_df.persist(StorageLevel.MEMORY_AND_DISK)
        test_df.persist(StorageLevel.MEMORY_AND_DISK)
        split_rows = train_df.count() + valid_df.count() + test_df.count()
        profiler.stop("data_split", rows=split_rows)
        lineage.append(
            StageLineage(
                stage="data_split",
                status="success",
                input_rows=feat_count,
                output_rows=split_rows,
                notes="Time-aware or random split (60/20/20 with tiny-sample fallback)",
            )
        )

        has_ground_truth = clean_df.filter(F.col("class") != "unknown").limit(1).count() > 0
        split_columns = ["timestamp", *feature_cols, "class"] if "timestamp" in train_df.columns else [*feature_cols, "class"]
        train_pdf = _to_pandas_cap(train_df, split_columns, cap_rows=120000)
        valid_pdf = _to_pandas_cap(valid_df, split_columns, cap_rows=60000)
        test_pdf = _to_pandas_cap(test_df, split_columns, cap_rows=60000)

        profiler.start("train_unsupervised")
        ml_metrics_df, anomaly_scores_df, cv_metrics_df, feature_importance_df, model_payloads = _run_unsupervised_models(
            train_pdf=train_pdf,
            eval_splits={"train": train_pdf, "validation": valid_pdf, "test": test_pdf},
            feature_cols=feature_cols,
            has_ground_truth=has_ground_truth,
        )
        profiler.stop("train_unsupervised", rows=len(train_pdf))

        for model_name, payload in model_payloads.items():
            with (models_dir / f"{model_name}_unsupervised.pkl").open("wb") as f:
                f.write(payload)

        if has_ground_truth:
            sk_df, sk_pickle = _sklearn_baseline(train_pdf[[*feature_cols, "class"]], feature_cols)
            with (models_dir / "sklearn_baseline.pkl").open("wb") as f:
                f.write(sk_pickle)
            ml_metrics_df = pd.concat([ml_metrics_df, sk_df], ignore_index=True)
        else:
            with (models_dir / "sklearn_baseline.pkl").open("wb") as f:
                f.write(pickle.dumps({"note": "skipped_supervised_baseline_no_ground_truth_labels"}))

        class_distribution = (
            enriched_df.groupBy("class").count().orderBy(F.desc("count")).toPandas().rename(columns={"count": "row_count"})
        )

        split_summary = (
            train_df.withColumn("split", F.lit("train"))
            .unionByName(valid_df.withColumn("split", F.lit("validation")))
            .unionByName(test_df.withColumn("split", F.lit("test")))
            .groupBy("split", "class")
            .count()
            .orderBy("split", "class")
            .toPandas()
            .rename(columns={"count": "row_count"})
        )

        family_risk_summary = (
            enriched_df.groupBy("operating_mode", "mode_risk", "class")
            .count()
            .orderBy(F.desc("count"))
            .toPandas()
            .rename(columns={"count": "row_count", "operating_mode": "app_family", "mode_risk": "family_risk"})
        )

        std_exprs = [F.stddev(F.col(c)).alias(c) for c in permission_cols]
        std_row = clean_df.agg(*std_exprs).toPandas().iloc[0].fillna(0.0)
        std_series = pd.to_numeric(std_row, errors="coerce").fillna(0.0)
        test_anomaly_rate = float(
            ml_metrics_df.loc[ml_metrics_df["split"] == "test", "anomaly_rate"].dropna().mean()
            if "anomaly_rate" in ml_metrics_df.columns
            else 0.0
        )
        business_insights = pd.DataFrame(
            [
                {
                    "insight": "estimated_anomaly_rate",
                    "value": round(test_anomaly_rate, 6),
                    "recommendation": "Increase alert review staffing" if test_anomaly_rate >= 0.05 else "Maintain current monitoring cadence",
                },
                {
                    "insight": "high_variability_sensor_count",
                    "value": int((std_series > float(std_series.median())).sum()),
                    "recommendation": "Review sensors with high variability for drift and calibration",
                },
            ]
        )
        business_metric_rows = []
        for _, r in ml_metrics_df[ml_metrics_df["split"] == "test"].iterrows():
            anomaly_rate = float(r.get("anomaly_rate") or 0.0)
            population = int(test_df.count())
            anomaly_count = int(round(population * anomaly_rate))
            review_cost = anomaly_count * 8.0
            business_metric_rows.append(
                {
                    "algorithm": r.get("algorithm", "unknown"),
                    "split": "test",
                    "tp": None,
                    "fp": None,
                    "fn": None,
                    "expected_profit_usd": round(-review_cost, 2),
                    "assumption": "Each anomaly alert review costs $8.00",
                }
            )

        strong_scaling = pd.DataFrame()
        weak_scaling = pd.DataFrame()
        cost_perf = pd.DataFrame()
        if run_scalability:
            strong_scaling, weak_scaling, cost_perf = _run_scalability(
                enriched_df,
                featured_df,
                executors=int(config.get("spark.executor.instances", 1)),
                cost_per_executor_hour=float(config.get("estimated_executor_cost_per_hour_usd", 0.25)),
            )

        resource_allocation = pd.DataFrame(
            [
                {
                    "app_name": config.get("app_name", "NATICUSdroidPipeline"),
                    "master": config.get("master", "local[*]"),
                    "shuffle_partitions": config.get("shuffle_partitions", 8),
                    "default_parallelism": config.get("default_parallelism", 8),
                    "executor_instances": config.get("spark.executor.instances", 1),
                    "executor_cores": config.get("spark.executor.cores", "N/A"),
                    "executor_memory": config.get("spark.executor.memory", "N/A"),
                    "driver_memory": config.get("spark.driver.memory", "N/A"),
                    "adaptive_query_execution": config.get("adaptive_query_execution", True),
                }
            ]
        )

        bottlenecks = pd.DataFrame(
            [
                {
                    "bottleneck": "I/O",
                    "observation": "CSV input incurs parsing overhead; parquet recommended for production reuse",
                    "mitigation": "Persist cleaned dataset to parquet and use predicate pushdown",
                },
                {
                    "bottleneck": "Shuffle",
                    "observation": "groupBy and split operations trigger shuffle",
                    "mitigation": "Tune spark.sql.shuffle.partitions and leverage AQE",
                },
                {
                    "bottleneck": "Computation",
                    "observation": "Unsupervised model training + scoring adds CPU usage on sampled data",
                    "mitigation": "Adjust sample caps and anomaly model complexity per hardware limits",
                },
            ]
        )

        # Tableau exports
        pd.DataFrame([m.__dict__ for m in profiler.records()]).to_csv(outputs / "etl_stage_timings.csv", index=False)
        pd.DataFrame(dq_metrics).to_csv(outputs / "data_quality_report.csv", index=False)
        pd.DataFrame([x.__dict__ for x in lineage]).to_csv(outputs / "pipeline_lineage.csv", index=False)
        class_distribution.to_csv(outputs / "class_distribution.csv", index=False)
        split_summary.to_csv(outputs / "split_summary.csv", index=False)
        ml_metrics_df.to_csv(outputs / "model_metrics.csv", index=False)
        if cv_metrics_df.empty:
            cv_metrics_df = pd.DataFrame(columns=["algorithm", "params", "cv_metric_auc", "selection_metric", "selection_value"])
        cv_metrics_df.to_csv(outputs / "cv_results.csv", index=False)
        if feature_importance_df.empty:
            feature_importance_df = pd.DataFrame(columns=["algorithm", "feature", "importance"])
        feature_importance_df.sort_values(["algorithm", "importance"], ascending=[True, False]).to_csv(outputs / "feature_importance.csv", index=False)
        anomaly_scores_df.to_csv(outputs / "anomaly_scores.csv", index=False)
        family_risk_summary.to_csv(outputs / "family_risk_distribution.csv", index=False)
        business_insights.to_csv(outputs / "business_insights.csv", index=False)
        pd.DataFrame(columns=["algorithm", "split", "metric", "ci_lower_95", "ci_upper_95", "bootstrap_iterations", "sample_rows"]).to_csv(
            outputs / "bootstrap_confidence_intervals.csv", index=False
        )
        pd.DataFrame(business_metric_rows).to_csv(outputs / "business_metric_alignment.csv", index=False)
        resource_allocation.to_csv(outputs / "resource_allocation.csv", index=False)
        bottlenecks.to_csv(outputs / "bottleneck_analysis.csv", index=False)

        if run_scalability:
            strong_scaling.to_csv(outputs / "scalability_strong.csv", index=False)
            weak_scaling.to_csv(outputs / "scalability_weak.csv", index=False)
            cost_perf.to_csv(outputs / "cost_performance.csv", index=False)

        profiler.dump_json(
            root / "data" / "spark_ui_jobs" / "spark_job_metrics.json",
            extra={
                "selected_stage": "full_pipeline",
                "data_path": data_path,
                "run_scalability": run_scalability,
            },
        )

        # Persist curated parquet output for downstream query efficiency.
        parquet_dir = root / "data" / "curated_parquet"
        if parquet_dir.exists():
            shutil.rmtree(parquet_dir)
        enriched_df.write.mode("overwrite").partitionBy("class").parquet(str(parquet_dir))

        # Explicit unpersist to release memory after expensive stages.
        for frame in [
            clean_df,
            enriched_df,
            featured_df,
            train_df,
            valid_df,
            test_df,
        ]:
            frame.unpersist(blocking=False)

        print(f"Pipeline complete. Tableau CSVs written to: {outputs}")

    except Exception as exc:
        lineage.append(
            StageLineage(
                stage="pipeline",
                status="failed",
                input_rows=0,
                output_rows=0,
                notes=str(exc),
            )
        )
        pd.DataFrame([x.__dict__ for x in lineage]).to_csv(outputs / "pipeline_lineage.csv", index=False)
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    args = parse_args()
    run(
        data_path=args.data_path,
        output_dir=args.output_dir,
        run_scalability=args.run_scalability,
    )
