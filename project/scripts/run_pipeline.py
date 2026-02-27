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
from pyspark.ml.classification import (
    GBTClassifier,
    LogisticRegression,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.param.shared import Param, Params
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from sklearn.ensemble import RandomForestClassifier as SkRandomForestClassifier
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

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

    if "class" not in raw.columns:
        # Build pseudo-labels for unlabeled sensor datasets using robust outlier counts.
        sensor_candidates = [c for c in ["dv_pressure", "reservoirs", "oil_temperature", "motor_current", "tp2", "tp3", "h1"] if c in permission_cols]
        if not sensor_candidates:
            sensor_candidates = permission_cols[: min(6, len(permission_cols))]

        bounds = {}
        for c in sensor_candidates:
            q = raw.approxQuantile(c, [0.25, 0.75], 0.01)
            if len(q) != 2:
                continue
            q1, q3 = q
            iqr = q3 - q1
            bounds[c] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

        anomaly_expr = F.lit(0)
        for c, (lo, hi) in bounds.items():
            anomaly_expr = anomaly_expr + F.when((F.col(c) < F.lit(lo)) | (F.col(c) > F.lit(hi)), F.lit(1)).otherwise(F.lit(0))

        raw = raw.withColumn("class", F.when(anomaly_expr >= F.lit(2), F.lit("malware")).otherwise(F.lit("benign")))
    else:
        # Normalize labels to benign/malware where possible.
        raw = raw.withColumn(
            "class",
            F.when(F.lower(F.col("class").cast("string")).isin("1", "true", "malware", "anomaly", "fault"), F.lit("malware"))
            .when(F.lower(F.col("class").cast("string")).isin("0", "false", "benign", "normal"), F.lit("benign"))
            .otherwise(F.lower(F.col("class").cast("string"))),
        )

    metrics = []
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

    out = out.withColumn("label", F.when(F.col("class") == F.lit("malware"), F.lit(1.0)).otherwise(F.lit(0.0)))

    feature_cols = permission_cols + ["sensor_risk_score", "mode_risk_score"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")
    out = assembler.transform(out)
    return out, feature_cols


def _bootstrap_metric_cis(
    predictions: DataFrame,
    algorithm: str,
    split_name: str = "test",
    max_rows: int = 50000,
    iterations: int = 200,
) -> list[dict[str, Any]]:
    pdf = predictions.select("label", "prediction").dropna().limit(max_rows).toPandas()
    if len(pdf) < 50:
        return []
    y_true = pdf["label"].astype(int).to_numpy()
    y_pred = pdf["prediction"].astype(int).to_numpy()
    n = len(pdf)
    rng = np.random.default_rng(42)

    acc_scores = []
    f1_scores = []
    for _ in range(iterations):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        yp = y_pred[idx]
        acc_scores.append(float((yt == yp).mean()))
        tp = float(((yt == 1) & (yp == 1)).sum())
        fp = float(((yt == 0) & (yp == 1)).sum())
        fn = float(((yt == 1) & (yp == 0)).sum())
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1_scores.append((2 * precision * recall) / max(precision + recall, 1e-9))

    rows = []
    for metric_name, vals in [("accuracy", acc_scores), ("f1", f1_scores)]:
        lo, hi = np.percentile(np.array(vals), [2.5, 97.5]).tolist()
        rows.append(
            {
                "algorithm": algorithm,
                "split": split_name,
                "metric": metric_name,
                "ci_lower_95": round(float(lo), 4),
                "ci_upper_95": round(float(hi), 4),
                "bootstrap_iterations": iterations,
                "sample_rows": n,
            }
        )
    return rows


def _business_metric_row(predictions: DataFrame, algorithm: str, split_name: str = "test") -> dict[str, Any]:
    c = predictions.select("label", "prediction").agg(
        F.sum(F.when((F.col("label") == 1) & (F.col("prediction") == 1), 1).otherwise(0)).alias("tp"),
        F.sum(F.when((F.col("label") == 0) & (F.col("prediction") == 1), 1).otherwise(0)).alias("fp"),
        F.sum(F.when((F.col("label") == 1) & (F.col("prediction") == 0), 1).otherwise(0)).alias("fn"),
    ).collect()[0]
    tp = int(c["tp"] or 0)
    fp = int(c["fp"] or 0)
    fn = int(c["fn"] or 0)
    expected_profit = (tp * 120.0) - (fp * 20.0) - (fn * 200.0)
    return {
        "algorithm": algorithm,
        "split": split_name,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "expected_profit_usd": round(expected_profit, 2),
        "assumption": "TP:+120, FP:-20, FN:-200",
    }


def _evaluate_predictions(predictions: DataFrame, split_name: str, algorithm: str, runtime_s: float) -> list[dict[str, Any]]:
    multiclass_metrics = {
        "accuracy": MulticlassClassificationEvaluator(metricName="accuracy"),
        "f1": MulticlassClassificationEvaluator(metricName="f1"),
        "weightedPrecision": MulticlassClassificationEvaluator(metricName="weightedPrecision"),
        "weightedRecall": MulticlassClassificationEvaluator(metricName="weightedRecall"),
    }

    # AUC can fail if a split has one class only.
    try:
        auc = BinaryClassificationEvaluator(metricName="areaUnderROC").evaluate(predictions)
    except Exception:
        auc = float("nan")

    row = {
        "algorithm": algorithm,
        "engine": "pyspark_mllib",
        "split": split_name,
        "runtime_seconds": round(runtime_s, 4),
        "auc": round(float(auc), 4) if auc == auc else None,
    }
    for name, evaluator in multiclass_metrics.items():
        row[name] = round(float(evaluator.evaluate(predictions)), 4)
    return [row]


def _train_cv_model(
    train_df: DataFrame,
    evaluator: BinaryClassificationEvaluator,
    algorithm: str,
    parallelism: int,
) -> tuple[Any, list[dict[str, Any]], float]:
    if algorithm == "logistic_regression":
        estimator = LogisticRegression(
            featuresCol="features",
            labelCol="label",
            maxIter=50,
            regParam=0.01,
            elasticNetParam=0.0,
        )
        grid = (
            ParamGridBuilder()
            .addGrid(estimator.regParam, [0.01, 0.05])
            .addGrid(estimator.elasticNetParam, [0.0, 0.5])
            .build()
        )
    elif algorithm == "random_forest":
        estimator = RandomForestClassifier(featuresCol="features", labelCol="label", seed=42)
        grid = (
            ParamGridBuilder()
            .addGrid(estimator.maxDepth, [4, 8])
            .addGrid(estimator.numTrees, [20, 40])
            .build()
        )
    elif algorithm == "gbt":
        estimator = GBTClassifier(featuresCol="features", labelCol="label", seed=42)
        grid = (
            ParamGridBuilder()
            .addGrid(estimator.maxDepth, [3, 5])
            .addGrid(estimator.maxIter, [20, 40])
            .build()
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    cv = CrossValidator(
        estimator=estimator,
        estimatorParamMaps=grid,
        evaluator=evaluator,
        numFolds=3,
        parallelism=max(1, parallelism),
        seed=42,
    )

    start = time.perf_counter()
    model = cv.fit(train_df)
    runtime = time.perf_counter() - start

    cv_rows = []
    for pmap, score in zip(grid, model.avgMetrics):
        serializable_params = {p.name: v for p, v in pmap.items()}
        cv_rows.append(
            {
                "algorithm": algorithm,
                "params": json.dumps(serializable_params, sort_keys=True),
                "cv_metric_auc": round(float(score), 6),
            }
        )
    return model, cv_rows, runtime


def _extract_feature_importance(model: Any, algorithm: str, feature_cols: list[str]) -> pd.DataFrame:
    if algorithm == "logistic_regression":
        coeffs = model.bestModel.coefficients.toArray().tolist()
        rows = [
            {"algorithm": algorithm, "feature": f, "importance": float(abs(v))}
            for f, v in zip(feature_cols, coeffs)
        ]
    else:
        imps = model.bestModel.featureImportances.toArray().tolist()
        rows = [{"algorithm": algorithm, "feature": f, "importance": float(v)} for f, v in zip(feature_cols, imps)]
    return pd.DataFrame(rows).sort_values("importance", ascending=False)


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
        _ = scaled.repartition(partitions).groupBy("label").count().collect()
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
        featured_df = _upsample_for_cv(featured_df, target_rows=120)
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
                notes="Stratified split by class (60/20/20 with tiny-sample fallback)",
            )
        )

        evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC", labelCol="label")
        ml_metrics: list[dict[str, Any]] = []
        cv_metrics: list[dict[str, Any]] = []
        fi_frames: list[pd.DataFrame] = []
        bootstrap_rows: list[dict[str, Any]] = []
        business_metric_rows: list[dict[str, Any]] = []

        for algorithm in ["logistic_regression", "random_forest", "gbt"]:
            profiler.start(f"train_{algorithm}")
            model, cv_rows, train_runtime = _train_cv_model(
                train_df,
                evaluator,
                algorithm=algorithm,
                parallelism=int(config.get("default_parallelism", 4)),
            )
            profiler.stop(f"train_{algorithm}", rows=train_df.count())

            model_path = models_dir / f"{algorithm}_spark_model"
            if model_path.exists():
                shutil.rmtree(model_path)
            model.bestModel.write().overwrite().save(str(model_path))

            train_pred = model.transform(train_df)
            valid_pred = model.transform(valid_df)
            test_pred = model.transform(test_df)
            ml_metrics.extend(_evaluate_predictions(train_pred, "train", algorithm, train_runtime))
            ml_metrics.extend(_evaluate_predictions(valid_pred, "validation", algorithm, train_runtime))
            ml_metrics.extend(_evaluate_predictions(test_pred, "test", algorithm, train_runtime))
            cv_metrics.extend(cv_rows)
            fi_frames.append(_extract_feature_importance(model, algorithm, feature_cols))
            bootstrap_rows.extend(_bootstrap_metric_cis(test_pred, algorithm, split_name="test"))
            business_metric_rows.append(_business_metric_row(test_pred, algorithm, split_name="test"))

        featured_count = feat_count
        baseline_cap = 120000
        frac = min(1.0, baseline_cap / max(featured_count, 1))
        pdf_for_sklearn = featured_df.select(*feature_cols, "class").sample(False, frac, seed=42).limit(baseline_cap).toPandas()
        sk_df, sk_pickle = _sklearn_baseline(pdf_for_sklearn, feature_cols)
        with (models_dir / "sklearn_baseline.pkl").open("wb") as f:
            f.write(sk_pickle)

        ml_metrics_df = pd.DataFrame(ml_metrics)
        ml_metrics_df = pd.concat([ml_metrics_df, sk_df], ignore_index=True)

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

        fault_rate = float((class_distribution.loc[class_distribution["class"] == "malware", "row_count"].sum() / max(class_distribution["row_count"].sum(), 1)))
        std_exprs = [F.stddev(F.col(c)).alias(c) for c in permission_cols]
        std_row = clean_df.agg(*std_exprs).toPandas().iloc[0].fillna(0.0)
        std_series = pd.to_numeric(std_row, errors="coerce").fillna(0.0)
        business_insights = pd.DataFrame(
            [
                {
                    "insight": "estimated_fault_rate",
                    "value": round(fault_rate, 6),
                    "recommendation": "Prioritize preventive maintenance" if fault_rate >= 0.05 else "Maintain current monitoring cadence",
                },
                {
                    "insight": "high_variability_sensor_count",
                    "value": int((std_series > float(std_series.median())).sum()),
                    "recommendation": "Review sensors with high variability for drift and calibration",
                },
            ]
        )
        business_metric_rows.extend(
            [
                {
                    "algorithm": "portfolio_summary",
                    "split": "test",
                    "tp": int(sum(r["tp"] for r in business_metric_rows)),
                    "fp": int(sum(r["fp"] for r in business_metric_rows)),
                    "fn": int(sum(r["fn"] for r in business_metric_rows)),
                    "expected_profit_usd": round(sum(r["expected_profit_usd"] for r in business_metric_rows), 2),
                    "assumption": "TP:+120, FP:-20, FN:-200",
                }
            ]
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
                    "observation": "groupBy and CV operations trigger shuffle",
                    "mitigation": "Tune spark.sql.shuffle.partitions and leverage AQE",
                },
                {
                    "bottleneck": "Computation",
                    "observation": "CrossValidator with 3 models increases CPU time",
                    "mitigation": "Bound grid size and raise CV parallelism within cluster limits",
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
        pd.DataFrame(cv_metrics).to_csv(outputs / "cv_results.csv", index=False)
        pd.concat(fi_frames, ignore_index=True).to_csv(outputs / "feature_importance.csv", index=False)
        family_risk_summary.to_csv(outputs / "family_risk_distribution.csv", index=False)
        business_insights.to_csv(outputs / "business_insights.csv", index=False)
        pd.DataFrame(bootstrap_rows).to_csv(outputs / "bootstrap_confidence_intervals.csv", index=False)
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
