from __future__ import annotations

import json
from pathlib import Path

import yaml


def test_core_project_files_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts" / "run_pipeline.py").exists()
    assert (root / "config" / "spark_config.yaml").exists()
    assert (root / "config" / "tableau_config.json").exists()


def test_spark_config_contains_resource_keys() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "config" / "spark_config.yaml").read_text(encoding="utf-8"))
    assert "shuffle_partitions" in cfg
    assert "default_parallelism" in cfg
    assert "spark.executor.instances" in cfg


def test_tableau_config_has_required_datasources() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "config" / "tableau_config.json").read_text(encoding="utf-8"))
    names = {x["name"] for x in cfg["datasources"]}
    required = {
        "etl_stage_timings",
        "data_quality_report",
        "pipeline_lineage",
        "model_metrics",
        "feature_importance",
        "scalability_strong",
        "scalability_weak",
    }
    assert required.issubset(names)
