from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass
class StageMetric:
    stage: str
    duration_seconds: float
    rows: int


class PerformanceProfiler:
    def __init__(self) -> None:
        self._started_at: dict[str, float] = {}
        self._records: list[StageMetric] = []

    def start(self, stage: str) -> None:
        self._started_at[stage] = time.perf_counter()

    def stop(self, stage: str, rows: int) -> StageMetric:
        started = self._started_at.pop(stage)
        metric = StageMetric(
            stage=stage,
            duration_seconds=round(time.perf_counter() - started, 4),
            rows=rows,
        )
        self._records.append(metric)
        return metric

    def records(self) -> list[StageMetric]:
        return list(self._records)

    def dump_json(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "stage_metrics": [asdict(x) for x in self._records],
            "extra": extra or {},
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
