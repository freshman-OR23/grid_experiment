from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict


class StageProfiler:
    """记录各个阶段耗时，便于后续判断瓶颈。"""

    def __init__(self) -> None:
        self.records: Dict[str, float] = {}

    @contextmanager
    def track(self, stage_name: str):
        start_time = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start_time
        self.records[stage_name] = self.records.get(stage_name, 0.0) + elapsed

    def summary(self) -> Dict[str, float]:
        return dict(self.records)
