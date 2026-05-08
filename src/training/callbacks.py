from __future__ import annotations


class EarlyStopping:
    """基于验证指标的 early stopping，支持最小提升阈值与预热验证轮数。"""

    def __init__(self, patience: int, min_delta: float = 0.0, warmup_validations: int = 0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.warmup_validations = warmup_validations
        self.best_score = None
        self.bad_rounds = 0
        self.num_validations = 0

    def step(self, score: float) -> bool:
        self.num_validations += 1
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.bad_rounds = 0
            return False
        if self.num_validations <= self.warmup_validations:
            return False
        self.bad_rounds += 1
        return self.bad_rounds >= self.patience
