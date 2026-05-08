from __future__ import annotations

import math
from typing import Dict, List


def recall_at_k(predictions: List[str], target_item: str, k: int) -> float:
    return 1.0 if target_item in predictions[:k] else 0.0


def ndcg_at_k(predictions: List[str], target_item: str, k: int) -> float:
    if target_item not in predictions[:k]:
        return 0.0
    rank = predictions[:k].index(target_item) + 1
    return 1.0 / math.log2(rank + 1)


def compute_ranking_metrics(batch_predictions: List[List[str]], batch_targets: List[str]) -> Dict[str, float]:
    metrics = {
        "Recall@5": 0.0,
        "Recall@10": 0.0,
        "NDCG@5": 0.0,
        "NDCG@10": 0.0,
    }
    if not batch_targets:
        return metrics

    for predictions, target_item in zip(batch_predictions, batch_targets, strict=True):
        metrics["Recall@5"] += recall_at_k(predictions, target_item, 5)
        metrics["Recall@10"] += recall_at_k(predictions, target_item, 10)
        metrics["NDCG@5"] += ndcg_at_k(predictions, target_item, 5)
        metrics["NDCG@10"] += ndcg_at_k(predictions, target_item, 10)

    total = len(batch_targets)
    return {name: value / total for name, value in metrics.items()}
