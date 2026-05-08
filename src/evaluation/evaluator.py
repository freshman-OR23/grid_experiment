from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_ranking_metrics
from src.models.decoding import BeamSearchConfig, decode_topk_items
from src.utils.progress import create_progress


def evaluate_model(
    model,
    dataloader: DataLoader,
    sid_to_items,
    item_sid_lookup,
    decode_config: BeamSearchConfig,
    device: torch.device,
    max_batches: int | None = None,
    sid_prefix_index: Dict[Tuple[int, ...], list[int]] | None = None,
) -> Dict[str, float]:
    metric_sums = {"Recall@5": 0.0, "Recall@10": 0.0, "NDCG@5": 0.0, "NDCG@10": 0.0}
    total_batches = 0

    for batch in create_progress(dataloader, desc="评估模型", leave=True):
        predictions = decode_topk_items(
            model=model,
            batch=batch,
            sid_to_items=sid_to_items,
            item_sid_lookup=item_sid_lookup,
            config=decode_config,
            device=device,
            sid_prefix_index=sid_prefix_index,
        )
        metrics = compute_ranking_metrics(predictions, batch["target_items"])
        for key, value in metrics.items():
            metric_sums[key] += value
        total_batches += 1
        if max_batches is not None and total_batches >= max_batches:
            break

    if total_batches == 0:
        return metric_sums
    return {metric_name: metric_value / total_batches for metric_name, metric_value in metric_sums.items()}
