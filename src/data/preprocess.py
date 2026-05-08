from __future__ import annotations

import ast
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.features.item_text_builder import build_item_text
from src.utils.io import ensure_dir, save_json
from src.utils.progress import create_progress


def _load_json_lines(path: str | Path) -> List[dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                records.append(ast.literal_eval(line))
    return records


def _iterative_k_core_filter(interactions: pd.DataFrame, min_interactions: int) -> pd.DataFrame:
    """迭代做 k-core filtering，直到用户和物品都满足最少交互数要求。"""
    filtered = interactions.copy()
    while True:
        user_counts = filtered["user_id"].value_counts()
        item_counts = filtered["item_id"].value_counts()
        valid_users = set(user_counts[user_counts >= min_interactions].index)
        valid_items = set(item_counts[item_counts >= min_interactions].index)
        next_filtered = filtered[
            filtered["user_id"].isin(valid_users) & filtered["item_id"].isin(valid_items)
        ]
        if len(next_filtered) == len(filtered):
            break
        filtered = next_filtered
    return filtered


def _build_splits(user_sequences: Dict[str, List[str]]) -> Tuple[List[dict], List[dict], List[dict]]:
    train_rows, valid_rows, test_rows = [], [], []
    for user_id, item_sequence in user_sequences.items():
        if len(item_sequence) < 3:
            continue
        train_rows.append({"user_id": user_id, "items": item_sequence[:-2]})
        # 验证集目标为倒数第二个 item，因此历史中不能包含它本身。
        valid_rows.append({"user_id": user_id, "items": item_sequence[:-2], "target": item_sequence[-2]})
        # 测试集目标为最后一个 item，历史允许包含验证目标。
        test_rows.append({"user_id": user_id, "items": item_sequence[:-1], "target": item_sequence[-1]})
    return train_rows, valid_rows, test_rows


def preprocess_amazon_beauty(
    review_json_path: str | Path,
    meta_json_path: str | Path,
    processed_dir: str | Path,
    text_fields: List[str],
    min_interactions: int,
) -> dict:
    """下载后的 Amazon Beauty 数据预处理：5-core、文本拼接、序列划分。"""
    processed_root = ensure_dir(processed_dir)

    reviews = _load_json_lines(review_json_path)
    meta_records = _load_json_lines(meta_json_path)

    meta_by_item = {}
    for item_meta in create_progress(meta_records, desc="整理 item 元数据", leave=True):
        item_id = item_meta.get("asin")
        if item_id is None:
            continue
        meta_by_item[item_id] = item_meta

    interaction_rows = []
    for review in create_progress(reviews, desc="整理用户交互", leave=True):
        user_id = review.get("reviewerID")
        item_id = review.get("asin")
        timestamp = review.get("unixReviewTime")
        if user_id is None or item_id is None or timestamp is None:
            continue
        interaction_rows.append(
            {
                "user_id": user_id,
                "item_id": item_id,
                "timestamp": int(timestamp),
            }
        )

    interactions = pd.DataFrame(interaction_rows)
    interactions = _iterative_k_core_filter(interactions, min_interactions=min_interactions)
    interactions = interactions.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    items_after_filter = sorted(interactions["item_id"].unique().tolist())
    item_texts = {}
    missing_meta_count = 0
    for item_id in create_progress(items_after_filter, desc="构造 item 文本", leave=True):
        item_meta = meta_by_item.get(item_id, {})
        if not item_meta:
            missing_meta_count += 1
        item_texts[item_id] = build_item_text(item_meta, text_fields=text_fields)

    user_sequences = defaultdict(list)
    for row in interactions.itertuples(index=False):
        user_sequences[row.user_id].append(row.item_id)

    train_rows, valid_rows, test_rows = _build_splits(user_sequences)

    pd.DataFrame(train_rows).to_json(processed_root / "train_sequences.jsonl", orient="records", lines=True)
    pd.DataFrame(valid_rows).to_json(processed_root / "valid_sequences.jsonl", orient="records", lines=True)
    pd.DataFrame(test_rows).to_json(processed_root / "test_sequences.jsonl", orient="records", lines=True)

    item_text_records = [{"item_id": item_id, "text": text} for item_id, text in item_texts.items()]
    pd.DataFrame(item_text_records).to_json(processed_root / "item_texts.jsonl", orient="records", lines=True)

    sequence_lengths = [len(sequence) for sequence in user_sequences.values()]
    summary = {
        "num_interactions": int(len(interactions)),
        "num_users": int(interactions["user_id"].nunique()),
        "num_items": int(interactions["item_id"].nunique()),
        "avg_sequence_length": float(sum(sequence_lengths) / len(sequence_lengths)) if sequence_lengths else 0.0,
        "min_interactions": int(min_interactions),
        "missing_meta_count": int(missing_meta_count),
    }
    save_json(summary, processed_root / "data_summary.json")

    return {
        "train_path": str(processed_root / "train_sequences.jsonl"),
        "valid_path": str(processed_root / "valid_sequences.jsonl"),
        "test_path": str(processed_root / "test_sequences.jsonl"),
        "item_text_path": str(processed_root / "item_texts.jsonl"),
        "summary_path": str(processed_root / "data_summary.json"),
        "summary": summary,
    }
