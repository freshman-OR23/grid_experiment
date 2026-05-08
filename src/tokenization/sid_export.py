from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.tokenization.base import BaseSIDTokenizer
from src.utils.io import ensure_dir, save_json


def export_item_sids(
    item_ids: List[str],
    sid_matrix: np.ndarray,
    output_dir: str | Path,
    append_dedup_token: bool = False,
    dedup_default_token: int = 0,
) -> Dict:
    """导出 item SID，并可选为碰撞 SID 追加一个局部去重 token。"""
    output_root = ensure_dir(output_dir)

    # 先按原始量化 SID 分组，后续再决定是否补一个 de-dup token。
    base_sid_to_items = defaultdict(list)
    for item_id, sid_tokens in zip(item_ids, sid_matrix, strict=True):
        sid_list = [int(token) for token in sid_tokens.tolist()]
        base_sid_string = BaseSIDTokenizer.sid_to_string(sid_list)
        base_sid_to_items[base_sid_string].append(item_id)

    sid_records = []
    final_sid_to_items = defaultdict(list)
    item_to_canonical = {}
    item_to_final_sid: Dict[str, List[int]] = {}
    max_dedup_token = dedup_default_token

    for base_sid_string, items in base_sid_to_items.items():
        base_sid_tokens = [int(token) for token in base_sid_string.split("-")]
        sorted_items = sorted(items)
        canonical_item = sorted_items[0]

        for dedup_index, item_id in enumerate(sorted_items):
            final_sid_tokens = list(base_sid_tokens)
            if append_dedup_token:
                # 仅在同一个碰撞组内分配局部编号；无碰撞样本统一走默认 token。
                dedup_token = dedup_default_token if len(sorted_items) == 1 else dedup_index
                final_sid_tokens.append(int(dedup_token))
                max_dedup_token = max(max_dedup_token, int(dedup_token))

            final_sid_string = BaseSIDTokenizer.sid_to_string(final_sid_tokens)
            sid_records.append(
                {
                    "item_id": item_id,
                    "sid_tokens": final_sid_tokens,
                    "sid_string": final_sid_string,
                }
            )
            final_sid_to_items[final_sid_string].append(item_id)
            item_to_canonical[item_id] = canonical_item
            item_to_final_sid[item_id] = final_sid_tokens

    layer_count = len(next(iter(item_to_final_sid.values()))) if item_to_final_sid else sid_matrix.shape[1]
    layer_distributions = [defaultdict(int) for _ in range(layer_count)]
    for item_id in item_ids:
        sid_tokens = item_to_final_sid[item_id]
        for layer_index, token_value in enumerate(sid_tokens):
            layer_distributions[layer_index][int(token_value)] += 1

    pd.DataFrame(sid_records).to_json(output_root / "item_sids.jsonl", orient="records", lines=True)

    base_collision_map = {sid: items for sid, items in base_sid_to_items.items() if len(items) > 1}
    collision_map = {sid: items for sid, items in final_sid_to_items.items() if len(items) > 1}

    stats = {
        "num_items": len(item_ids),
        "num_unique_sid": len(final_sid_to_items),
        "num_collisions": len(collision_map),
        "collision_items": sum(len(items) for items in collision_map.values()),
        "base_num_unique_sid": len(base_sid_to_items),
        "base_num_collisions": len(base_collision_map),
        "base_collision_items": sum(len(items) for items in base_collision_map.values()),
        "append_dedup_token": append_dedup_token,
        "dedup_default_token": dedup_default_token,
        "dedup_vocab_size": int(max_dedup_token) + 1 if append_dedup_token else 0,
        "max_collision_group_size": max((len(items) for items in base_sid_to_items.values()), default=0),
        "layer_distributions": [
            {str(token): count for token, count in sorted(layer_distribution.items())}
            for layer_distribution in layer_distributions
        ],
    }
    save_json(stats, output_root / "sid_stats.json")
    save_json(item_to_canonical, output_root / "item_to_canonical.json")
    save_json(collision_map, output_root / "sid_collisions.json")
    save_json(base_collision_map, output_root / "base_sid_collisions.json")

    return {
        "sid_path": str(output_root / "item_sids.jsonl"),
        "sid_stats_path": str(output_root / "sid_stats.json"),
        "item_to_canonical_path": str(output_root / "item_to_canonical.json"),
        "collision_path": str(output_root / "sid_collisions.json"),
        "base_collision_path": str(output_root / "base_sid_collisions.json"),
        "stats": stats,
    }
