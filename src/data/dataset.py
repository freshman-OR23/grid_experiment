from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from torch.utils.data import Dataset


@dataclass
class SequenceDatasetConfig:
    history_window: int
    sliding_window: bool
    sid_token_offset: int


def load_item_sid_map(path: str | Path) -> Dict[str, List[int]]:
    sid_map: Dict[str, List[int]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            sid_map[record["item_id"]] = [int(token) for token in record["sid_tokens"]]
    return sid_map


def infer_sid_sequence_length(item_sid_map: Dict[str, List[int]]) -> int:
    """从磁盘 SID 映射中推断当前 item SID 长度，兼容不同码本层数与 de-dup 方案。"""
    if not item_sid_map:
        return 0
    return len(next(iter(item_sid_map.values())))


class GenerativeSIDDataset(Dataset):
    """把用户 item 序列展开成“历史 SID 前缀 -> 下一个 item SID”的样本。"""

    def __init__(
        self,
        sequence_path: str | Path,
        item_sid_map: Dict[str, List[int]],
        config: SequenceDatasetConfig,
    ) -> None:
        self.samples: List[Dict[str, str | List[str]]] = []
        self.item_sid_map = item_sid_map
        self.config = config
        self._build_samples(sequence_path)

    def _build_samples(self, sequence_path: str | Path) -> None:
        with Path(sequence_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                item_ids = [item_id for item_id in record["items"] if item_id in self.item_sid_map]
                explicit_target = record.get("target")
                if explicit_target is not None and explicit_target not in self.item_sid_map:
                    explicit_target = None

                if self.config.sliding_window:
                    if len(item_ids) < 2:
                        continue
                    for end_index in range(1, len(item_ids)):
                        history = item_ids[max(0, end_index - self.config.history_window):end_index]
                        target = item_ids[end_index]
                        self.samples.append({"history_items": history, "target_item": target})
                    continue

                if explicit_target is not None:
                    history = item_ids[max(0, len(item_ids) - self.config.history_window):]
                    target = explicit_target
                else:
                    if len(item_ids) < 2:
                        continue
                    history = item_ids[max(0, len(item_ids) - 1 - self.config.history_window):-1]
                    target = item_ids[-1]
                if not history:
                    continue
                self.samples.append({"history_items": history, "target_item": target})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, List[int] | str]:
        sample = self.samples[index]
        history_tokens: List[int] = []

        # history_window 控制单个样本最多保留多少历史 item，避免序列过长拖慢训练。
        for item_id in sample["history_items"]:
            history_tokens.extend(
                token + self.config.sid_token_offset for token in self.item_sid_map[item_id]
            )

        # 真实 SID token 从 sid_token_offset 开始编号，避开 pad/bos/eos 的特殊 token。
        target_tokens = [
            token + self.config.sid_token_offset
            for token in self.item_sid_map[sample["target_item"]]
        ]
        return {
            "history_tokens": history_tokens,
            "target_tokens": target_tokens,
            "target_item": sample["target_item"],
        }
