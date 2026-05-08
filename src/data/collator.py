from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass
class CollatorConfig:
    pad_token_id: int
    bos_token_id: int
    eos_token_id: int


class SIDSeq2SeqCollator:
    """构造 encoder 输入和 decoder teacher-forcing 所需张量。"""

    def __init__(self, config: CollatorConfig) -> None:
        self.config = config

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        history_lengths = [len(sample["history_tokens"]) for sample in batch]
        decoder_lengths = [len(sample["target_tokens"]) + 1 for sample in batch]

        max_history = max(history_lengths) if history_lengths else 0
        max_decoder = max(decoder_lengths) + 1 if decoder_lengths else 0

        input_ids, attention_mask = [], []
        decoder_input_ids, decoder_attention_mask, labels = [], [], []
        target_items = []

        for sample in batch:
            history_tokens = sample["history_tokens"] or [self.config.pad_token_id]
            target_tokens = sample["target_tokens"]

            padded_history = history_tokens + [self.config.pad_token_id] * (max_history - len(history_tokens))
            history_mask = [1] * len(history_tokens) + [0] * (max_history - len(history_tokens))

            # decoder 训练采用 teacher forcing：输入 BOS + target，标签为 target + EOS。
            decoder_in = [self.config.bos_token_id] + target_tokens
            decoder_label = target_tokens + [self.config.eos_token_id]
            decoder_in = decoder_in + [self.config.pad_token_id] * (max_decoder - len(decoder_in))
            decoder_label = decoder_label + [-100] * (max_decoder - len(decoder_label))
            decoder_mask = [1] * (len(target_tokens) + 1) + [0] * (max_decoder - len(target_tokens) - 1)

            input_ids.append(padded_history)
            attention_mask.append(history_mask)
            decoder_input_ids.append(decoder_in)
            decoder_attention_mask.append(decoder_mask)
            labels.append(decoder_label)
            target_items.append(sample["target_item"])

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "decoder_input_ids": torch.tensor(decoder_input_ids, dtype=torch.long),
            "decoder_attention_mask": torch.tensor(decoder_attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "target_items": target_items,
        }
