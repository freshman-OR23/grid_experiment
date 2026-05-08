from __future__ import annotations

import torch
from torch import nn


def build_seq2seq_loss(ignore_index: int = -100) -> nn.Module:
    """next-token prediction 的标准交叉熵损失。"""
    return nn.CrossEntropyLoss(ignore_index=ignore_index)
