from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.utils.progress import create_progress


@dataclass
class TextEncoderConfig:
    model_name: str
    batch_size: int
    max_length: int
    device: str
    normalize_embeddings: bool


class PretrainedTextEncoder:
    """封装预训练文本编码器，便于后续替换成更强模型做消融。"""

    def __init__(self, config: TextEncoderConfig) -> None:
        self.config = config
        self.device = self._resolve_device(config.device)
        self.model = SentenceTransformer(config.model_name, device=self.device)
        if hasattr(self.model, "max_seq_length"):
            self.model.max_seq_length = config.max_length

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def encode(self, texts: List[str]) -> np.ndarray:
        """对 item 文本列表进行批量编码，输出二维语义向量数组。"""
        embeddings = []
        total_batches = (len(texts) + self.config.batch_size - 1) // self.config.batch_size
        for batch_index in create_progress(range(total_batches), desc="生成 item embedding", leave=True):
            start = batch_index * self.config.batch_size
            end = min(len(texts), start + self.config.batch_size)
            batch_texts = texts[start:end]
            batch_embeddings = self.model.encode(
                batch_texts,
                batch_size=len(batch_texts),
                show_progress_bar=False,
                normalize_embeddings=self.config.normalize_embeddings,
                convert_to_numpy=True,
            )
            embeddings.append(batch_embeddings)
        return np.concatenate(embeddings, axis=0) if embeddings else np.empty((0, 0), dtype=np.float32)
