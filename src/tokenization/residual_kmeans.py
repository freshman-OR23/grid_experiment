from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from src.tokenization.base import BaseSIDTokenizer
from src.utils.progress import create_progress


@dataclass
class ResidualKMeansConfig:
    num_layers: int
    codebook_size: int
    codebook_sizes: List[int] | None
    normalize_residuals: bool
    random_state: int


class ResidualKMeansTokenizer(BaseSIDTokenizer):
    """用逐层残差 K-Means 构造紧凑 SID，支持每层不同的码本大小。"""

    def __init__(self, config: ResidualKMeansConfig) -> None:
        self.config = config
        self.codebooks: List[np.ndarray] = []
        self.models: List[MiniBatchKMeans] = []
        self.layer_codebook_sizes = self._resolve_layer_codebook_sizes()

    def fit(self, embeddings: np.ndarray) -> None:
        residuals = embeddings.astype(np.float32).copy()
        self.codebooks.clear()
        self.models.clear()

        for layer_index in create_progress(range(self.config.num_layers), desc="训练 RK-Means tokenizer", leave=True):
            if self.config.normalize_residuals:
                residuals = self._normalize_rows(residuals)

            n_clusters = self.layer_codebook_sizes[layer_index]
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=self.config.random_state + layer_index,
                batch_size=min(4096, len(residuals)),
                n_init="auto",
            )
            kmeans.fit(residuals)
            labels = kmeans.predict(residuals)
            codebook = kmeans.cluster_centers_.astype(np.float32)

            self.models.append(kmeans)
            self.codebooks.append(codebook)
            residuals = residuals - codebook[labels]

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        residuals = embeddings.astype(np.float32).copy()
        sid_layers = []

        for layer_index, codebook in enumerate(self.codebooks):
            if self.config.normalize_residuals:
                residuals = self._normalize_rows(residuals)
            labels = self.models[layer_index].predict(residuals).astype(np.int32)
            sid_layers.append(labels)
            residuals = residuals - codebook[labels]

        return np.stack(sid_layers, axis=1)

    def export_state(self) -> Dict:
        return {
            "num_layers": self.config.num_layers,
            "codebook_size": self.config.codebook_size,
            "codebook_sizes": self.layer_codebook_sizes,
            "normalize_residuals": self.config.normalize_residuals,
            "random_state": self.config.random_state,
            "codebooks": [codebook.tolist() for codebook in self.codebooks],
        }

    def _resolve_layer_codebook_sizes(self) -> List[int]:
        if self.config.codebook_sizes is not None:
            if len(self.config.codebook_sizes) != self.config.num_layers:
                raise ValueError("codebook_sizes 的长度必须与 num_layers 一致。")
            return [int(size) for size in self.config.codebook_sizes]
        return [int(self.config.codebook_size)] * self.config.num_layers

    @staticmethod
    def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return matrix / norms
