from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np


class BaseSIDTokenizer(ABC):
    """Semantic ID tokenizer 抽象基类，便于后续替换成 R-VQ / RQ-VAE。"""

    @abstractmethod
    def fit(self, embeddings: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def export_state(self) -> Dict:
        raise NotImplementedError

    @staticmethod
    def sid_to_string(sid_tokens: List[int]) -> str:
        return "-".join(str(token) for token in sid_tokens)
