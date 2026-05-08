from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.tokenization.base import BaseSIDTokenizer
from src.utils.progress import create_progress


@dataclass
class RQVAEConfig:
    num_layers: int
    codebook_sizes: List[int]
    hidden_dim: int
    latent_dim: int
    batch_size: int
    learning_rate: float
    num_epochs: int
    commitment_beta: float
    random_state: int
    device: str


class _RQVAEModel(nn.Module):
    def __init__(self, input_dim: int, config: RQVAEConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, input_dim),
        )
        self.codebooks = nn.ParameterList(
            [nn.Parameter(torch.empty(size, config.latent_dim)) for size in config.codebook_sizes]
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class RQVAETokenizer(BaseSIDTokenizer):
    """轻量 RQ-VAE tokenizer，支持 K-Means 码本初始化与分层残差量化。"""

    def __init__(self, config: RQVAEConfig) -> None:
        self.config = config
        self.device = self._resolve_device(config.device)
        self.model: _RQVAEModel | None = None
        self.input_dim: int | None = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def fit(self, embeddings: np.ndarray) -> None:
        self.input_dim = int(embeddings.shape[1])
        self.model = _RQVAEModel(self.input_dim, self.config).to(self.device)
        dataset = TensorDataset(torch.tensor(embeddings, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=False)

        self._initialize_codebooks(embeddings)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)

        for epoch in create_progress(range(self.config.num_epochs), desc="训练 RQ-VAE tokenizer", leave=True):
            epoch_loss = 0.0
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                z_e = self.model.encode(batch_x)
                z_q, _, rq_loss = self._quantize(z_e)
                recon = self.model.decode(z_q)
                recon_loss = F.mse_loss(recon, batch_x)
                loss = recon_loss + rq_loss
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("RQ-VAE tokenizer 尚未 fit。")

        self.model.eval()
        sid_rows: List[np.ndarray] = []
        dataset = TensorDataset(torch.tensor(embeddings, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False, drop_last=False)

        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                z_e = self.model.encode(batch_x)
                _, indices, _ = self._quantize(z_e)
                sid_rows.append(indices.cpu().numpy())
        return np.concatenate(sid_rows, axis=0) if sid_rows else np.empty((0, self.config.num_layers), dtype=np.int32)

    def export_state(self) -> Dict:
        if self.model is None:
            raise RuntimeError("RQ-VAE tokenizer 尚未 fit。")
        return {
            "type": "rqvae",
            "num_layers": self.config.num_layers,
            "codebook_sizes": self.config.codebook_sizes,
            "hidden_dim": self.config.hidden_dim,
            "latent_dim": self.config.latent_dim,
            "batch_size": self.config.batch_size,
            "learning_rate": self.config.learning_rate,
            "num_epochs": self.config.num_epochs,
            "commitment_beta": self.config.commitment_beta,
            "random_state": self.config.random_state,
            "input_dim": self.input_dim,
            "codebooks": [codebook.detach().cpu().tolist() for codebook in self.model.codebooks],
        }

    def _initialize_codebooks(self, embeddings: np.ndarray) -> None:
        assert self.model is not None
        self.model.eval()
        sample_count = min(len(embeddings), 4096)
        sample = torch.tensor(embeddings[:sample_count], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            encoded = self.model.encode(sample).cpu().numpy()

        residuals = encoded.copy()
        for layer_index, codebook_size in enumerate(self.config.codebook_sizes):
            kmeans = MiniBatchKMeans(
                n_clusters=codebook_size,
                random_state=self.config.random_state + layer_index,
                batch_size=min(1024, sample_count),
                n_init="auto",
            )
            kmeans.fit(residuals)
            centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=self.device)
            self.model.codebooks[layer_index].data.copy_(centers)
            labels = kmeans.predict(residuals)
            residuals = residuals - kmeans.cluster_centers_[labels]

    def _quantize(self, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.model is not None
        residual = z_e
        quantized_sum = torch.zeros_like(z_e)
        all_indices = []
        rq_loss = torch.tensor(0.0, device=z_e.device)

        for layer_index, codebook in enumerate(self.model.codebooks):
            distances = (
                residual.pow(2).sum(dim=1, keepdim=True)
                - 2 * residual @ codebook.t()
                + codebook.pow(2).sum(dim=1).unsqueeze(0)
            )
            indices = torch.argmin(distances, dim=1)
            quantized = codebook[indices]
            quantized_st = residual + (quantized - residual).detach()
            quantized_sum = quantized_sum + quantized_st
            rq_loss = rq_loss + F.mse_loss(quantized, residual.detach()) + self.config.commitment_beta * F.mse_loss(
                residual, quantized.detach()
            )
            residual = residual - quantized
            all_indices.append(indices.unsqueeze(1))

        return quantized_sum, torch.cat(all_indices, dim=1), rq_loss
