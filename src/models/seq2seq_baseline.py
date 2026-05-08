from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.models.modules import PositionalEncoding


@dataclass
class Seq2SeqConfig:
    sid_vocab_size: int
    d_model: int
    n_heads: int
    num_encoder_layers: int
    num_decoder_layers: int
    dim_feedforward: int
    dropout: float
    pad_token_id: int
    max_seq_len: int


class Seq2SeqSIDRecommender(nn.Module):
    """轻量 encoder-decoder Transformer，用于根据历史 SID 生成下一个 item SID。"""

    def __init__(self, config: Seq2SeqConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(
            num_embeddings=config.sid_vocab_size,
            embedding_dim=config.d_model,
            padding_idx=config.pad_token_id,
        )
        # 历史长度是“item 数”，而真正输入给 Transformer 的是“history item 数 × 每个 item 的 SID token 数”。
        # 这里给更充足的上界，避免更换码本层数或追加 de-dup token 后位置编码长度不够。
        self.position_encoding = PositionalEncoding(config.d_model, config.dropout, max_len=max(config.max_seq_len * 8, 512))

        self.transformer = nn.Transformer(
            d_model=config.d_model,
            nhead=config.n_heads,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.output_projection = nn.Linear(config.d_model, config.sid_vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        src = self.position_encoding(self.token_embedding(input_ids))
        tgt = self.position_encoding(self.token_embedding(decoder_input_ids))

        src_key_padding_mask = attention_mask == 0
        tgt_key_padding_mask = decoder_attention_mask == 0
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(decoder_input_ids.size(1), device=decoder_input_ids.device)

        hidden = self.transformer(
            src=src,
            tgt=tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.output_projection(hidden)
