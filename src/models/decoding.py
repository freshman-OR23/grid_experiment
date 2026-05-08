from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from src.tokenization.base import BaseSIDTokenizer


@dataclass
class BeamSearchConfig:
    beam_width: int
    max_decode_len: int
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    sid_token_offset: int
    sid_vocab_size: int
    topk_eval: int
    filter_invalid_sid: bool
    deduplicate_predictions: bool
    prefix_pruning: bool = True
    candidate_multiplier: int = 4


def build_sid_prefix_index(sid_to_items: Dict[str, List[str]]) -> Dict[Tuple[int, ...], List[int]]:
    """为合法 SID 建前缀索引，方便 beam search 只扩展可行分支。"""
    prefix_index: Dict[Tuple[int, ...], set[int]] = {}
    for sid_string in sid_to_items:
        sid_tokens = tuple(int(token) for token in sid_string.split("-"))
        for prefix_length in range(len(sid_tokens)):
            prefix = sid_tokens[:prefix_length]
            next_token = sid_tokens[prefix_length]
            prefix_index.setdefault(prefix, set()).add(next_token)
    return {prefix: sorted(tokens) for prefix, tokens in prefix_index.items()}


def decode_topk_items(
    model,
    batch: Dict[str, torch.Tensor],
    sid_to_items: Dict[str, List[str]],
    item_sid_lookup: Dict[str, List[int]],
    config: BeamSearchConfig,
    device: torch.device,
    sid_prefix_index: Dict[Tuple[int, ...], List[int]] | None = None,
) -> List[List[str]]:
    """使用带前缀剪枝的 beam search 生成多个 SID 候选，再映射回 item。"""
    model.eval()
    batch_size = batch["input_ids"].size(0)
    all_predictions: List[List[str]] = []

    with torch.no_grad():
        for sample_index in range(batch_size):
            input_ids = batch["input_ids"][sample_index : sample_index + 1].to(device)
            attention_mask = batch["attention_mask"][sample_index : sample_index + 1].to(device)
            predicted_items = _beam_search_single(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                sid_to_items=sid_to_items,
                config=config,
                device=device,
                sid_prefix_index=sid_prefix_index,
            )
            all_predictions.append(predicted_items[: config.topk_eval])
    return all_predictions


def _beam_search_single(
    model,
    input_ids,
    attention_mask,
    sid_to_items,
    config: BeamSearchConfig,
    device: torch.device,
    sid_prefix_index: Dict[Tuple[int, ...], List[int]] | None = None,
) -> List[str]:
    beams: List[Tuple[List[int], float]] = [([config.bos_token_id], 0.0)]
    invalid_token_ids = {config.pad_token_id, config.bos_token_id}

    for _ in range(config.max_decode_len):
        active_beams = [beam for beam in beams if not beam[0] or beam[0][-1] != config.eos_token_id]
        completed_beams = [beam for beam in beams if beam[0] and beam[0][-1] == config.eos_token_id]
        if not active_beams:
            break

        decoder_input_ids = torch.tensor([tokens for tokens, _ in active_beams], dtype=torch.long, device=device)
        decoder_attention_mask = torch.ones_like(decoder_input_ids, device=device)
        expanded_input_ids = input_ids.expand(decoder_input_ids.size(0), -1)
        expanded_attention_mask = attention_mask.expand(decoder_input_ids.size(0), -1)

        logits = model(
            input_ids=expanded_input_ids,
            attention_mask=expanded_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )
        next_token_logits = logits[:, -1, :]
        next_token_logits[:, list(invalid_token_ids)] = float("-inf")
        next_token_log_probs = F.log_softmax(next_token_logits, dim=-1)

        new_beams: List[Tuple[List[int], float]] = list(completed_beams)
        per_beam_topk = min(config.sid_vocab_size, max(config.beam_width * config.candidate_multiplier, config.beam_width))

        for beam_index, (tokens, score) in enumerate(active_beams):
            candidate_log_probs = next_token_log_probs[beam_index]
            allowed_tokens = _allowed_next_tokens(
                tokens=tokens,
                config=config,
                sid_prefix_index=sid_prefix_index,
            )
            if allowed_tokens is not None:
                allowed_token_ids = [token + config.sid_token_offset for token in allowed_tokens]
                mask = torch.full_like(candidate_log_probs, float("-inf"))
                mask[allowed_token_ids] = candidate_log_probs[allowed_token_ids]
                candidate_log_probs = mask

            top_values, top_indices = torch.topk(candidate_log_probs, k=per_beam_topk, dim=-1)
            added = 0
            for value, index in zip(top_values.tolist(), top_indices.tolist(), strict=True):
                if value == float("-inf") or index >= config.sid_vocab_size:
                    continue
                new_beams.append((tokens + [index], score + float(value)))
                added += 1
                if added >= config.beam_width:
                    break

        new_beams.sort(key=lambda item: item[1], reverse=True)
        beams = new_beams[: config.beam_width]

    predicted_items: List[str] = []
    seen_items = set()
    for tokens, _ in beams:
        sid_tokens = [
            token - config.sid_token_offset
            for token in tokens[1:]
            if token not in (config.eos_token_id, config.pad_token_id) and token >= config.sid_token_offset
        ]
        if len(sid_tokens) != config.max_decode_len:
            continue
        sid_string = BaseSIDTokenizer.sid_to_string(sid_tokens)
        item_ids = sid_to_items.get(sid_string, [])
        if config.filter_invalid_sid and not item_ids:
            continue
        for item_id in item_ids:
            if config.deduplicate_predictions and item_id in seen_items:
                continue
            predicted_items.append(item_id)
            seen_items.add(item_id)
    return predicted_items


def _allowed_next_tokens(
    tokens: List[int],
    config: BeamSearchConfig,
    sid_prefix_index: Dict[Tuple[int, ...], List[int]] | None,
) -> List[int] | None:
    if not config.prefix_pruning or sid_prefix_index is None:
        return None
    sid_prefix = tuple(
        token - config.sid_token_offset
        for token in tokens[1:]
        if token >= config.sid_token_offset
    )
    return sid_prefix_index.get(sid_prefix)
