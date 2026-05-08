from __future__ import annotations

from typing import Dict, List


def infer_sid_vocab_size(item_sid_map: Dict[str, List[int]], sid_token_offset: int) -> int:
    """从当前 SID 数据自动推断模型词表大小，避免配置与磁盘数据不一致。"""
    max_sid_token = 0
    for sid_tokens in item_sid_map.values():
        if sid_tokens:
            max_sid_token = max(max_sid_token, max(int(token) for token in sid_tokens))
    return max_sid_token + 1 + sid_token_offset
