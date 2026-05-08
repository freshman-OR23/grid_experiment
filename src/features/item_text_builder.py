from __future__ import annotations

from typing import Dict, Iterable


def _safe_to_string(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " > ".join(str(item) for item in value if item is not None)
    return str(value)


def build_item_text(item_meta: Dict, text_fields: Iterable[str]) -> str:
    """按照配置把 item 的多个字段拼成一段文本，供语义编码器使用。"""
    pieces = []
    for field_name in text_fields:
        if field_name == "title":
            value = item_meta.get("title", "")
        elif field_name == "categories":
            value = item_meta.get("categories", "")
        elif field_name == "description":
            value = item_meta.get("description", "")
        elif field_name == "price":
            value = item_meta.get("price", "")
        else:
            value = item_meta.get(field_name, "")
        text = _safe_to_string(value).strip()
        if text:
            pieces.append(f"{field_name}: {text}")
    return " ; ".join(pieces)
