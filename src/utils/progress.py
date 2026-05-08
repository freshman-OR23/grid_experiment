from __future__ import annotations

from tqdm.auto import tqdm


def create_progress(iterable=None, total=None, desc: str = "", leave: bool = False):
    """统一包装 tqdm，便于后续替换进度条实现。"""
    return tqdm(iterable=iterable, total=total, desc=desc, leave=leave)
