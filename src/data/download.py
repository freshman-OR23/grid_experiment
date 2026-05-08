from __future__ import annotations

import gzip
import shutil
from pathlib import Path

import requests

from src.utils.io import ensure_dir
from src.utils.progress import create_progress


AMAZON_BEAUTY_META_URL = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz"
AMAZON_BEAUTY_REVIEWS_URL = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz"


def _download_file(url: str, output_path: Path) -> None:
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    with output_path.open("wb") as handle, create_progress(total=total_size, desc=f"下载 {output_path.name}", leave=True) as progress:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            handle.write(chunk)
            progress.update(len(chunk))


def _gunzip_file(input_path: Path, output_path: Path) -> None:
    with gzip.open(input_path, "rb") as src_handle, output_path.open("wb") as dst_handle:
        shutil.copyfileobj(src_handle, dst_handle)


def download_amazon_beauty(raw_data_dir: str | Path) -> dict:
    """下载 Amazon Beauty 所需的评论和元数据文件。"""
    raw_dir = ensure_dir(raw_data_dir)

    review_gz = raw_dir / "reviews_Beauty_5.json.gz"
    meta_gz = raw_dir / "meta_Beauty.json.gz"
    review_json = raw_dir / "reviews_Beauty_5.json"
    meta_json = raw_dir / "meta_Beauty.json"

    if not review_gz.exists():
        _download_file(AMAZON_BEAUTY_REVIEWS_URL, review_gz)
    if not meta_gz.exists():
        _download_file(AMAZON_BEAUTY_META_URL, meta_gz)

    if not review_json.exists():
        _gunzip_file(review_gz, review_json)
    if not meta_json.exists():
        _gunzip_file(meta_gz, meta_json)

    return {
        "review_json": str(review_json),
        "meta_json": str(meta_json),
    }
