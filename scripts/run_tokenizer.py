from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.download import download_amazon_beauty
from src.data.preprocess import preprocess_amazon_beauty
from src.features.text_encoder import PretrainedTextEncoder, TextEncoderConfig
from src.tokenization.residual_kmeans import ResidualKMeansConfig, ResidualKMeansTokenizer
from src.tokenization.rqvae import RQVAEConfig, RQVAETokenizer
from src.tokenization.sid_export import export_item_sids
from src.utils.config import ensure_project_relative_paths, load_yaml_config
from src.utils.io import ensure_dir, save_json
from src.utils.logging import setup_logger
from src.utils.profiler import StageProfiler
from src.utils.seed import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    config = ensure_project_relative_paths(load_yaml_config(args.config), PROJECT_ROOT)
    set_global_seed(config["seed"])
    logger = setup_logger(config["paths"]["log_dir"], name="run_tokenizer")
    profiler = StageProfiler()

    with profiler.track("download_data"):
        download_info = download_amazon_beauty(config["paths"]["raw_data_dir"])

    with profiler.track("preprocess_data"):
        processed_info = preprocess_amazon_beauty(
            review_json_path=download_info["review_json"],
            meta_json_path=download_info["meta_json"],
            processed_dir=config["paths"]["processed_data_dir"],
            text_fields=config["dataset"]["text_fields"],
            min_interactions=config["dataset"]["min_interactions"],
        )
    logger.info("数据摘要：%s", processed_info["summary"])

    item_text_df = pd.read_json(processed_info["item_text_path"], lines=True)
    if args.smoke_test:
        item_text_df = item_text_df.head(256)

    encoder = PretrainedTextEncoder(
        TextEncoderConfig(
            model_name=config["text_encoder"]["model_name"],
            batch_size=config["text_encoder"]["batch_size"],
            max_length=config["text_encoder"]["max_length"],
            device=config["text_encoder"]["device"],
            normalize_embeddings=config["text_encoder"]["normalize_embeddings"],
        )
    )

    with profiler.track("generate_embeddings"):
        start_time = time.perf_counter()
        embeddings = encoder.encode(item_text_df["text"].tolist())
        elapsed = time.perf_counter() - start_time
        embedding_dir = ensure_dir(Path(config["paths"]["processed_data_dir"]) / "item_embeddings")
        np.save(embedding_dir / "item_embeddings.npy", embeddings)
        item_text_df[["item_id"]].to_json(embedding_dir / "item_ids.jsonl", orient="records", lines=True)
        save_json(
            {
                "num_items": len(item_text_df),
                "embedding_dim": int(embeddings.shape[1]) if embeddings.size else 0,
                "batch_size": config["text_encoder"]["batch_size"],
                "device": encoder.device,
                "total_seconds": elapsed,
                "seconds_per_item": elapsed / max(len(item_text_df), 1),
            },
            embedding_dir / "embedding_stats.json",
        )

    tokenizer = build_tokenizer(config)

    with profiler.track("train_tokenizer"):
        tokenizer.fit(embeddings)
    with profiler.track("export_sid"):
        sid_matrix = tokenizer.transform(embeddings)
        sid_info = export_item_sids(
            item_ids=item_text_df["item_id"].tolist(),
            sid_matrix=sid_matrix,
            output_dir=config["paths"]["sid_dir"],
            append_dedup_token=config["tokenizer"].get("append_dedup_token", False),
            dedup_default_token=config["tokenizer"].get("dedup_default_token", 0),
        )
        save_json(tokenizer.export_state(), Path(config["paths"]["sid_dir"]) / "tokenizer_state.json")

    logger.info("SID 统计：%s", sid_info["stats"])
    save_json(profiler.summary(), Path(config["paths"]["profiling_dir"]) / "tokenizer_profile.json")


def build_tokenizer(config):
    tokenizer_type = config["tokenizer"]["type"].lower()
    if tokenizer_type == "rkmeans":
        return ResidualKMeansTokenizer(
            ResidualKMeansConfig(
                num_layers=config["tokenizer"]["num_layers"],
                codebook_size=config["tokenizer"]["codebook_size"],
                codebook_sizes=config["tokenizer"].get("codebook_sizes"),
                normalize_residuals=config["tokenizer"]["normalize_residuals"],
                random_state=config["tokenizer"]["random_state"],
            )
        )
    if tokenizer_type == "rqvae":
        return RQVAETokenizer(
            RQVAEConfig(
                num_layers=config["tokenizer"]["num_layers"],
                codebook_sizes=config["tokenizer"]["codebook_sizes"],
                hidden_dim=config["tokenizer"]["hidden_dim"],
                latent_dim=config["tokenizer"]["latent_dim"],
                batch_size=config["tokenizer"]["train_batch_size"],
                learning_rate=config["tokenizer"]["learning_rate"],
                num_epochs=config["tokenizer"]["num_epochs"],
                commitment_beta=config["tokenizer"]["commitment_beta"],
                random_state=config["tokenizer"]["random_state"],
                device=config["tokenizer"].get("device", "auto"),
            )
        )
    raise ValueError(f"不支持的 tokenizer 类型: {tokenizer_type}")


if __name__ == "__main__":
    main()
