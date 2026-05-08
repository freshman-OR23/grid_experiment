from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.collator import CollatorConfig, SIDSeq2SeqCollator
from src.data.dataset import GenerativeSIDDataset, SequenceDatasetConfig, infer_sid_sequence_length, load_item_sid_map
from src.evaluation.evaluator import evaluate_model
from src.models.decoding import BeamSearchConfig, build_sid_prefix_index
from src.models.seq2seq_baseline import Seq2SeqConfig, Seq2SeqSIDRecommender
from src.tokenization.utils import infer_sid_vocab_size
from src.utils.config import ensure_project_relative_paths, load_yaml_config
from src.utils.io import load_json, save_json
from src.utils.logging import setup_logger
from src.utils.profiler import StageProfiler
from src.utils.seed import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    config = ensure_project_relative_paths(load_yaml_config(args.config), PROJECT_ROOT)
    set_global_seed(config["seed"])
    logger = setup_logger(config["paths"]["log_dir"], name="run_eval")
    profiler = StageProfiler()

    item_sid_map = load_item_sid_map(Path(config["paths"]["sid_dir"]) / "item_sids.jsonl")
    sid_sequence_length = infer_sid_sequence_length(item_sid_map)
    sid_collision_map = load_json(Path(config["paths"]["sid_dir"]) / "sid_collisions.json")
    sid_to_items = {sid: items for sid, items in sid_collision_map.items()}
    if not sid_to_items:
        for item_id, sid_tokens in item_sid_map.items():
            sid_string = "-".join(str(token) for token in sid_tokens)
            sid_to_items.setdefault(sid_string, []).append(item_id)
    sid_prefix_index = build_sid_prefix_index(sid_to_items)

    dataset = GenerativeSIDDataset(
        sequence_path=Path(config["paths"]["processed_data_dir"]) / f"{args.split}_sequences.jsonl",
        item_sid_map=item_sid_map,
        config=SequenceDatasetConfig(
            history_window=config["data"]["history_window"],
            sliding_window=False,
            sid_token_offset=config["data"]["sid_token_offset"],
        ),
    )
    if args.smoke_test:
        dataset.samples = dataset.samples[:64]

    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["eval_batch_size"],
        shuffle=False,
        collate_fn=SIDSeq2SeqCollator(
            CollatorConfig(
                pad_token_id=config["data"]["pad_token_id"],
                bos_token_id=config["data"]["bos_token_id"],
                eos_token_id=config["data"]["eos_token_id"],
            )
        ),
    )

    sid_vocab_size = infer_sid_vocab_size(item_sid_map, config["data"]["sid_token_offset"])
    model = Seq2SeqSIDRecommender(
        Seq2SeqConfig(
            sid_vocab_size=sid_vocab_size,
            d_model=config["model"]["d_model"],
            n_heads=config["model"]["n_heads"],
            num_encoder_layers=config["model"]["num_encoder_layers"],
            num_decoder_layers=config["model"]["num_decoder_layers"],
            dim_feedforward=config["model"]["dim_feedforward"],
            dropout=config["model"]["dropout"],
            pad_token_id=config["data"]["pad_token_id"],
            max_seq_len=config["model"]["max_seq_len"],
        )
    )
    checkpoint_path = Path(config["paths"]["checkpoint_dir"]) / "best_model.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with profiler.track("evaluate_model"):
        metrics = evaluate_model(
            model=model,
            dataloader=dataloader,
            sid_to_items=sid_to_items,
            item_sid_lookup=item_sid_map,
            decode_config=BeamSearchConfig(
                beam_width=config["decoding"]["beam_width"],
                max_decode_len=sid_sequence_length or config["decoding"]["max_decode_len"],
                bos_token_id=config["data"]["bos_token_id"],
                eos_token_id=config["data"]["eos_token_id"],
                pad_token_id=config["data"]["pad_token_id"],
                sid_token_offset=config["data"]["sid_token_offset"],
                sid_vocab_size=sid_vocab_size,
                topk_eval=config["decoding"]["topk_eval"],
                filter_invalid_sid=config["decoding"]["filter_invalid_sid"],
                deduplicate_predictions=config["decoding"]["deduplicate_predictions"],
                prefix_pruning=config["decoding"].get("prefix_pruning", True),
                candidate_multiplier=config["decoding"].get("candidate_multiplier", 4),
            ),
            device=device,
            sid_prefix_index=sid_prefix_index,
        )

    logger.info("%s 指标：%s", args.split, metrics)
    save_json(metrics, Path(config["paths"]["metrics_dir"]) / f"{args.split}_metrics.json")
    save_json(profiler.summary(), Path(config["paths"]["profiling_dir"]) / f"{args.split}_profile.json")


if __name__ == "__main__":
    main()
