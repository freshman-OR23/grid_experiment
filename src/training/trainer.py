from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.evaluation.evaluator import evaluate_model
from src.models.decoding import BeamSearchConfig
from src.training.callbacks import EarlyStopping
from src.utils.io import ensure_dir
from src.utils.progress import create_progress


@dataclass
class TrainerConfig:
    learning_rate: float
    weight_decay: float
    num_epochs: int
    gradient_accumulation_steps: int
    validate_every_n_steps: int
    validation_max_batches: int | None
    patience: int
    use_amp: bool
    early_stopping_min_delta: float = 0.0
    early_stopping_warmup_validations: int = 0


class Seq2SeqTrainer:
    """负责模型训练、验证和 checkpoint 保存的轻量训练器。"""

    def __init__(
        self,
        model,
        loss_fn,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        sid_to_items,
        item_sid_lookup,
        sid_prefix_index: Dict[Tuple[int, ...], List[int]] | None,
        decode_config: BeamSearchConfig,
        trainer_config: TrainerConfig,
        checkpoint_dir: str | Path,
        logger,
        profiler,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.sid_to_items = sid_to_items
        self.item_sid_lookup = item_sid_lookup
        self.sid_prefix_index = sid_prefix_index
        self.decode_config = decode_config
        self.config = trainer_config
        self.checkpoint_dir = ensure_dir(checkpoint_dir)
        self.logger = logger
        self.profiler = profiler
        self.device = device

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.use_amp and device.type == "cuda")
        self.early_stopper = EarlyStopping(
            patience=self.config.patience,
            min_delta=self.config.early_stopping_min_delta,
            warmup_validations=self.config.early_stopping_warmup_validations,
        )
        self.training_history: List[Dict[str, float | int]] = []

    def train(self) -> Dict[str, float]:
        global_step = 0
        best_metrics: Dict[str, float] = {}
        best_checkpoint = self.checkpoint_dir / "best_model.pt"
        self.optimizer.zero_grad(set_to_none=True)

        for epoch_index in range(self.config.num_epochs):
            self.model.train()
            running_loss = 0.0
            step_count = 0

            with self.profiler.track(f"train_epoch_{epoch_index + 1}"):
                for batch in create_progress(self.train_loader, desc=f"训练 Epoch {epoch_index + 1}", leave=True):
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    decoder_input_ids = batch["decoder_input_ids"].to(self.device)
                    decoder_attention_mask = batch["decoder_attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    with torch.cuda.amp.autocast(enabled=self.config.use_amp and self.device.type == "cuda"):
                        logits = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            decoder_input_ids=decoder_input_ids,
                            decoder_attention_mask=decoder_attention_mask,
                        )
                        loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                        loss = loss / self.config.gradient_accumulation_steps

                    self.scaler.scale(loss).backward()

                    if (global_step + 1) % self.config.gradient_accumulation_steps == 0:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)

                    running_loss += float(loss.item()) * self.config.gradient_accumulation_steps
                    step_count += 1
                    global_step += 1

                    if global_step % self.config.validate_every_n_steps == 0:
                        with self.profiler.track("validation"):
                            valid_metrics = evaluate_model(
                                model=self.model,
                                dataloader=self.valid_loader,
                                sid_to_items=self.sid_to_items,
                                item_sid_lookup=self.item_sid_lookup,
                                decode_config=self.decode_config,
                                device=self.device,
                                max_batches=self.config.validation_max_batches,
                                sid_prefix_index=self.sid_prefix_index,
                            )
                        train_loss = running_loss / max(step_count, 1)
                        self.logger.info(
                            "Epoch %d Step %d | train_loss=%.4f | valid Recall@10=%.4f",
                            epoch_index + 1,
                            global_step,
                            train_loss,
                            valid_metrics["Recall@10"],
                        )
                        self.training_history.append(
                            {
                                "epoch": epoch_index + 1,
                                "global_step": global_step,
                                "train_loss": train_loss,
                                **valid_metrics,
                            }
                        )
                        if not best_metrics or valid_metrics["Recall@10"] > best_metrics.get("Recall@10", -math.inf):
                            best_metrics = valid_metrics
                            torch.save(self.model.state_dict(), best_checkpoint)
                        if self.early_stopper.step(valid_metrics["Recall@10"]):
                            self.logger.info("触发 early stopping，停止训练。")
                            return best_metrics

            epoch_loss = running_loss / max(step_count, 1)
            self.logger.info("Epoch %d 结束，平均训练损失 %.4f", epoch_index + 1, epoch_loss)

        if not best_checkpoint.exists():
            torch.save(self.model.state_dict(), best_checkpoint)
        return best_metrics
