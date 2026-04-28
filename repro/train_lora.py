import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

from repro.common import (
    DEFAULT_WANDB_ENTITY,
    DEFAULT_WANDB_PROJECT,
    LORA_TARGET_MODULES,
    build_sft_features,
    ensure_dir,
    load_instruction_records,
    resolve_wandb_mode,
    set_seed,
    setup_logger,
)


class SFTDataset(Dataset):
    def __init__(self, features: List[Dict[str, List[int]]]) -> None:
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        return self.features[index]


@dataclass
class SupervisedDataCollator:
    tokenizer: AutoTokenizer

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.tensor(feature["input_ids"], dtype=torch.long) for feature in features]
        attention_masks = [torch.tensor(feature["attention_mask"], dtype=torch.long) for feature in features]
        labels = [torch.tensor(feature["labels"], dtype=torch.long) for feature in features]

        batch_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        batch_attention = torch.nn.utils.rnn.pad_sequence(
            attention_masks,
            batch_first=True,
            padding_value=0,
        )
        batch_labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100,
        )

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention,
            "labels": batch_labels,
        }


class FileLoggingCallback(TrainerCallback):
    def __init__(self, logger, estimated_steps: Optional[int]) -> None:
        self.logger = logger
        self.estimated_steps = estimated_steps

    def on_train_begin(self, args, state, control, **kwargs):
        self.logger.info(
            "Trainer loop started | max_steps=%s num_train_epochs=%s per_device_batch=%s grad_accum=%s",
            state.max_steps,
            args.num_train_epochs,
            args.per_device_train_batch_size,
            args.gradient_accumulation_steps,
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        step = state.global_step
        total_steps = self.estimated_steps or state.max_steps or 0
        percent = (100.0 * step / total_steps) if total_steps else None
        percent_text = f"{percent:.2f}" if percent is not None else "n/a"
        self.logger.info(
            "Train log | step=%s/%s progress_pct=%s epoch=%.4f loss=%s grad_norm=%s learning_rate=%s",
            step,
            total_steps if total_steps else "unknown",
            percent_text,
            state.epoch or 0.0,
            logs.get("loss"),
            logs.get("grad_norm"),
            logs.get("learning_rate"),
        )

    def on_save(self, args, state, control, **kwargs):
        self.logger.info("Checkpoint saved | global_step=%s epoch=%.4f", state.global_step, state.epoch or 0.0)

    def on_train_end(self, args, state, control, **kwargs):
        self.logger.info("Trainer loop ended | final_global_step=%s final_epoch=%.4f", state.global_step, state.epoch or 0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA SFT for round-1 CPQS comparisons.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--group_name", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--backbone", choices=["qwen", "llama"], default="qwen")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--wandb_project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb_entity", default=DEFAULT_WANDB_ENTITY)
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument("--log_file", default="")
    return parser.parse_args()


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    ensure_dir(output_dir)
    log_file = Path(cfg.log_file) if cfg.log_file else output_dir / "train_lora.log"
    logger = setup_logger("repro.train_lora", log_file)
    logger.info("LoRA training started")
    logger.info("Config | group=%s seed=%s output_dir=%s", cfg.group_name, cfg.seed, output_dir)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    records = load_instruction_records(cfg.train_data)
    features: List[Dict[str, List[int]]] = []
    for record in records:
        feature = build_sft_features(
            tokenizer=tokenizer,
            instruction=record["instruction"],
            input_text=record["input"],
            answer=record["output"],
            backbone=cfg.backbone,
            max_length=cfg.max_length,
        )
        if feature is not None:
            features.append(feature)
    logger.info("Training data prepared | raw_records=%s usable_features=%s", len(records), len(features))

    dataset = SFTDataset(features)
    updates_per_epoch = math.ceil(
        len(dataset) / (cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps)
    )
    estimated_steps = math.ceil(updates_per_epoch * cfg.epochs)
    logger.info(
        "Training schedule | dataset_size=%s updates_per_epoch=%s estimated_total_steps=%s",
        len(dataset),
        updates_per_epoch,
        estimated_steps,
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=resolve_wandb_mode(not cfg.disable_wandb),
        run_name=f"{cfg.group_name}-seed{cfg.seed}",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        save_total_limit=1,
        seed=cfg.seed,
    )

    if not cfg.disable_wandb:
        import os

        os.environ["WANDB_PROJECT"] = cfg.wandb_project
        os.environ["WANDB_ENTITY"] = cfg.wandb_entity

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=SupervisedDataCollator(tokenizer),
        callbacks=[FileLoggingCallback(logger, estimated_steps)],
    )
    trainer.train()
    logger.info("Trainer finished")
    trainer.save_model(str(output_dir / "final_adapter"))
    tokenizer.save_pretrained(str(output_dir / "final_adapter"))
    logger.info("Adapter saved | path=%s", output_dir / "final_adapter")

    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                **vars(cfg),
                "num_train_records": len(records),
                "num_train_features": len(features),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("LoRA training finished")


if __name__ == "__main__":
    main()
