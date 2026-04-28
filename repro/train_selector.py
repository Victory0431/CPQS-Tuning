import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

from repro.common import (
    DEFAULT_WANDB_ENTITY,
    DEFAULT_WANDB_PROJECT,
    DeepTextCNN,
    build_question,
    ensure_dir,
    extract_response_hidden_states,
    load_backbone,
    load_selector_examples,
    resolve_wandb_mode,
    selected_num_layers,
    set_seed,
    setup_logger,
    stratified_split,
    total_layers,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CPQS selector with validation metrics.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--pos_train_path", required=True)
    parser.add_argument("--neg_dataset1_path", required=True)
    parser.add_argument("--neg_dataset2_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--backbone", choices=["qwen", "llama"], default="qwen")
    parser.add_argument("--use_layers", choices=["all", "last"], default="all")
    parser.add_argument("--use_part", choices=["front", "middle", "back", "full"], default="full")
    parser.add_argument("--device_cnn", default="cuda:0")
    parser.add_argument("--device_llm", default="cuda:1")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_examples", type=int, default=15000)
    parser.add_argument("--wandb_project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb_entity", default=DEFAULT_WANDB_ENTITY)
    parser.add_argument("--wandb_run_name", default="selector-train")
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument("--log_file", default="")
    return parser.parse_args()


def compute_metrics(golds: List[int], probs: List[float], preds: List[int]) -> Dict[str, float]:
    auc = roc_auc_score(golds, probs) if len(set(golds)) > 1 else 0.0
    return {
        "accuracy": accuracy_score(golds, preds),
        "f1": f1_score(golds, preds),
        "auc": auc,
    }


def evaluate(
    examples: List[Dict[str, object]],
    tokenizer,
    model,
    classifier: nn.Module,
    cfg: argparse.Namespace,
) -> Dict[str, float]:
    golds: List[int] = []
    probs: List[float] = []
    preds: List[int] = []
    losses: List[float] = []
    criterion = nn.CrossEntropyLoss()
    classifier.eval()

    with torch.no_grad():
        for example in tqdm(examples, desc="Validation", leave=False):
            question = build_question(example)
            answer = str(example["answer"])
            label = int(example["label"])
            hidden = extract_response_hidden_states(
                tokenizer=tokenizer,
                model=model,
                question=question,
                answer=answer,
                backbone=cfg.backbone,
                use_layers=cfg.use_layers,
                use_part=cfg.use_part,
            )
            hidden = hidden.unsqueeze(0).to(cfg.device_cnn)
            labels = torch.tensor([label], device=cfg.device_cnn)
            logits = classifier(hidden)
            loss = criterion(logits, labels)
            prob = F.softmax(logits, dim=1)[0, 1].item()
            pred = int(torch.argmax(logits, dim=1).item())

            golds.append(label)
            probs.append(prob)
            preds.append(pred)
            losses.append(loss.item())

    metrics = compute_metrics(golds, probs, preds)
    metrics["loss"] = sum(losses) / max(len(losses), 1)
    return metrics


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    ensure_dir(output_dir)
    ensure_dir(output_dir / "checkpoints")
    log_file = Path(cfg.log_file) if cfg.log_file else output_dir / "train_selector.log"
    logger = setup_logger("repro.train_selector", log_file)
    logger.info("Selector training started")
    logger.info("Config | output_dir=%s seed=%s epochs=%s lr=%s", output_dir, cfg.seed, cfg.epochs, cfg.learning_rate)

    if not cfg.disable_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            config=vars(cfg),
        )

    tokenizer, backbone = load_backbone(cfg.model_path, cfg.device_llm)
    layer_count = total_layers(backbone)
    num_layers = selected_num_layers(layer_count, cfg.use_layers, cfg.use_part)
    logger.info("Backbone loaded | total_layers=%s selected_layers=%s", layer_count, num_layers)

    classifier = DeepTextCNN(
        embedding_dim=backbone.config.hidden_size,
        num_classes=2,
        num_layers=num_layers,
    ).to(cfg.device_cnn)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=cfg.learning_rate)
    scaler = torch.amp.GradScaler("cuda")
    criterion = nn.CrossEntropyLoss()

    all_examples = load_selector_examples(
        pos_path=cfg.pos_train_path,
        neg1_path=cfg.neg_dataset1_path,
        neg2_path=cfg.neg_dataset2_path,
        seed=cfg.seed,
    )[: cfg.max_train_examples]
    train_examples, val_examples = stratified_split(all_examples, cfg.train_ratio, cfg.seed)
    logger.info("Dataset prepared | train=%s val=%s total=%s", len(train_examples), len(val_examples), len(all_examples))

    (output_dir / "split_summary.json").write_text(
        json.dumps(
            {
                "train_size": len(train_examples),
                "val_size": len(val_examples),
                "num_layers": num_layers,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    best_auc = float("-inf")
    train_log: List[Dict[str, float]] = []
    global_step = 0

    for epoch in range(cfg.epochs):
        classifier.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, example in enumerate(tqdm(train_examples, desc=f"Epoch {epoch + 1}/{cfg.epochs}"), start=1):
            question = build_question(example)
            answer = str(example["answer"])
            label = int(example["label"])

            hidden = extract_response_hidden_states(
                tokenizer=tokenizer,
                model=backbone,
                question=question,
                answer=answer,
                backbone=cfg.backbone,
                use_layers=cfg.use_layers,
                use_part=cfg.use_part,
            )
            hidden = hidden.unsqueeze(0).to(cfg.device_cnn)
            labels = torch.tensor([label], device=cfg.device_cnn)

            with torch.amp.autocast("cuda"):
                logits = classifier(hidden)
                loss = criterion(logits, labels) / cfg.grad_accum_steps

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * cfg.grad_accum_steps
            global_step += 1

            if step % cfg.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        if len(train_examples) % cfg.grad_accum_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        train_loss = epoch_loss / max(len(train_examples), 1)
        val_metrics = evaluate(val_examples, tokenizer, backbone, classifier, cfg)
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            **val_metrics,
        }
        train_log.append(epoch_metrics)
        logger.info("Epoch done | metrics=%s", epoch_metrics)

        if not cfg.disable_wandb:
            wandb.log(epoch_metrics, step=global_step)

        checkpoint_path = output_dir / "checkpoints" / f"selector_epoch_{epoch + 1}.pth"
        torch.save(classifier.state_dict(), checkpoint_path)

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_path = output_dir / "checkpoints" / "best_selector.pth"
            torch.save(classifier.state_dict(), best_path)
            (output_dir / "best_metrics.json").write_text(
                json.dumps(epoch_metrics, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("New best checkpoint saved | epoch=%s auc=%.6f", epoch + 1, val_metrics["auc"])

    (output_dir / "training_log.json").write_text(
        json.dumps(train_log, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if not cfg.disable_wandb:
        wandb.finish()
    logger.info("Selector training finished")


if __name__ == "__main__":
    main()
