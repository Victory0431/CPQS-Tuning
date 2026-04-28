import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from tqdm import tqdm

from repro.common import (
    DeepTextCNN,
    append_jsonl,
    extract_batched_response_hidden_states,
    build_question,
    ensure_dir,
    load_backbone,
    load_instruction_records,
    selected_num_layers,
    setup_logger,
    total_layers,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score candidate SFT data with a trained CPQS selector.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--cnn_checkpoint", required=True)
    parser.add_argument("--predict_data", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--backbone", choices=["qwen", "llama"], default="qwen")
    parser.add_argument("--use_layers", choices=["all", "last"], default="all")
    parser.add_argument("--use_part", choices=["front", "middle", "back", "full"], default="full")
    parser.add_argument("--device_cnn", default="cuda:0")
    parser.add_argument("--device_llm", default="cuda:1")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--progress_log_every", type=int, default=500)
    parser.add_argument("--log_file", default="")
    return parser.parse_args()


def main() -> None:
    cfg = parse_args()
    output_dir = Path(cfg.output_dir)
    ensure_dir(output_dir)
    log_file = Path(cfg.log_file) if cfg.log_file else output_dir / "score_candidates.log"
    logger = setup_logger("repro.score_candidates", log_file)
    logger.info("Candidate scoring started")
    logger.info(
        "Config | model=%s checkpoint=%s predict_data=%s limit=%s batch_size=%s",
        cfg.model_path,
        cfg.cnn_checkpoint,
        cfg.predict_data,
        cfg.limit,
        cfg.batch_size,
    )

    logger.info("Loading backbone")
    tokenizer, llm = load_backbone(cfg.model_path, cfg.device_llm)
    total_hidden_layers = total_layers(llm)
    num_layers = selected_num_layers(total_hidden_layers, cfg.use_layers, cfg.use_part)
    logger.info("Backbone loaded | total_hidden_layers=%s selected_layers=%s", total_hidden_layers, num_layers)

    classifier = DeepTextCNN(
        embedding_dim=llm.config.hidden_size,
        num_classes=2,
        num_layers=num_layers,
    ).to(cfg.device_cnn)
    classifier.load_state_dict(torch.load(cfg.cnn_checkpoint, map_location=cfg.device_cnn))
    classifier.eval()
    logger.info("CNN checkpoint loaded")

    records = load_instruction_records(cfg.predict_data)
    if cfg.limit > 0:
        records = records[: cfg.limit]
    logger.info("Candidate records loaded | count=%s", len(records))

    scored_records: List[Dict[str, object]] = []
    partial_jsonl = output_dir / "scored_candidates.partial.jsonl"
    if partial_jsonl.exists():
        partial_jsonl.unlink()

    total_records = len(records)
    total_batches = (total_records + cfg.batch_size - 1) // cfg.batch_size
    start_time = time.time()

    for batch_index, start in enumerate(range(0, total_records, cfg.batch_size), start=1):
        batch = records[start : start + cfg.batch_size]
        questions = [build_question(record) for record in batch]
        answers = [str(record["output"]) for record in batch]
        hidden = extract_batched_response_hidden_states(
            tokenizer=tokenizer,
            model=llm,
            questions=questions,
            answers=answers,
            backbone=cfg.backbone,
            use_layers=cfg.use_layers,
            use_part=cfg.use_part,
        )
        hidden = hidden.to(cfg.device_cnn)
        with torch.no_grad():
            logits = classifier(hidden)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

        for local_index, record in enumerate(batch):
            answer = str(record["output"])
            scored_record = {
                **record,
                "response_length": len(answer),
                "cpqs_score": float(probabilities[local_index, 1].item()),
                "predicted_class": int(predictions[local_index].item()),
                "probability_negative": float(probabilities[local_index, 0].item()),
                "probability_positive": float(probabilities[local_index, 1].item()),
            }
            scored_records.append(scored_record)
            append_jsonl(partial_jsonl, scored_record)

        processed = min(start + len(batch), total_records)
        if processed % cfg.progress_log_every == 0 or batch_index == total_batches:
            elapsed = time.time() - start_time
            throughput = processed / elapsed if elapsed > 0 else 0.0
            eta_seconds = (total_records - processed) / throughput if throughput > 0 else -1.0
            logger.info(
                "Progress | processed=%s/%s batches=%s/%s throughput=%.2f samples/s eta_minutes=%.2f",
                processed,
                total_records,
                batch_index,
                total_batches,
                throughput,
                eta_seconds / 60 if eta_seconds >= 0 else -1.0,
            )

    scored_records.sort(key=lambda item: item["cpqs_score"], reverse=True)
    logger.info("Sorting completed | total_records=%s", len(scored_records))

    with open(output_dir / "scored_candidates.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "source",
                "response_length",
                "cpqs_score",
                "predicted_class",
                "probability_negative",
                "probability_positive",
                "instruction",
                "input",
                "output",
            ],
        )
        writer.writeheader()
        for record in scored_records:
            writer.writerow(
                {
                    key: record.get(key)
                    for key in [
                        "sample_id",
                        "source",
                        "response_length",
                        "cpqs_score",
                        "predicted_class",
                        "probability_negative",
                        "probability_positive",
                        "instruction",
                        "input",
                        "output",
                    ]
                }
            )

    with open(output_dir / "scored_candidates.json", "w", encoding="utf-8") as handle:
        json.dump(scored_records, handle, ensure_ascii=False, indent=2)
    logger.info("Candidate scoring finished | saved_json=%s saved_csv=%s", output_dir / "scored_candidates.json", output_dir / "scored_candidates.csv")


if __name__ == "__main__":
    main()
