import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from tqdm import tqdm

from repro.common import (
    DeepTextCNN,
    build_question,
    ensure_dir,
    extract_response_hidden_states,
    load_backbone,
    load_instruction_records,
    selected_num_layers,
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
    return parser.parse_args()


def main() -> None:
    cfg = parse_args()
    output_dir = Path(cfg.output_dir)
    ensure_dir(output_dir)

    tokenizer, llm = load_backbone(cfg.model_path, cfg.device_llm)
    total_hidden_layers = total_layers(llm)
    num_layers = selected_num_layers(total_hidden_layers, cfg.use_layers, cfg.use_part)

    classifier = DeepTextCNN(
        embedding_dim=llm.config.hidden_size,
        num_classes=2,
        num_layers=num_layers,
    ).to(cfg.device_cnn)
    classifier.load_state_dict(torch.load(cfg.cnn_checkpoint, map_location=cfg.device_cnn))
    classifier.eval()

    records = load_instruction_records(cfg.predict_data)
    if cfg.limit > 0:
        records = records[: cfg.limit]

    scored_records: List[Dict[str, object]] = []

    for record in tqdm(records, desc="Scoring candidate data"):
        question = build_question(record)
        answer = str(record["output"])
        hidden = extract_response_hidden_states(
            tokenizer=tokenizer,
            model=llm,
            question=question,
            answer=answer,
            backbone=cfg.backbone,
            use_layers=cfg.use_layers,
            use_part=cfg.use_part,
        )
        hidden = hidden.unsqueeze(0).to(cfg.device_cnn)
        with torch.no_grad():
            logits = classifier(hidden)
            probabilities = F.softmax(logits, dim=1).squeeze(0)

        scored_records.append(
            {
                **record,
                "response_length": len(answer),
                "cpqs_score": float(probabilities[1].item()),
                "predicted_class": int(torch.argmax(probabilities).item()),
                "probability_negative": float(probabilities[0].item()),
                "probability_positive": float(probabilities[1].item()),
            }
        )

    scored_records.sort(key=lambda item: item["cpqs_score"], reverse=True)

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


if __name__ == "__main__":
    main()
