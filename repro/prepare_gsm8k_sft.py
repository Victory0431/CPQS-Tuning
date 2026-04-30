import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_from_disk

from repro.common import ensure_dir, setup_logger


DEFAULT_INSTRUCTION = (
    "Solve the following math word problem carefully. "
    "Show the reasoning clearly, and end the final line with exactly: #### <answer>"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare GSM8K train split for LoRA SFT.")
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--log_file", default="")
    return parser.parse_args()


def main() -> None:
    cfg = parse_args()
    output_path = Path(cfg.output_path)
    ensure_dir(output_path.parent)
    log_path = Path(cfg.log_file) if cfg.log_file else output_path.with_suffix(".log")
    logger = setup_logger("repro.prepare_gsm8k_sft", log_path)
    logger.info("GSM8K SFT preparation started")
    logger.info(
        "Config | dataset_root=%s output_path=%s limit=%s",
        cfg.dataset_root,
        output_path,
        cfg.limit,
    )

    dataset = load_from_disk(str(Path(cfg.dataset_root) / "gsm8k"))["train"]
    rows: List[Dict[str, Any]] = [dataset[index] for index in range(len(dataset))]
    if cfg.limit > 0:
        rows = rows[: cfg.limit]

    records: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        records.append(
            {
                "sample_id": index,
                "instruction": cfg.instruction,
                "input": row["question"],
                "output": row["answer"],
                "source": "gsm8k_train",
            }
        )

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)

    logger.info("GSM8K SFT preparation finished | records=%s | saved=%s", len(records), output_path)


if __name__ == "__main__":
    main()
