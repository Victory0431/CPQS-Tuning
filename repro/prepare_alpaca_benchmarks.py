import argparse
from pathlib import Path

from datasets import DatasetDict, load_dataset

from repro.common import ensure_dir, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare automatic Alpaca-domain benchmarks on local disk.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--log_file", default="")
    return parser.parse_args()


def save_if_missing(dataset_dict: DatasetDict, output_path: Path, logger) -> None:
    if output_path.exists():
        logger.info("Dataset already prepared | %s", output_path)
        return
    ensure_dir(output_path.parent)
    dataset_dict.save_to_disk(str(output_path))
    logger.info("Dataset saved | %s", output_path)


def main() -> None:
    cfg = parse_args()
    output_root = Path(cfg.output_root)
    ensure_dir(output_root)
    log_file = Path(cfg.log_file) if cfg.log_file else output_root / "prepare_alpaca_benchmarks.log"
    logger = setup_logger("repro.prepare_alpaca_benchmarks", log_file)
    logger.info("Benchmark preparation started | output_root=%s", output_root)

    logger.info("Loading HellaSwag from Hugging Face datasets")
    hellaswag = load_dataset("hellaswag")
    logger.info(
        "Loaded HellaSwag | train=%s validation=%s test=%s",
        len(hellaswag["train"]),
        len(hellaswag["validation"]),
        len(hellaswag["test"]),
    )
    save_if_missing(DatasetDict({"validation": hellaswag["validation"], "test": hellaswag["test"]}), output_root / "hellaswag", logger)

    logger.info("Loading TruthfulQA multiple-choice from Hugging Face datasets")
    truthfulqa_mc = load_dataset("truthful_qa", "multiple_choice")
    logger.info("Loaded TruthfulQA MC | validation=%s", len(truthfulqa_mc["validation"]))
    save_if_missing(DatasetDict({"validation": truthfulqa_mc["validation"]}), output_root / "truthfulqa_mc", logger)

    logger.info("Benchmark preparation finished")


if __name__ == "__main__":
    main()
