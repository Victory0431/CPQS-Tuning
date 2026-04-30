import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List

from repro.common import ensure_dir, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GSM8K Top/Bottom/Random subsets from scored candidates.")
    parser.add_argument("--scored_candidates", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--k", type=int, default=500)
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--log_file", default="")
    return parser.parse_args()


def trim_records(records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return [
        {
            "sample_id": record["sample_id"],
            "instruction": record["instruction"],
            "input": record["input"],
            "output": record["output"],
            "source": record.get("source", ""),
            "cpqs_score": record.get("cpqs_score"),
            "response_length": record.get("response_length"),
        }
        for record in records
    ]


def write_subset(path: Path, records: List[Dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)


def main() -> None:
    cfg = parse_args()
    output_dir = Path(cfg.output_dir)
    ensure_dir(output_dir)
    log_file = Path(cfg.log_file) if cfg.log_file else output_dir / "build_gsm8k_subsets.log"
    logger = setup_logger("repro.build_gsm8k_subsets", log_file)
    logger.info(
        "GSM8K subset building started | scored_candidates=%s k=%s random_seed=%s",
        cfg.scored_candidates,
        cfg.k,
        cfg.random_seed,
    )

    records = json.loads(Path(cfg.scored_candidates).read_text(encoding="utf-8"))
    total = len(records)
    if cfg.k > total:
        raise ValueError(f"K={cfg.k} exceeds total scored records {total}")

    top_k = trim_records(records[: cfg.k])
    bottom_k = trim_records(records[-cfg.k :])
    rng = random.Random(cfg.random_seed)
    random_k = trim_records(rng.sample(records, cfg.k))

    top_path = output_dir / f"cnn_top_{cfg.k}.json"
    bottom_path = output_dir / f"cnn_bottom_{cfg.k}.json"
    random_path = output_dir / f"random_{cfg.k}_seed_{cfg.random_seed}.json"
    write_subset(top_path, top_k)
    write_subset(bottom_path, bottom_k)
    write_subset(random_path, random_k)

    manifest = [
        {"group": "cnn_top", "seed": 1, "size": len(top_k), "path": top_path.name},
        {"group": "cnn_bottom", "seed": 1, "size": len(bottom_k), "path": bottom_path.name},
        {"group": "random", "seed": cfg.random_seed, "size": len(random_k), "path": random_path.name},
    ]
    with open(output_dir / "subset_manifest.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group", "seed", "size", "path"])
        writer.writeheader()
        writer.writerows(manifest)

    logger.info("GSM8K subset building finished | top=%s bottom=%s random=%s", top_path, bottom_path, random_path)


if __name__ == "__main__":
    main()
