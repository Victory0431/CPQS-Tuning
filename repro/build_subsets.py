import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List

from repro.common import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Top-K, Bottom-K, Random-K, and Full SFT subsets.")
    parser.add_argument("--scored_candidates", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--k", type=int, default=5000)
    parser.add_argument("--random_seeds", type=int, nargs="+", default=[1, 2, 3])
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

    with open(cfg.scored_candidates, "r", encoding="utf-8") as handle:
        records = json.load(handle)

    total = len(records)
    if cfg.k > total:
        raise ValueError(f"K={cfg.k} is larger than total candidate size {total}.")

    top_k = trim_records(records[: cfg.k])
    bottom_k = trim_records(records[-cfg.k :])
    full = trim_records(records)

    write_subset(output_dir / "full.json", full)
    write_subset(output_dir / f"cnn_top_{cfg.k}.json", top_k)
    write_subset(output_dir / f"cnn_bottom_{cfg.k}.json", bottom_k)

    manifest: List[Dict[str, object]] = [
        {"group": "full", "seed": 1, "size": len(full), "path": "full.json"},
        {"group": "cnn_top", "seed": 1, "size": len(top_k), "path": f"cnn_top_{cfg.k}.json"},
        {"group": "cnn_bottom", "seed": 1, "size": len(bottom_k), "path": f"cnn_bottom_{cfg.k}.json"},
    ]

    for seed in cfg.random_seeds:
        rng = random.Random(seed)
        sample = trim_records(rng.sample(records, cfg.k))
        filename = f"random_{cfg.k}_seed_{seed}.json"
        write_subset(output_dir / filename, sample)
        manifest.append(
            {"group": "random", "seed": seed, "size": len(sample), "path": filename}
        )

    with open(output_dir / "subset_manifest.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group", "seed", "size", "path"])
        writer.writeheader()
        writer.writerows(manifest)


if __name__ == "__main__":
    main()
