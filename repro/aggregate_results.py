import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate round-1 benchmark scores.")
    parser.add_argument("--results_root", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    cfg = parse_args()
    results_root = Path(cfg.results_root)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_run_rows: List[Dict[str, Any]] = []
    for path in sorted(results_root.rglob("run_scores.json")):
        with open(path, "r", encoding="utf-8") as handle:
            per_run_rows.extend(json.load(handle))

    with open(output_dir / "per_run_scores.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group", "seed", "dataset", "score", "adapter_path"])
        writer.writeheader()
        writer.writerows(per_run_rows)

    grouped: Dict[Tuple[str, str], List[float]] = {}
    for row in per_run_rows:
        grouped.setdefault((row["group"], row["dataset"]), []).append(float(row["score"]))

    summary_rows: List[Dict[str, Any]] = []
    for (group, dataset), scores in sorted(grouped.items()):
        mean = statistics.mean(scores)
        std = statistics.stdev(scores) if len(scores) > 1 else 0.0
        summary_rows.append(
            {
                "group": group,
                "dataset": dataset,
                "num_runs": len(scores),
                "mean": mean,
                "std": std,
            }
        )

    with open(output_dir / "group_mean_std.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group", "dataset", "num_runs", "mean", "std"])
        writer.writeheader()
        writer.writerows(summary_rows)


if __name__ == "__main__":
    main()
