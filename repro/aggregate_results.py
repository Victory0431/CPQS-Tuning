import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple

from repro.common import ensure_dir, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate round-1 benchmark scores.")
    parser.add_argument("--results_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--log_file", default="")
    return parser.parse_args()


def main() -> None:
    cfg = parse_args()
    results_root = Path(cfg.results_root)
    output_dir = Path(cfg.output_dir)
    ensure_dir(output_dir)
    log_file = Path(cfg.log_file) if cfg.log_file else output_dir / "aggregate_results.log"
    logger = setup_logger("repro.aggregate_results", log_file)
    logger.info("Aggregation started | results_root=%s output_dir=%s", results_root, output_dir)

    per_run_rows: List[Dict[str, Any]] = []
    for path in sorted(results_root.rglob("run_scores.json")):
        with open(path, "r", encoding="utf-8") as handle:
            per_run_rows.extend(json.load(handle))
    logger.info("Collected run score files | rows=%s", len(per_run_rows))

    with open(output_dir / "per_run_scores.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group", "seed", "dataset", "score", "adapter_path"])
        writer.writeheader()
        writer.writerows(per_run_rows)
    logger.info("Saved per-run table | path=%s", output_dir / "per_run_scores.csv")

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
    logger.info("Saved summary table | path=%s rows=%s", output_dir / "group_mean_std.csv", len(summary_rows))


if __name__ == "__main__":
    main()
