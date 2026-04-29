import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


TARGET_METRICS = {
    "mmlu": "acc,none",
    "arc_challenge": "acc_norm,none",
    "hellaswag": "acc_norm,none",
    "truthfulqa_mc1": "acc,none",
}

GROUP_LABELS = {
    "base_lm_eval_vllm": "Base",
    "full": "Full",
    "random_k5000": "Random-K",
    "cnn_top_k5000": "CNN Top-K",
    "cnn_bottom_k5000": "CNN Bottom-K",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate lm-eval + vLLM results for Alpaca-GPT4 evaluation."
    )
    parser.add_argument("--results_root", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def infer_group_seed(eval_dir: Path) -> tuple[str, int]:
    name = eval_dir.name
    if name == "base_lm_eval_vllm":
        return "Base", 1
    parts = name.split("_seed")
    if len(parts) == 2 and parts[1].isdigit():
        return GROUP_LABELS.get(parts[0], parts[0]), int(parts[1])
    return GROUP_LABELS.get(name, name), 1


def find_result_jsons(results_root: Path) -> list[Path]:
    result_jsons = []
    for path in sorted(results_root.glob("*/__*/*results_*.json")):
        if "smoke" in str(path):
            continue
        result_jsons.append(path)
    return result_jsons


def load_score_rows(result_json: Path) -> list[dict]:
    eval_dir = result_json.parents[1]
    group, seed = infer_group_seed(eval_dir)
    payload = json.loads(result_json.read_text())
    rows = []
    for dataset, metric_key in TARGET_METRICS.items():
        value = payload["results"][dataset][metric_key]
        rows.append(
            {
                "group": group,
                "seed": seed,
                "dataset": dataset,
                "score": value,
                "result_json": str(result_json),
            }
        )
    return rows


def mean_std(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)

    per_run_rows = []
    for result_json in find_result_jsons(results_root):
        per_run_rows.extend(load_score_rows(result_json))
    per_run_rows.sort(key=lambda row: (row["group"], row["seed"], row["dataset"]))

    grouped = defaultdict(list)
    for row in per_run_rows:
        grouped[(row["group"], row["dataset"])].append(float(row["score"]))

    summary_rows = []
    for (group, dataset), values in sorted(grouped.items()):
        mean, std = mean_std(values)
        summary_rows.append(
            {
                "group": group,
                "dataset": dataset,
                "num_runs": len(values),
                "mean": mean,
                "std": std,
            }
        )

    write_csv(
        output_dir / "alpaca_auto_per_run_scores.csv",
        ["group", "seed", "dataset", "score", "result_json"],
        per_run_rows,
    )
    write_csv(
        output_dir / "alpaca_auto_group_mean_std.csv",
        ["group", "dataset", "num_runs", "mean", "std"],
        summary_rows,
    )


if __name__ == "__main__":
    main()
