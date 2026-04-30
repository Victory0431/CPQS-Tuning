import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt

from repro.common import ensure_dir, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot CPQS score distribution for GSM8K candidate data.")
    parser.add_argument("--scored_candidates", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_k", type=int, default=500)
    parser.add_argument("--log_file", default="")
    return parser.parse_args()


def load_scores(path: Path) -> List[float]:
    records = json.loads(path.read_text(encoding="utf-8"))
    return [float(record["cpqs_score"]) for record in records]


def main() -> None:
    cfg = parse_args()
    output_dir = Path(cfg.output_dir)
    ensure_dir(output_dir)
    log_file = Path(cfg.log_file) if cfg.log_file else output_dir / "plot_scores.log"
    logger = setup_logger("repro.plot_gsm8k_selector_scores", log_file)
    logger.info("Score plotting started")
    logger.info(
        "Config | scored_candidates=%s output_dir=%s top_k=%s",
        cfg.scored_candidates,
        output_dir,
        cfg.top_k,
    )

    scores = load_scores(Path(cfg.scored_candidates))
    total = len(scores)
    top_k = min(cfg.top_k, total)
    top_threshold = scores[top_k - 1]
    bottom_threshold = scores[-top_k]

    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=50, color="#4C78A8", edgecolor="white")
    plt.axvline(top_threshold, color="#F58518", linestyle="--", linewidth=2, label=f"Top-{top_k} threshold")
    plt.axvline(bottom_threshold, color="#E45756", linestyle="--", linewidth=2, label=f"Bottom-{top_k} threshold")
    plt.xlabel("CPQS score")
    plt.ylabel("Count")
    plt.title("GSM8K candidate CPQS score distribution")
    plt.legend()
    hist_path = output_dir / "gsm8k_score_histogram.png"
    plt.tight_layout()
    plt.savefig(hist_path, dpi=200)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(range(1, total + 1), scores, color="#72B7B2", linewidth=1.5)
    plt.axvline(top_k, color="#F58518", linestyle="--", linewidth=2, label=f"Top-{top_k} boundary")
    plt.axvline(total - top_k, color="#E45756", linestyle="--", linewidth=2, label=f"Bottom-{top_k} boundary")
    plt.xlabel("Rank (descending by CPQS)")
    plt.ylabel("CPQS score")
    plt.title("GSM8K candidate CPQS score curve")
    plt.legend()
    curve_path = output_dir / "gsm8k_score_curve.png"
    plt.tight_layout()
    plt.savefig(curve_path, dpi=200)
    plt.close()

    summary = {
        "total_records": total,
        "top_k": top_k,
        "top_threshold": top_threshold,
        "bottom_threshold": bottom_threshold,
        "max_score": max(scores),
        "min_score": min(scores),
        "mean_score": sum(scores) / total,
    }
    (output_dir / "score_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "Score plotting finished | histogram=%s curve=%s summary=%s",
        hist_path,
        curve_path,
        output_dir / "score_summary.json",
    )


if __name__ == "__main__":
    main()
