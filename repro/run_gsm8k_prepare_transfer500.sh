#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 <scored_candidates_json> <plot_output_dir> <subset_output_dir> <k>" >&2
  exit 1
fi

SCORED_CANDIDATES="$1"
PLOT_OUTPUT_DIR="$2"
SUBSET_OUTPUT_DIR="$3"
K="$4"

python -m repro.plot_gsm8k_selector_scores \
  --scored_candidates "$SCORED_CANDIDATES" \
  --output_dir "$PLOT_OUTPUT_DIR" \
  --top_k "$K" \
  --log_file /home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/logs/gsm8k_plot_transfer500.log

python -m repro.build_gsm8k_subsets \
  --scored_candidates "$SCORED_CANDIDATES" \
  --output_dir "$SUBSET_OUTPUT_DIR" \
  --k "$K" \
  --random_seed 1 \
  --log_file /home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/logs/gsm8k_build_subsets_k${K}.log
