#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "Usage: $0 <gpu_id> <adapter_path> <group_name> <output_dir> <log_file>" >&2
  exit 2
fi

GPU_ID="$1"
ADAPTER_PATH="$2"
GROUP_NAME="$3"
OUTPUT_DIR="$4"
LOG_FILE="$5"

MODEL_PATH="/home/qjh/llm_learning/models/Qwen2.5-1.5B-Instruct"
BENCHMARKS_ROOT="/home/qjh/llm_learning/CPQS_lab/data/benchmarks"
MMLU_PATH="/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/eval/Open LLM Leaderboard/MMLU/mmlu_data/mmlu_test.json"

CUDA_VISIBLE_DEVICES="$GPU_ID" python -m repro.evaluate_round1 \
  --model_path "$MODEL_PATH" \
  --adapter_path "$ADAPTER_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --group_name "$GROUP_NAME" \
  --seed 1 \
  --benchmarks_root "$BENCHMARKS_ROOT" \
  --mmlu_path "$MMLU_PATH" \
  --benchmarks gsm8k \
  --enable_thinking false \
  --do_sample false \
  --temperature 0 \
  --batch_size_gsm8k 32 \
  --sample_dump_count 30 \
  --log_file "$LOG_FILE"
