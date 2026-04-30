#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "Usage: $0 <gpu_id> <train_data_json> <group_name> <lora_output_dir> <eval_output_dir>" >&2
  exit 1
fi

GPU_ID="$1"
TRAIN_DATA="$2"
GROUP_NAME="$3"
LORA_OUTPUT_DIR="$4"
EVAL_OUTPUT_DIR="$5"

LOG_ROOT="/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/logs"
MODEL_PATH="/home/qjh/llm_learning/base_model/qwen3_8B"
BENCHMARKS_ROOT="/home/qjh/llm_learning/CPQS_lab/data/benchmarks"
MMLU_PATH="/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/eval/Open LLM Leaderboard/MMLU/mmlu_data/mmlu_test.json"

CUDA_VISIBLE_DEVICES="$GPU_ID" python -m repro.train_lora \
  --model_path "$MODEL_PATH" \
  --train_data "$TRAIN_DATA" \
  --output_dir "$LORA_OUTPUT_DIR" \
  --group_name "$GROUP_NAME" \
  --seed 1 \
  --backbone qwen \
  --max_length 2048 \
  --epochs 3 \
  --learning_rate 5e-5 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --lora_rank 16 \
  --lora_alpha 8 \
  --log_file "$LOG_ROOT/${GROUP_NAME}_train.log"

CUDA_VISIBLE_DEVICES="$GPU_ID" python -m repro.evaluate_round1 \
  --model_path "$MODEL_PATH" \
  --adapter_path "$LORA_OUTPUT_DIR/final_adapter" \
  --output_dir "$EVAL_OUTPUT_DIR" \
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
  --log_file "$LOG_ROOT/${GROUP_NAME}_eval.log"
