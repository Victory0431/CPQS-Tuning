# Round-1 CPQS Reproduction

This folder contains a clean reproduction layer for the first critical comparison:

- `Base`
- `Full SFT`
- `Random-K` with `K=5000`, 3 seeds
- `CNN Top-K` with `K=5000`
- `CNN Bottom-K` with `K=5000`

The original repo scripts are kept intact. New code lives here so every step is easier to rerun, log, and compare in your fork.

## Design choices

- The selector still follows the original hidden-state extraction logic:
  - use the same base model
  - build `user-only` and `user+assistant` prompts
  - find the assistant start index from the `user-only` tokenized length
  - keep only assistant response hidden states
  - feed those response hidden states into the CNN selector
- First-round benchmark scope is intentionally narrow:
  - `GSM8K`
  - `MATH-500`
  - `ARC-Challenge`
  - `MMLU subset`
- Scoring is script-based only. No LLM judge is used.

## Files

- `train_selector.py`: train the CNN selector with validation accuracy/F1/AUC
- `score_candidates.py`: score candidate SFT data with the trained selector
- `build_subsets.py`: materialize `Full`, `Top-K`, `Bottom-K`, `Random-K`
- `train_lora.py`: run LoRA SFT with fixed hyperparameters
- `evaluate_round1.py`: automatic scoring on the four selected benchmarks
- `aggregate_results.py`: generate the per-run score table and the group mean/std table
- `configs/round1_experiment.json`: fixed experiment definition

## Expected outputs

Raw run-level table:

- `per_run_scores.csv`

Group-level summary table:

- `group_mean_std.csv`

## Suggested command order

### 1. Train selector

```bash
conda activate cpqs-tuning
cd /home/qjh/llm_learning/CPQS_lab/CPQS-Tuning

python -m repro.train_selector \
  --model_path /home/qjh/llm_learning/base_model/qwen3_8B \
  --pos_train_path /home/qjh/llm_learning/CPQS_lab/data/raw_data/alpaca_gpt4_data.json \
  --neg_dataset1_path /home/qjh/llm_learning/CPQS_lab/data/raw_data/alpaca_gpt4_data_llama.json \
  --neg_dataset2_path /home/qjh/llm_learning/CPQS_lab/data/raw_data/alpaca_gpt4_data_qwen251.json \
  --output_dir /home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/selector_round1 \
  --backbone qwen \
  --use_layers all \
  --use_part full \
  --device_cnn cuda:0 \
  --device_llm cuda:1 \
  --wandb_run_name selector-round1
```

### 2. Score candidate data

```bash
python -m repro.score_candidates \
  --model_path /home/qjh/llm_learning/base_model/qwen3_8B \
  --cnn_checkpoint /home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/selector_round1/checkpoints/best_selector.pth \
  --predict_data /home/qjh/llm_learning/CPQS_lab/data/candidate_data/alpaca_gpt4_data.json \
  --output_dir /home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/scored_alpaca \
  --backbone qwen \
  --use_layers all \
  --use_part full \
  --device_cnn cuda:0 \
  --device_llm cuda:1
```

### 3. Build subsets

```bash
python -m repro.build_subsets \
  --scored_candidates /home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/scored_alpaca/scored_candidates.json \
  --output_dir /home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/subsets_round1 \
  --k 5000 \
  --random_seeds 1 2 3
```

### 4. Train LoRA runs

Full:

```bash
python -m repro.train_lora \
  --model_path /home/qjh/llm_learning/base_model/qwen3_8B \
  --train_data /home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/subsets_round1/full.json \
  --output_dir /home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/lora/full/seed_1 \
  --group_name full \
  --seed 1
```

Random-K seed 1 example:

```bash
python -m repro.train_lora \
  --model_path /home/qjh/llm_learning/base_model/qwen3_8B \
  --train_data /home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/subsets_round1/random_5000_seed_1.json \
  --output_dir /home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/lora/random_k/seed_1 \
  --group_name random_k \
  --seed 1
```

CNN Top-K seed 1 example:

```bash
python -m repro.train_lora \
  --model_path /home/qjh/llm_learning/base_model/qwen3_8B \
  --train_data /home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/subsets_round1/cnn_top_5000.json \
  --output_dir /home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/lora/cnn_top_k/seed_1 \
  --group_name cnn_top_k \
  --seed 1
```

CNN Bottom-K seed 1 example:

```bash
python -m repro.train_lora \
  --model_path /home/qjh/llm_learning/base_model/qwen3_8B \
  --train_data /home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/subsets_round1/cnn_bottom_5000.json \
  --output_dir /home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/lora/cnn_bottom_k/seed_1 \
  --group_name cnn_bottom_k \
  --seed 1
```

### 5. Evaluate each run

Base model:

```bash
python -m repro.evaluate_round1 \
  --model_path /home/qjh/llm_learning/base_model/qwen3_8B \
  --output_dir /home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/eval/base \
  --group_name base \
  --seed 1 \
  --benchmarks_root /home/qjh/llm_learning/CPQS_lab/data/benchmarks \
  --mmlu_path "/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/eval/Open LLM Leaderboard/MMLU/mmlu_data/mmlu_test.json"
```

LoRA adapter example:

```bash
python -m repro.evaluate_round1 \
  --model_path /home/qjh/llm_learning/base_model/qwen3_8B \
  --adapter_path /home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/lora/random_k/seed_1/final_adapter \
  --output_dir /home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/eval/random_k_seed_1 \
  --group_name random_k \
  --seed 1 \
  --benchmarks_root /home/qjh/llm_learning/CPQS_lab/data/benchmarks \
  --mmlu_path "/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/eval/Open LLM Leaderboard/MMLU/mmlu_data/mmlu_test.json"
```

### 6. Aggregate results

```bash
python -m repro.aggregate_results \
  --results_root /home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/eval \
  --output_dir /home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/tables
```

## Notes

- `MMLU subset` is built deterministically inside `evaluate_round1.py`:
  - all subjects are included
  - each subject contributes `8` examples by default
  - subset seed is fixed to `42`
- The `MATH-500` scorer is automatic and symbolic where possible. It is stricter than an LLM judge and may undercount edge cases with unusual formatting. That is acceptable for round 1 because all compared runs use the same scorer.
