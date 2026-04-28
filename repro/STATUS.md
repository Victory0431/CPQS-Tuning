# CPQS Reproduction Status

Last updated: 2026-04-28 16:46 CST

## Repository

- Local repo: `/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning`
- Fork remote: `origin -> https://github.com/Victory0431/CPQS-Tuning`
- Upstream remote: `upstream -> https://github.com/renllll/CPQS-Tuning`

## Completed

- Bound local repo to the user fork and preserved upstream.
- Created a clean `repro/` pipeline for round-1 experiments.
- Added scripts for:
  - selector training
  - candidate scoring
  - subset building
  - LoRA SFT
  - automatic evaluation
  - result aggregation
- Added experiment config in `repro/configs/round1_experiment.json`.
- Smoke-tested:
  - selector training pipeline
  - automatic evaluation pipeline
- Installed and verified `wandb` in the `cpqs-tuning` environment.

## W&B Status

- `wandb` package is installed in `cpqs-tuning`.
- API access is working through `/home/qjh/.netrc`.
- Verified account:
  - username: `jiahongqin1`
  - entity: `jiahongqin1-ucas-hias`
- Target project:
  - `https://wandb.ai/jiahongqin1-ucas-hias/CPQS_research`

## Runtime Logging

The `repro/` long-running scripts now use timestamped logs.

- `train_selector.py`
- `train_lora.py`
- `score_candidates.py`
- `evaluate_round1.py`
- `build_subsets.py`
- `aggregate_results.py`

Current logging policy:

- every long-running job writes both to stdout and to a timestamped log file path
- benchmark evaluation saves outputs benchmark-by-benchmark instead of only at the very end
- candidate scoring writes progress logs and a partial JSONL stream during execution
- LoRA SFT now also writes trainer loop progress to file, including:
  - estimated total steps
  - trainer loop start
  - periodic `step/loss/epoch/learning_rate`
  - checkpoint save events

This is now the default expectation for follow-up experiment code.

## vLLM Status

- `vllm` is not currently installed in the `cpqs-tuning` environment
- migrating evaluation to `vLLM` is therefore a code-and-environment change, not just a flag switch

## Round-1 Experimental Scope

Fixed comparison groups:

- `Base`
- `Full SFT`, seed 1 first
- `Random-K`, `K=5000`, seeds 1/2/3
- `CNN Top-K`, `K=5000`, seed 1 first, then seeds 2/3
- `CNN Bottom-K`, `K=5000`, seed 1 first, then seeds 2/3

Benchmarks:

- `GSM8K`
- `MATH-500`
- `ARC-Challenge`
- `MMLU subset`

Scoring:

- script-based only
- no LLM judge

### Current Candidate Choice

For the current round-1 critical comparison, the scored candidate pool is:

- `alpaca_gpt4_data.json`

Reason for choosing it first:

- smaller and cheaper to score than `reasoning-deepseek-r1-146k.json`
- better for a first closed-loop replication where the main goal is:
  - `CNN Top-K vs Random-K`
  - `CNN Bottom-K vs Random-K`
  - `Full vs Base`
- it reduces the confound of switching immediately into a strongly reasoning-specialized candidate pool

Planned follow-up after the first closed loop is stable:

- score and run the same pipeline on `DeepSeek-R1` candidate data as the second general-domain candidate pool

## Paper-Style Hyperparameters Used For Round 1

Execution baseline is aligned to the paper settings summarized in `/home/qjh/llm_learning/CPQS_lab/orders.txt`:

- LoRA
- `bf16`
- `3` epochs
- learning rate `5e-5`
- effective batch size `16`
- max sequence length `2048`
- LoRA rank `16`

Additional implementation note:

- CPQS selector training keeps the original repo logic for hidden-state extraction:
  - same base model as the target model family
  - build `user-only` and `user+assistant` prompts
  - compute the assistant start boundary from the `user-only` prompt length
  - keep only assistant response hidden states
  - feed response hidden states to the CNN selector

## Current Execution Plan

Stage 1:

1. Train the selector on the 15k high/low-quality set.
2. Score `alpaca_gpt4_data.json`.
3. Build `Full / Random-K / CNN Top-K / CNN Bottom-K`.

Stage 2:

1. Run `Base` evaluation.
2. Start LoRA SFT runs.
3. Evaluate every finished adapter immediately.

Stage 3:

1. Aggregate:
   - per-run raw score table
   - group mean/std summary table
2. Focus on:
   - `CNN Top-K vs Random-K`
   - `CNN Bottom-K vs Random-K`
   - `Full vs Base`
   - `CNN Top-K vs Full`

## Current Outputs

Smoke-test artifacts already exist under:

- `repro_outputs/smoke_selector`
- `repro_outputs/smoke_eval_base`

Formal round-1 artifacts will be written under:

- `repro_outputs/selector_round1`
- `repro_outputs/scored_alpaca`
- `repro_outputs/subsets_round1`
- `repro_outputs/lora`
- `repro_outputs/eval`
- `repro_outputs/tables`

## Live Run Status

### Selector

Formal selector training is complete.

- run name: `selector-round1`
- local output dir:
  - `repro_outputs/selector_round1`
- key artifacts:
  - `repro_outputs/selector_round1/checkpoints/best_selector.pth`
  - `repro_outputs/selector_round1/best_metrics.json`
- validation metrics:
  - accuracy: `0.8254`
  - F1: `0.7734`
  - AUC: `0.9291`
  - validation loss: `0.3352`

### Base Eval

`Base` evaluation was started and produced valid progress logs on `GSM8K`, but it is not currently running.

- output dir:
  - `repro_outputs/eval/base`
- log file:
  - `repro_outputs/logs/base_eval.log`
- latest confirmed progress:
  - `GSM8K 216 / 1319`
  - observed throughput around `0.29-0.31 samples/s`
- current configuration from the last launch:
  - `gsm8k batch=4`
  - `math500 batch=4`
  - `arc batch=8`
  - `mmlu batch=8`
- note:
  - no final benchmark score has been produced yet
- next step:
  - relaunch with the newer logging version after a GPU slot becomes available

### Candidate Scoring

Formal candidate scoring is complete.

- candidate dataset:
  - `/home/qjh/llm_learning/CPQS_lab/data/candidate_data/alpaca_gpt4_data.json`
- candidate count:
  - `52,002`
- output dir:
  - `repro_outputs/scored_alpaca`
- log file:
  - `repro_outputs/logs/score_candidates.log`
- current configuration:
  - `batch_size=8`
- observed throughput:
  - roughly `26.7-27.0 samples/s`
- finish time:
  - completed at `2026-04-28 14:46`
- key outputs:
  - `repro_outputs/scored_alpaca/scored_candidates.json`
  - `repro_outputs/scored_alpaca/scored_candidates.csv`
  - `repro_outputs/scored_alpaca/scored_candidates.partial.jsonl`

### Subset Construction

Round-1 subsets are now fully materialized.

- output dir:
  - `repro_outputs/subsets_round1`
- log file:
  - `repro_outputs/logs/build_subsets_round1.log`
- generated subsets:
  - `full.json`
  - `cnn_top_5000.json`
  - `cnn_bottom_5000.json`
  - `random_5000_seed_1.json`
  - `random_5000_seed_2.json`
  - `random_5000_seed_3.json`
- manifest:
  - `repro_outputs/subsets_round1/subset_manifest.csv`

## Active Training

### Full LoRA

Formal `Full seed 1` LoRA training is running on `GPU0`.

- output dir:
  - `repro_outputs/lora/full/seed_1`
- log file:
  - `repro_outputs/logs/lora_full_seed1.log`
- hyperparameters:
  - `bf16`
  - `epochs=3`
  - `lr=5e-5`
  - `max_length=2048`
  - `lora_rank=16`
  - `lora_alpha=8`
  - effective batch size `16`
- current state:
  - log confirms data preparation completed for `52,002` records
  - custom file logging for per-step trainer progress was added after this run had already started, so this specific run still has limited local visibility
  - `wandb` is attached for live tracking

### Random-K LoRA

Formal `Random-K seed 1` LoRA training is now running on `GPU1`.

- output dir:
  - `repro_outputs/lora/random_k5000/seed_1`
- log file:
  - `repro_outputs/logs/lora_random_k5000_seed1.log`
- hyperparameters:
  - `bf16`
  - `epochs=3`
  - `lr=5e-5`
  - `max_length=2048`
  - `lora_rank=16`
  - `lora_alpha=8`
  - effective batch size `16`
- current state:
  - training data prepared for `5,000` records
  - estimated total steps: `936`
  - trainer loop has started successfully
  - first logged train step:
    - `step=10/939`
    - `epoch=0.0320`
    - `loss=2.1247`

## Current Bottlenecks

- both GPUs are currently occupied by LoRA training, so `CNN Top-K`, `CNN Bottom-K`, and `Base eval` must wait for the next free slot
- `Full seed 1` predates the improved per-step file logging, so W&B remains the best live visibility source for that run

## Immediate Next Actions

1. Let `Full seed 1` and `Random-K seed 1` continue in parallel on the two H200 GPUs.
2. Launch `CNN Top-K seed 1` as soon as one GPU frees up.
3. Launch `CNN Bottom-K seed 1` immediately after that.
4. Resume `Base` evaluation with the improved logging path after a GPU slot is available.
5. Evaluate each finished adapter immediately and aggregate:
   - per-run raw scores
   - group mean/std
