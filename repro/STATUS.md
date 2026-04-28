# CPQS Reproduction Status

Last updated: 2026-04-28 23:16 CST

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

`Base` evaluation is currently running on `GPU1`.

- output dir:
  - `repro_outputs/eval/base`
- log file:
  - `repro_outputs/logs/base_eval.log`
- completed so far:
  - `GSM8K` finished with score `0.359363`
  - predictions saved to `repro_outputs/eval/base/gsm8k_predictions.json`
  - `MATH-500` finished with score `0.122000`
  - predictions saved to `repro_outputs/eval/base/math500_predictions.json`
- current live progress:
  - `ARC-Challenge` finished with score `0.243174`
  - predictions saved to `repro_outputs/eval/base/arc_challenge_predictions.json`
  - current benchmark: `MMLU subset`
- current configuration from the last launch:
  - `gsm8k batch=4`
  - `math500 batch=4`
  - `arc batch=8`
  - `mmlu batch=8`
- note:
  - exact resume was not possible because the first attempt left progress logs only and no partial prediction files
  - the current restarted run is the valid one to track
  - after entering `MMLU subset`, file log updates are sparser because `progress_log_every_batches=20` and the subset is relatively small

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

Formal `Full seed 1` LoRA training is still running on `GPU0`.

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
  - latest confirmed saved checkpoint:
    - `step=3251/9750`
    - `epoch=1.0`
    - checkpoint write time `2026-04-28 19:34:22 CST`
  - no newer epoch checkpoint has appeared yet

### Random-K LoRA

Formal `Random-K seed 1` LoRA training is complete.

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
  - completed at `2026-04-28 18:14:45 CST`
  - final adapter saved successfully

### CNN Top-K LoRA

Formal `CNN Top-K seed 1` LoRA training is complete.

- output dir:
  - `repro_outputs/lora/cnn_top_k5000/seed_1`
- log file:
  - `repro_outputs/logs/lora_cnn_top_k5000_seed1.log`
- hyperparameters:
  - `bf16`
  - `epochs=3`
  - `lr=5e-5`
  - `max_length=2048`
  - `lora_rank=16`
  - `lora_alpha=8`
  - effective batch size `16`
- current state:
  - completed at `2026-04-28 21:56:11 CST`
  - final adapter saved successfully

### CNN Bottom-K LoRA

Formal `CNN Bottom-K seed 1` LoRA training is complete.

- output dir:
  - `repro_outputs/lora/cnn_bottom_k5000/seed_1`
- log file:
  - `repro_outputs/logs/lora_cnn_bottom_k5000_seed1.log`
- hyperparameters:
  - `bf16`
  - `epochs=3`
  - `lr=5e-5`
  - `max_length=2048`
  - `lora_rank=16`
  - `lora_alpha=8`
  - effective batch size `16`
- current state:
  - completed at `2026-04-28 21:53:26 CST`
  - final adapter saved successfully

## Active Evaluation

### Random-K Eval

Formal `Random-K seed 1` evaluation started at `2026-04-28 22:25 CST` on `GPU0`.

- output dir:
  - `repro_outputs/eval/random_k5000_seed1`
- log file:
  - `repro_outputs/logs/eval_random_k5000_seed1.log`
- current state:
  - model and tokenizer loaded
  - currently running `gsm8k`
  - latest confirmed progress: `960 / 1319`

### CNN Top-K Eval

Formal `CNN Top-K seed 1` evaluation started at `2026-04-28 22:25 CST` on `GPU1`.

- output dir:
  - `repro_outputs/eval/cnn_top_k5000_seed1`
- log file:
  - `repro_outputs/logs/eval_cnn_top_k5000_seed1.log`
- current state:
  - model and tokenizer loaded
  - currently running `gsm8k`
  - latest confirmed progress: `480 / 1319`

### CNN Bottom-K Eval

Formal `CNN Bottom-K seed 1` evaluation started at `2026-04-28 22:25 CST` on `GPU0`.

- output dir:
  - `repro_outputs/eval/cnn_bottom_k5000_seed1`
- log file:
  - `repro_outputs/logs/eval_cnn_bottom_k5000_seed1.log`
- current state:
  - model and tokenizer loaded
  - currently running `gsm8k`
  - latest confirmed progress: `720 / 1319`

## Current Bottlenecks

- `Full seed 1` is still the longest remaining training job on the critical path
- both GPUs are now busy with concurrent evaluation jobs, so throughput per run will be lower than single-job mode
- `Full seed 1` predates the improved per-step file logging, so W&B remains the best live visibility source for that run
- `nvidia-smi` snapshots can temporarily hide one eval process view, but current `ps` and log checks confirm:
  - `Base eval` is still alive
  - `Random-K eval` is still alive
  - `CNN Top-K eval` is still alive
  - `CNN Bottom-K eval` is still alive

## Immediate Next Actions

1. Let `Base / Random-K / CNN Top-K / CNN Bottom-K` evaluations continue in parallel.
2. As soon as any of the three adapter evals produces benchmark outputs, keep monitoring throughput and stability.
3. After `Full seed 1` finishes, run `Full` evaluation.
4. Then aggregate:
   - per-run raw scores
   - group mean/std
