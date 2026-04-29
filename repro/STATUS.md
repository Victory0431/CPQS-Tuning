# CPQS Reproduction Status

Last updated: 2026-04-29 09:46 CST

## Repository And Environment

- Local repo: `/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning`
- Fork remote: `origin -> https://github.com/Victory0431/CPQS-Tuning`
- Upstream remote: `upstream -> https://github.com/renllll/CPQS-Tuning`
- Conda env: `cpqs-tuning`
- Base model: `/home/qjh/llm_learning/base_model/qwen3_8B`
- W&B project: `https://wandb.ai/jiahongqin1-ucas-hias/CPQS_research`

## Scope Of This Round

This round is the user-approved minimal closed loop, not the full paper evaluation package.

- candidate pool: `alpaca_gpt4_data.json`
- groups:
  - `Base`
  - `Full`
  - `Random-K (K=5000)`
  - `CNN Top-K (K=5000)`
  - `CNN Bottom-K (K=5000)`
- benchmarks:
  - `GSM8K`
  - `MATH-500`
  - `ARC-Challenge`
  - `MMLU subset`
- scoring:
  - script-based only
  - no LLM judge
  - no AlpacaEval

## Paper-Alignment Notes

- LoRA SFT hyperparameters are fixed across groups:
  - same base model
  - `3` epochs
  - learning rate `5e-5`
  - max length `2048`
  - LoRA rank `16`
  - same prompt format
  - only the training subset changes
- Selector hidden-state extraction follows the original repo logic:
  - build `user-only` and `user+assistant` prompts
  - compute the assistant start boundary from the `user-only` prompt length
  - keep only assistant response hidden states
  - feed those response hidden states into the CNN selector
- Evaluation is intentionally narrower than the paper's Alpaca evaluation protocol.
  - current results are valid for the internal `Base / Full / Random / Top / Bottom` comparison
  - current results are not yet a strict reproduction of the paper's Alpaca benchmark section

## Completed Pipeline Stages

### Selector Training

Completed under `repro_outputs/selector_round1`.

- best checkpoint:
  - `repro_outputs/selector_round1/checkpoints/best_selector.pth`
- best metrics:
  - accuracy: `0.8254`
  - F1: `0.7734`
  - AUC: `0.9291`
  - validation loss: `0.3352`

### Candidate Scoring

Completed under `repro_outputs/scored_alpaca`.

- main outputs:
  - `repro_outputs/scored_alpaca/scored_candidates.json`
  - `repro_outputs/scored_alpaca/scored_candidates.csv`

### Subset Building

Completed under `repro_outputs/subsets_round1`.

- `full.json`
- `random_5000_seed_1.json`
- `random_5000_seed_2.json`
- `random_5000_seed_3.json`
- `cnn_top_5000.json`
- `cnn_bottom_5000.json`
- `subset_manifest.csv`

### LoRA Training

Completed:

- `Random-K seed 1`
  - adapter: `repro_outputs/lora/random_k5000/seed_1/final_adapter`
- `CNN Top-K seed 1`
  - adapter: `repro_outputs/lora/cnn_top_k5000/seed_1/final_adapter`
- `CNN Bottom-K seed 1`
  - adapter: `repro_outputs/lora/cnn_bottom_k5000/seed_1/final_adapter`

Incomplete:

- `Full seed 1`
  - latest checkpoint: `repro_outputs/lora/full/seed_1/checkpoint-3251`
  - latest confirmed trainer state:
    - global step `3251 / 9750`
    - epoch `1.0 / 3.0`
  - no `final_adapter` exists yet
  - this run must be resumed, not treated as complete

## Evaluation Status

### Base

Completed under `repro_outputs/eval/base`.

- `gsm8k = 0.3593631539044731`
- `math500 = 0.122`
- `arc_challenge = 0.2431740614334471`
- `mmlu_subset = 0.2543859649122807`

Files:

- `repro_outputs/eval/base/run_scores.json`
- `repro_outputs/eval/base/run_scores.csv`
- per-benchmark prediction JSONs are present

### Random-K Seed 1

Partially completed under `repro_outputs/eval/random_k5000_seed1`.

- finished:
  - `gsm8k = 0.8453373768006065`
- missing:
  - `math500`
  - `arc_challenge`
  - `mmlu_subset`

Latest log evidence:

- `2026-04-28 23:53:11 CST`
- progress reached `math500 160 / 500`
- process later stopped before writing additional benchmark outputs

### CNN Bottom-K Seed 1

Partially completed under `repro_outputs/eval/cnn_bottom_k5000_seed1`.

- finished:
  - `gsm8k = 0.8544351781652767`
- missing:
  - `math500`
  - `arc_challenge`
  - `mmlu_subset`

Latest log evidence:

- `2026-04-28 23:57:57 CST`
- progress reached `math500 80 / 500`
- process later stopped before writing additional benchmark outputs

### CNN Top-K Seed 1

Interrupted before any benchmark result was saved.

- output dir exists:
  - `repro_outputs/eval/cnn_top_k5000_seed1`
- no `run_scores.json` exists
- latest log evidence:
  - `2026-04-28 23:56:40 CST`
  - progress reached `gsm8k 1040 / 1319`

### Full Seed 1

Not started because `Full seed 1` training is not finished.

## Evaluation Script Bug Already Fixed

The original `evaluate_round1.py` had a bug at `mmlu_subset` entry.

- root cause:
  - `evaluate_mmlu_subset()` did not accept `logger` and `progress_log_every_batches`
  - caller passed both arguments
- consequence:
  - runs launched before the fix could crash on entering `mmlu_subset`
- remediation already committed:
  - fixed function signature
  - added `--benchmarks` so a run can resume benchmark-by-benchmark
  - existing `run_scores.json` is preserved for unselected benchmarks

This fix was used successfully to recover `Base -> mmlu_subset`.

## Logging Status

Long-running scripts now write timestamped logs.

- selector:
  - `repro_outputs/logs/selector_round1.log`
- scoring:
  - `repro_outputs/logs/score_candidates.log`
- subset build:
  - `repro_outputs/logs/build_subsets_round1.log`
- LoRA:
  - `repro_outputs/logs/lora_full_seed1.log`
  - `repro_outputs/logs/lora_random_k5000_seed1.log`
  - `repro_outputs/logs/lora_cnn_top_k5000_seed1.log`
  - `repro_outputs/logs/lora_cnn_bottom_k5000_seed1.log`
- eval:
  - `repro_outputs/logs/base_eval.log`
  - `repro_outputs/logs/base_eval_resume_mmlu.log`
  - `repro_outputs/logs/eval_random_k5000_seed1.log`
  - `repro_outputs/logs/eval_cnn_top_k5000_seed1.log`
  - `repro_outputs/logs/eval_cnn_bottom_k5000_seed1.log`

## Current Machine State

Machine state changed after the idle check.

### Idle Snapshot

As of `2026-04-29 09:36 CST`:

- `nvidia-smi` showed both H200 GPUs idle
- no CPQS training or evaluation process was running

That confirmed the previous long jobs had not completed end-to-end and required explicit relaunch.

### Active Snapshot

As of `2026-04-29 09:44 CST`, the following jobs were relaunched and are active:

- `Full seed 1` resume training
  - GPU: `GPU0`
  - PID: `565634`
  - session: `tmux cpqs_full_seed1_resume`
  - checkpoint: `checkpoint-3251`
- `Random-K seed 1` eval resume
  - GPU: `GPU0`
  - PID: `565640`
  - session: `tmux cpqs_eval_random_seed1_resume`
  - benchmarks: `math500, arc_challenge, mmlu_subset`
- `CNN Top-K seed 1` eval restart
  - GPU: `GPU1`
  - PID: `565643`
  - session: `tmux cpqs_eval_top_seed1`
  - benchmarks: `gsm8k, math500, arc_challenge, mmlu_subset`
- `CNN Bottom-K seed 1` eval resume
  - GPU: `GPU1`
  - PID: `565646`
  - session: `tmux cpqs_eval_bottom_seed1_resume`
  - benchmarks: `math500, arc_challenge, mmlu_subset`

Current eval batch sizes for all relaunched eval jobs:

- `gsm8k = 8`
- `math500 = 8`
- `arc_challenge = 12`
- `mmlu_subset = 12`

## Remaining Work For The Minimal Closed Loop

1. Resume `Full seed 1` from `checkpoint-3251`.
2. Finish `CNN Top-K seed 1` evaluation from scratch.
3. Resume `Random-K seed 1` evaluation for:
   - `math500`
   - `arc_challenge`
   - `mmlu_subset`
4. Resume `CNN Bottom-K seed 1` evaluation for:
   - `math500`
   - `arc_challenge`
   - `mmlu_subset`
5. Run `Full seed 1` evaluation after training finishes.
6. Generate result tables:
   - per-run raw score table
   - group mean/std summary table

## Next Expansion After Minimal Closure

Not started yet:

- `Random-K seed 2`
- `Random-K seed 3`
- `CNN Top-K seed 2`
- `CNN Top-K seed 3`
- `CNN Bottom-K seed 2`
- `CNN Bottom-K seed 3`
- eval for all additional seeds

## Latest Code Update

`repro/train_lora.py` now supports checkpoint resume.

- new args:
  - `--resume_from_checkpoint`
  - `--auto_resume_latest_checkpoint`

This is required to continue `Full seed 1` without discarding the already finished first epoch.
