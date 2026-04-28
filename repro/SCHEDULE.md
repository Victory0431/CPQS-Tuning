# CPQS Round-1 Remaining Tasks And Dual-GPU Schedule

Last updated: 2026-04-28 23:28 CST

## Current State

- `Selector` training is complete.
- `Candidate scoring` on `alpaca_gpt4_data.json` is complete.
- `Full / Random-K / CNN Top-K / CNN Bottom-K` subsets are complete.
- `Random-K seed 1` LoRA training is complete.
- `CNN Top-K seed 1` LoRA training is complete.
- `CNN Bottom-K seed 1` LoRA training is complete.
- `Base eval` has finished `GSM8K`, `MATH-500`, and `ARC-Challenge`, and is currently running `MMLU subset`.
- `Random-K seed 1` eval has started.
- `CNN Top-K seed 1` eval has started.
- `CNN Bottom-K seed 1` eval has started.
- the original `evaluate_round1.py` had a bug at `mmlu_subset` entry and has now been patched

## Active Jobs

- `Full seed 1` is running on `GPU0`.
  - dataset: `52,002` records
  - latest confirmed checkpoint: `step 3251 / 9750`
  - last checkpoint write time: `2026-04-28 19:34:22 CST`
  - interpretation: first epoch is complete, about two thirds of training remain
- `CNN Top-K seed 1` finished at `2026-04-28 21:56:11 CST`.
- `CNN Bottom-K seed 1` finished at `2026-04-28 21:53:26 CST`.
- `Base eval` restarted from scratch on `GPU1` at `2026-04-28 20:26 CST`.
  - exact resume was not possible because the previous run had progress logs only and no partial prediction files
  - `GSM8K` is complete with score `0.359363`
  - `MATH-500` is complete with score `0.122000`
  - `ARC-Challenge` is complete with score `0.243174`
  - the original process exited at `MMLU subset` because of the `evaluate_mmlu_subset()` argument mismatch bug
  - `Base` `mmlu_subset` has been relaunched on the patched script at `2026-04-28 23:27 CST`
- `Random-K seed 1` eval started on `GPU0` at `2026-04-28 22:25 CST`.
  - latest confirmed progress: `960 / 1319` on `gsm8k`
- `CNN Top-K seed 1` eval started on `GPU1` at `2026-04-28 22:25 CST`.
  - latest confirmed progress: `480 / 1319` on `gsm8k`
- `CNN Bottom-K seed 1` eval started on `GPU0` at `2026-04-28 22:25 CST`.
  - latest confirmed progress: `720 / 1319` on `gsm8k`

## Remaining Tasks

Minimal round-1 closed loop still needed:

- finish `Full seed 1`
- finish `Base` evaluation
- finish evaluation for:
  - `Random-K seed 1`
  - `CNN Top-K seed 1`
  - `CNN Bottom-K seed 1`
  - `Full seed 1`
- aggregate:
  - per-run raw score table
  - group mean/std summary table

Additional round-1 expansion after the first closed loop:

- `Random-K seed 2`
- `Random-K seed 3`
- `CNN Top-K seed 2`
- `CNN Top-K seed 3`
- `CNN Bottom-K seed 2`
- `CNN Bottom-K seed 3`
- evaluate all additional runs
- refresh mean/std tables with all seeds

## Time Assumptions

- `Top-K / Bottom-K / Random-K` LoRA on `5,000` samples:
  - about `1.4-1.7` hours when given a full GPU
  - about `1.5-2.0` hours when sharing or under contention
- `Full` LoRA on `52,002` samples:
  - first epoch took about `5` hours
  - remaining training is best estimated at about `10-11` more hours from `2026-04-28 20:22 CST`
- one full evaluation run over:
  - `GSM8K (1319)`
  - `MATH-500 (500)`
  - `ARC-Challenge (1172)`
  - `MMLU subset (456)`
  - expected wall time: about `2.0-3.0` hours per run with the current HF generation path
- result aggregation:
  - about `10-20` minutes

## Dual-GPU Schedule

Best-effort schedule from the current state:

- `GPU0`
  - now until `Full seed 1` finishes:
    - `Full seed 1`
    - `Random-K seed 1` evaluation
    - `CNN Bottom-K seed 1` evaluation
  - after `Full seed 1` finishes:
    - `Full` evaluation

- `GPU1`
  - now until the patched `Base` `mmlu_subset` recovery finishes:
    - `Base` `mmlu_subset` recovery
    - `CNN Top-K seed 1` evaluation

## Expected Finish Times

Best estimate for the minimal round-1 closed loop:

- if the current training ETAs hold and evaluation starts immediately after GPU slots free up:
  - `Base eval` now only needs to finish the patched `MMLU subset` recovery, so it should close sooner than the prior estimate
  - `Random-K / Top-K / Bottom-K` evals are already in flight, so the minimal loop may close earlier than the previous sequential estimate if concurrent throughput stays stable
  - those three adapter evals were launched before the `mmlu_subset` fix, so each may still need a short `mmlu_subset`-only recovery run
  - `Full seed 1` training remains the dominant unknown and likely still extends into `2026-04-29`
  - `Full` evaluation plus table aggregation remain the last step of the minimal closed loop

Best estimate for the expanded three-seed round-1 package:

- training for the six extra seed runs likely needs about `5-7` more GPU-hours after the minimal loop
- evaluation for those six extra runs likely needs about `12-18` more wall-clock hours unless evaluation throughput is improved
- full three-seed package is therefore more realistically a `2026-04-29` to `2026-04-30` task, not a same-night finish

## Logging Notes

- new LoRA runs now write timestamped file logs with:
  - `step`
  - `epoch`
  - `loss`
  - `grad_norm`
  - `learning_rate`
  - `elapsed_minutes`
  - `avg_step_seconds`
  - `eta_minutes`
- active logs:
  - `repro_outputs/logs/lora_full_seed1.log`
  - `repro_outputs/logs/lora_cnn_top_k5000_seed1.log`
  - `repro_outputs/logs/lora_cnn_bottom_k5000_seed1.log`
  - `repro_outputs/logs/base_eval.log`
  - `repro_outputs/logs/base_eval_resume_mmlu.log`
  - `repro_outputs/logs/eval_random_k5000_seed1.log`
  - `repro_outputs/logs/eval_cnn_top_k5000_seed1.log`
  - `repro_outputs/logs/eval_cnn_bottom_k5000_seed1.log`
