# CPQS Round-1 Remaining Tasks And Dual-GPU Schedule

Last updated: 2026-04-28 20:27 CST

## Current State

- `Selector` training is complete.
- `Candidate scoring` on `alpaca_gpt4_data.json` is complete.
- `Full / Random-K / CNN Top-K / CNN Bottom-K` subsets are complete.
- `Random-K seed 1` LoRA training is complete.
- `Base eval` was started earlier but is not currently running.

## Active Jobs

- `Full seed 1` is running on `GPU0`.
  - dataset: `52,002` records
  - latest confirmed checkpoint: `step 3251 / 9750`
  - last checkpoint write time: `2026-04-28 19:34:22 CST`
  - interpretation: first epoch is complete, about two thirds of training remain
- `CNN Top-K seed 1` started on `GPU1` at `2026-04-28 20:16 CST`.
  - latest ETA from training log at `step 30`: about `84` minutes remaining
  - best current finish estimate: `2026-04-28 21:40-21:50 CST`
- `Base eval` restarted from scratch on `GPU1` at `2026-04-28 20:26 CST`.
  - exact resume was not possible because the previous run had progress logs only and no partial prediction files
  - it is currently sharing `GPU1` with `CNN Top-K seed 1`
- `CNN Bottom-K seed 1` started on `GPU0` at `2026-04-28 20:16 CST`.
  - latest ETA from training log at `step 20`: about `92` minutes remaining
  - best current finish estimate: `2026-04-28 21:50-22:00 CST`
  - note: this job is sharing `GPU0` with `Full seed 1`, so ETA is less stable than `Top-K`

## Remaining Tasks

Minimal round-1 closed loop still needed:

- finish `Full seed 1`
- finish `CNN Top-K seed 1`
- finish `CNN Bottom-K seed 1`
- rerun `Base` evaluation
- run evaluation for:
  - `Full seed 1`
  - `Random-K seed 1`
  - `CNN Top-K seed 1`
  - `CNN Bottom-K seed 1`
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
  - `2026-04-28 20:16` to about `2026-04-28 21:50`:
    - `Full seed 1`
    - `CNN Bottom-K seed 1`
  - about `2026-04-28 21:50` to about `2026-04-29 06:30-07:30`:
    - `Full seed 1` continues alone
  - after `Full seed 1` finishes:
    - run `Full` evaluation

- `GPU1`
  - `2026-04-28 20:16` to about `2026-04-28 21:40-21:50`:
    - `CNN Top-K seed 1`
    - `Base` evaluation in parallel from `2026-04-28 20:26`
  - then recommended order:
    - `Random-K seed 1` evaluation
    - `CNN Top-K seed 1` evaluation
    - `CNN Bottom-K seed 1` evaluation once the adapter is ready

## Expected Finish Times

Best estimate for the minimal round-1 closed loop:

- if the current training ETAs hold and evaluation starts immediately after GPU slots free up:
  - first four evals (`Base / Random-K / Top-K / Bottom-K`) can likely finish by `2026-04-29 06:00-08:00 CST`
  - `Full seed 1` training should finish around `2026-04-29 06:30-07:30 CST`
  - `Full` evaluation plus table aggregation should finish around `2026-04-29 09:00-10:30 CST`

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
