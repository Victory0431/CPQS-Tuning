# CPQS Remaining Tasks And Dual-GPU Schedule

Last updated: 2026-04-29 09:46 CST

## What Is Actually Done

- selector training: complete
- candidate scoring on `alpaca_gpt4_data.json`: complete
- subset build: complete
- `Base` eval: complete
- `Random-K seed 1` LoRA: complete
- `CNN Top-K seed 1` LoRA: complete
- `CNN Bottom-K seed 1` LoRA: complete

## What Is Not Done Yet

- `Full seed 1` LoRA: interrupted at `checkpoint-3251`
- `Random-K seed 1` eval: only `gsm8k` finished
- `CNN Bottom-K seed 1` eval: only `gsm8k` finished
- `CNN Top-K seed 1` eval: interrupted during `gsm8k`
- `Full seed 1` eval: not started
- aggregated result tables: not generated

## Current Active Schedule

Relaunched at `2026-04-29 09:43 CST`.

### GPU0

- `Full seed 1` resume training from `checkpoint-3251`
- `Random-K seed 1` eval resume on:
  - `math500`
  - `arc_challenge`
  - `mmlu_subset`

### GPU1

- `CNN Top-K seed 1` eval from scratch
- `CNN Bottom-K seed 1` eval resume on:
  - `math500`
  - `arc_challenge`
  - `mmlu_subset`

Next queued job after `Full seed 1` training finishes:

- `Full seed 1` eval

## Resume Strategy

### Full Training

Use checkpoint resume, not restart from zero.

- script now supports:
  - `--resume_from_checkpoint`
  - `--auto_resume_latest_checkpoint`

### Random-K Eval

Resume only missing benchmarks:

- `math500`
- `arc_challenge`
- `mmlu_subset`

Existing `gsm8k` score in `run_scores.json` should be preserved.

### CNN Bottom-K Eval

Resume only missing benchmarks:

- `math500`
- `arc_challenge`
- `mmlu_subset`

Existing `gsm8k` score in `run_scores.json` should be preserved.

### CNN Top-K Eval

Restart the full evaluation from scratch.

Reason:

- no benchmark-level score file was saved before interruption
- there is no partial `gsm8k` prediction file to resume from

## Time Expectations

These are current best-effort estimates, not guarantees.

- `Full seed 1` remaining training:
  - about `8-12` hours, depending on whether it gets a dedicated GPU
- `Random-K seed 1` remaining eval:
  - about `1.5-2.5` hours
- `CNN Bottom-K seed 1` remaining eval:
  - about `1.5-2.5` hours
- `CNN Top-K seed 1` full eval restart:
  - about `2.5-4.0` hours
- `Full seed 1` eval:
  - about `2.5-4.0` hours
- aggregation and doc refresh:
  - about `10-20` minutes

## Best-Case Minimal Closure

If all relaunched jobs run cleanly:

1. the three seed-1 adapter evals can finish today
2. `Full seed 1` training is still the dominant path
3. once `Full` finishes, its eval plus aggregation closes the minimal round

## After Minimal Closure

Second-wave runs still pending:

- `Random-K seed 2`
- `Random-K seed 3`
- `CNN Top-K seed 2`
- `CNN Top-K seed 3`
- `CNN Bottom-K seed 2`
- `CNN Bottom-K seed 3`

Those should start only after the minimal seed-1 comparison is complete and verified.
