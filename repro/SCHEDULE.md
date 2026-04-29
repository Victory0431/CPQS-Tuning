# CPQS Remaining Tasks And Dual-GPU Schedule

Last updated: 2026-04-29 11:08 CST

## What Is Actually Done

- selector training: complete
- candidate scoring on `alpaca_gpt4_data.json`: complete
- subset build: complete
- `Base` eval: complete
- `Random-K seed 1` LoRA: complete
- `CNN Top-K seed 1` LoRA: complete
- `CNN Bottom-K seed 1` LoRA: complete

## What Is Not Done Yet

- `Full seed 1` LoRA: resumed and still running
- `CNN Top-K seed 1` eval: `gsm8k` done, remaining benchmarks still running
- `Full seed 1` eval: not started
- aggregated result tables: not generated
- second-wave multi-seed expansion: mostly not started

## Current Active Schedule

Current state after the 2026-04-29 morning relaunches and follow-up continuation:

### GPU0

- `Full seed 1` resume training from `checkpoint-3251`
- `CNN Top-K seed 2` LoRA training started at `2026-04-29 11:06 CST`

### GPU1

- `CNN Top-K seed 1` eval from scratch
- `Random-K seed 2` LoRA training started at `2026-04-29 11:00 CST`

Next queued job after `Full seed 1` training finishes:

- `Full seed 1` eval

## Time Expectations

These are current best-effort estimates, not guarantees.

- `Full seed 1` remaining training:
  - about `8-12` hours, depending on whether it gets a dedicated GPU
- `CNN Top-K seed 1` remaining eval:
  - about `2-3` hours from the current `math500` stage
- `Full seed 1` eval:
  - about `2.5-4.0` hours
- `Random-K seed 2` LoRA:
  - about `1.5-2.0` hours
- `CNN Top-K seed 2` LoRA:
  - about `1.5-2.0` hours
- aggregation and doc refresh:
  - about `10-20` minutes

## Best-Case Minimal Closure

If all relaunched jobs run cleanly:

1. `Random-K seed 1` eval is complete
2. `CNN Bottom-K seed 1` eval is complete
3. `CNN Top-K seed 1` is the only seed-1 adapter eval still running
4. `Full seed 1` training is still the dominant path
5. once `Full` finishes, its eval plus aggregation closes the minimal round

## After Minimal Closure

Second-wave runs still pending:

- `Random-K seed 2` is now running
- `CNN Top-K seed 2` is now running
- `Random-K seed 3`
- `CNN Top-K seed 3`
- `CNN Bottom-K seed 2`
- `CNN Bottom-K seed 3`

Those should start only after the minimal seed-1 comparison is complete and verified.
