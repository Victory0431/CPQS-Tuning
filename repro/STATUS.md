# CPQS Reproduction Status

Last updated: 2026-04-28

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

## Next Immediate Action

Start formal selector training with W&B logging enabled, then continue into candidate scoring and subset construction.
