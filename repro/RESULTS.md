# CPQS 当前结果

最后更新：2026-04-29 22:03 CST

## 当前正式口径

- 候选池：`Alpaca-GPT4`
- 评测框架：`lm-evaluation-harness + vLLM`
- 基础模型：`Qwen3-8B`
- 推理设置：
  - `enable_thinking=false`
  - `temperature=0`
  - `bf16`
  - `max_model_len=2048`

## 当前有效结果

| group | seed | MMLU | ARC-Challenge acc_norm | HellaSwag acc_norm | TruthfulQA MC1 | 状态 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Base | 1 | 0.4964 | 0.4224 | 0.5856 | 0.3599 | 已完成 |
| Full | 1 | 0.7223 | 0.5017 | 0.6988 | 0.4259 | 已完成 |
| CNN Top-K | 1 | 0.6887 | 0.5000 | 0.6917 | 0.3880 | 已完成 |

## 结果文件

- 原始结果目录：
  - [base_lm_eval_vllm](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/eval/base_lm_eval_vllm)
- 原始结果 JSON：
  - [results_2026-04-29T21-17-31.610048.json](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/eval/base_lm_eval_vllm/__home__qjh__llm_learning__base_model__qwen3_8B/results_2026-04-29T21-17-31.610048.json)
- 汇总表：
  - [alpaca_auto_per_run_scores.csv](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/tables/alpaca_auto_per_run_scores.csv)
  - [alpaca_auto_group_mean_std.csv](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/tables/alpaca_auto_group_mean_std.csv)

## 说明

- 当前已经完成的正式结果有：
  - `Base`
  - `Full seed 1`
  - `CNN Top-K seed 1`
- 仍在运行：
  - `Random-K seed 1`
  - `CNN Bottom-K seed 1`
- 运行中的任务与排程统一维护在 [SCHEDULE.md](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro/SCHEDULE.md)。
