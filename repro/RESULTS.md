# CPQS 正式结果

最后更新：2026-04-30 10:20 CST

## 正式评测口径

- 候选池：`Alpaca-GPT4`
- 评测框架：`lm-evaluation-harness + vLLM`
- 基础模型：`Qwen3-8B`
- 推理设置：
  - `enable_thinking=false`
  - `temperature=0`
  - `bf16`
  - `max_model_len=2048`

## 论文 Table 1 风格主表

下表按论文 `Alpaca-GPT4` 自动评测部分的列顺序整理：
`MMLU / ARC / TruthfulQA / HellaSwag`。

| Method | MMLU | ARC-Challenge | TruthfulQA MC1 | HellaSwag |
| --- | ---: | ---: | ---: | ---: |
| Base | 0.4964 | 0.4224 | 0.3599 | 0.5856 |
| Full | 0.7223 | 0.5017 | 0.4259 | 0.6988 |
| Random-K (K=5000, seed 1) | 0.7209 | 0.5068 | 0.3941 | 0.6899 |
| CNN Top-K (K=5000, seed 1) | 0.6887 | 0.5000 | 0.3880 | 0.6917 |
| CNN Bottom-K (K=5000, seed 1) | 0.7162 | 0.5205 | 0.3807 | 0.6973 |

## 当前原始结果表

| group | seed | MMLU | ARC-Challenge acc_norm | HellaSwag acc_norm | TruthfulQA MC1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Base | 1 | 0.4964 | 0.4224 | 0.5856 | 0.3599 |
| Full | 1 | 0.7223 | 0.5017 | 0.6988 | 0.4259 |
| Random-K | 1 | 0.7209 | 0.5068 | 0.6899 | 0.3941 |
| CNN Top-K | 1 | 0.6887 | 0.5000 | 0.6917 | 0.3880 |
| CNN Bottom-K | 1 | 0.7162 | 0.5205 | 0.6973 | 0.3807 |

## 当前结论

- `Full` 相比 `Base` 在四个 benchmark 上都有明显提升。
- `Random-K seed 1` 与 `Full seed 1` 非常接近，在 `ARC-Challenge` 上还略高。
- `CNN Top-K seed 1` 本轮低于 `Random-K seed 1`。
- `CNN Bottom-K seed 1` 在 `ARC-Challenge` 和 `HellaSwag` 上不差，但 `TruthfulQA MC1` 低于 `Random-K seed 1`。
- 当前仍只有 `seed 1` 正式结果，论文要求的 `mean/std` 比较还需要继续补 `seed 2 / seed 3`。

## 结果文件

- 原始结果目录：
  - [base_lm_eval_vllm](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/eval/base_lm_eval_vllm)
  - [full_seed1](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/eval/full_seed1)
  - [random_k5000_seed1](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/eval/random_k5000_seed1)
  - [cnn_top_k5000_seed1](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/eval/cnn_top_k5000_seed1)
  - [cnn_bottom_k5000_seed1](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/eval/cnn_bottom_k5000_seed1)
- 每个 run 原始分数表：
  - [alpaca_auto_per_run_scores.csv](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/tables/alpaca_auto_per_run_scores.csv)
- group mean/std 汇总表：
  - [alpaca_auto_group_mean_std.csv](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/tables/alpaca_auto_group_mean_std.csv)
