# CPQS 正式结果

最后更新：2026-04-30 14:35 CST

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

## GSM8K 数学线正式结果

### 当前正式口径

- 数据集：`GSM8K test`
- 模型：`Qwen3-8B`
- 模式：`non-thinking`
- 推理参数：
  - `temperature=0`
  - `do_sample=false`
  - `max_new_tokens=512`
  - `batch_size_gsm8k=32`
- 自动评分：
  - gold 从 `#### answer` 提取
  - prediction 统一做数字归一化后 exact match

### Base 正式结果

| Method | GSM8K |
| --- | ---: |
| Base | 0.9310 |

### Base vs Full 正式对照

| Method | GSM8K |
| --- | ---: |
| Base | 0.9310 |
| Full seed 1 | 0.8271 |

### 当前直接结论

- 当前 `Full seed 1 = 0.8271`，明显低于 `Base = 0.9310`。
- 这说明在当前这版 `GSM8K Full SFT` 协议下，模型性能出现了明显回退。
- 因此当前不适合立刻继续推进：
  - `Random-K`
  - `CNN Top-K`
  - `CNN Bottom-K`
- 更合理的顺序是先排查：
  - `GSM8K` 训练 prompt 与 `Base` 推理 prompt 是否足够一致
  - `Qwen3` 在该监督格式下是否发生了过拟合或行为漂移
  - 是否需要改用更贴论文的模型与任务口径

### 与论文表 3 的口径关系

- 论文表 3 是“在 `GSM8K` 训练后，再在 `GSM8K` 上评测不同数据选择方法”。
- 我们现在这条结果是数学线正式基线：
  - 还没有经过 `GSM8K Full SFT`
  - 也还没有进入 `Random-K / CNN Top-K / CNN Bottom-K`
- 因此它当前只用于回答：
  - `Base` 的 `GSM8K` 评测链路是否正常
  - 后续 `Full / Random / CNN` 是否有一个可信起点

### 当前判断

- `Base = 0.9310`，说明这一版 `GSM8K` 自动评测链路是正常的。
- 结合已落盘样例看，当前不存在“提示模板严重污染”或“答案抽取大面积失效”的迹象。
- 但 `Full seed 1 = 0.8271` 明显低于 `Base`，所以当前优先级应调整为：
  - 先分析 `Full` 回退原因
  - 暂缓进入 `CNN` 选择器实验

### 数学线结果文件

- 正式分数表：
  - [gsm8k_base_full_run_scores.csv](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/eval/gsm8k_base_full/run_scores.csv)
- 正式日志：
  - [gsm8k_base_full.log](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/logs/gsm8k_base_full.log)
- 正式样例：
  - [gsm8k_samples.json](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/eval/gsm8k_base_full/samples/gsm8k_samples.json)
- 全量预测：
  - [gsm8k_predictions.json](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/eval/gsm8k_base_full/gsm8k_predictions.json)
- `Full seed 1` 分数表：
  - [gsm8k_full_seed1_run_scores.csv](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/eval/gsm8k_full_seed1/run_scores.csv)
- `Full seed 1` 日志：
  - [gsm8k_full_eval_seed1.log](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/logs/gsm8k_full_eval_seed1.log)
- `Full seed 1` 样例：
  - [gsm8k_samples.json](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/eval/gsm8k_full_seed1/samples/gsm8k_samples.json)
- `Full seed 1` 全量预测：
  - [gsm8k_predictions.json](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/eval/gsm8k_full_seed1/gsm8k_predictions.json)
