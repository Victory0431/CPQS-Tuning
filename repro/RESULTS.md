# CPQS 正式结果

最后更新：2026-04-30 16:55 CST

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
- 这并不等于后续 `Random-500 / CNN Top-500 / CNN Bottom-500` 没有分析价值。
- 当前已经转入一条更明确的探索线：
  - 直接用现有 selector 对 `GSM8K train` 全量打分
  - 构造 `Top-500 / Bottom-500 / Random-500`
  - 比较小规模子集在数学任务上的迁移效果
- `Full < Base` 的首轮排查文档仍保留作为背景依据：
  - [GSM8K_FULL_REGRESSION_AUDIT.md](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro/GSM8K_FULL_REGRESSION_AUDIT.md)

### 与论文表 3 的口径关系

- 论文表 3 是“在 `GSM8K` 训练后，再在 `GSM8K` 上评测不同数据选择方法”。
- 我们当前这轮正在执行的是一个探索版迁移实验：
  - selector 不是在 `GSM8K` 域内重训
  - 而是直接复用现有 `Alpaca` 主线训练得到的 selector
  - 然后迁移到 `GSM8K train` 上做全量打分
- 因此这轮结果的解释应当是：
  - “现有 CPQS selector 迁移到数学域后，对 `Top-500 / Bottom-500 / Random-500` 的区分是否有用”
  - 而不是“论文表 3 的严格同构复现”

### 当前判断

- `Base = 0.9310`，说明这一版 `GSM8K` 自动评测链路是正常的。
- 结合已落盘样例看，当前不存在“提示模板严重污染”或“答案抽取大面积失效”的迹象。
- `Full seed 1` 明显低于 `Base`，说明当前 `Qwen3-8B + GSM8K Full SFT` 协议本身存在性能回退风险。
- 也正因为如此，`Top-500 / Bottom-500 / Random-500` 小数据实验更值得看：
  - 它可以帮助我们判断“回退是否来自全量监督本身”
  - 也可以帮助我们判断“selector 迁移后是否至少能优于随机采样”

### GSM8K selector 迁移打分完成

这一步已经完成，当前使用的是现有 selector：

- checkpoint：
  - [best_selector.pth](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/selector_round1/checkpoints/best_selector.pth)
- 候选池：
  - [full_train.json](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/gsm8k/full_train.json)
- 总样本数：
  - `7473`

打分输出已经落盘：

- JSON：
  - [scored_candidates.json](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/gsm8k/scored_existing_selector/scored_candidates.json)
- CSV：
  - [scored_candidates.csv](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/gsm8k/scored_existing_selector/scored_candidates.csv)
- 日志：
  - [gsm8k_score_existing_selector.log](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/logs/gsm8k_score_existing_selector.log)

分数摘要如下：

| 指标 | 数值 |
| --- | ---: |
| 总样本数 | 7473 |
| 最高分 | 0.7053 |
| 最低分 | 0.0017 |
| 平均分 | 0.4013 |
| Top-500 阈值 | 0.5648 |
| Bottom-500 阈值 | 0.1999 |

### GSM8K 分数分布图

- 直方图：
  - [gsm8k_score_histogram.png](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/gsm8k/plots_transfer500/gsm8k_score_histogram.png)
- 排序曲线：
  - [gsm8k_score_curve.png](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/gsm8k/plots_transfer500/gsm8k_score_curve.png)
- 摘要：
  - [score_summary.json](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/gsm8k/plots_transfer500/score_summary.json)

从当前分布看：

- 高分段和低分段之间存在可见间隔，不是完全挤在一起。
- 但这仍然只是“selector 分数分离度”，不能直接等价为“下游训练效果差异”。
- 是否真的有用，还要看后续 `Top-500 / Bottom-500 / Random-500` 的正式 `GSM8K` 评测。

### GSM8K Top-500 / Bottom-500 / Random-500 子集已构造

当前 3 个训练子集都已经准备好：

- [cnn_top_500.json](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/gsm8k/subsets_transfer500/cnn_top_500.json)
- [cnn_bottom_500.json](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/gsm8k/subsets_transfer500/cnn_bottom_500.json)
- [random_500_seed_1.json](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/gsm8k/subsets_transfer500/random_500_seed_1.json)
- 清单：
  - [subset_manifest.csv](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/gsm8k/subsets_transfer500/subset_manifest.csv)

### GSM8K transfer-500 当前待完成项

当前还没有完成正式结果的组别是：

| Method | 训练集 | 状态 |
| --- | --- | --- |
| Random-500 seed 1 | 随机 500 条 | 待训练评测 |
| CNN Top-500 seed 1 | 分数最高 500 条 | 待训练评测 |
| CNN Bottom-500 seed 1 | 分数最低 500 条 | 待训练评测 |

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
