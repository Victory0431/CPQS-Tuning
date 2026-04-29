# CPQS 当前结果汇总

最后更新：2026-04-29 21:42 CST

## 一、当前哪些结果是可信的

`2026-04-29` 之后，当前仓库里可信结果分两类：

- 历史修复版最小闭环基线：
  - `Base v2 non-thinking`
  - 对应 `GSM8K / MATH-500 / ARC-Challenge / MMLU subset`
- 当前论文 `Alpaca-GPT4` 自动评测主线基线：
  - `Base | lm-eval + vLLM`
  - 对应 `MMLU / ARC-Challenge / HellaSwag / TruthfulQA`

其中，后者才是当前这条 `Alpaca-GPT4` 复现线后续要继续扩展到 `Full / Random-K / CNN Top-K / CNN Bottom-K` 的正式口径。

## 二、当前正式 Alpaca 自动评测基线

本次正式基线使用：

- `lm-evaluation-harness + vLLM`
- `Qwen3-8B`
- `enable_thinking=false`
- `temperature=0`
- `bf16`
- `max_model_len=2048`

结果目录：

- [base_lm_eval_vllm](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/eval/base_lm_eval_vllm)

日志：

- [base_lm_eval_vllm.log](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/logs/base_lm_eval_vllm.log)

原始结果 JSON：

- [results_2026-04-29T21-17-31.610048.json](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/eval/base_lm_eval_vllm/__home__qjh__llm_learning__base_model__qwen3_8B/results_2026-04-29T21-17-31.610048.json)

正式基线如下：

| 组别 | seed | MMLU | ARC-Challenge acc_norm | HellaSwag acc_norm | TruthfulQA MC1 | 状态 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Base lm-eval vLLM | 1 | 0.4964 | 0.4224 | 0.5856 | 0.3599 | 已完成 |

对应表格：

- [alpaca_auto_per_run_scores.csv](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/tables/alpaca_auto_per_run_scores.csv)
- [alpaca_auto_group_mean_std.csv](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/tables/alpaca_auto_group_mean_std.csv)

## 三、与论文口径的关系

这组结果的意义是：

- benchmark 集合已经对齐到论文 `Alpaca-GPT4` 自动评测部分
- 推理后端已经切到 `vLLM`
- 后续 `Full / Random-K / CNN Top-K / CNN Bottom-K` 可以直接沿用同一套协议

但也要注意两点：

1. 论文 Table 1 主要报告的是不同数据选择策略与 `Full` 的比较，不是直接给出一个完全同设定的 `Qwen3-8B Base` 行。
2. 我们当前底座模型是 `Qwen3-8B`，论文该表的代表模型并不是同一个，所以这里更适合作为“当前复现线内部统一基线”，而不是直接做跨论文数字对齐。

## 四、历史修复版最小闭环基线

这组结果保留作“旧脚本修复完成”的留痕，不再作为 `Alpaca-GPT4` 正式主表基线：

| 组别 | seed | GSM8K | MATH-500 | ARC-Challenge | MMLU subset | 状态 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Base v2 non-thinking | 1 | 0.9310 | 0.4680 | 0.8942 | 0.7083 | 已完成 |

对应目录：

- [base_v2_nonthinking](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/eval/base_v2_nonthinking)

## 五、当前结论

当前最稳妥的结论是：

- `Alpaca-GPT4` 这条线的正式评测协议已经稳定到 `lm-eval + vLLM`
- `Base` 正式基线已经跑完，可以作为后续 adapter 对比起点
- `2026-04-29 21:42 CST` 当前排程为：
  - 正在正式评测：`Full seed 1`、`CNN Top-K seed 1`
  - 已挂自动接力队列：`Random-K seed 1`、`CNN Bottom-K seed 1`
- 已验证：在 `max_model_len=2048` 的正式协议下，“每卡 2 个 vLLM 评测同时跑” 会因为
  KV cache 显存不足而失败，因此当前采用“每卡 1 个正式评测 + 1 个排队任务”的稳妥方案

在这些 adapter 全部完成前，不对最终优劣下结论。
