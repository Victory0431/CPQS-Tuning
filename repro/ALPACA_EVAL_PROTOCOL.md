# Alpaca 自动评测协议

最后更新：2026-04-29 19:36 CST

## 一、适用范围

这份协议用于：

- 候选池使用 `Alpaca-GPT4`
- 第一轮只做**自动评分**
- 暂不使用 `GPT-4o pairwise judge`
- 暂不使用 `AlpacaEval`

## 二、论文对应的自动评测集

根据论文 `4.4 Evaluation Metrics`，`Alpaca-GPT4` 这条线在不依赖 LLM judge 的情况下，第一优先应该跑：

- `MMLU`
- `ARC-Challenge`
- `HellaSwag`
- `TruthfulQA`

这四个 benchmark 都属于论文里通过 `lm-evaluation-harness` 评测的自动指标部分。

## 三、当前本地准备情况

### 已有本地数据

- `ARC-Challenge`
  - `/home/qjh/llm_learning/CPQS_lab/data/benchmarks/arc_challenge`
- `MMLU full`
  - `/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/eval/Open LLM Leaderboard/MMLU/mmlu_data/mmlu_test.json`

### 新增本地数据

- `HellaSwag`
  - `/home/qjh/llm_learning/CPQS_lab/data/benchmarks/hellaswag`
- `TruthfulQA multiple-choice`
  - `/home/qjh/llm_learning/CPQS_lab/data/benchmarks/truthfulqa_mc`

准备脚本：

- [prepare_alpaca_benchmarks.py](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro/prepare_alpaca_benchmarks.py)

## 四、当前脚本支持

评测脚本：

- [evaluate_round1.py](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro/evaluate_round1.py)

新增支持的 benchmark 标识：

- `mmlu_full`
- `arc_challenge`
- `hellaswag`
- `truthfulqa_mc1`

说明：

- `truthfulqa_mc1` 当前使用的是 `truthful_qa / multiple_choice` 的自动多选评分
- 这是不依赖大模型 judge 的版本

## 五、推荐运行顺序

### 1. 先做 smoke test

建议每次改完脚本后，先跑：

- `limit=20`
- `sample_dump_count=10`

目的：

- 检查 prompt 格式
- 检查自动抽取
- 检查日志是否有持续输出
- 检查样例文件是否能人工核对

### 2. smoke test 通过后再开全量

正式跑法建议：

- `enable_thinking=false`
- `do_sample=true`
- `temperature=0.7`
- `top_p=0.8`
- `top_k=20`

说明：

- 当前 `Qwen3-8B` 在这套自动 benchmark 上，`non-thinking` 更稳定
- `thinking` 模式更容易在 `max_new_tokens=512` 内截断，从而影响自动评分

## 六、当前不建议做的事

在 `Alpaca-GPT4` 这条线上，当前不建议：

- 把 `GSM8K / MATH-500` 当成主评测集
- 用新 `Base` 去和旧脚本下的 adapter 分数混合比较
- 在没做 smoke test 的情况下直接开全量长跑

## 七、后续扩展

如果后续要严格补齐论文的另一半：

- `AlpacaEval`
- `GPT-4o pairwise judge`

可以在自动 benchmark 稳定后单独加，不要和第一轮自动评测混在一起。
