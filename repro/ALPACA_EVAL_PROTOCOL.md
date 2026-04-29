# Alpaca 自动评测协议

最后更新：2026-04-29 20:55 CST

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
  - 由 `lm-evaluation-harness` 自动管理与缓存

### 新增本地数据

- `HellaSwag`
  - `/home/qjh/llm_learning/CPQS_lab/data/benchmarks/hellaswag`
- `TruthfulQA multiple-choice`
  - `/home/qjh/llm_learning/CPQS_lab/data/benchmarks/truthfulqa_mc`

准备脚本：

- [prepare_alpaca_benchmarks.py](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro/prepare_alpaca_benchmarks.py)

## 四、当前正式实现

当前已经切换为：

- `lm-evaluation-harness + vLLM`

统一入口脚本：

- [run_lm_eval_vllm.py](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro/run_lm_eval_vllm.py)
- [smoke_test_vllm.py](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro/smoke_test_vllm.py)

说明：

- `truthfulqa_mc1` 使用 `truthful_qa / multiple_choice` 的自动多选评分
- 不依赖大模型 judge
- 与论文“部署和推理使用 `vLLM`”的描述更一致

## 五、推荐运行顺序

### 1. 先做 `vLLM` 单条推理 smoke

目的：

- 检查 `Qwen3` chat template 是否正确
- 检查 `enable_thinking=false` 是否生效
- 检查 `vLLM` 与模型版本是否兼容

### 2. 再做 `lm-eval + vLLM` 小批量 smoke

建议：

- `limit=10`
- benchmark 直接覆盖：
  - `MMLU`
  - `ARC-Challenge`
  - `HellaSwag`
  - `TruthfulQA MC1`

目的：

- 检查 benchmark 名称与数据下载是否正常
- 检查 `apply_chat_template` 是否正确接入
- 检查日志是否有持续输出
- 检查 `MMLU` 子任务样例是否能正常落盘

### 3. smoke test 通过后再开全量

正式跑法建议：

- `enable_thinking=false`
- `temperature=0`
- `bf16`
- `max_model_len=2048`
- `batch_size=auto:4`
- `max_batch_size=64`
- smoke 已经把四个 benchmark 缓存到本地后，正式跑建议加：
  - `--hf_offline`

说明：

- `Qwen3-8B` 在这条自动 benchmark 主线上，`non-thinking` 更稳
- `auto:4 + max_batch_size=64` 已在 `H200` 上通过 smoke
- 第一次全量 `MMLU` 会有冷启动缓存成本，后续会更快

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
