# CPQS 复现状态

最后更新：2026-04-29 19:36 CST

## 一、环境与仓库

- 本地仓库：
  - `/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning`
- 用户 fork：
  - `origin -> https://github.com/Victory0431/CPQS-Tuning`
- 上游仓库：
  - `upstream -> https://github.com/renllll/CPQS-Tuning`
- conda 环境：
  - `cpqs-tuning`
- 基础模型：
  - `/home/qjh/llm_learning/base_model/qwen3_8B`
- W&B 项目：
  - `https://wandb.ai/jiahongqin1-ucas-hias/CPQS_research`

## 二、已经完成的核心阶段

### 1. 选择器训练

已完成，目录：

- `repro_outputs/selector_round1`

关键指标：

- Accuracy：`0.8254`
- F1：`0.7734`
- AUC：`0.9291`
- Val loss：`0.3352`

### 2. 候选数据打分

已完成，目录：

- `repro_outputs/scored_alpaca`

主要文件：

- `scored_candidates.json`
- `scored_candidates.csv`

### 3. 第一轮子集构造

已完成，目录：

- `repro_outputs/subsets_round1`

已生成：

- `full.json`
- `random_5000_seed_1.json`
- `random_5000_seed_2.json`
- `random_5000_seed_3.json`
- `cnn_top_5000.json`
- `cnn_bottom_5000.json`

### 4. 已完成的 LoRA 训练

- `Random-K seed 1`
- `Random-K seed 2`
- `CNN Top-K seed 1`
- `CNN Top-K seed 2`
- `CNN Bottom-K seed 1`

## 三、Base 评测修复结论

`2026-04-29` 已确认旧版 `Base` 评测脚本存在错误，主要包括：

- 固定 `do_sample=False`
- 没有暴露 `enable_thinking`
- `left padding` 下批量生成输出切分错误
- 对 `<think>...</think>`、`#### ...`、`<answer>...</answer>` 处理不充分

已经完成的修复：

- 新增 `enable_thinking / do_sample / temperature / top_p / top_k`
- 修复输出切分逻辑
- 增加样例落盘
- 加强自动答案抽取

修复版 `Base v2 non-thinking` 已完成全量评测：

- `GSM8K = 0.9310`
- `MATH-500 = 0.4680`
- `ARC-Challenge = 0.8942`
- `MMLU subset = 0.7083`

对应目录：

- `repro_outputs/eval/base_v2_nonthinking`

说明：

- 旧 `Base` 分数已作废
- 旧脚本下跑出的 adapter 结果暂不再与新 `Base` 混合解释

## 四、论文口径修正

当前已经明确：

- 如果候选池继续使用 `Alpaca-GPT4`
- 且第一轮不使用 LLM judge

那么论文里最应该先跑的自动评测集是：

- `MMLU`
- `ARC-Challenge`
- `HellaSwag`
- `TruthfulQA`

不应该再把：

- `GSM8K`
- `MATH-500`

当作 `Alpaca-GPT4` 主线的核心评测集。

## 五、已完成的评测准备

### 1. 新 benchmark 已准备到本地

目录：

- `/home/qjh/llm_learning/CPQS_lab/data/benchmarks/hellaswag`
- `/home/qjh/llm_learning/CPQS_lab/data/benchmarks/truthfulqa_mc`

准备脚本：

- [prepare_alpaca_benchmarks.py](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro/prepare_alpaca_benchmarks.py)

日志：

- `/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/logs/prepare_alpaca_benchmarks.log`

### 2. 评测脚本已扩展支持

当前 `evaluate_round1.py` 已支持：

- `mmlu_full`
- `arc_challenge`
- `hellaswag`
- `truthfulqa_mc1`

以及原有：

- `gsm8k`
- `math500`
- `mmlu_subset`

### 3. 小批量 smoke test 已完成

本地 smoke 目录：

- `repro_outputs/smoke_eval_alpaca_auto`

本次 smoke 仅测前 `20` 条，目的是验证：

- prompt 是否正确
- 自动评分是否通
- 时间戳日志是否正常
- 样例文件是否正常落盘

smoke 结果：

- `ARC-Challenge = 0.8000`
- `HellaSwag = 0.7500`
- `TruthfulQA MC1 = 0.9000`
- `MMLU full = 1.0000`

说明：

- 这组数字**不是正式结果**
- 它只表示脚本链路已经跑通

## 六、当前运行中的任务

截至 `2026-04-29 19:23 CST`，当前还能确认在跑的 GPU 任务只有：

### GPU0

- `Full seed 1` 训练
  - PID：`565634`
  - tmux：`cpqs_full_seed1_resume`
  - 日志：
    - `repro_outputs/logs/lora_full_seed1.log`

### GPU1

- 当前空闲

## 七、接下来的正确路线

下一阶段建议按这个顺序推进：

1. 固定 `Alpaca-GPT4` 自动评测协议为：
   - `MMLU`
   - `ARC-Challenge`
   - `HellaSwag`
   - `TruthfulQA`
2. 用统一的新脚本重跑：
   - `Base`
   - `Full`
   - `Random-K`
   - `CNN Top-K`
   - `CNN Bottom-K`
3. 在这套统一协议上重新做：
   - 原始分数表
   - mean/std 汇总表
4. 之后如果要跑：
   - `GSM8K / MATH-500`
   - `HumanEval / GPQA`
   
   应切换到 `Reasoning-DeepSeek` 路线，而不是继续沿用 `Alpaca-GPT4`

## 八、提醒文档

为避免后续再次犯同类错误，已经新增两份文档：

- [ALPACA_EVAL_PROTOCOL.md](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro/ALPACA_EVAL_PROTOCOL.md)
- [ATTENTION_FOR_FUTURE_RUNS.md](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro/ATTENTION_FOR_FUTURE_RUNS.md)
