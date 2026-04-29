# CPQS 项目历程与排程

最后更新：2026-04-29 21:56 CST

## 用途

这份文档专门给后续 agent 接手时使用，要求持续增量更新，不做历史覆盖。

建议维护规则：

- 每次关键决策、脚本切换、任务中断原因，都追加到“项目历程”
- 当前正在跑什么，只更新“当前排程”
- 不把结果细节堆在这里，结果统一看 [RESULTS.md](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro/RESULTS.md)

## 项目历程

### 2026-04-29 之前

- 已完成选择器训练、候选数据打分、第一轮子集构造。
- 已完成部分 LoRA 训练：
  - `Random-K seed 1`
  - `Random-K seed 2`
  - `CNN Top-K seed 1`
  - `CNN Top-K seed 2`
  - `CNN Bottom-K seed 1`

### 2026-04-29 Base 评测修复

- 发现旧版 Base 评测脚本存在错误：
  - 输出切分错误
  - `enable_thinking` 未暴露
  - 答案抽取不充分
- 修复后得到 `Base v2 non-thinking`，但该结果只保留作旧链路修复留痕。

### 2026-04-29 Alpaca 正式评测主线切换

- 论文 `Alpaca-GPT4` 自动评测主线确定为：
  - `MMLU`
  - `ARC-Challenge`
  - `HellaSwag`
  - `TruthfulQA`
- 正式评测框架切换为：
  - `lm-evaluation-harness + vLLM`
- 正式 Base 已完成，作为当前统一基线。

### 2026-04-29 并行策略验证

- 尝试“每卡 2 个正式 `vLLM` 评测同时跑”。
- 实测失败，第二个实例在初始化 KV cache 时退出。
- 关键报错：
  - `ValueError: No available memory for the cache blocks`
- 结论：
  - 当前正式协议下不能稳定每卡双开 `vLLM`
  - 改为“每卡 1 个正式评测 + 1 个排队任务”

### 2026-04-29 队列脚本化

- 不再依赖临时 `tmux while-loop` 作为长期方案。
- 新增正式排队脚本：
  - [queue_after_tmux.py](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro/queue_after_tmux.py)
- 脚本职责：
  - 等待指定 `tmux session` 结束
  - 自动启动后继任务
  - 全程写时间戳日志
- 当前已实际接管两个队列会话：
  - `cpqs_queue_gpu0`
  - `cpqs_queue_gpu1`

## 当前排程

### GPU0

- 正在运行：
  - `Full seed 1` 正式评测
  - tmux：`cpqs_eval_full_seed1`
  - 日志：
    - [full_seed1_lm_eval_vllm.log](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/logs/full_seed1_lm_eval_vllm.log)
- 排队等待：
  - `Random-K seed 1` 正式评测
  - 队列 tmux：`cpqs_queue_gpu0`
  - 队列日志：
    - [queue_gpu0.log](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/logs/queue_gpu0.log)

### GPU1

- 正在运行：
  - `CNN Top-K seed 1` 正式评测
  - tmux：`cpqs_eval_cnn_top_seed1`
  - 日志：
    - [cnn_top_k5000_seed1_lm_eval_vllm.log](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/logs/cnn_top_k5000_seed1_lm_eval_vllm.log)
- 排队等待：
  - `CNN Bottom-K seed 1` 正式评测
  - 队列 tmux：`cpqs_queue_gpu1`
  - 队列日志：
    - [queue_gpu1.log](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/logs/queue_gpu1.log)

## 当前统一协议

- benchmark：
  - `MMLU`
  - `ARC-Challenge`
  - `HellaSwag`
  - `TruthfulQA MC1`
- 推理参数：
  - `temperature=0`
  - `bf16`
  - `max_model_len=2048`
  - `batch_size=auto:2`
  - `max_batch_size=16`
  - `gpu_memory_utilization=0.4`
  - `hf_offline=true`

## 下一步

- 等 `Full seed 1 / CNN Top-K seed 1` 先完成。
- 自动接续 `Random-K seed 1 / CNN Bottom-K seed 1`。
- 四个 seed1 全部结束后：
  - 重新聚合 `per-run` 表
  - 重新聚合 `mean/std` 表
  - 更新 [RESULTS.md](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro/RESULTS.md)
