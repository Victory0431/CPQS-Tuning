# CPQS 项目历程与排程

最后更新：2026-05-01 11:30 CST

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

### 2026-04-29 第一批正式结果落盘

- 已完成正式评测：
  - `Base`
  - `Full seed 1`
  - `CNN Top-K seed 1`
- 已更新正式汇总表：
  - `alpaca_auto_per_run_scores.csv`
  - `alpaca_auto_group_mean_std.csv`
- 剩余未完成：
  - `Random-K seed 1`
  - `CNN Bottom-K seed 1`

### 2026-04-30 seed 1 正式评测收尾

- `Random-K seed 1` 已完成正式评测。
- `CNN Bottom-K seed 1` 已完成正式评测。
- 本轮 seed 1 五组正式结果现已齐全：
  - `Base`
  - `Full`
  - `Random-K`
  - `CNN Top-K`
  - `CNN Bottom-K`
- `RESULTS.md` 已整理为论文 Table 1 风格主表。

### 2026-04-30 实现审计与主线调整

- 已完成对论文、原仓库、当前复现实现的逐项审计。
- 审计结论：
  - `CNN` 结构基本贴论文与原仓库
  - 选择器训练流程大体贴近，但不是完全原样复现
  - 当前最大偏差在 `LoRA SFT`，不是 `CNN` 本体
- 已确认两个高影响问题：
  - `Full seed 1` 使用了 `lora_alpha=16`，与其他组 `lora_alpha=8` 不一致
  - `SFT full_prompt` 构造会把答案后额外追加的 assistant 起始标记一起纳入监督
- 结论：
  - 当前 Alpaca 主线结果不能直接支持论文结论
  - 也不能据此直接反驳论文结论
  - 需要先修复 `SFT` 再继续解释方法效果
- 已新增文档：
  - [IMPLEMENTATION_AUDIT.md](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro/IMPLEMENTATION_AUDIT.md)
  - [GSM8K_EXPERIMENT_PLAN.md](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro/GSM8K_EXPERIMENT_PLAN.md)

### 2026-04-30 GSM8K Base 正式基线完成

- 已完成 `GSM8K Base` 正式全量评测。
- 正式结果：
  - `Base = 0.9310`
- 正式日志显示：
  - `1319` 条测试样本全部完成
  - 最终 throughput 约 `0.88 samples/s`
- 已落盘：
  - 正式分数：
    - [run_scores.csv](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/eval/gsm8k_base_full/run_scores.csv)
  - 正式样例：
    - [gsm8k_samples.json](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/eval/gsm8k_base_full/samples/gsm8k_samples.json)
  - 正式日志：
    - [gsm8k_base_full.log](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/logs/gsm8k_base_full.log)
- 当前判断：
  - 数学线 `GSM8K` 评测链路正常
  - 可以继续拿同一协议比较 `Full / Random-K / CNN Top-K / CNN Bottom-K`

### 2026-04-30 GSM8K Full seed 1 完成

- `GSM8K Full seed 1` 已完成训练与正式评测。
- 正式结果：
  - `Full seed 1 = 0.8271`
- 与当前正式基线对比：
  - `Base = 0.9310`
  - `Full` 明显低于 `Base`
- 这意味着：
  - 当前数学线不是“Full 提升后再比较 Top-K/Random-K”的状态
  - 而是“Full 已经先发生性能回退”
- 当前最合理的动作：
  - 暂缓直接进入 `CNN` 选择器实验
  - 先排查 `Full` 为什么会把 `GSM8K` 从 `0.9310` 拉低到 `0.8271`

### 2026-04-30 GSM8K Full 回退首轮排查

- 已完成一轮小型 ablation：
  - 对齐 `Base` 与 `Full` 全量预测
  - 重点查看 `Base 正确 / Full 错误` 的样本
- 当前统计：
  - `Full` 错误总数：`228`
  - `Base 正确 / Full 错误`：`163`
- 目前未发现以下问题：
  - 答案抽取失败
  - 输出被截断
  - `<think>` 污染
  - 非标准 non-thinking 模板输出
- 当前最像的原因：
  - `Full` 训练后数学推理本身退化
  - 尤其体现在百分比、单位换算、多步约束链条
- 已新增排查文档：
  - [GSM8K_FULL_REGRESSION_AUDIT.md](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro/GSM8K_FULL_REGRESSION_AUDIT.md)

### 2026-04-30 GSM8K Top/Bottom/Random-500 探索版启动

- 用户要求继续推进论文 `GSM8K` 下游任务风格实验：
  - `Top-500`
  - `Bottom-500`
  - `Random-500`
- 当前执行版说明：
  - 先不重训 `GSM8K` 专用 selector
  - 直接使用现有 `selector_round1` 的 `best_selector.pth`
  - 对全部 `GSM8K train (7473)` 做迁移打分
- 当前这一步已经完成：
  - `GSM8K train (7473)` 全量迁移打分
  - 分数分布图生成
  - `Top-500 / Bottom-500 / Random-500` 三个子集构造
- 已落盘结果：
  - [scored_candidates.json](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/gsm8k/scored_existing_selector/scored_candidates.json)
  - [gsm8k_score_histogram.png](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/gsm8k/plots_transfer500/gsm8k_score_histogram.png)
  - [gsm8k_score_curve.png](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/gsm8k/plots_transfer500/gsm8k_score_curve.png)
  - [cnn_top_500.json](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/gsm8k/subsets_transfer500/cnn_top_500.json)
  - [cnn_bottom_500.json](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/gsm8k/subsets_transfer500/cnn_bottom_500.json)
  - [random_500_seed_1.json](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/gsm8k/subsets_transfer500/random_500_seed_1.json)
- 本轮新增脚本：
  - [plot_gsm8k_selector_scores.py](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro/plot_gsm8k_selector_scores.py)
  - [build_gsm8k_subsets.py](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro/build_gsm8k_subsets.py)
  - [run_gsm8k_prepare_transfer500.sh](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro/run_gsm8k_prepare_transfer500.sh)
  - [run_gsm8k_train_eval.sh](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro/run_gsm8k_train_eval.sh)

### 2026-04-30 GSM8K transfer-500 双卡排程

- 当前正式执行排程如下：
  - `GPU0`：
    - `cnn_top_500 seed 1` 训练 + 评测
  - `GPU1`：
    - `random_500 seed 1` 训练 + 评测
  - 排队任务：
    - `cnn_bottom_500 seed 1` 在任一空闲卡上接续启动
- 统一协议：
  - 模型：`Qwen3-8B`
  - epoch：`3`
  - lr：`5e-5`
  - LoRA：`r=16, alpha=8`
  - batch：`per_device=1, grad_acc=16`
  - eval：`GSM8K test`
  - decode：`non-thinking, temperature=0, batch_size_gsm8k=32`
- 日志要求：
  - 所有训练和评测均写入 `repro_outputs/logs/`
  - 队列等待也必须有独立时间戳日志
- 2026-04-30 16:42 CST 已确认实际启动：
  - `tmux session`：
    - `gsm8k_top500_pipeline`
    - `gsm8k_random500_pipeline`
    - `gsm8k_bottom500_queue`
  - `wandb` 在线同步已确认正常：
    - `gsm8k_cnn_top_500_seed1-seed1`
    - `gsm8k_random_500_seed1-seed1`
  - 当前日志入口：
    - [gsm8k_cnn_top_500_seed1_train.log](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/logs/gsm8k_cnn_top_500_seed1_train.log)
    - [gsm8k_random_500_seed1_train.log](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/logs/gsm8k_random_500_seed1_train.log)
    - [gsm8k_bottom500_queue.log](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/logs/gsm8k_bottom500_queue.log)

### 2026-05-01 GSM8K transfer-500 结果收尾

- 三组探索实验已全部完成：
  - `Random-500 seed 1 = 0.8461`
  - `CNN Top-500 seed 1 = 0.8431`
  - `CNN Bottom-500 seed 1 = 0.8143`
- 结合已完成结果：
  - `Base = 0.9310`
  - `Full = 0.8271`
- 当前可得结论：
  - 迁移 selector 有一定“识别坏数据”的能力
  - 但还没有表现出 `Top-500 > Random-500`
- 因此主线决策调整为：
  - 不继续在这版迁移 selector 上堆更多 seed
  - 转向训练 `GSM8K` 域内 selector
  - 用域内 selector 重新做 `Top-500 / Bottom-500 / Random-500`

### 2026-05-01 下一步任务切换

- 当前下一步正式切换为：
  - 准备 `GSM8K` 域内 selector 训练数据
  - 候选正样本：
    - `GSM8K train` 原始高质量答案
  - 候选负样本：
    - `Qwen3-8B Base` 在 `GSM8K train` 上的生成结果
    - `CNN Bottom-500 LoRA` 在 `GSM8K train` 上的生成结果
- 这样做的原因：
  - 当前只有 `bad data` 区分信号，没有 `top > random` 信号
  - 需要让 selector 在数学域内学习“什么是更好/更差的解答”
- 本轮新增脚本：
  - [prepare_gsm8k_selector_data.py](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro/prepare_gsm8k_selector_data.py)

### 2026-05-01 域内 selector 负样本假设纠偏

- 在最初的域内 selector 方案里，曾尝试把以下生成结果直接当作负样本：
  - `Qwen3-8B Base` 在 `GSM8K train` 上的输出
  - `CNN Bottom-500 adapter` 在 `GSM8K train` 上的输出
- 该设定已被主动叫停，原因是：
  - 论文并没有给出“强模型在旧数学数据集上的生成结果可直接视为低质量样本”的依据
  - 尤其 `Qwen3-8B Base` 在当前 `GSM8K test` 上有 `0.9310`，不能被武断地归类为“坏答案”
  - `GSM8K` 数据较早，也不能简单假设其参考解答一定优于更强新模型的全部输出
- 因此：
  - 这条“Base/Bottom 输出直接做负样本”的分支已停止
  - 当前不再继续消耗显卡生成这类伪负样本
- 当前修正后的原则是：
  - 只有在有明确标签依据时，才能把样本放入 selector 的正负监督集
  - 否则最多只能把这类输出作为待比较候选，而不能直接当真负样本

## 当前排程

- `GSM8K Base` 正式评测已经完成。
- `GSM8K Full seed 1` 已完成。
- `GSM8K transfer-500` 三组探索实验也已完成。
- 当前主任务已切换为：
  - 训练 `GSM8K` 域内 selector
  - 再做一轮更贴论文逻辑的 `Top-500 / Bottom-500 / Random-500`
- 当前不再建议回去补旧 `Alpaca` 主线多 seed。

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

- 当前最重要的下一步是：
  - 生成 `GSM8K` 域内 selector 训练所需的正负样本
  - 启动域内 selector 训练
  - 用新 selector 重新打分 `GSM8K train`
  - 再比较 `Top-500 / Bottom-500 / Random-500`
- 这会是比当前迁移实验更接近论文逻辑的一轮。
