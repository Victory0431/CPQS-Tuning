# CPQS 复现状态

最后更新：2026-04-29 16:33 CST

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

## 二、当前实验范围

本轮是用户确认过的“最小闭环 + 第二波 seed 扩展并行推进”，不是论文全部评测协议的完整复现。

固定比较组：

- `Base`
- `Full`
- `Random-K (K=5000)`
- `CNN Top-K (K=5000)`
- `CNN Bottom-K (K=5000)`

固定评测集：

- `GSM8K`
- `MATH-500`
- `ARC-Challenge`
- `MMLU subset`

评分原则：

- 全部使用脚本自动评分
- 不使用 LLM judge

## 三、已经完成的阶段

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

对应 `final_adapter` 已存在。

### 5. 已完成的评测

已完成评测的组：

- `Base`
- `Random-K seed 1`
- `CNN Top-K seed 1`
- `CNN Bottom-K seed 1`

完整表格见：

- [RESULTS.md](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro/RESULTS.md)

CSV 文件见：

- `repro_outputs/tables/per_run_scores.csv`
- `repro_outputs/tables/group_mean_std.csv`

### 6. Base 评测排查结论

`2026-04-29` 对 `Qwen3-8B Base` 的评测异常做了专项排查，已经确认旧版 `Base` 分数不可信，主要问题来自评测脚本而不是模型本身：

- 旧版 `evaluate_round1.py` 固定使用 `do_sample=False`
- 旧版脚本没有暴露 `enable_thinking`
- 旧版批量生成在 `left padding` 下切分输出的方式有 bug，会把半截 prompt 混进 `raw_output`
- 旧版答案抽取对 `<think>...</think>`、`#### ...`、`<answer>...</answer>` 处理不充分

已经完成的修复：

- 新增 `enable_thinking / do_sample / temperature / top_p / top_k` 参数
- 修复批量生成输出切分逻辑
- 新增 `sample_dump_count / sample_dump_dir`
- 统一为四个 benchmark 保存可人工核查的样例
- 加强数字与数学答案抽取规则

旧版 `Base` 结果中的异常证据：

- `gsm8k` 旧结果中约 `17.8%` 样本存在 prompt 泄漏痕迹
- `math500` 旧结果中约 `44.4%` 样本存在 prompt 泄漏痕迹
- 旧 `gsm8k` 中有 `981/1319` 条没有走到最终答案标记

30 条小样本 debug 结果：

- `non-thinking`
  - `GSM8K = 0.9333`
  - `MATH-500 = 0.4667`
  - `ARC-Challenge = 0.8667`
  - `MMLU subset = 0.6333`
- `thinking`
  - `GSM8K = 0.5667`
  - `MATH-500 = 0.1000`
  - `ARC/MMLU` 仍在跑完中时已可见输出更长、更容易在 `max_new_tokens=512` 内截断

当前判断：

- 第一轮正式重评测应优先采用 `non-thinking`
- 旧版 `Base` 分数暂时不应用于解释 `Top-K / Random-K / Bottom-K`
- 已启动新版正式重评测：
  - `repro_outputs/eval/base_v2_nonthinking`
  - 日志：`repro_outputs/logs/base_eval_v2_nonthinking.log`

## 四、当前进行中的任务

截至 `2026-04-29 16:33 CST`，当前主要任务如下：

### GPU0

- `Full seed 1` 继续训练
  - PID：`565634`
  - tmux：`cpqs_full_seed1_resume`
  - 已从 `checkpoint-3251` 恢复
  - 已出现更新后的更高 checkpoint：
    - `checkpoint-6500`
- `CNN Top-K seed 2` 自动评测
  - PID：`1190267`
  - tmux：`cpqs_eval_top_seed2`
  - 日志：
    - `repro_outputs/logs/eval_cnn_top_k5000_seed2.log`
- `Base thinking 30 条 debug`
  - PID：`1269631`
  - tmux：`cpqs_base_debug_think`
  - 日志：
    - `repro_outputs/logs/base_debug_thinking.log`

### GPU1

- `Random-K seed 2` 自动评测
  - PID：`1190275`
  - tmux：`cpqs_eval_random_seed2`
  - 日志：
    - `repro_outputs/logs/eval_random_k5000_seed2.log`
- `Base v2 non-thinking 正式重评测`
  - PID：`1282377`
  - tmux：`cpqs_eval_base_v2`
  - 日志：
    - `repro_outputs/logs/base_eval_v2_nonthinking.log`

说明：

- `Base non-thinking 30 条 debug` 已完成
- `CNN Bottom-K seed 2` 训练进程已不在当前任务列表，需要结合产物与日志再确认是已完成还是已中断
- 现在最关键的新任务是 `Base v2 non-thinking` 全量重评测

## 五、当前未完成项

最小闭环还差：

- `Full seed 1` 训练完成
- `Full seed 1` 评测
- `Base v2 non-thinking` 全量评测完成并更新基线表

第二波 seed 扩展还差：

- `Random-K seed 2` 评测完成
- `CNN Top-K seed 2` 评测完成
- `CNN Bottom-K seed 2` 训练完成并评测
- `Random-K seed 3` 训练与评测
- `CNN Top-K seed 3` 训练与评测
- `CNN Bottom-K seed 3` 训练与评测

## 六、结果解读现状

当前已经拿到的完整 seed1 结果，可以先支持以下比较：

- `CNN Top-K seed1 vs Random-K seed1`
- `CNN Bottom-K seed1 vs Random-K seed1`
- `Base vs Random-K seed1`
- `Base vs CNN Top-K seed1`
- `Base vs CNN Bottom-K seed1`

当前还不能完成的重点比较：

- `Full vs Base`
- `CNN Top-K vs Full`

原因是 `Full seed 1` 训练和评测还没结束。

## 七、日志与产物约定

当前长任务默认都具备带时间戳日志，主要目录：

- `repro_outputs/logs`
- `repro_outputs/lora`
- `repro_outputs/eval`
- `repro_outputs/tables`

从现在开始，`repro/` 下文档统一使用中文维护。
