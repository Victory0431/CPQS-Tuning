# GSM8K 数学实验方案

最后更新：2026-04-30 18:40 CST

## 目标

从当前 `Alpaca_GPT4` 主线暂时抽身，先做一个更直接、更容易自动判分的数学闭环：

- 用 `GSM8K` 训练集做数据筛选与 `LoRA SFT`
- 用 `GSM8K` 测试集做自动评分
- 快速判断 `CPQS` 在数学领域是否能稳定优于随机抽样

如果这条线也不成立，那么当前方法至少在：

- 通用指令能力
- 数学推理能力

这两条线上都没有显现稳定收益，后续就应更谨慎评估论文方法的泛化性。

## 与论文的关系

论文表 3 是“数学领域中，不同数据选择方法在 `GSM8K` 数据集上训练模型的性能评估”。

论文表 3 的严格对照对象包括：

- `Original`
- `Self`
- `MoDs`
- `Alpagasus`
- `Superfiltering`
- `Full`

我们当前仓库直接具备的是：

- `Base`
- `Full`
- `Random-K`
- `CNN Top-K`
- `CNN Bottom-K`

因此建议分两阶段推进。

## 阶段 A：先做最小闭环

先不强求把论文表 3 的所有外部 baseline 一次做全，先验证我们当前 `CPQS` 线本身：

- `Base`
- `Full`
- `Random-K`
- `CNN Top-K`
- `CNN Bottom-K`

重点比较：

- `Full vs Base`
- `CNN Top-K vs Random-K`
- `CNN Bottom-K vs Random-K`
- `CNN Top-K vs Full`

这是最快能回答“当前方法在数学数据上到底有没有用”的方案。

## 当前实际执行版

截至 2026-04-30，本轮实际执行的是一个“迁移 selector 到 GSM8K”的探索版：

- 不重新训练 `GSM8K` 域内 selector
- 直接复用现有 `selector_round1` 的最优 checkpoint
- 对 `GSM8K train (7473)` 全量打分
- 构造：
  - `CNN Top-500`
  - `CNN Bottom-500`
  - `Random-500`
- 然后在统一 `LoRA SFT` 协议下分别训练并在 `GSM8K test` 上评测

这轮实验的作用是：

- 快速判断现有 `CPQS` 选择器迁移到数学域后，是否仍然能区分“更有帮助”和“更无帮助”的样本
- 判断 `CNN Top-500` 至少是否能优于 `Random-500`
- 判断 `CNN Bottom-500` 是否会显著差于 `Random-500`

这不是论文表 3 的最严格同构复现，因为：

- selector 不是在 `GSM8K` 域内训练出来的
- 外部 baseline 例如 `Self / MoDs / Alpagasus / Superfiltering` 这轮还未纳入

但它仍然能非常快地回答一个重要问题：

- 现有 selector 是否具备跨域迁移价值

## 当前状态

当前已完成：

- `GSM8K Base = 0.9310`
- `GSM8K Full seed 1 = 0.8271`
- `GSM8K train` 全量打分完成
- 分数分布图完成
- `Top-500 / Bottom-500 / Random-500` 子集构造完成

当前待完成：

- `CNN Top-500 seed 1` 训练 + 评测
- `CNN Bottom-500 seed 1` 训练 + 评测
- `Random-500 seed 1` 训练 + 评测

这三组结束后，结果表将统一比较：

- `Base`
- `Full`
- `Random-500`
- `CNN Top-500`
- `CNN Bottom-500`

## 阶段 B：再补论文表 3 外部 baseline

如果阶段 A 结果有希望，再补以下方法：

- `Original`
- `Self`
- `MoDs`
- `Alpagasus`

这一步需要额外实现或接入对应算法，不适合在当前主线仍未稳定前就同时展开。

## 模型选择建议

### 方案 1：严格贴论文

优先模型：

- `Qwen2.5-7B-Instruct`

原因：

- 论文表 3 明确给出了 `Qwen 2.5-7B-Instruct` 在 `GSM8K` 上的结果
- 这样更容易和论文数值直接比对

### 方案 2：延续当前环境快速验证

备选模型：

- `Qwen3-8B`

原因：

- 当前环境、脚本、LoRA 流水线都已接通
- 可以更快得到“方法是否有效”的内部验证

### 建议

主建议是：

1. 先用 `Qwen3-8B` 跑通修复后的 `GSM8K` 最小闭环
2. 如果看见趋势，再切到 `Qwen2.5-7B-Instruct` 做论文对照版

这样效率和论文贴合度能兼顾。

## 数据定义

### 候选池

- `GSM8K train`
- 规模约 `7.5K`

### 评测集

- `GSM8K test`
- 规模约 `1K`

### 评分方式

必须使用自动评分，不使用 `LLM judge`。

标准规则：

- gold 从 `#### answer` 中提取
- prediction 也统一提取最终数字答案
- 用归一化后的数字做 exact match

## 子集设计

### 第一轮推荐设置

- `Full`: 全量 `GSM8K train`
- `Random-K`: `K=3000` 或 `K=5000`
- `CNN Top-K`: 同样 `K=3000` 或 `K=5000`
- `CNN Bottom-K`: 同样 `K=3000` 或 `K=5000`

### 为什么建议先用 `K=3000`

- `GSM8K train` 只有约 `7.5K`
- 直接 `K=5000` 与 `Full` 距离已经不算很大
- `K=3000` 更容易拉开数据筛选方法之间的差异

### 如果更想保持和上一轮一致

也可以先继续：

- `K=5000`

优点：

- 和前面 `Alpaca_GPT4` 主线更一致
- 更少新变量

建议排序：

1. 第一优先：`K=3000`
2. 第二优先：`K=5000`

## seed 设计

### 正式建议

- `Base`: 1 次
- `Full`: 1 次
- `Random-K`: 3 seeds
- `CNN Top-K`: 3 seeds
- `CNN Bottom-K`: 3 seeds

这样既能控制总成本，也能算出：

- per-run 原始分数表
- group `mean/std`

## 训练协议

### 统一要求

所有 `LoRA SFT` 组必须保持完全一致，只允许训练数据子集不同。

统一项：

- 同一 base model
- 同一 epoch
- 同一 learning rate
- 同一 max length
- 同一 prompt 模板
- 同一 evaluation prompt
- 同一 decoding 参数

### 当前建议超参

按论文 4.3 口径统一：

- `epochs=3`
- `learning_rate=5e-5`
- `effective_batch_size=16`
  - 例如 `per_device_train_batch_size=1`
  - `gradient_accumulation_steps=16`
- `max_length=2048`
- `LoRA rank=16`
- `LoRA alpha=8`
- `bf16`

### 训练前必须修的实现问题

在新实验开始前，必须先修下面两点：

1. `Full` 组不能再单独使用 `lora_alpha=16`
2. `SFT full_prompt` 不能继续使用会额外追加 assistant 起始标记的构造方式

否则新一轮数学实验仍然会被旧偏差污染。

## 选择器训练协议

### 保守方案

为了先验证方法效果，建议第一轮延续当前 `CNN` 结构，但把训练协议写死并单独记录：

- backbone 固定
- hidden state 提取方式固定
- `Adam`
- `lr=1e-4`
- `grad_accum=16`
- `AMP`
- `max_train_examples=15000` 的逻辑在 `GSM8K` 线下改成“使用全部可用候选样本”

### 更贴原论文的处理

由于 `GSM8K train` 本身不足 `15000`，这里建议：

- 直接用全量 `GSM8K train`
- 正负样本构造方法单独写清
- 如果正负样本定义依赖外部模型输出，要先固定生成源

这一段在真正启动前还需要再明确一次数据构造口径。

## Prompt 与输出格式

为保证自动判分稳定，建议统一强约束输出格式：

- 训练 prompt 明确要求：
  - “请逐步推理，并在最后一行输出 `#### 最终答案`”
- 评测 prompt 也保持同样要求

这样能减少：

- 空答案
- 多答案
- 提取失败

## 评测协议

### benchmark

- 正式主评测只跑 `GSM8K`

### 推理设置

为了先做稳定比较，建议：

- `temperature=0`
- `bf16`
- `max_model_len=2048`

如果模型是 `Qwen3` 且需要考虑 thinking mode，则分两条线：

- 主线：`enable_thinking=false`
- 扩展线：`enable_thinking=true`

但第一轮正式结果建议只保留一条主线，不要混着解释。

## smoke test 方案

正式全量训练和评测前，必须先过两个小测试。

### smoke 1：SFT 标签检查

随机抽 3 条样本，打印：

- `user_prompt`
- `full_prompt`
- `supervised token span`
- `decoded labels`

确认监督目标只覆盖真实答案区间。

### smoke 2：GSM8K 小批量评测

对 `Base` 先跑前 `50` 条测试样本，落盘：

- question
- gold
- raw_output
- extracted_answer
- correct / incorrect

人工抽看至少 `20` 条。

只有 smoke 通过，才开全量。

## 输出表格

### 表 1：每个 run 的原始分数表

建议字段：

- `group`
- `seed`
- `train_size`
- `gsm8k_exact_match`
- `model`
- `epochs`
- `learning_rate`
- `lora_rank`
- `lora_alpha`

### 表 2：每个 group 的 mean/std 汇总表

建议字段：

- `group`
- `num_runs`
- `gsm8k_mean`
- `gsm8k_std`

## 成功判据

第一轮数学实验最重要的不是追求最好绝对分数，而是回答下面两个问题：

1. `Full` 是否稳定优于 `Base`
2. `CNN Top-K` 是否稳定优于 `Random-K`

如果这两个都不成立，就说明当前方法至少在这条数学线下没有显示出预期优势。

## 建议执行顺序

1. 修复 `SFT` 高风险实现偏差
2. 写 `GSM8K` 数据准备与自动评分脚本
3. 做 `Base` 的 `50` 条 smoke eval
4. 跑 `Base` 正式 `GSM8K`
5. 跑 `Full`
6. 训练选择器并构造 `Random-K / CNN Top-K / CNN Bottom-K`
7. 正式跑 3 seeds
8. 汇总 per-run 和 mean/std

## 当前建议

如果只选一个最务实的起点，我建议：

- 先修 `SFT`
- 用 `Qwen3-8B`
- 在 `GSM8K` 上做
  - `Base`
  - `Full`
  - `Random-K 3 seeds`
  - `CNN Top-K 3 seeds`
  - `CNN Bottom-K 3 seeds`
- `K` 先取 `3000`

这会是最快看到方法是否还有希望的一轮实验。

## 当前实际执行版

截至 `2026-04-30`，当前准备执行的是一个更快的探索版，而不是最严格的论文同构版。

### 当前执行设置

- 候选池：`GSM8K train`
- 评分器：
  - 先使用已经训练好的现有 `CNN selector`
  - 即 `repro_outputs/selector_round1/checkpoints/best_selector.pth`
- 评分目标：
  - 对全部 `7473` 条 `GSM8K train` 打 `CPQS score`
- 子集规模：
  - `Top-500`
  - `Bottom-500`
  - `Random-500`
- 训练模型：
  - 当前先用 `Qwen3-8B`
- 评测集：
  - `GSM8K test`

### 为什么先这样做

- 当前 `Full < Base` 的现象已经出现
- 如果现在再额外重训一个 `GSM8K` 专用 selector，会把排查周期拉长
- 所以先用现有 selector 做一版迁移打分，快速回答：
  - `Top-500` 有没有比 `Random-500` 更好
  - `Bottom-500` 是否明显更差

### 这一版和严格复刻的差异

严格复刻更接近：

- 在 `GSM8K` 域内重新定义正负样本
- 重新训练领域 selector
- 再对 `GSM8K train` 打分

而当前执行版是：

- 直接把现有 selector 迁移到 `GSM8K train`

因此这轮结果应解释为：

- `CPQS selector` 的迁移打分探索实验

而不是：

- 论文 `GSM8K` 下游任务的最严格复刻

### 当前新增产物

本轮会新增：

- `GSM8K train` 全量打分结果
- `Top-500 / Bottom-500 / Random-500` 三个训练子集
- 分数分布图：
  - 直方图
  - 排序曲线图

这些结果会同步更新到：

- [RESULTS.md](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro/RESULTS.md)
- [SCHEDULE.md](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro/SCHEDULE.md)
