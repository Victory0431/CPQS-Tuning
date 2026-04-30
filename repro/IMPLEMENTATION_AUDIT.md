# CPQS 实现审计

最后更新：2026-04-30 18:40 CST

## 审计范围

本次审计对照了三部分材料：

- 论文：`7489_CPQS_Tuning_A_Model_Self_.docx/.pdf`
- 原开源代码：
  - `train_cnn.py`
  - `predict.py`
  - `select_data.py`
- 我们当前复现实现：
  - `repro/common.py`
  - `repro/train_selector.py`
  - `repro/score_candidates.py`
  - `repro/build_subsets.py`
  - `repro/train_lora.py`

本次重点回答三个问题：

1. `CNN` 结构是否贴原文
2. `CNN` 训练流程是否贴原文
3. 数据筛选后接 `LoRA SFT` 是否贴论文超参和训练流程

## 一页结论

- `CNN` 结构本身基本贴论文和原仓库，不是当前最主要问题。
- 选择器训练流程大体贴近，但不是完全原样复现；主要差异是我们做了 `train/val` 切分和 best checkpoint 选择。
- 当前最大偏差在 `LoRA SFT` 这一段，而不是 `CNN` 本体。
- 其中最严重的两点是：
  - `Full` 组实际用了 `lora_alpha=16`，而其他组用了 `lora_alpha=8`
  - 我们当前 `SFT` prompt 构造会把答案后额外追加的 assistant 起始标记一起纳入监督
- 因此，当前 `Full / Random-K / CNN Top-K / CNN Bottom-K` 的比较不能视为论文口径下的严格结论。

## 1. CNN 结构对照

### 结论

`CNN` 结构基本一致，风险低。

### 对照证据

- 原仓库 `DeepTextCNN`：
  - [train_cnn.py](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/train_cnn.py#L37)
- 我们当前 `DeepTextCNN`：
  - [common.py](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro/common.py#L40)

两边的核心结构一致：

- `Conv2d` 三个卷积核：`(3, 4, 5)`
- `num_filters=256`
- 后接两层 `Conv1d(kernel_size=3, padding=1)`
- `adaptive_max_pool1d`
- `dropout=0.5`
- `Linear(3*256 -> 512 -> 2)`

这和论文里“`2D + 1D conv + adaptive max pooling`”的描述是对齐的。

## 2. hidden state 提取对照

### 结论

主干逻辑贴近论文与原仓库，但有一个中风险细节。

### 一致点

- 都是把 `Instruction + Input` 作为 user
- 都是把 `Response` 拼成 assistant
- 都是前向跑完整对话，再用 `start_idx` 截取 response 对应 hidden states

对照位置：

- 原仓库：
  - [train_cnn.py](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/train_cnn.py#L306)
  - [predict.py](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/predict.py#L221)
- 我们当前实现：
  - [common.py](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro/common.py#L232)
  - [common.py](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro/common.py#L253)

### 中风险差异

对于 `Qwen` 路线，原仓库和我们当前实现都对 `full_dialog` 使用了：

- `tokenizer.apply_chat_template(..., add_generation_prompt=True)`

对应位置：

- 原仓库：
  - [train_cnn.py](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/train_cnn.py#L310)
- 我们实现：
  - [common.py](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro/common.py#L160)

这会导致 response hidden slice 的尾部可能多出一个 assistant 起始标记。它和原仓库是一致的，所以不构成“我们自己额外引入的新偏差”；但它和论文字面上的“只取 Response”并不完全等价。

综合判断：

- 这点有影响，但不是当前最主要问题。

## 3. 选择器训练流程对照

### 结论

总体贴近论文，但不是原样复现，风险中等。

### 与论文/原代码一致的部分

- `Adam`
- `learning_rate=1e-4`
- `grad_accum_steps=16`
- `AMP`
- `CrossEntropyLoss`
- 训练总量上限 `15000`

对应位置：

- 原仓库训练参数：
  - [train_cnn.py](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/train_cnn.py#L133)
  - [train_cnn.py](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/train_cnn.py#L285)
- 我们当前训练参数：
  - [train_selector.py](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro/train_selector.py#L43)
  - [train_selector.py](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro/train_selector.py#L143)

### 关键差异 1：我们切了验证集

原仓库 `train_cnn.py` 是直接把 `15000` 条数据拿来训练，没有显式验证集。

我们当前实现会做：

- `train_ratio=0.9`
- `train=13499`
- `val=1501`

证据：

- 代码：
  - [train_selector.py](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro/train_selector.py#L46)
  - [train_selector.py](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro/train_selector.py#L153)
- 实际落盘：
  - [split_summary.json](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/selector_round1/split_summary.json)

影响判断：

- 这会让真正参与梯度更新的样本数少于原仓库。
- 同时我们会根据 `val auc` 选 `best checkpoint`，而原仓库更像“训完即用”。
- 这会改变选择器的最优点，属于中等影响差异。

### 关键差异 2：训练轮数控制不同

- 论文和原仓库更接近“单轮扫过固定条数”
- 我们当前实现默认 `epochs=1`

证据：

- [train_selector.py](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro/train_selector.py#L47)

影响判断：

- 如果数据顺序与抽样方式不同，`1 epoch` 下最终参数会比原仓库更敏感。

## 4. LoRA SFT 对照

### 结论

这是当前偏差最大的部分，也是最可能影响最终结论的部分，风险高。

## 4.1 模型主线已偏离论文主表

论文 `Alpaca_GPT4` 主表对应的是：

- `Llama2-7B`
- `Llama-Factory`

而我们当前主线是：

- `Qwen3-8B`
- 自写 `PEFT + Trainer`

对应位置：

- 训练脚本：
  - [train_lora.py](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro/train_lora.py#L216)
  - [train_lora.py](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro/train_lora.py#L234)

影响判断：

- 这意味着我们当前实验从一开始就不是论文 Table 1 的严格复现。
- 这不代表实验没有价值，但它更像“方法迁移验证”，而不是“论文主表对齐”。

## 4.2 `Full` 组超参和其他组不一致

论文 4.3 明确写的是：

- `LoRA rank = 16`
- `LoRA alpha = 8`
- `3 epochs`
- `learning rate = 5e-5`
- `batch size = 16`
- `max length = 2048`

我们当前 `Random / CNN Top / CNN Bottom` 基本符合这个口径，但 `Full` 组不一致：

- `Full seed 1`：`lora_alpha=16`
- 其他组：`lora_alpha=8`

证据：

- [full run_config.json](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/lora/full/seed_1/run_config.json)
- [random run_config.json](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/lora/random_k5000/seed_1/run_config.json)
- [cnn top run_config.json](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/lora/cnn_top_k5000/seed_1/run_config.json)
- [cnn bottom run_config.json](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/lora/cnn_bottom_k5000/seed_1/run_config.json)

影响判断：

- 这是高影响问题。
- `Full vs Base`
- `CNN Top-K vs Full`
- `Random-K vs Full`

这几类比较都会被污染。

## 4.3 当前 SFT 标签构造不够标准

我们当前 `SFT` 数据构造流程是：

1. `user_prompt = apply_chat_template(user_only, add_generation_prompt=True)`
2. `full_prompt = apply_chat_template(full_dialog, add_generation_prompt=True)`
3. 用 `user_prompt` 长度去 mask `labels[:start_idx]`
4. 剩余 token 全部参与监督

证据：

- [common.py](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro/common.py#L160)
- [common.py](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro/common.py#L377)

这意味着什么：

- `full_dialog` 末尾如果额外追加了 assistant 起始标记
- 这个额外标记也会被纳入 `labels`
- 模型会被训练去预测“答案后继续起一个 assistant 块”

影响判断：

- 这是高影响实现偏差。
- 它会影响所有组，而不只是 `Full`。
- 这会直接改变 `SFT` 监督目标，可能压低整体训练质量，也会让不同子集之间的真实差异更难显现。

## 4.4 超参数里哪些是对齐的

除去上面两点，其他大项大体是对齐的：

- `epochs=3`
- `learning_rate=5e-5`
- `max_length=2048`
- `LoRA rank=16`
- 有效 batch size 约为 `1 * 16 = 16`

对应位置：

- [train_lora.py](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro/train_lora.py#L129)
- [train_lora.py](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro/train_lora.py#L234)

## 5. TruthfulQA 低分是否像“打分坏了”

### 结论

目前没有发现“答案抽取脚本坏掉”的证据，但存在“任务模板与模型模式不完全匹配”的风险。

### 已确认的事实

- 现在的 `TruthfulQA MC1` 是 `lm-eval` 标准多选对数似然评测，不是我们自写自由生成打分。
- 样例文件中可以看到：
  - 每个候选答案单独做 loglikelihood
  - 不是靠字符串抽答案

样例证据：

- [samples_truthfulqa_mc1_2026-04-29T21-17-31.610048.jsonl](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/eval/base_lm_eval_vllm/__home__qjh__llm_learning__base_model__qwen3_8B/samples_truthfulqa_mc1_2026-04-29T21-17-31.610048.jsonl)

### 当前更像什么问题

样例 prompt 末尾带有空的：

- `<think>\n\n</think>`

这说明 `Qwen3` 的 chat template / 模型模式与 `truthfulqa_mc1` 的标准 loglikelihood 口径之间可能存在适配问题。

影响判断：

- 这更像“评测模板兼容性问题”或者“模型模式差异”，而不是“我们答案抽取器写坏了”。
- 所以当前 `TruthfulQA` 低分值得谨慎解释，但不能直接据此说 `lm-eval` 打分失效。

## 6. 最终判断

### 哪些部分基本没问题

- `CNN` 网络结构
- hidden-state 主体抽取逻辑
- 选择器大框架

### 哪些部分是高影响问题

- `Full` 组 `lora_alpha` 配置错误
- `SFT full_prompt` 仍使用 `add_generation_prompt=True`
- 当前模型/框架主线已偏离论文主表口径

### 对当前结论的影响

- 现在的结果不能直接支持论文结论。
- 现在的结果也不能干净地反驳论文结论。
- 更准确的说法是：当前实现下，`CNN` 没有显示出预期优势；但由于 `SFT` 段存在高影响偏差，我们还不能把责任直接归到 `CPQS` 方法本身。

## 7. 建议的修复优先级

### P0：必须先修

1. 统一所有组 `LoRA alpha=8`
2. 修正 `SFT` prompt 构造：
   - `user_prompt` 保留 `add_generation_prompt=True`
   - `full_prompt` 不再额外追加 generation prompt
3. 训练前先做 token 级 smoke test，确认监督区间只覆盖真实答案

### P1：如果要做更严格论文复现

1. `Alpaca_GPT4` 主线切回论文同款模型
2. 尽量使用 `Llama-Factory` 或严格对齐其训练配置
3. 评测保持论文同类 benchmark 口径

### P2：如果转向数学线

优先做 `GSM8K` 单任务训练和评测，先验证：

- `Full` 是否显著优于 `Base`
- `CNN Top-K` 是否优于 `Random-K`

具体方案见：

- [GSM8K_EXPERIMENT_PLAN.md](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro/GSM8K_EXPERIMENT_PLAN.md)
