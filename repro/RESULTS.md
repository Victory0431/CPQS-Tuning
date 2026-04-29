# CPQS 当前结果汇总

最后更新：2026-04-29 19:36 CST

## 一、当前哪些结果是可信的

`2026-04-29` 已确认旧版评测脚本存在输出切分与答案抽取问题，因此：

- 旧 `Base` 分数已作废，不再引用
- 旧脚本下跑出的 `Random-K / CNN Top-K / CNN Bottom-K` 评测结果暂时只保留作历史记录
- 当前**唯一可以直接作为可信基线引用**的是修复版 `Base v2 non-thinking`

修复版结果目录：

- [base_v2_nonthinking](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/eval/base_v2_nonthinking)

修复版日志：

- [base_eval_v2_nonthinking.log](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/logs/base_eval_v2_nonthinking.log)

本地结果表也已同步替换：

- `repro_outputs/tables/per_run_scores.csv`
- `repro_outputs/tables/group_mean_std.csv`

## 二、修复版 Base 基线

| 组别 | seed | GSM8K | MATH-500 | ARC-Challenge | MMLU subset | 状态 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Base v2 non-thinking | 1 | 0.9310 | 0.4680 | 0.8942 | 0.7083 | 已完成 |

对应原始 JSON：

- [run_scores.json](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/eval/base_v2_nonthinking/run_scores.json)

## 三、与论文基线的对比

这里必须分成两层来看，不能混成一句“对齐/不对齐”。

### 1. 当前这组 `GSM8K / MATH-500` 可以参考论文 Table 2 的 `Reasoning-DeepSeek Base`

论文 Table 2 报告的 `Qwen2.5-7B-Instruct Base` 为：

- `GSM8K = 76.27`
- `Math 500 = 73.40`

我们当前修复版 `Qwen3-8B non-thinking Base` 为：

- `GSM8K = 93.10`
- `MATH-500 = 46.80`

简单数值差：

- `GSM8K`: `+16.83`
- `MATH-500`: `-26.60`

但这**不是严格 apples-to-apples 对比**，原因有三点：

- 论文这里的模型是 `Qwen2.5-7B-Instruct`，我们现在是 `Qwen3-8B`
- 论文这里对应的是 `Reasoning-DeepSeek` 路线，而我们当前训练候选池是 `Alpaca-GPT4`
- 论文对 `Math 500` 的说明里提到生成长度设置会明显影响结果

因此，这一组对比只能用来说明“当前脚本已经不再是明显坏掉的状态”，不能当作严格复现实验结论。

### 2. 当前这组 `ARC / MMLU subset` 不能直接和论文 Table 1 的“Base”对比

原因很简单：论文 `Alpaca-GPT4` 的 Table 1 **没有单独报告 base model 行**，它报告的是：

- `Self` 不同子集大小
- `Superfiltering`
- `MoDs`
- `Alpagasus`
- `Full (52k)`

而且 Table 1 用的是：

- `Llama2-7B`
- `MMLU / ARC / TruthfulQA / HellaSwag / AlpacaEval`

所以：

- 当前 `ARC-Challenge = 89.42`
- 当前 `MMLU subset = 70.83`

不能直接写成“高于/低于论文 base”，因为论文在这条线里没有对应的 base 行。

## 四、论文口径上的重要修正

如果我们继续沿用 `Alpaca-GPT4` 作为候选池，那么论文里自动评测、且**不需要 LLM judge** 的第一优先 benchmark 应该是：

- `MMLU`
- `ARC-Challenge`
- `HellaSwag`
- `TruthfulQA`

而不是当前这套：

- `GSM8K`
- `MATH-500`
- `ARC-Challenge`
- `MMLU subset`

也就是说，当前修复版 `Base v2` 虽然已经把脚本错误修好了，但它仍然属于：

- “链路修复后的可用基线”
- 不是“论文 Alpaca-GPT4 主表 1 的最终复现基线”

## 五、旧 adapter 结果的处理方式

下面这些历史结果是在旧评测脚本下得到的：

- `Random-K seed 1`
- `CNN Top-K seed 1`
- `CNN Bottom-K seed 1`

它们暂时不应该与修复版 `Base v2` 混合解释，也不应该再用于写结论。

在新的 Alpaca 自动评测协议 (`MMLU / ARC / HellaSwag / TruthfulQA`) 确认后，应统一使用**同一版脚本**重新评测：

- `Base`
- `Full`
- `Random-K`
- `CNN Top-K`
- `CNN Bottom-K`

## 六、当前结论

当前最稳妥的结论只有三条：

- `Base` 的旧分数确实是错的，已经被修复版 `Base v2 non-thinking` 替换
- 论文 `Alpaca-GPT4` 的自动评测口径应切换到 `MMLU / ARC / HellaSwag / TruthfulQA`
- 在 adapter 全部按新评测协议重跑之前，不再解释 `Top-K / Random-K / Bottom-K` 的优劣
