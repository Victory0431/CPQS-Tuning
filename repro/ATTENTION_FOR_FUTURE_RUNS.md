# 后续运行注意事项

最后更新：2026-04-29 19:36 CST

这份文档专门记录本轮已经踩过的坑，供后续模型和后续实验直接复用。

## 1. 不要再使用旧版 Base 分数

旧版 `Base` 评测结果已经确认无效，原因包括：

- `left padding` 下批量生成输出切分错误
- 输出里混入 prompt 残片
- 对 `<think>...</think>`、`#### ...`、`<answer>...</answer>` 处理不完整

结论：

- 旧 `Base` 分数不能再写进任何对比表
- 旧脚本下的 adapter 结果也不要再和新 `Base` 混合解释

## 2. Base 和 adapter 必须用同一版评测脚本

如果 `Base` 用新脚本、adapter 用旧脚本，那么：

- 分数不可比
- 结论无效

后续每次要做横向比较前，先确认：

- benchmark 集合一致
- prompt 一致
- decoding 参数一致
- 抽取逻辑一致
- 日志与样例落盘一致

## 3. 先做 smoke test，再开全量

每次脚本、prompt、benchmark 或 decoding 参数发生变化时，必须先跑：

- `limit=20`
- `sample_dump_count=10`

确认以下四件事：

- raw output 没有混入 prompt
- 最终答案能被正确抽取
- 日志持续刷新，不是长时间静默
- 样例文件可人工核查

## 4. Qwen3 当前优先使用 non-thinking

在当前自动 benchmark 脚本下：

- `enable_thinking=false` 更稳定
- `enable_thinking=true` 更容易因为输出过长而截断

如果后续一定要测 `thinking`，至少要重新确认：

- `max_new_tokens` 是否足够
- thinking 内容剥离是否稳定
- 最终答案抽取是否仍然准确

## 5. Alpaca-GPT4 不要再用 GSM8K / MATH-500 当主评测

如果候选池是 `Alpaca-GPT4`，第一轮自动评测应优先用：

- `MMLU`
- `ARC-Challenge`
- `HellaSwag`
- `TruthfulQA`

如果候选池是 `Reasoning-DeepSeek`，再去重点看：

- `GSM8K`
- `Math 500`
- `HumanEval`
- `GPQA`

不要把这两条路线混在一起解释。

## 6. 所有长任务都要有时间戳日志

必须保证：

- 每个长任务都有独立日志文件
- 日志里要有配置行
- 日志里要有 benchmark 级别的开始/结束记录
- 日志里要有按 batch 的进度输出
- 样例文件要单独落盘

## 7. 文档更新要和代码更新同步

每次发生以下情况时，都要同步更新仓库文档：

- 评测协议改动
- 基线被修正
- benchmark 集合变更
- 任务排程变化
- 关键实验被判定无效

否则后续很容易再次引用错误结果。
