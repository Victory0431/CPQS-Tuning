# 后续运行注意事项

最后更新：2026-04-29 20:55 CST

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

- `vLLM` 单条推理 smoke
- `lm-eval + vLLM` 小批量 smoke

确认以下几件事：

- chat template 没有失效
- raw output 没有混入 prompt
- benchmark 名称与数据下载无误
- 日志持续刷新，不是长时间静默
- 样例文件可人工核查

## 4. Qwen3 当前优先使用 non-thinking

在当前自动 benchmark 主线上：

- `enable_thinking=false` 更稳定
- `enable_thinking=true` 更容易因为输出过长而截断

如果后续一定要测 `thinking`，至少要重新确认：

- `max_gen_toks` 是否足够
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

## 8. `vLLM` 评测环境必须固定版本

不要直接信任 `pip` 自动解析出的最新版本组合。

本轮已经踩到的坑：

- `vllm 0.8.5.post1`
- `transformers 5.7.0`
- `tokenizers 0.22.2`

这个组合会在 `Qwen3` 上报错：

- `AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended`

当前已验证可用的组合是：

- `vllm 0.8.5.post1`
- `lm-eval 0.4.9.1`
- `transformers 4.51.3`
- `tokenizers 0.21.4`

后续如果重建环境，优先直接固定这组版本，不要重新盲装一次再排错。

## 9. 正式评测前要清理失效代理变量

本轮还遇到一个隐性减速点：

- 进程继承了失效的本地代理
- `lm-eval` 访问 Hugging Face 元数据时会先尝试连接 `127.0.0.1:7890`
- 失败后才回退到缓存

表现为：

- 任务没有挂
- GPU 已经占内存
- 但日志里反复出现 `MaxRetryError` 和 `ProxyError`
- 整体会被大量无效重试拖慢

处理方式：

- 在正式评测脚本里清理：
  - `http_proxy`
  - `https_proxy`
  - `HTTP_PROXY`
  - `HTTPS_PROXY`
  - `all_proxy`
  - `ALL_PROXY`

如果后续又看到类似 `127.0.0.1:7890 refused` 的日志，优先检查代理环境变量，而不是先怀疑模型或 benchmark。

## 10. `Full` 组 LoRA 超参必须和子集组完全一致

本轮已经确认一个高影响错误：

- `Full seed 1` 实际使用了 `lora_alpha=16`
- `Random-K / CNN Top-K / CNN Bottom-K` 使用的是 `lora_alpha=8`

这会直接污染：

- `Full vs Base`
- `Full vs Random-K`
- `Full vs CNN Top-K`

后续任何正式对比前，必须逐个检查每个 run 的：

- `epochs`
- `learning_rate`
- `max_length`
- `lora_rank`
- `lora_alpha`
- prompt 模板

只有训练数据子集允许不同。

## 11. `SFT` 监督区间必须只覆盖真实答案

本轮还确认了另一个高影响问题：

- `user_prompt` 用 `add_generation_prompt=True` 是合理的
- 但 `full_prompt` 如果也继续用 `add_generation_prompt=True`
- 就可能在答案后额外追加 assistant 起始标记
- 这些额外 token 会一起进入 `labels`

结果是：

- 模型不只是学答案
- 还会被训练去预测“答案后再起一个 assistant 块”

后续任何新训练启动前，必须先做 token 级 smoke test，至少核对：

- `user_prompt`
- `full_prompt`
- `start_idx`
- `labels` 解码后的可见文本

如果监督区间里包含了答案外的模板残片，这轮训练就不应继续。

## 12. 文档每次修改后都要提交并推送仓库

从现在开始，这条规则必须固定执行：

- 只要修改了仓库中的文档
- 就不要只停留在本地工作区
- 必须完成：
  - `git status` 自检
  - `git add`
  - `git commit`
  - `git push origin`

适用文档包括但不限于：

- `RESULTS.md`
- `SCHEDULE.md`
- `ATTENTION_FOR_FUTURE_RUNS.md`
- `README.md`
- 各类实验方案、审计说明、状态说明

原因：

- 需要让用户随时能从远程仓库看到最新状态
- 需要给后续模型保留连续留痕
- 避免出现“本地文档已更新，但远程仓库仍是旧状态”的信息错位

如果当前还有代码改动未准备好一起推送，也至少要明确说明：

- 哪些文档已改
- 为什么暂时未推送
- 预计何时补推
