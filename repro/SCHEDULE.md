# CPQS 当前排程

最后更新：2026-04-29 15:48 CST

## 一、目标划分

当前工作分成两条线并行推进：

### 主线 A：先闭合最小实验闭环

- 完成 `Full seed 1` 训练
- 完成 `Full seed 1` 评测
- 生成完整的第一轮结果表

### 主线 B：提前推进第二波 seed 扩展

- `Random-K seed 2`
- `CNN Top-K seed 2`
- `CNN Bottom-K seed 2`
- 后续再接 `seed 3`

## 二、当前显卡排程

### GPU0

当前任务：

1. `Full seed 1` 训练继续跑
2. `CNN Top-K seed 2` 评测

后续接力：

1. `Full seed 1` 训练结束后，立即跑 `Full seed 1` 评测
2. 再根据空闲情况补 `Random-K seed 3` 或 `CNN Top-K seed 3`

### GPU1

当前任务：

1. `Random-K seed 2` 评测
2. `CNN Bottom-K seed 2` 训练

后续接力：

1. `CNN Bottom-K seed 2` 训练结束后，立刻跑它的评测
2. 再补 `CNN Bottom-K seed 3` 或 `Random-K seed 3`

## 三、当前优先级

按优先级排序：

1. `Full seed 1` 训练完成
2. `Full seed 1` 评测完成
3. `seed2` 这三条线尽快闭环
4. 启动 `seed3` 三组训练
5. 等 seed 数量足够后刷新 mean/std 表

## 四、时间预估

以下是当前保守估计，不是严格保证：

- `Full seed 1` 剩余训练：
  - 仍是当前最慢的一项
  - 从现有进度看，大约还需要数小时
- 单个 `5000` 子集 LoRA 训练：
  - 大约 `1.5-2` 小时
- 单个 adapter 的四 benchmark 评测：
  - 大约 `2.5-4` 小时
- 聚合表格与文档更新：
  - 大约 `10-20` 分钟

## 五、下一批待启动任务

当前未启动的后续任务：

- `Random-K seed 3` 训练
- `CNN Top-K seed 3` 训练
- `CNN Bottom-K seed 3` 训练

待 `CNN Bottom-K seed 2` 训练完成后要接的任务：

- `CNN Bottom-K seed 2` 评测

待 `Full seed 1` 训练完成后要接的任务：

- `Full seed 1` 评测

## 六、排程原则

当前默认遵循以下原则：

- 每张卡至少维持 `2` 个真实任务
- 优先不打断健康长任务
- 优先把已训练完成的 adapter 尽快送去评测
- 评测与训练尽量穿插，以便尽早拿到可比较结果
- 文档与结果表持续更新，并优先使用中文
