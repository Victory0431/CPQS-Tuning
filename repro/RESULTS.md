# CPQS 当前结果汇总

最后更新：2026-04-29 15:48 CST

## 一、说明

以下表格基于当前已经**完整完成评测**的运行结果整理：

- `Base`
- `Random-K seed 1`
- `CNN Top-K seed 1`
- `CNN Bottom-K seed 1`

当前仍未纳入表格的关键项：

- `Full seed 1` 训练与评测尚未完成
- `seed2 / seed3` 的大部分评测还在进行中

对应 CSV 文件：

- 原始分数表：
  - `repro_outputs/tables/per_run_scores.csv`
- 分组 mean/std 表：
  - `repro_outputs/tables/group_mean_std.csv`

## 二、每个运行的原始分数表

| 组别 | seed | GSM8K | MATH-500 | ARC-Challenge | MMLU subset | 状态 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Base | 1 | 0.3594 | 0.1220 | 0.2432 | 0.2544 | 已完成 |
| Random-K | 1 | 0.8453 | 0.4700 | 0.3174 | 0.2895 | 已完成 |
| CNN Top-K | 1 | 0.7892 | 0.2220 | 0.2654 | 0.2544 | 已完成 |
| CNN Bottom-K | 1 | 0.8544 | 0.4340 | 0.2858 | 0.2544 | 已完成 |

## 三、按组汇总的 mean/std 表

当前每组只有 1 个完整 run，因此 `std = 0.0000`。

| 组别 | GSM8K mean±std | MATH-500 mean±std | ARC-Challenge mean±std | MMLU subset mean±std | 完整 run 数 |
| --- | --- | --- | --- | --- | ---: |
| Base | 0.3594 ± 0.0000 | 0.1220 ± 0.0000 | 0.2432 ± 0.0000 | 0.2544 ± 0.0000 | 1 |
| Random-K | 0.8453 ± 0.0000 | 0.4700 ± 0.0000 | 0.3174 ± 0.0000 | 0.2895 ± 0.0000 | 1 |
| CNN Top-K | 0.7892 ± 0.0000 | 0.2220 ± 0.0000 | 0.2654 ± 0.0000 | 0.2544 ± 0.0000 | 1 |
| CNN Bottom-K | 0.8544 ± 0.0000 | 0.4340 ± 0.0000 | 0.2858 ± 0.0000 | 0.2544 ± 0.0000 | 1 |

## 四、当前可直接观察到的对比

### 1. CNN Top-K vs Random-K

| 数据集 | CNN Top-K | Random-K | 差值（Top - Random） |
| --- | ---: | ---: | ---: |
| GSM8K | 0.7892 | 0.8453 | -0.0561 |
| MATH-500 | 0.2220 | 0.4700 | -0.2480 |
| ARC-Challenge | 0.2654 | 0.3174 | -0.0520 |
| MMLU subset | 0.2544 | 0.2895 | -0.0351 |

### 2. CNN Bottom-K vs Random-K

| 数据集 | CNN Bottom-K | Random-K | 差值（Bottom - Random） |
| --- | ---: | ---: | ---: |
| GSM8K | 0.8544 | 0.8453 | +0.0091 |
| MATH-500 | 0.4340 | 0.4700 | -0.0360 |
| ARC-Challenge | 0.2858 | 0.3174 | -0.0316 |
| MMLU subset | 0.2544 | 0.2895 | -0.0351 |

### 3. Base vs Random-K

| 数据集 | Base | Random-K | 差值（Random - Base） |
| --- | ---: | ---: | ---: |
| GSM8K | 0.3594 | 0.8453 | +0.4860 |
| MATH-500 | 0.1220 | 0.4700 | +0.3480 |
| ARC-Challenge | 0.2432 | 0.3174 | +0.0742 |
| MMLU subset | 0.2544 | 0.2895 | +0.0351 |

## 五、当前还缺的关键结果

以下关键比较仍未闭环：

- `Full vs Base`
- `CNN Top-K vs Full`
- 三 seed 条件下的 `mean/std` 稳定比较

缺失原因：

- `Full seed 1` 训练与评测尚未结束
- `seed2 / seed3` 还在推进中

## 六、下一次更新时将补充的内容

下一个版本会补：

- `Full seed 1` 的完整评测结果
- `Random-K seed 2`
- `CNN Top-K seed 2`
- `CNN Bottom-K seed 2`
- 若 `seed3` 也跑出来，则刷新真正有统计意义的 `mean/std`
