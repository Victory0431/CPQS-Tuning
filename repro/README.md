# CPQS 第一轮复现实验说明

本目录用于维护当前这轮 CPQS 复现实验的独立流程，目标是先跑通一个最小闭环，并把每一步的日志、产物、结果表和排程记录清楚。

当前第一轮固定比较组为：

- `Base`
- `Full SFT`
- `Random-K (K=5000)`
- `CNN Top-K (K=5000)`
- `CNN Bottom-K (K=5000)`

当前第一轮固定评测集为：

- `GSM8K`
- `MATH-500`
- `ARC-Challenge`
- `MMLU subset`

评分方式固定为：

- 全部使用脚本自动评分
- 不使用 LLM judge

## 设计原则

### 1. 保持与论文核心工艺一致

选择器部分保持原论文代码思路，尤其是隐藏状态提取方式：

- 使用与目标模型一致的基础模型
- 分别构造 `user-only` 和 `user+assistant` 提示
- 通过 `user-only` 提示的 token 长度定位 assistant 响应起点
- 仅保留 assistant 响应区间的隐藏状态
- 将响应隐藏状态送入 CNN 选择器

### 2. 先跑最小闭环

为了尽快拿到可以比较的结果，当前这一轮先不扩展到论文中的全部评测协议，而是优先完成：

- `Base vs Full`
- `Random-K vs CNN Top-K`
- `Random-K vs CNN Bottom-K`
- `CNN Top-K vs Full`

### 3. 全流程写日志

长任务默认都写带时间戳的日志，便于：

- 追踪训练和评测进度
- 排查异常退出
- 回顾每轮实验的实际耗时

## 目录中的主要脚本

- `train_selector.py`
  - 训练 CNN 选择器
- `score_candidates.py`
  - 对候选指令数据打分
- `build_subsets.py`
  - 生成 `Full / Random / Top / Bottom` 训练子集
- `train_lora.py`
  - 运行 LoRA SFT
- `evaluate_round1.py`
  - 对四个 benchmark 自动评测
- `aggregate_results.py`
  - 聚合生成原始分数表与均值方差表
- `configs/round1_experiment.json`
  - 第一轮实验的固定配置

## 当前主要输出目录

- `repro_outputs/selector_round1`
  - 选择器训练产物
- `repro_outputs/scored_alpaca`
  - 候选数据打分结果
- `repro_outputs/subsets_round1`
  - 第一轮训练子集
- `repro_outputs/lora`
  - LoRA 训练产物
- `repro_outputs/eval`
  - 自动评测结果
- `repro_outputs/tables`
  - 聚合表格
- `repro_outputs/logs`
  - 全部日志

## 结果文件

当前聚合后的表格会写到：

- `repro_outputs/tables/per_run_scores.csv`
- `repro_outputs/tables/group_mean_std.csv`

文档版结果汇总见：

- [RESULTS.md](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro/RESULTS.md)

## 常用命令顺序

### 1. 训练选择器

```bash
conda activate cpqs-tuning
cd /home/qjh/llm_learning/CPQS_lab/CPQS-Tuning

python -m repro.train_selector \
  --model_path /home/qjh/llm_learning/base_model/qwen3_8B \
  --pos_train_path /home/qjh/llm_learning/CPQS_lab/data/raw_data/alpaca_gpt4_data.json \
  --neg_dataset1_path /home/qjh/llm_learning/CPQS_lab/data/raw_data/alpaca_gpt4_data_llama.json \
  --neg_dataset2_path /home/qjh/llm_learning/CPQS_lab/data/raw_data/alpaca_gpt4_data_qwen251.json \
  --output_dir /home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/selector_round1 \
  --backbone qwen \
  --use_layers all \
  --use_part full \
  --device_cnn cuda:0 \
  --device_llm cuda:1 \
  --wandb_run_name selector-round1
```

### 2. 对候选数据打分

```bash
python -m repro.score_candidates \
  --model_path /home/qjh/llm_learning/base_model/qwen3_8B \
  --cnn_checkpoint /home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/selector_round1/checkpoints/best_selector.pth \
  --predict_data /home/qjh/llm_learning/CPQS_lab/data/candidate_data/alpaca_gpt4_data.json \
  --output_dir /home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/scored_alpaca \
  --backbone qwen \
  --use_layers all \
  --use_part full \
  --device_cnn cuda:0 \
  --device_llm cuda:1
```

### 3. 构造训练子集

```bash
python -m repro.build_subsets \
  --scored_candidates /home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/scored_alpaca/scored_candidates.json \
  --output_dir /home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/subsets_round1 \
  --k 5000 \
  --random_seeds 1 2 3
```

### 4. 训练 LoRA

```bash
python -m repro.train_lora \
  --model_path /home/qjh/llm_learning/base_model/qwen3_8B \
  --train_data /home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/subsets_round1/random_5000_seed_1.json \
  --output_dir /home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/lora/random_k5000/seed_1 \
  --group_name random_k5000 \
  --seed 1
```

`Full / CNN Top / CNN Bottom` 组只需要更换 `train_data`、`output_dir` 和 `group_name`。

### 5. 自动评测

```bash
python -m repro.evaluate_round1 \
  --model_path /home/qjh/llm_learning/base_model/qwen3_8B \
  --adapter_path /home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/lora/random_k5000/seed_1/final_adapter \
  --output_dir /home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/eval/random_k5000_seed1 \
  --group_name random_k5000 \
  --seed 1 \
  --benchmarks_root /home/qjh/llm_learning/CPQS_lab/data/benchmarks \
  --mmlu_path "/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/eval/Open LLM Leaderboard/MMLU/mmlu_data/mmlu_test.json"
```

### 6. 聚合结果

```bash
python -m repro.aggregate_results \
  --results_root /home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/eval \
  --output_dir /home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/tables
```

## 当前文档约定

从现在开始，本目录下所有实验文档统一使用中文撰写，并尽量包含以下内容：

- 更新时间
- 当前运行任务
- 已完成结果
- 表格化对比
- 后续排程
- 日志与产物路径
