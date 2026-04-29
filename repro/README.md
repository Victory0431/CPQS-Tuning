# CPQS 第一轮复现实验说明

本目录用于维护当前这轮 CPQS 复现实验的独立流程，目标是先把链路跑稳、把错误留痕清楚，再逐步靠近论文口径。

当前第一轮固定比较组为：

- `Base`
- `Full SFT`
- `Random-K (K=5000)`
- `CNN Top-K (K=5000)`
- `CNN Bottom-K (K=5000)`

当前文档中的评测分成两类：

- 历史最小闭环评测：
  - `GSM8K`
  - `MATH-500`
  - `ARC-Challenge`
  - `MMLU subset`
- 论文 `Alpaca-GPT4` 自动评测主线：
  - `MMLU`
  - `ARC-Challenge`
  - `HellaSwag`
  - `TruthfulQA`

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

### 2. 先把链路跑稳，再对齐论文协议

前面为了尽快发现问题，先跑过一轮最小闭环；现在已经确认：

- 旧 `Base` 评测结果有误，已修复
- `Alpaca-GPT4` 不应再以 `GSM8K / MATH-500` 作为主评测
- `Alpaca-GPT4` 的正式自动评测主线更适合切到 `lm-evaluation-harness + vLLM`

因此当前更优先的是：

- 用统一脚本重跑 `Base / Full / Random-K / CNN Top-K / CNN Bottom-K`
- 自动评测先对齐到 `MMLU / ARC / HellaSwag / TruthfulQA`

### 3. 全流程写日志

长任务默认都写带时间戳的日志，便于：

- 追踪训练和评测进度
- 排查异常退出
- 回顾每轮实验的实际耗时

## 环境划分

从 `2026-04-29` 起，训练与正式评测分环境执行：

- 训练环境：
  - `cpqs-tuning`
- 评测环境：
  - `cpqs-eval-vllm`

当前已验证可用的评测环境关键版本：

- `vllm==0.8.5.post1`
- `lm-eval==0.4.9.1`
- `torch==2.6.0+cu124`
- `transformers==4.51.3`
- `tokenizers==0.21.4`

注意：

- 不要在 `cpqs-tuning` 里直接安装 `lm-eval` 或 `vllm`
- 不要让 `cpqs-eval-vllm` 里的 `transformers` 漂移到 `5.x`

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
  - 历史最小闭环自动评测脚本
- `prepare_alpaca_benchmarks.py`
  - 准备 `HellaSwag / TruthfulQA` 本地 benchmark 数据
- `smoke_test_vllm.py`
  - 对 `Qwen3` chat template 与 `vLLM` 推理做最小单条 smoke test
- `run_lm_eval_vllm.py`
  - 使用 `lm-evaluation-harness + vLLM` 运行正式评测，并写时间戳日志
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
  - 历史自动评测结果
- `repro_outputs/eval_lm_eval_smoke`
  - `lm-eval + vLLM` 小批量 smoke 结果
- `repro_outputs/smoke`
  - `vLLM` 单条推理 smoke 输出
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

评测协议说明见：

- [ALPACA_EVAL_PROTOCOL.md](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro/ALPACA_EVAL_PROTOCOL.md)

错误复盘与注意事项见：

- [ATTENTION_FOR_FUTURE_RUNS.md](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro/ATTENTION_FOR_FUTURE_RUNS.md)

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

### 5. 历史最小闭环自动评测

```bash
python -m repro.evaluate_round1 \
  --model_path /home/qjh/llm_learning/base_model/qwen3_8B \
  --adapter_path /home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/lora/random_k5000/seed_1/final_adapter \
  --output_dir /home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/eval/random_k5000_seed1 \
  --group_name random_k5000 \
  --seed 1 \
  --benchmarks_root /home/qjh/llm_learning/CPQS_lab/data/benchmarks \
  --mmlu_path "/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/eval/Open LLM Leaderboard/MMLU/mmlu_data/mmlu_test.json" \
  --benchmarks mmlu_full arc_challenge hellaswag truthfulqa_mc1 \
  --enable_thinking false \
  --do_sample true \
  --temperature 0.7 \
  --top_p 0.8 \
  --top_k 20 \
  --sample_dump_count 30
```

### 6. `vLLM` 单条推理 smoke

```bash
conda activate cpqs-eval-vllm
cd /home/qjh/llm_learning/CPQS_lab/CPQS-Tuning

python -m repro.smoke_test_vllm \
  --model_path /home/qjh/llm_learning/base_model/qwen3_8B \
  --output_path /home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/smoke/vllm_smoke_base.json \
  --log_path /home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/logs/vllm_smoke_base.log \
  --gpu 1 \
  --max_model_len 2048 \
  --temperature 0
```

### 7. `lm-eval + vLLM` 自动评测

```bash
conda activate cpqs-eval-vllm
cd /home/qjh/llm_learning/CPQS_lab/CPQS-Tuning

python -m repro.run_lm_eval_vllm \
  --model_path /home/qjh/llm_learning/base_model/qwen3_8B \
  --output_dir /home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/eval/base_lm_eval_vllm \
  --log_path /home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/logs/base_lm_eval_vllm.log \
  --tasks mmlu arc_challenge hellaswag truthfulqa_mc1 \
  --gpu 1 \
  --batch_size auto:4 \
  --max_batch_size 64 \
  --max_model_len 2048 \
  --max_gen_toks 2048 \
  --temperature 0 \
  --top_p 1.0 \
  --hf_offline
```

### 8. 聚合结果

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
