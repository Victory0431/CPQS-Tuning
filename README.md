# CPQS-Tuning

这是论文 **CPQS-Tuning: A Model Self-Perception-Based Data Filtering Algorithm for Efficient Instruction Fine-Tuning** 的代码仓库。该方法提出了新的数据质量指标 **Contrastive Perception Quality Score (CPQS)**，直接利用大语言模型的隐藏状态来评估指令微调数据质量，并据此筛选高质量子集。

仓库主要包含两部分内容：

- **CPQS 打分与筛选代码**
  - 用于提取隐藏表示、计算每条样本的 CPQS 分数，并按分数筛选数据。
- **训练与评测脚本**
  - 用于把 CPQS-Tuning 接入 LoRA / PEFT 微调流程，并完成效果验证。

## 项目概述

指令微调是释放大语言模型能力的重要步骤，但训练数据中常常存在低质量样本和冗余样本。传统数据筛选方法通常依赖外部评价模型或人工设计指标，不仅带来额外计算开销，也可能与模型自身的学习动态不一致。

**CPQS-Tuning** 的核心思想是：让模型利用自己的隐藏状态来“感知”数据质量。具体来说，**CPQS** 通过对比模型对不同样本的内部表征，给出一个内生的数据质量分数。论文结果显示，在 Alpaca\_GPT4 和 DeepSeek-R1 数据上，仅选择不足 10% 的高质量子集，就可以达到甚至超过使用全量数据训练的效果，并优于已有主流数据筛选方法。

项目提供了复现实验所需的主要代码、数据筛选脚本与部分中间结果，便于继续开展高效、自感知指令微调研究。

下图展示了方法流程：

<img width="1204" alt="Method_diagram (1)" src="https://github.com/user-attachments/assets/7ce8b4b8-9d1e-4758-8188-69de305834da" />

## 数据准备

训练 CNN 选择器和验证算法效果所需的数据可从以下位置获取：

- 训练数据位于 `raw_data` 目录
- 验证数据位于 `train_data` 目录

下载地址：

- **百度网盘**
  - 链接：<https://pan.baidu.com/s/1J9WhQorhaLhBb84LckQQtA?pwd=9c62>
  - 提取码：`9c62`
- **Hugging Face**
  - <https://huggingface.co/datasets/renlll/CPQS_data/tree/main/data>

## 环境配置

可以按以下步骤准备运行环境：

```bash
# 克隆仓库
git clone https://github.com/renllll/CPQS-Tuning.git

# 进入项目目录
cd CPQS-Tuning

# 创建 conda 环境
conda create -n cpqs-tuning python=3.10

# 激活环境
conda activate cpqs-tuning

# 安装依赖
pip install -r requirements.txt
```

## 运行说明

### 1. 训练 CNN 选择器

```bash
python train_cnn.py
```

需要重点检查的 `parse_args` 参数：

- `--model_path`：基础大模型权重路径
- `--pos_train_path`：正样本 JSON 路径
- `--neg_dataset1_path`：第一份负样本 JSON 路径
- `--neg_dataset2_path`：第二份负样本 JSON 路径
- `--local_data_path`：本地缓存或预合并数据路径
- `--backbone`：骨干模型类型，支持 `qwen` 或 `llama`

### 2. 预测数据质量

```bash
python predict.py
```

需要重点检查的 `parse_args` 参数：

- `--model_path`：基础大模型权重路径
- `--cnn_checkpoint`：训练好的 TextCNN `.pth` 权重路径
- `--predict_data`：待打分 JSON 数据路径
- `--output_path`：预测结果保存路径
- `--failed_path`：失败样本保存路径
- `--backbone`：骨干模型类型，支持 `qwen` 或 `llama`

运行后，每条样本会得到一个 `CPQS_score`。

### 3. 筛选数据集

```bash
python process_cpqs.py \
  -i predict.json \
  -o sorted_full.json \
  -n 1000
```

参数说明：

- `--input_file` / `-i`：原始 JSON 路径
- `--output_file` / `-o`：完整排序后的 JSON 输出路径
- `--top_n` / `-n`：可选参数。如果 `N > 0`，会额外生成一个 `*_top_N.json` 文件，只保留前几项核心字段，便于下游微调

运行完成后，即可得到筛选后的高质量子集。

## 复现实验文档

本仓库当前额外维护了一套中文复现实验文档，位于 `repro/` 目录，主要包括：

- [README.md](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro/README.md)
- [STATUS.md](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro/STATUS.md)
- [SCHEDULE.md](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro/SCHEDULE.md)
- [RESULTS.md](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro/RESULTS.md)

这些文档会持续用中文记录当前复现进度、排程、结果表格和后续任务。
