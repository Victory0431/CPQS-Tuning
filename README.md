# CPQS-Tuning
This is the repo for CPQS-Tuning: A Model Self-Perception-Based Data Filtering Algorithm for Efficient Instruction Fine-Tuning, which introduces a novel metric—the Contrastive Perception Quality Score (CPQS)—to identify and select the highest-quality subset of instruction-tuning data directly from a  LLM’s hidden states.

The repo contains:

- **CPQS Scoring & Filtering Code**

Scripts to extract hidden representations, compute CPQS for each example, and filter out the top k % of data.

- **Training & Evaluation Scripts**

Example pipelines demonstrating how to integrate CPQS-Tuning into your LoRA/PEFT workflows.




## **Overview**

**CPQS-tuning**

Instruction fine-tuning is essential to unlocking the full potential of large language models (LLMs), yet it often suffers from low-quality and redundant data. Traditional filtering methods rely on external evaluation models or hand-crafted metrics, which not only add computation overhead but may also misalign with the model’s own learning dynamics. In **CPQS-Tuning**, we introduce a **self-perception-based filtering** approach: the **Contrastive Perception Quality Score (CPQS)** leverages the hidden states of an LLM to intrinsically assess instruction-data quality. By selecting under 10 % of the original Alpaca\_GPT4 and DeepSeek-R1 datasets based on CPQS, our filtered subset not only matches but **outperforms** models trained on the full datasets and surpasses current state-of-the-art data-selection techniques.

Extensive experiments in both general and downstream domains—covering benchmarks such as GSM8K, HumanEval, and HumanEval-Plus—demonstrate that CPQS-Tuning delivers an **average performance gain of over 3.6 %** against leading filtering algorithms. This project provides all necessary code, data-selection scripts, and pretrained checkpoints to reproduce our results and facilitate further research into efficient, self-aware instruction fine-tuning.

The figure below illustrates the flowchart of our method.

<img width="1204" alt="Method_diagram (1)" src="https://github.com/user-attachments/assets/7ce8b4b8-9d1e-4758-8188-69de305834da" />

## **Install**




百度网盘：链接: https://pan.baidu.com/s/1J9WhQorhaLhBb84LckQQtA?pwd=9c62 提取码: 9c62 

hugginface ：https://huggingface.co/datasets/renlll/CPQS_data/tree/main/data
