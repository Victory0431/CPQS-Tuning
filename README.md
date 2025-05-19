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

## **Data Preparation**

The datasets for training the CNN model and for validating our algorithm’s performance are available at the following locations. The **training set** is located in the `raw_data` folder, and the **validation set** is in the `train_data` folder:

* **Baidu Netdisk**
  Link: [https://pan.baidu.com/s/1J9WhQorhaLhBb84LckQQtA?pwd=9c62](https://pan.baidu.com/s/1J9WhQorhaLhBb84LckQQtA?pwd=9c62)
  Extract code: `9c62`

* **Hugging Face**
  [https://huggingface.co/datasets/renlll/CPQS\_data/tree/main/data](https://huggingface.co/datasets/renlll/CPQS_data/tree/main/data)



## **Environment Setup**

Since our algorithm is straightforward, you can set up the environment by running the following commands:

```bash
# Clone the repository
git clone https://github.com/renllll/CPQS-Tuning.git

# Enter the project directory
cd CPQS-Tuning

# Create a new conda environment with Python 3.10
conda create -n cpqs-tuning python=3.10

# Activate the environment
conda activate cpqs-tuning

# Install required packages
pip install -r requirements.txt
```

## **Run Code**
1.训练CNN模型

python train_cnn.py

你需要修改parse_args中的参数

--model_path: Path to the LLM checkpoint  
--pos_train_path: Path to positive-sample JSON file  
--neg_dataset1_path: Path to first negative-sample JSON file  
--neg_dataset2_path: Path to second negative-sample JSON file  
--local_data_path: Path to cached merged dataset  用于加载已保存的训练数据
--backbone: Backbone type (qwen or llama)  你可以选择哪种模型的提示模板

2.预测数据质量

python predict.py

你需要修改parse_args中的参数

--model_path: Path to the LLM checkpoint  
--cnn_checkpoint: Path to the trained TextCNN .pth file  
--predict_data: Path to the JSON file for inference  
--output_path: Path to save prediction results (JSON)  
--failed_path: Path to save failed predictions (JSON)  
--backbone: Backbone type (“qwen” or “llama”)  
运行以上代码后你就会得到输出对每个条目评估CPQS_score得分的文件

3.筛选数据集
