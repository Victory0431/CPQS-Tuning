#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepTextCNN inference script (Qwen-2 or Llama-2 backbone).

Example
-------
# Qwen-2-7B
python predict_textcnn.py \
    --backbone qwen \
    --model_path /path/to/Qwen2-7B-Instruct \
    --cnn_checkpoint textcnn_step_15000_full.pth \
    --predict_data something.json

# Llama-2-7B
python predict_textcnn.py \
    --backbone llama \
    --model_path /path/to/Llama-2-7b-hf \
    --cnn_checkpoint textcnn_step_15000_full.pth \
    --predict_data something.json
"""
import argparse
import gc
import json
import random
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ────────────────────────────────────────────────────────────────────
# Model definition（保持与训练脚本一致）
# ────────────────────────────────────────────────────────────────────
class DeepTextCNN(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        num_layers: int,
        kernel_sizes: Tuple[int, int, int] = (3, 4, 5),
        num_filters: int = 256,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(num_layers, num_filters, kernel_size=(k, embedding_dim))
                for k in kernel_sizes
            ]
        )
        self.deep_convs = nn.ModuleList(
            [nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=1) for _ in range(2)]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(len(kernel_sizes) * num_filters, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, T, D)
        feats: List[torch.Tensor] = []
        for conv, k in zip(self.convs, self.kernel_sizes):
            seq_len = x.size(2)
            if seq_len < k:
                pad_amt = k - seq_len
                x_pad = F.pad(x, (0, 0, 0, pad_amt))
                out = F.relu(conv(x_pad)).squeeze(3)
            else:
                out = F.relu(conv(x)).squeeze(3)
            for deep in self.deep_convs:
                out = F.relu(deep(out))
            pooled = F.adaptive_max_pool1d(out, 1).squeeze(2)
            feats.append(pooled)
        feats = torch.cat(feats, dim=1)
        feats = self.dropout(feats)
        return self.fc(feats)  # (B, 2)

# ────────────────────────────────────────────────────────────────────
# Utils
# ────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("DeepTextCNN Inference")
    # 路径
    p.add_argument(
        "--model_path",
        type=str,
        default="path/Llama-2-7b-hf",  # ← 改这里
        help="LLM checkpoint",
    )
    p.add_argument(
        "--cnn_checkpoint",
        type=str,
        default="textcnn_step_15000_full.pth",  # ← 改这里
        help="Trained TextCNN .pth",
    )
    p.add_argument(
        "--predict_data",
        type=str,
        default="alpaca_gpt4_data.json",  # ← 改这里
        help="JSON list to infer",
    )
    p.add_argument("--output_path", type=str, default="predict.json")
    p.add_argument("--failed_path", type=str, default="predict_failed.json")
    # 运行参数
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--use_layers", choices=["all", "last"], default="all")
    p.add_argument("--use_part", choices=["front", "middle", "back", "full"], default="full")
    p.add_argument("--backbone", choices=["qwen", "llama"], default="llama")
    # 设备
    p.add_argument("--device_cnn", type=str, default="cuda:0")
    p.add_argument("--device_llm", type=str, default="cuda:0")
    return p.parse_args()


def load_backbone(model_path: str, device_llm: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map={"": device_llm}
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    return tokenizer, model


def select_hidden_states(
    hidden_states: torch.Tensor, use_layers: str, use_part: str, total_layers: int
) -> torch.Tensor:
    if use_layers == "last":
        return hidden_states[-1:].contiguous()
    third = total_layers // 3
    if use_part == "front":
        return hidden_states[:third]
    if use_part == "middle":
        return hidden_states[third : 2 * third]
    if use_part == "back":
        return hidden_states[2 * third :]
    return hidden_states  # full


def build_prompts(question: str, answer: str, backbone: str):
    """
    返回 (user_only_prompt, full_dialog_prompt)
    - Qwen: list[dict]
    - Llama: str
    """
    if backbone == "qwen":
        user_only = [{"role": "user", "content": question}]
        full_dialog = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        return user_only, full_dialog
    else:  # llama
        user_only = f"<s> [INST] {question} [/INST]"
        full_dialog = f"{user_only} {answer} </s>"
        return user_only, full_dialog


# ────────────────────────────────────────────────────────────────────
# Main prediction routine
# ────────────────────────────────────────────────────────────────────
def predict(cfg: argparse.Namespace) -> None:
    # 设备
    device_cnn = torch.device(cfg.device_cnn)
    device_llm = torch.device(cfg.device_llm)

    # 加载骨干
    tokenizer, llm = load_backbone(cfg.model_path, cfg.device_llm)
    total_layers = llm.config.num_hidden_layers + 1

    # 计算选层数，初始化 CNN
    num_layers_sel = (
        1
        if cfg.use_layers == "last"
        else {
            "front": total_layers // 3,
            "middle": total_layers // 3,
            "back": total_layers - 2 * (total_layers // 3),
            "full": total_layers,
        }[cfg.use_part]
    )
    print(f"[Info] Using {num_layers_sel} layers ({cfg.use_layers}, {cfg.use_part})")

    classifier = DeepTextCNN(
        embedding_dim=llm.config.hidden_size,
        num_classes=2,
        num_layers=num_layers_sel,
    ).to(device_cnn)
    classifier.load_state_dict(torch.load(cfg.cnn_checkpoint, map_location=device_cnn))
    classifier.eval()

    # 读取待预测数据
    with open(cfg.predict_data, "r", encoding="utf-8") as f:
        raw_items = json.load(f)
    random.shuffle(raw_items)

    predictions, failed = [], []
    bs = cfg.batch_size

    for idx in tqdm(range(0, len(raw_items), bs), desc="Predicting"):
        batch = raw_items[idx : idx + bs]
        # 构造批次 prompt
        user_prompts, full_prompts, items = [], [], []
        for item in batch:
            q = f"{item.get('instruction','')}\nInput:{item.get('input','')}"
            a = item.get("output", "")
            u_only, full = build_prompts(q, a, cfg.backbone)
            if cfg.backbone == "qwen":
                user_prompts.append(
                    tokenizer.apply_chat_template(u_only, tokenize=False, add_generation_prompt=True)
                )
                full_prompts.append(
                    tokenizer.apply_chat_template(full, tokenize=False, add_generation_prompt=True)
                )
            else:  # llama
                user_prompts.append(u_only)
                full_prompts.append(full)
            items.append(item)

        # 逐条处理（batch_size 通常为 1，保持逻辑简单）
        for uprompt, fprompt, item in zip(user_prompts, full_prompts, items):
            try:
                up_ids = tokenizer([uprompt], return_tensors="pt")
                fp_ids = tokenizer([fprompt], return_tensors="pt").to(device_llm)
                start_idx = len(up_ids["input_ids"][0])

                with torch.no_grad():
                    out = llm(fp_ids["input_ids"], output_hidden_states=True)
                hidden = torch.stack(out.hidden_states, dim=0).squeeze(1)  # (L, T, D)
                hidden = hidden[:, start_idx:]  # 仅答案部分
                hidden = select_hidden_states(hidden, cfg.use_layers, cfg.use_part, total_layers)
                hidden = hidden.unsqueeze(0).to(device_cnn).float()  # (1, L_sel, T, D)

                with torch.no_grad():
                    logits = classifier(hidden)
                    probs = F.softmax(logits, dim=1)
                    pred = int(torch.argmax(probs, dim=1).item())
                item["predicted_class"] = pred
                item["probabilities"] = probs.squeeze(0).tolist()
                item["CPQS_score"] = probs[0, 1].item()
                predictions.append(item)

            except torch.cuda.OutOfMemoryError:
                print(f"[OOM] index {idx} skipped.")
                failed.append(item)
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                print(f"[Error] index {idx}: {e}")
                failed.append(item)
                torch.cuda.empty_cache()
                gc.collect()

    # 保存结果
    Path(cfg.output_path).write_text(json.dumps(predictions, ensure_ascii=False, indent=2), "utf-8")
    print(f"[OK] Saved {len(predictions)} predictions → {cfg.output_path}")
    if failed:
        Path(cfg.failed_path).write_text(json.dumps(failed, ensure_ascii=False, indent=2), "utf-8")
        print(f"[Warn] {len(failed)} samples failed → {cfg.failed_path}")

# ────────────────────────────────────────────────────────────────────
# Entry
# ────────────────────────────────────────────────────────────────────
def main():
    cfg = parse_args()
    predict(cfg)

if __name__ == "__main__":
    main()
