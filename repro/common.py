import json
import logging
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sympy import simplify
from sympy.parsing.latex import parse_latex
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_WANDB_ENTITY = "jiahongqin1-ucas-hias"
DEFAULT_WANDB_PROJECT = "CPQS_research"
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0


class DeepTextCNN(nn.Module):
    """TextCNN matching the original repo's selector architecture."""

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
                nn.Conv2d(
                    in_channels=num_layers,
                    out_channels=num_filters,
                    kernel_size=(kernel_size, embedding_dim),
                )
                for kernel_size in kernel_sizes
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features: List[torch.Tensor] = []
        for conv, kernel_size in zip(self.convs, self.kernel_sizes):
            seq_len = x.size(2)
            if seq_len < kernel_size:
                x_local = F.pad(x, (0, 0, 0, kernel_size - seq_len))
            else:
                x_local = x
            out = F.relu(conv(x_local)).squeeze(3)
            for deep_conv in self.deep_convs:
                out = F.relu(deep_conv(out))
            pooled = F.adaptive_max_pool1d(out, 1).squeeze(2)
            features.append(pooled)
        merged = torch.cat(features, dim=1)
        merged = self.dropout(merged)
        return self.fc(merged)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def setup_logger(name: str, log_path: Path) -> logging.Logger:
    ensure_dir(log_path.parent)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_question(item: Dict[str, Any]) -> str:
    return f"{item.get('instruction', '')}\nInput:{item.get('input', '')}"


def build_prompts(question: str, answer: str, backbone: str) -> Tuple[Any, Any]:
    if backbone == "qwen":
        user_only = [{"role": "user", "content": question}]
        full_dialog = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        return user_only, full_dialog
    user_only = f"<s> [INST] {question} [/INST]"
    full_dialog = f"{user_only} {answer} </s>"
    return user_only, full_dialog


def render_prompts(tokenizer: AutoTokenizer, question: str, answer: str, backbone: str) -> Tuple[str, str]:
    user_only, full_dialog = build_prompts(question, answer, backbone)
    if backbone == "qwen":
        user_prompt = tokenizer.apply_chat_template(
            user_only,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_prompt = tokenizer.apply_chat_template(
            full_dialog,
            tokenize=False,
            add_generation_prompt=True,
        )
        return user_prompt, full_prompt
    return user_only, full_dialog


def load_backbone(
    model_path: str,
    device: str,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map={"": device},
        trust_remote_code=True,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    tokenizer.padding_side = "right"
    return tokenizer, model


def total_layers(model: AutoModelForCausalLM) -> int:
    return model.config.num_hidden_layers + 1


def selected_num_layers(total_hidden_layers: int, use_layers: str, use_part: str) -> int:
    if use_layers == "last":
        return 1
    third = total_hidden_layers // 3
    if use_part in {"front", "middle"}:
        return third
    if use_part == "back":
        return total_hidden_layers - 2 * third
    return total_hidden_layers


def select_hidden_states(
    hidden_states: torch.Tensor,
    use_layers: str,
    use_part: str,
    total_hidden_layers: int,
) -> torch.Tensor:
    if use_layers == "last":
        return hidden_states[-1:].contiguous()
    third = total_hidden_layers // 3
    if use_part == "front":
        return hidden_states[:third]
    if use_part == "middle":
        return hidden_states[third : 2 * third]
    if use_part == "back":
        return hidden_states[2 * third :]
    return hidden_states


def extract_response_hidden_states(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    question: str,
    answer: str,
    backbone: str,
    use_layers: str,
    use_part: str,
) -> torch.Tensor:
    user_prompt, full_prompt = render_prompts(tokenizer, question, answer, backbone)
    user_inputs = tokenizer([user_prompt], return_tensors="pt")
    full_inputs = tokenizer([full_prompt], return_tensors="pt").to(model.device)
    start_idx = len(user_inputs["input_ids"][0])
    with torch.no_grad():
        outputs = model(full_inputs["input_ids"], output_hidden_states=True)
    hidden = torch.stack(outputs.hidden_states, dim=0).squeeze(1)
    hidden = hidden[:, start_idx:]
    hidden = select_hidden_states(hidden, use_layers, use_part, total_layers(model))
    return hidden.float()


def extract_batched_response_hidden_states(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    questions: Sequence[str],
    answers: Sequence[str],
    backbone: str,
    use_layers: str,
    use_part: str,
) -> torch.Tensor:
    user_prompts: List[str] = []
    full_prompts: List[str] = []
    for question, answer in zip(questions, answers):
        user_prompt, full_prompt = render_prompts(tokenizer, question, answer, backbone)
        user_prompts.append(user_prompt)
        full_prompts.append(full_prompt)

    user_inputs = tokenizer(user_prompts, return_tensors="pt", padding=True)
    full_inputs = tokenizer(full_prompts, return_tensors="pt", padding=True).to(model.device)
    user_lengths = user_inputs["attention_mask"].sum(dim=1).tolist()
    full_lengths = full_inputs["attention_mask"].sum(dim=1).tolist()

    with torch.no_grad():
        outputs = model(
            full_inputs["input_ids"],
            attention_mask=full_inputs["attention_mask"],
            output_hidden_states=True,
        )
    hidden = torch.stack(outputs.hidden_states, dim=0)
    hidden = select_hidden_states(hidden, use_layers, use_part, total_layers(model)).float()
    hidden = hidden.permute(1, 0, 2, 3).contiguous()

    response_lengths = [max(1, int(full_len - user_len)) for user_len, full_len in zip(user_lengths, full_lengths)]
    max_response_len = max(response_lengths)
    batch_size, num_layers, _, hidden_size = hidden.shape
    padded_hidden = torch.zeros(
        batch_size,
        num_layers,
        max_response_len,
        hidden_size,
        dtype=hidden.dtype,
        device=hidden.device,
    )

    for index, (start_idx, full_len) in enumerate(zip(user_lengths, full_lengths)):
        response_hidden = hidden[index, :, int(start_idx) : int(full_len), :]
        response_len = response_hidden.shape[1]
        if response_len == 0:
            response_hidden = hidden[index, :, int(full_len) - 1 : int(full_len), :]
            response_len = 1
        padded_hidden[index, :, :response_len, :] = response_hidden
    return padded_hidden


def load_selector_examples(
    pos_path: str,
    neg1_path: str,
    neg2_path: str,
    seed: int,
) -> List[Dict[str, Any]]:
    pos_data = load_json(pos_path)
    neg1_data = load_json(neg1_path)
    neg2_data = load_json(neg2_path)

    examples: List[Dict[str, Any]] = []
    for idx, item in enumerate(pos_data):
        examples.append(
            {
                "example_id": f"pos_{idx}",
                "instruction": item["instruction"],
                "input": item["input"],
                "answer": item["output"],
                "label": 1,
                "source": "alpaca_gpt4",
            }
        )
    for idx, item in enumerate(neg1_data):
        examples.append(
            {
                "example_id": f"neg_llama_{idx}",
                "instruction": item["instruction"],
                "input": item["input"],
                "answer": item["模型输出"],
                "label": 0,
                "source": "llama_low_quality",
            }
        )
    for idx, item in enumerate(neg2_data):
        examples.append(
            {
                "example_id": f"neg_qwen_{idx}",
                "instruction": item["instruction"],
                "input": item["input"],
                "answer": item["模型输出"],
                "label": 0,
                "source": "qwen_low_quality",
            }
        )

    rng = random.Random(seed)
    rng.shuffle(examples)
    return examples


def stratified_split(
    examples: Sequence[Dict[str, Any]],
    train_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    positives = [example for example in examples if example["label"] == 1]
    negatives = [example for example in examples if example["label"] == 0]
    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)

    pos_cut = int(len(positives) * train_ratio)
    neg_cut = int(len(negatives) * train_ratio)

    train_examples = positives[:pos_cut] + negatives[:neg_cut]
    val_examples = positives[pos_cut:] + negatives[neg_cut:]
    rng.shuffle(train_examples)
    rng.shuffle(val_examples)
    return train_examples, val_examples


def build_sft_features(
    tokenizer: AutoTokenizer,
    instruction: str,
    input_text: str,
    answer: str,
    backbone: str,
    max_length: int,
) -> Optional[Dict[str, List[int]]]:
    question = f"{instruction}\nInput:{input_text}"
    user_prompt, full_prompt = render_prompts(tokenizer, question, answer, backbone)
    user_inputs = tokenizer([user_prompt], return_tensors="pt")
    full_inputs = tokenizer([full_prompt], return_tensors="pt")

    input_ids = full_inputs["input_ids"][0][:max_length]
    attention_mask = full_inputs["attention_mask"][0][:max_length]
    labels = input_ids.clone()

    start_idx = len(user_inputs["input_ids"][0])
    label_cut = min(start_idx, labels.shape[0])
    labels[:label_cut] = -100
    if torch.all(labels.eq(-100)):
        return None

    return {
        "input_ids": input_ids.tolist(),
        "attention_mask": attention_mask.tolist(),
        "labels": labels.tolist(),
    }


def load_instruction_records(path: str) -> List[Dict[str, Any]]:
    records = load_json(path)
    normalized: List[Dict[str, Any]] = []
    for idx, item in enumerate(records):
        normalized.append(
            {
                "sample_id": item.get("sample_id", idx),
                "instruction": item.get("instruction", ""),
                "input": item.get("input", ""),
                "output": item.get("output", ""),
                "source": item.get("source", Path(path).stem),
            }
        )
    return normalized


def format_mc_options(options: Dict[str, Any]) -> str:
    if isinstance(options, dict) and "label" in options and "text" in options:
        return "\n".join(
            f"{label}. {text}" for label, text in zip(options["label"], options["text"])
        )
    return "\n".join(f"{label}. {text}" for label, text in options.items())


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_final_choice(text: str) -> str:
    patterns = [
        r"Final answer\s*[:：]\s*([A-D])\b",
        r"Answer\s*[:：]\s*([A-D])\b",
        r"\b([A-D])\b",
    ]
    cleaned = text.upper()
    for pattern in patterns:
        match = re.search(pattern, cleaned)
        if match:
            return match.group(1)
    return ""


def extract_gsm8k_gold(answer: str) -> str:
    if "####" in answer:
        return normalize_number(answer.split("####")[-1])
    return normalize_number(answer)


def normalize_number(text: str) -> str:
    cleaned = text.strip()
    cleaned = cleaned.replace(",", "")
    cleaned = cleaned.replace("$", "")
    cleaned = cleaned.replace("%", "")
    cleaned = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", cleaned)
    numbers = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
    if numbers:
        candidate = numbers[-1]
    else:
        candidate = cleaned
    try:
        value = float(candidate)
        if math.isclose(value, round(value)):
            return str(int(round(value)))
        return f"{value:.10f}".rstrip("0").rstrip(".")
    except ValueError:
        return normalize_whitespace(candidate)


def extract_math_candidate(text: str) -> str:
    boxed = re.findall(r"\\boxed\{(.+?)\}", text)
    if boxed:
        return boxed[-1]
    final_match = re.search(r"Final answer\s*[:：]\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if final_match:
        return final_match.group(1).strip()
    return text.strip()


def strip_latex_wrappers(text: str) -> str:
    stripped = text.strip()
    stripped = stripped.replace("\\left", "")
    stripped = stripped.replace("\\right", "")
    stripped = stripped.replace("$", "")
    stripped = stripped.replace("\\!", "")
    stripped = normalize_whitespace(stripped)
    return stripped


def try_sympy_equal(left: str, right: str) -> bool:
    try:
        left_expr = parse_latex(left)
        right_expr = parse_latex(right)
        return bool(simplify(left_expr - right_expr) == 0)
    except Exception:
        return False


def split_top_level_items(text: str) -> List[str]:
    items: List[str] = []
    current: List[str] = []
    depth = 0
    for char in text:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        if char == "," and depth == 0:
            items.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    if current:
        items.append("".join(current).strip())
    return items


def math_answers_equal(prediction: str, gold: str) -> bool:
    pred = strip_latex_wrappers(extract_math_candidate(prediction))
    target = strip_latex_wrappers(gold)

    if pred == target:
        return True

    pred_simple = pred.strip("()[]")
    target_simple = target.strip("()[]")
    pred_parts = split_top_level_items(pred_simple)
    target_parts = split_top_level_items(target_simple)
    if len(pred_parts) > 1 and len(pred_parts) == len(target_parts):
        return all(
            math_answers_equal(pred_part, target_part)
            for pred_part, target_part in zip(pred_parts, target_parts)
        )

    if normalize_number(pred) == normalize_number(target):
        return True

    if try_sympy_equal(pred, target):
        return True

    return False


def resolve_wandb_mode(enable_wandb: bool) -> List[str]:
    return ["wandb"] if enable_wandb else []
