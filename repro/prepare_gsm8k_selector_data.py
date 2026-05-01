import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from repro.common import (
    ensure_dir,
    extract_gsm8k_gold,
    load_instruction_records,
    normalize_number,
    setup_logger,
    strip_thinking_content,
)
from repro.evaluate_round1 import build_chat_prompt, build_gsm8k_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare positive/negative data for GSM8K-domain selector training."
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--adapter_path", default="")
    parser.add_argument("--input_data", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--source_name", required=True)
    parser.add_argument("--label", type=int, choices=[0, 1], required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--do_sample", choices=["true", "false"], default="false")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--enable_thinking", choices=["true", "false"], default="false")
    parser.add_argument("--log_file", default="")
    return parser.parse_args()


def parse_bool_flag(value: str) -> bool:
    return value.lower() == "true"


def load_model_and_tokenizer(model_path: str, adapter_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return tokenizer, model


def build_generation_kwargs(
    tokenizer,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
) -> Dict[str, Any]:
    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
        if top_k > 0:
            generation_kwargs["top_k"] = top_k
    return generation_kwargs


def generate_batch(tokenizer, model, prompts: Sequence[str], generation_kwargs: Dict[str, Any]) -> List[str]:
    encoded = tokenizer(list(prompts), return_tensors="pt", padding=True).to(model.device)
    prompt_length = encoded["input_ids"].shape[1]
    with torch.no_grad():
        generated = model.generate(**encoded, **generation_kwargs)
    outputs: List[str] = []
    for index in range(generated.shape[0]):
        completion = generated[index, prompt_length:]
        outputs.append(tokenizer.decode(completion, skip_special_tokens=True).strip())
    return outputs


def main() -> None:
    cfg = parse_args()
    output_path = Path(cfg.output_path)
    ensure_dir(output_path.parent)
    log_file = Path(cfg.log_file) if cfg.log_file else output_path.parent / f"{output_path.stem}.log"
    logger = setup_logger("repro.prepare_gsm8k_selector_data", log_file)
    logger.info("GSM8K selector data preparation started")
    logger.info(
        "Config | input=%s output=%s source=%s label=%s adapter=%s batch_size=%s limit=%s",
        cfg.input_data,
        output_path,
        cfg.source_name,
        cfg.label,
        cfg.adapter_path or "<base>",
        cfg.batch_size,
        cfg.limit,
    )

    tokenizer, model = load_model_and_tokenizer(cfg.model_path, cfg.adapter_path)
    generation_kwargs = build_generation_kwargs(
        tokenizer=tokenizer,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=parse_bool_flag(cfg.do_sample),
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
    )

    records = load_instruction_records(cfg.input_data)
    if cfg.limit > 0:
        records = records[: cfg.limit]
    logger.info("Records loaded | count=%s", len(records))

    enable_thinking = parse_bool_flag(cfg.enable_thinking)
    prepared: List[Dict[str, Any]] = []
    start_time = time.time()
    total = len(records)
    total_batches = (total + cfg.batch_size - 1) // cfg.batch_size

    for batch_index in range(total_batches):
        batch = records[batch_index * cfg.batch_size : (batch_index + 1) * cfg.batch_size]
        prompt_texts = [build_gsm8k_prompt(record["input"]) for record in batch]
        prompts = [build_chat_prompt(tokenizer, prompt_text, enable_thinking) for prompt_text in prompt_texts]
        outputs = generate_batch(tokenizer, model, prompts, generation_kwargs)

        for record, prompt_text, output in zip(batch, prompt_texts, outputs):
            prepared.append(
                {
                    "sample_id": record["sample_id"],
                    "instruction": record["instruction"],
                    "input": record["input"],
                    "output": strip_thinking_content(output),
                    "source": cfg.source_name,
                    "label": cfg.label,
                    "gold_answer": extract_gsm8k_gold(str(record["output"])),
                    "predicted_answer": normalize_number(output),
                    "prompt": prompt_text,
                    "raw_output": output,
                }
            )

        if (batch_index + 1) % 10 == 0 or batch_index + 1 == total_batches:
            processed = min((batch_index + 1) * cfg.batch_size, total)
            elapsed = time.time() - start_time
            throughput = processed / elapsed if elapsed > 0 else 0.0
            logger.info(
                "Progress | processed=%s/%s batches=%s/%s throughput=%.2f samples/s",
                processed,
                total,
                batch_index + 1,
                total_batches,
                throughput,
            )

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(prepared, handle, ensure_ascii=False, indent=2)
    logger.info("GSM8K selector data preparation finished | saved=%s rows=%s", output_path, len(prepared))


if __name__ == "__main__":
    main()
