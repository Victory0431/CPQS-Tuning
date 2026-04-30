import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from transformers import AutoTokenizer

from repro.common import ensure_dir, load_instruction_records, render_prompts, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect SFT supervised token spans.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--backbone", choices=["qwen", "llama"], default="qwen")
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--log_file", default="")
    return parser.parse_args()


def main() -> None:
    cfg = parse_args()
    output_path = Path(cfg.output_path)
    ensure_dir(output_path.parent)
    log_path = Path(cfg.log_file) if cfg.log_file else output_path.with_suffix(".log")
    logger = setup_logger("repro.smoke_sft_labels", log_path)
    logger.info("SFT label smoke started")
    logger.info(
        "Config | model_path=%s train_data=%s backbone=%s num_samples=%s",
        cfg.model_path,
        cfg.train_data,
        cfg.backbone,
        cfg.num_samples,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = load_instruction_records(cfg.train_data)[: cfg.num_samples]
    report_rows: List[Dict[str, Any]] = []

    for index, record in enumerate(records, start=1):
        question = f"{record['instruction']}\nInput:{record['input']}"
        user_prompt, full_prompt = render_prompts(
            tokenizer,
            question,
            record["output"],
            cfg.backbone,
            full_add_generation_prompt=False,
        )
        user_inputs = tokenizer([user_prompt], return_tensors="pt")
        full_inputs = tokenizer([full_prompt], return_tensors="pt")
        input_ids = full_inputs["input_ids"][0]
        start_idx = len(user_inputs["input_ids"][0])
        supervised_ids = input_ids[start_idx:]
        report_rows.append(
            {
                "sample_index": index,
                "sample_id": record.get("sample_id", index - 1),
                "user_prompt": user_prompt,
                "full_prompt": full_prompt,
                "user_token_count": int(start_idx),
                "full_token_count": int(input_ids.shape[0]),
                "supervised_token_count": int(supervised_ids.shape[0]),
                "supervised_text": tokenizer.decode(supervised_ids, skip_special_tokens=False),
                "answer_preview": str(record["output"])[:500],
            }
        )
        logger.info(
            "Sample inspected | sample_index=%s sample_id=%s user_tokens=%s full_tokens=%s supervised_tokens=%s",
            index,
            record.get("sample_id", index - 1),
            start_idx,
            int(input_ids.shape[0]),
            int(supervised_ids.shape[0]),
        )

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report_rows, handle, ensure_ascii=False, indent=2)

    logger.info("SFT label smoke finished | saved=%s", output_path)


if __name__ == "__main__":
    main()
