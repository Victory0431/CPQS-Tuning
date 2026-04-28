import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from datasets import load_from_disk
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from repro.common import (
    ensure_dir,
    extract_final_choice,
    extract_gsm8k_gold,
    format_mc_options,
    math_answers_equal,
    normalize_number,
    setup_logger,
    set_seed,
)

BENCHMARK_CHOICES = ["gsm8k", "math500", "arc_challenge", "mmlu_subset"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automatic benchmark scoring for round-1 CPQS runs.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--adapter_path", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--group_name", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--benchmarks_root", required=True)
    parser.add_argument("--mmlu_path", required=True)
    parser.add_argument("--mmlu_examples_per_subject", type=int, default=8)
    parser.add_argument("--mmlu_subset_seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--batch_size_gsm8k", type=int, default=4)
    parser.add_argument("--batch_size_math500", type=int, default=4)
    parser.add_argument("--batch_size_arc", type=int, default=8)
    parser.add_argument("--batch_size_mmlu", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress_log_every_batches", type=int, default=20)
    parser.add_argument("--benchmarks", nargs="+", choices=BENCHMARK_CHOICES, default=BENCHMARK_CHOICES)
    parser.add_argument("--log_file", default="")
    return parser.parse_args()


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
    if hasattr(model, "generation_config"):
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
    return tokenizer, model


def batched(items: Sequence[Dict[str, Any]], batch_size: int) -> Sequence[Sequence[Dict[str, Any]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def generate_batch(tokenizer, model, prompts: List[str], max_new_tokens: int) -> List[str]:
    encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
    outputs: List[str] = []
    input_lengths = encoded["attention_mask"].sum(dim=1).tolist()
    for index, input_length in enumerate(input_lengths):
        completion = generated[index, int(input_length) :]
        outputs.append(tokenizer.decode(completion, skip_special_tokens=True).strip())
    return outputs


def build_chat_prompt(tokenizer, user_content: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def evaluate_gsm8k(tokenizer, model, dataset_root: str, batch_size: int, max_new_tokens: int, limit: int, logger, progress_log_every_batches: int) -> Dict[str, Any]:
    dataset = load_from_disk(str(Path(dataset_root) / "gsm8k"))["test"]
    rows = [dataset[index] for index in range(len(dataset))]
    if limit > 0:
        rows = rows[:limit]

    scored_rows: List[Dict[str, Any]] = []
    correct = 0
    total_batches = (len(rows) + batch_size - 1) // batch_size
    start_time = time.time()
    for batch_index, batch in enumerate(batched(rows, batch_size), start=1):
        prompts = [
            build_chat_prompt(
                tokenizer,
                "Solve the math word problem carefully. End your response with 'Final answer: <number>'.\n\n"
                f"Question: {row['question']}",
            )
            for row in batch
        ]
        outputs = generate_batch(tokenizer, model, prompts, max_new_tokens)
        for row, output in zip(batch, outputs):
            prediction = normalize_number(output)
            gold = extract_gsm8k_gold(row["answer"])
            is_correct = prediction == gold
            correct += int(is_correct)
            scored_rows.append(
                {
                    "question": row["question"],
                    "gold": gold,
                    "prediction": prediction,
                    "raw_output": output,
                    "correct": is_correct,
                }
            )
        if batch_index % progress_log_every_batches == 0 or batch_index == total_batches:
            elapsed = time.time() - start_time
            processed = min(batch_index * batch_size, len(rows))
            throughput = processed / elapsed if elapsed > 0 else 0.0
            logger.info("Benchmark progress | gsm8k | processed=%s/%s batches=%s/%s throughput=%.2f samples/s", processed, len(rows), batch_index, total_batches, throughput)
    return {"dataset": "gsm8k", "score": correct / len(scored_rows), "rows": scored_rows}


def evaluate_math500(tokenizer, model, dataset_root: str, batch_size: int, max_new_tokens: int, limit: int, logger, progress_log_every_batches: int) -> Dict[str, Any]:
    dataset = load_from_disk(str(Path(dataset_root) / "math500"))["test"]
    rows = [dataset[index] for index in range(len(dataset))]
    if limit > 0:
        rows = rows[:limit]

    scored_rows: List[Dict[str, Any]] = []
    correct = 0
    total_batches = (len(rows) + batch_size - 1) // batch_size
    start_time = time.time()
    for batch_index, batch in enumerate(batched(rows, batch_size), start=1):
        prompts = [
            build_chat_prompt(
                tokenizer,
                "Solve the problem carefully. End your response with 'Final answer: <answer>'.\n\n"
                f"Problem: {row['problem']}",
            )
            for row in batch
        ]
        outputs = generate_batch(tokenizer, model, prompts, max_new_tokens)
        for row, output in zip(batch, outputs):
            is_correct = math_answers_equal(output, row["answer"])
            correct += int(is_correct)
            scored_rows.append(
                {
                    "problem": row["problem"],
                    "gold": row["answer"],
                    "raw_output": output,
                    "correct": is_correct,
                }
            )
        if batch_index % progress_log_every_batches == 0 or batch_index == total_batches:
            elapsed = time.time() - start_time
            processed = min(batch_index * batch_size, len(rows))
            throughput = processed / elapsed if elapsed > 0 else 0.0
            logger.info("Benchmark progress | math500 | processed=%s/%s batches=%s/%s throughput=%.2f samples/s", processed, len(rows), batch_index, total_batches, throughput)
    return {"dataset": "math500", "score": correct / len(scored_rows), "rows": scored_rows}


def evaluate_arc(tokenizer, model, dataset_root: str, batch_size: int, max_new_tokens: int, limit: int, logger, progress_log_every_batches: int) -> Dict[str, Any]:
    dataset = load_from_disk(str(Path(dataset_root) / "arc_challenge"))["test"]
    rows = [dataset[index] for index in range(len(dataset))]
    if limit > 0:
        rows = rows[:limit]

    scored_rows: List[Dict[str, Any]] = []
    correct = 0
    total_batches = (len(rows) + batch_size - 1) // batch_size
    start_time = time.time()
    for batch_index, batch in enumerate(batched(rows, batch_size), start=1):
        prompts = [
            build_chat_prompt(
                tokenizer,
                "Answer the multiple-choice question. End your response with 'Final answer: <A/B/C/D>'.\n\n"
                f"Question: {row['question']}\n{format_mc_options(row['choices'])}",
            )
            for row in batch
        ]
        outputs = generate_batch(tokenizer, model, prompts, max_new_tokens)
        for row, output in zip(batch, outputs):
            prediction = extract_final_choice(output)
            is_correct = prediction == row["answerKey"]
            correct += int(is_correct)
            scored_rows.append(
                {
                    "id": row["id"],
                    "gold": row["answerKey"],
                    "prediction": prediction,
                    "raw_output": output,
                    "correct": is_correct,
                }
            )
        if batch_index % progress_log_every_batches == 0 or batch_index == total_batches:
            elapsed = time.time() - start_time
            processed = min(batch_index * batch_size, len(rows))
            throughput = processed / elapsed if elapsed > 0 else 0.0
            logger.info("Benchmark progress | arc_challenge | processed=%s/%s batches=%s/%s throughput=%.2f samples/s", processed, len(rows), batch_index, total_batches, throughput)
    return {"dataset": "arc_challenge", "score": correct / len(scored_rows), "rows": scored_rows}


def build_mmlu_subset(mmlu_path: str, examples_per_subject: int, seed: int) -> List[Dict[str, Any]]:
    with open(mmlu_path, "r", encoding="utf-8") as handle:
        rows = json.load(handle)
    by_subject: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_subject.setdefault(row["subject"], []).append(row)

    rng = random.Random(seed)
    subset: List[Dict[str, Any]] = []
    for subject in sorted(by_subject):
        subject_rows = list(by_subject[subject])
        rng.shuffle(subject_rows)
        subset.extend(subject_rows[:examples_per_subject])
    return subset


def evaluate_mmlu_subset(
    tokenizer,
    model,
    mmlu_path: str,
    examples_per_subject: int,
    subset_seed: int,
    batch_size: int,
    max_new_tokens: int,
    limit: int,
    logger,
    progress_log_every_batches: int,
) -> Dict[str, Any]:
    rows = build_mmlu_subset(mmlu_path, examples_per_subject, subset_seed)
    if limit > 0:
        rows = rows[:limit]

    scored_rows: List[Dict[str, Any]] = []
    correct = 0
    total_batches = (len(rows) + batch_size - 1) // batch_size
    start_time = time.time()
    for batch_index, batch in enumerate(batched(rows, batch_size), start=1):
        prompts = [
            build_chat_prompt(
                tokenizer,
                "Answer the multiple-choice question. End your response with 'Final answer: <A/B/C/D>'.\n\n"
                f"Subject: {row['subject']}\nQuestion: {row['question']}\n{format_mc_options(row['options'])}",
            )
            for row in batch
        ]
        outputs = generate_batch(tokenizer, model, prompts, max_new_tokens)
        for row, output in zip(batch, outputs):
            prediction = extract_final_choice(output)
            is_correct = prediction == row["answer"]
            correct += int(is_correct)
            scored_rows.append(
                {
                    "subject": row["subject"],
                    "gold": row["answer"],
                    "prediction": prediction,
                    "raw_output": output,
                    "correct": is_correct,
                }
            )
        if batch_index % progress_log_every_batches == 0 or batch_index == total_batches:
            elapsed = time.time() - start_time
            processed = min(batch_index * batch_size, len(rows))
            throughput = processed / elapsed if elapsed > 0 else 0.0
            logger.info("Benchmark progress | mmlu_subset | processed=%s/%s batches=%s/%s throughput=%.2f samples/s", processed, len(rows), batch_index, total_batches, throughput)
    return {
        "dataset": "mmlu_subset",
        "score": correct / len(scored_rows),
        "rows": scored_rows,
    }


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    ensure_dir(output_dir)
    log_file = Path(cfg.log_file) if cfg.log_file else output_dir / "evaluate_round1.log"
    logger = setup_logger("repro.evaluate_round1", log_file)
    logger.info("Evaluation started")
    logger.info(
        "Config | group=%s seed=%s max_new_tokens=%s adapter=%s benchmarks=%s batch_sizes=(gsm8k:%s math:%s arc:%s mmlu:%s)",
        cfg.group_name,
        cfg.seed,
        cfg.max_new_tokens,
        cfg.adapter_path or "<base>",
        ",".join(cfg.benchmarks),
        cfg.batch_size_gsm8k,
        cfg.batch_size_math500,
        cfg.batch_size_arc,
        cfg.batch_size_mmlu,
    )

    logger.info("Loading model and tokenizer")
    tokenizer, model = load_model_and_tokenizer(cfg.model_path, cfg.adapter_path)
    logger.info("Model and tokenizer loaded")

    all_benchmark_jobs = [
        ("gsm8k", lambda: evaluate_gsm8k(tokenizer, model, cfg.benchmarks_root, cfg.batch_size_gsm8k, cfg.max_new_tokens, cfg.limit, logger, cfg.progress_log_every_batches)),
        ("math500", lambda: evaluate_math500(tokenizer, model, cfg.benchmarks_root, cfg.batch_size_math500, cfg.max_new_tokens, cfg.limit, logger, cfg.progress_log_every_batches)),
        ("arc_challenge", lambda: evaluate_arc(tokenizer, model, cfg.benchmarks_root, cfg.batch_size_arc, cfg.max_new_tokens, cfg.limit, logger, cfg.progress_log_every_batches)),
        (
            "mmlu_subset",
            lambda: evaluate_mmlu_subset(
                tokenizer,
                model,
                cfg.mmlu_path,
                cfg.mmlu_examples_per_subject,
                cfg.mmlu_subset_seed,
                cfg.batch_size_mmlu,
                cfg.max_new_tokens,
                cfg.limit,
                logger,
                cfg.progress_log_every_batches,
            ),
        ),
    ]
    selected = set(cfg.benchmarks)
    benchmark_jobs = [(name, fn) for name, fn in all_benchmark_jobs if name in selected]

    csv_path = output_dir / "run_scores.csv"
    json_path = output_dir / "run_scores.json"
    existing_rows: List[Dict[str, Any]] = []
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as handle:
            existing_rows = json.load(handle)
        logger.info("Loaded existing run scores | rows=%s", len(existing_rows))
    run_rows: List[Dict[str, Any]] = [row for row in existing_rows if row.get("dataset") not in selected]
    if existing_rows:
        logger.info("Keeping prior benchmark results | rows=%s", len(run_rows))

    for benchmark_name, benchmark_fn in benchmark_jobs:
        logger.info("Benchmark start | %s", benchmark_name)
        result = benchmark_fn()
        dataset_name = result["dataset"]
        with open(output_dir / f"{dataset_name}_predictions.json", "w", encoding="utf-8") as handle:
            json.dump(result["rows"], handle, ensure_ascii=False, indent=2)
        row = {
            "group": cfg.group_name,
            "seed": cfg.seed,
            "dataset": dataset_name,
            "score": result["score"],
            "adapter_path": cfg.adapter_path,
        }
        run_rows.append(row)
        with open(csv_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["group", "seed", "dataset", "score", "adapter_path"])
            writer.writeheader()
            writer.writerows(run_rows)
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(run_rows, handle, ensure_ascii=False, indent=2)
        logger.info(
            "Benchmark done | %s | rows=%s | score=%.6f | saved=%s",
            dataset_name,
            len(result["rows"]),
            result["score"],
            output_dir / f"{dataset_name}_predictions.json",
        )

    logger.info("Evaluation finished | total_benchmarks=%s", len(run_rows))


if __name__ == "__main__":
    main()
