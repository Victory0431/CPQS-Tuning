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
from transformers import AutoModelForCausalLM, AutoTokenizer

from repro.common import (
    ensure_dir,
    extract_math_candidate,
    extract_final_choice,
    extract_gsm8k_gold,
    format_mc_options,
    math_answers_equal,
    normalize_number,
    strip_thinking_content,
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
    parser.add_argument("--enable_thinking", choices=["true", "false"], default="false")
    parser.add_argument("--do_sample", choices=["true", "false"], default="false")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--batch_size_gsm8k", type=int, default=4)
    parser.add_argument("--batch_size_math500", type=int, default=4)
    parser.add_argument("--batch_size_arc", type=int, default=8)
    parser.add_argument("--batch_size_mmlu", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample_dump_dir", default="")
    parser.add_argument("--sample_dump_count", type=int, default=0)
    parser.add_argument("--progress_log_every_batches", type=int, default=20)
    parser.add_argument("--benchmarks", nargs="+", choices=BENCHMARK_CHOICES, default=BENCHMARK_CHOICES)
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


def batched(items: Sequence[Dict[str, Any]], batch_size: int) -> Sequence[Sequence[Dict[str, Any]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


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


def generate_batch(tokenizer, model, prompts: List[str], generation_kwargs: Dict[str, Any]) -> List[str]:
    encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    prompt_length = encoded["input_ids"].shape[1]
    with torch.no_grad():
        generated = model.generate(**encoded, **generation_kwargs)
    outputs: List[str] = []
    for index in range(generated.shape[0]):
        completion = generated[index, prompt_length:]
        outputs.append(tokenizer.decode(completion, skip_special_tokens=True).strip())
    return outputs


def build_chat_prompt(tokenizer, user_content: str, enable_thinking: bool) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def build_gsm8k_prompt(question: str) -> str:
    return (
        "Solve the following math word problem carefully.\n"
        "End the final line with exactly: #### <answer>\n\n"
        f"Question: {question}"
    )


def build_math500_prompt(problem: str) -> str:
    return (
        "Solve the following math problem carefully.\n"
        "End the final line with exactly: \\boxed{answer}\n\n"
        f"Problem: {problem}"
    )


def build_arc_prompt(question: str, choices: Dict[str, Any]) -> str:
    return (
        "Answer the multiple-choice question.\n"
        "Output only one uppercase letter on the final line: A, B, C, or D.\n\n"
        f"Question: {question}\n{format_mc_options(choices)}"
    )


def build_mmlu_prompt(subject: str, question: str, options: Dict[str, Any]) -> str:
    return (
        "Answer the multiple-choice question.\n"
        "Output only one uppercase letter on the final line: A, B, C, or D.\n\n"
        f"Subject: {subject}\nQuestion: {question}\n{format_mc_options(options)}"
    )


def dump_sample_rows(sample_dump_dir: Path, dataset_name: str, rows: List[Dict[str, Any]], count: int, logger) -> None:
    if count <= 0:
        return
    ensure_dir(sample_dump_dir)
    dump_path = sample_dump_dir / f"{dataset_name}_samples.json"
    with open(dump_path, "w", encoding="utf-8") as handle:
        json.dump(rows[:count], handle, ensure_ascii=False, indent=2)
    logger.info("Sample dump saved | %s | rows=%s", dump_path, min(len(rows), count))


def evaluate_gsm8k(
    tokenizer,
    model,
    dataset_root: str,
    batch_size: int,
    generation_kwargs: Dict[str, Any],
    enable_thinking: bool,
    limit: int,
    logger,
    progress_log_every_batches: int,
) -> Dict[str, Any]:
    dataset = load_from_disk(str(Path(dataset_root) / "gsm8k"))["test"]
    rows = [dataset[index] for index in range(len(dataset))]
    if limit > 0:
        rows = rows[:limit]

    scored_rows: List[Dict[str, Any]] = []
    correct = 0
    total_batches = (len(rows) + batch_size - 1) // batch_size
    start_time = time.time()
    for batch_index, batch in enumerate(batched(rows, batch_size), start=1):
        prompt_texts = [build_gsm8k_prompt(row["question"]) for row in batch]
        prompts = [build_chat_prompt(tokenizer, prompt_text, enable_thinking) for prompt_text in prompt_texts]
        outputs = generate_batch(tokenizer, model, prompts, generation_kwargs)
        for row, prompt_text, output in zip(batch, prompt_texts, outputs):
            prediction = normalize_number(output)
            gold = extract_gsm8k_gold(row["answer"])
            is_correct = prediction == gold
            correct += int(is_correct)
            scored_rows.append(
                {
                    "question": row["question"],
                    "prompt": prompt_text,
                    "gold_answer": gold,
                    "extracted_answer": prediction,
                    "final_response": strip_thinking_content(output),
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


def evaluate_math500(
    tokenizer,
    model,
    dataset_root: str,
    batch_size: int,
    generation_kwargs: Dict[str, Any],
    enable_thinking: bool,
    limit: int,
    logger,
    progress_log_every_batches: int,
) -> Dict[str, Any]:
    dataset = load_from_disk(str(Path(dataset_root) / "math500"))["test"]
    rows = [dataset[index] for index in range(len(dataset))]
    if limit > 0:
        rows = rows[:limit]

    scored_rows: List[Dict[str, Any]] = []
    correct = 0
    total_batches = (len(rows) + batch_size - 1) // batch_size
    start_time = time.time()
    for batch_index, batch in enumerate(batched(rows, batch_size), start=1):
        prompt_texts = [build_math500_prompt(row["problem"]) for row in batch]
        prompts = [build_chat_prompt(tokenizer, prompt_text, enable_thinking) for prompt_text in prompt_texts]
        outputs = generate_batch(tokenizer, model, prompts, generation_kwargs)
        for row, prompt_text, output in zip(batch, prompt_texts, outputs):
            prediction = extract_math_candidate(output)
            is_correct = math_answers_equal(prediction, row["answer"])
            correct += int(is_correct)
            scored_rows.append(
                {
                    "problem": row["problem"],
                    "prompt": prompt_text,
                    "gold_answer": row["answer"],
                    "extracted_answer": prediction,
                    "final_response": strip_thinking_content(output),
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


def evaluate_arc(
    tokenizer,
    model,
    dataset_root: str,
    batch_size: int,
    generation_kwargs: Dict[str, Any],
    enable_thinking: bool,
    limit: int,
    logger,
    progress_log_every_batches: int,
) -> Dict[str, Any]:
    dataset = load_from_disk(str(Path(dataset_root) / "arc_challenge"))["test"]
    rows = [dataset[index] for index in range(len(dataset))]
    if limit > 0:
        rows = rows[:limit]

    scored_rows: List[Dict[str, Any]] = []
    correct = 0
    total_batches = (len(rows) + batch_size - 1) // batch_size
    start_time = time.time()
    for batch_index, batch in enumerate(batched(rows, batch_size), start=1):
        prompt_texts = [build_arc_prompt(row["question"], row["choices"]) for row in batch]
        prompts = [build_chat_prompt(tokenizer, prompt_text, enable_thinking) for prompt_text in prompt_texts]
        outputs = generate_batch(tokenizer, model, prompts, generation_kwargs)
        for row, prompt_text, output in zip(batch, prompt_texts, outputs):
            prediction = extract_final_choice(output)
            is_correct = prediction == row["answerKey"]
            correct += int(is_correct)
            scored_rows.append(
                {
                    "id": row["id"],
                    "prompt": prompt_text,
                    "gold_answer": row["answerKey"],
                    "extracted_answer": prediction,
                    "final_response": strip_thinking_content(output),
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
    limit: int,
    logger,
    progress_log_every_batches: int,
    generation_kwargs: Dict[str, Any],
    enable_thinking: bool,
) -> Dict[str, Any]:
    rows = build_mmlu_subset(mmlu_path, examples_per_subject, subset_seed)
    if limit > 0:
        rows = rows[:limit]

    scored_rows: List[Dict[str, Any]] = []
    correct = 0
    total_batches = (len(rows) + batch_size - 1) // batch_size
    start_time = time.time()
    for batch_index, batch in enumerate(batched(rows, batch_size), start=1):
        prompt_texts = [build_mmlu_prompt(row["subject"], row["question"], row["options"]) for row in batch]
        prompts = [build_chat_prompt(tokenizer, prompt_text, enable_thinking) for prompt_text in prompt_texts]
        outputs = generate_batch(tokenizer, model, prompts, generation_kwargs)
        for row, prompt_text, output in zip(batch, prompt_texts, outputs):
            prediction = extract_final_choice(output)
            is_correct = prediction == row["answer"]
            correct += int(is_correct)
            scored_rows.append(
                {
                    "subject": row["subject"],
                    "prompt": prompt_text,
                    "gold_answer": row["answer"],
                    "extracted_answer": prediction,
                    "final_response": strip_thinking_content(output),
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
    enable_thinking = parse_bool_flag(cfg.enable_thinking)
    do_sample = parse_bool_flag(cfg.do_sample)
    logger.info(
        "Config | group=%s seed=%s adapter=%s benchmarks=%s thinking=%s do_sample=%s temperature=%s top_p=%s top_k=%s max_new_tokens=%s batch_sizes=(gsm8k:%s math:%s arc:%s mmlu:%s)",
        cfg.group_name,
        cfg.seed,
        cfg.adapter_path or "<base>",
        ",".join(cfg.benchmarks),
        enable_thinking,
        do_sample,
        cfg.temperature,
        cfg.top_p,
        cfg.top_k,
        cfg.max_new_tokens,
        cfg.batch_size_gsm8k,
        cfg.batch_size_math500,
        cfg.batch_size_arc,
        cfg.batch_size_mmlu,
    )

    logger.info("Loading model and tokenizer")
    tokenizer, model = load_model_and_tokenizer(cfg.model_path, cfg.adapter_path)
    generation_kwargs = build_generation_kwargs(
        tokenizer=tokenizer,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=do_sample,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
    )
    logger.info("Model and tokenizer loaded")
    sample_dump_dir = Path(cfg.sample_dump_dir) if cfg.sample_dump_dir else output_dir / "samples"

    all_benchmark_jobs = [
        ("gsm8k", lambda: evaluate_gsm8k(tokenizer, model, cfg.benchmarks_root, cfg.batch_size_gsm8k, generation_kwargs, enable_thinking, cfg.limit, logger, cfg.progress_log_every_batches)),
        ("math500", lambda: evaluate_math500(tokenizer, model, cfg.benchmarks_root, cfg.batch_size_math500, generation_kwargs, enable_thinking, cfg.limit, logger, cfg.progress_log_every_batches)),
        ("arc_challenge", lambda: evaluate_arc(tokenizer, model, cfg.benchmarks_root, cfg.batch_size_arc, generation_kwargs, enable_thinking, cfg.limit, logger, cfg.progress_log_every_batches)),
        (
            "mmlu_subset",
            lambda: evaluate_mmlu_subset(
                tokenizer,
                model,
                cfg.mmlu_path,
                cfg.mmlu_examples_per_subject,
                cfg.mmlu_subset_seed,
                cfg.batch_size_mmlu,
                cfg.limit,
                logger,
                cfg.progress_log_every_batches,
                generation_kwargs,
                enable_thinking,
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
        dump_sample_rows(sample_dump_dir, dataset_name, result["rows"], cfg.sample_dump_count, logger)
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
