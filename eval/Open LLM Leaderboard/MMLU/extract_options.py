import os
import json
import re
import argparse
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model predictions on multiple-choice questions stored in a JSON file"
    )
    parser.add_argument(
        "--json_file",
        type=str,
        default=r"path/json_file",
        help="Path to the JSON file containing question records"
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=r"path/Qwen2___5-7B-Instruct",
        help="Path to the pretrained model"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for generation"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.8,
        help="Top-p (nucleus) sampling parameter"
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.05,
        help="Repetition penalty to avoid loops"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=3000,
        help="Maximum number of tokens to generate per prompt"
    )
    args = parser.parse_args()

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_path,
        use_fast=True,
        trust_remote_code=True
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens
    )
    llm = LLM(
        model=args.pretrained_model_path,
        max_model_len=args.max_tokens,
        tensor_parallel_size=1,
        trust_remote_code=True
    )

    # Load records from the specified JSON file
    with open(args.json_file, 'r', encoding='utf-8') as f:
        records = json.load(f)

    prompts = []
    for rec in records:
        options = rec.get('options', '')
        analysis = rec.get('model_output', '')
        prompt = (
            f"#### Original options: {options}\n"
            f"#### Model analysis: {analysis}\n"
            "Above is the analysis of a single-choice question. "
            "Extract and output only the correct option letter:"
        )
        messages = [{"role": "user", "content": prompt}]
        query = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(query)

    # Generate predictions
    outputs = llm.generate(prompts, sampling_params)

    correct = 0
    count = 0
    for rec, out in zip(records, outputs):
        text = out.outputs[0].text
        match = re.search(r'[A-Za-z]', text)
        if not match:
            rec['result'] = False
            rec['predicted_option'] = ''
            continue

        letter = match.group().upper()
        count += 1
        true_answer = rec.get('answer', '').upper()
        rec['predicted_option'] = letter
        rec['result'] = (letter == true_answer)
        if letter == true_answer:
            correct += 1

    # Compute and print accuracy
    accuracy = (correct / count * 100) if count > 0 else 0.0
    print(f"{os.path.basename(args.json_file)}: {accuracy:.2f}%")

    # Save updated records back to the same file
    with open(args.json_file, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()