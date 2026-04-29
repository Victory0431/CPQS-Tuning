import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal vLLM smoke test with Qwen chat template.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--log_path", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--prompt", default="请用一句话介绍你自己。")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--enable_thinking", action="store_true")
    return parser.parse_args()


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("VLLM_USE_V1", "1")
    for proxy_key in (
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "all_proxy",
        "ALL_PROXY",
    ):
        os.environ.pop(proxy_key, None)

    output_path = Path(args.output_path)
    log_path = Path(args.log_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"{now()} | INFO | vLLM smoke start\n")
        log_file.write(
            f"{now()} | INFO | config | {json.dumps(vars(args), ensure_ascii=False, sort_keys=True)}\n"
        )
        log_file.flush()

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=True,
        )
        messages = [{"role": "user", "content": args.prompt}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )
        log_file.write(f"{now()} | INFO | prompt_ready | chars={len(prompt)}\n")
        log_file.flush()

        llm = LLM(
            model=args.model_path,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=True,
        )
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
        outputs = llm.generate([prompt], sampling_params=sampling_params)
        text = outputs[0].outputs[0].text

        result = {
            "generated_at": now(),
            "model_path": args.model_path,
            "enable_thinking": args.enable_thinking,
            "prompt": args.prompt,
            "formatted_prompt": prompt,
            "raw_output": text,
        }
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log_file.write(f"{now()} | INFO | vLLM smoke finished | saved={output_path}\n")
        log_file.flush()


if __name__ == "__main__":
    main()
