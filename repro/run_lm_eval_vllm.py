import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run lm-evaluation-harness with the vLLM backend and timestamped logs."
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--log_path", required=True)
    parser.add_argument(
        "--tasks",
        nargs="+",
        required=True,
        help="lm-eval task names, e.g. mmlu arc_challenge hellaswag truthfulqa_mc1",
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--batch_size", default="8")
    parser.add_argument("--max_batch_size", type=int, default=None)
    parser.add_argument("--limit", default=None)
    parser.add_argument("--seed", default="42")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--max_gen_toks", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--data_parallel_size", type=int, default=1)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--think_end_token", default="</think>")
    parser.add_argument("--hf_offline", action="store_true")
    parser.add_argument("--no_apply_chat_template", action="store_true")
    parser.add_argument("--no_log_samples", action="store_true")
    parser.add_argument("--extra_model_arg", action="append", default=[])
    parser.add_argument("--extra_gen_kwarg", action="append", default=[])
    parser.add_argument("--extra_cli_arg", action="append", default=[])
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def build_command(args: argparse.Namespace) -> list[str]:
    thinking_flag = "True" if args.enable_thinking else "False"
    do_sample_flag = "True" if args.temperature > 0 else "False"
    model_args = [
        f"pretrained={args.model_path}",
        f"dtype={args.dtype}",
        f"gpu_memory_utilization={args.gpu_memory_utilization}",
        f"tensor_parallel_size={args.tensor_parallel_size}",
        f"data_parallel_size={args.data_parallel_size}",
        f"max_model_len={args.max_model_len}",
        f"enable_thinking={thinking_flag}",
    ]
    if args.enable_thinking:
        model_args.append(f'think_end_token="{args.think_end_token}"')
    model_args.extend(args.extra_model_arg)

    gen_kwargs = [
        f"temperature={args.temperature}",
        f"top_p={args.top_p}",
        f"do_sample={do_sample_flag}",
        f"max_gen_toks={args.max_gen_toks}",
    ]
    gen_kwargs.extend(args.extra_gen_kwarg)

    command = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        "vllm",
        "--model_args",
        ",".join(model_args),
        "--tasks",
        ",".join(args.tasks),
        "--batch_size",
        str(args.batch_size),
        "--device",
        "cuda",
        "--gen_kwargs",
        ",".join(gen_kwargs),
        "--seed",
        str(args.seed),
        "--output_path",
        args.output_dir,
        "--trust_remote_code",
    ]
    if not args.no_apply_chat_template:
        command.append("--apply_chat_template")
        command.append("--fewshot_as_multiturn")
    if not args.no_log_samples:
        command.append("--log_samples")
    if args.max_batch_size is not None:
        command.extend(["--max_batch_size", str(args.max_batch_size)])
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    for item in args.extra_cli_arg:
        command.extend(shlex.split(item))
    return command


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_header(log_file, args: argparse.Namespace, command: list[str]) -> None:
    meta = {
        "started_at": timestamp(),
        "model_path": args.model_path,
        "output_dir": args.output_dir,
        "log_path": args.log_path,
        "tasks": args.tasks,
        "gpu": args.gpu,
        "batch_size": args.batch_size,
        "max_batch_size": args.max_batch_size,
        "limit": args.limit,
        "seed": args.seed,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "max_gen_toks": args.max_gen_toks,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": args.tensor_parallel_size,
        "data_parallel_size": args.data_parallel_size,
        "enable_thinking": args.enable_thinking,
        "hf_offline": args.hf_offline,
    }
    log_file.write(f"{timestamp()} | INFO | lm-eval run start\n")
    log_file.write(
        f"{timestamp()} | INFO | config | {json.dumps(meta, ensure_ascii=False, sort_keys=True)}\n"
    )
    log_file.write(
        f"{timestamp()} | INFO | command | {shlex.join(command)}\n"
    )
    log_file.flush()


def stream_process(
    args: argparse.Namespace, command: list[str], log_path: Path, env: dict[str, str]
) -> int:
    with log_path.open("a", encoding="utf-8") as log_file:
        write_header(log_file, args, command)
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
            log_file.flush()
        returncode = process.wait()
        log_file.write(
            f"{timestamp()} | INFO | lm-eval run finished | returncode={returncode}\n"
        )
        log_file.flush()
        return returncode


if __name__ == "__main__":
    args = parse_args()
    output_dir = Path(args.output_dir)
    log_path = Path(args.log_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_parent(log_path)

    command = build_command(args)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("VLLM_USE_V1", "1")
    for proxy_key in (
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "all_proxy",
        "ALL_PROXY",
    ):
        env.pop(proxy_key, None)
    if args.hf_offline:
        env["HF_DATASETS_OFFLINE"] = "1"
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"

    exit_code = stream_process(args, command, log_path, env)
    raise SystemExit(exit_code)
