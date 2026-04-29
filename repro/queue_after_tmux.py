import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wait for a tmux session to finish, then launch a queued command with timestamped logs."
    )
    parser.add_argument("--wait_session", required=True, help="tmux session name to wait for")
    parser.add_argument("--command", required=True, help="shell command to run after the session ends")
    parser.add_argument("--log_path", required=True, help="queue log path")
    parser.add_argument("--name", required=True, help="human-readable queue job name")
    parser.add_argument("--workdir", default=None, help="working directory for the queued command")
    parser.add_argument("--poll_seconds", type=int, default=300, help="poll interval in seconds")
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def session_exists(session_name: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", session_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def log_line(handle, message: str) -> None:
    line = f"{timestamp()} | {message}\n"
    handle.write(line)
    handle.flush()
    sys.stdout.write(line)
    sys.stdout.flush()


def main() -> int:
    args = parse_args()
    log_path = Path(args.log_path)
    ensure_parent(log_path)
    workdir = args.workdir or os.getcwd()

    with log_path.open("a", encoding="utf-8") as handle:
        log_line(handle, f"QUEUE START | name={args.name} | wait_session={args.wait_session}")
        log_line(handle, f"QUEUE COMMAND | {args.command}")
        while session_exists(args.wait_session):
            log_line(handle, f"WAITING | session={args.wait_session} still active")
            time.sleep(args.poll_seconds)

        log_line(handle, f"LAUNCH | session={args.wait_session} ended, starting queued command")
        process = subprocess.Popen(
            args.command,
            shell=True,
            cwd=workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            executable="/bin/bash",
        )
        assert process.stdout is not None
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
        returncode = process.wait()
        log_line(handle, f"QUEUE FINISH | name={args.name} | returncode={returncode}")
        return returncode


if __name__ == "__main__":
    raise SystemExit(main())
