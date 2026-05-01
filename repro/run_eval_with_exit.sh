#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <launch_log> <workdir> <command...>" >&2
  exit 2
fi

LAUNCH_LOG="$1"
WORKDIR="$2"
shift 2

mkdir -p "$(dirname "$LAUNCH_LOG")"
{
  echo "$(date '+%F %T') | START | workdir=${WORKDIR}"
  echo "$(date '+%F %T') | COMMAND | $*"
} >> "$LAUNCH_LOG"

cd "$WORKDIR"
set +e
"$@"
CODE=$?
set -e
echo "$(date '+%F %T') | EXIT | code=${CODE}" >> "$LAUNCH_LOG"
exit "$CODE"
