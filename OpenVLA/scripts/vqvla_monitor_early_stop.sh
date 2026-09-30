#!/usr/bin/env bash
# Stop one evaluator only when its flushed log proves the configured failure
# budget has been exhausted.  This is for already-launched evaluators that do
# not have run_libero_eval.py's --max_failures option.
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 <pid> <evaluator-log> <max-failures>" >&2
    exit 2
fi

pid="$1"
log_path="$2"
max_failures="$3"

while kill -0 "$pid" 2>/dev/null; do
    read -r episodes successes < <(
        awk '
            /# episodes completed so far:/ { episodes = $NF }
            /# successes:/ { successes = $3 }
            END { if (episodes != "" && successes != "") print episodes, successes }
        ' "$log_path"
    ) || true

    if [[ -n "${episodes:-}" && -n "${successes:-}" ]] \
        && (( episodes - successes >= max_failures )); then
        echo "Stopping PID $pid: $((episodes - successes)) failures reached limit $max_failures." >&2
        kill -TERM "$pid"
        exit 0
    fi
    sleep 30
done
