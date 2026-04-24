#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs/task_k_gold
timestamp="$(date +%Y%m%d-%H%M%S)"
run_log="logs/task_k_gold/task-k-gold-${timestamp}.log"

echo "Starting Task K gold run. stdout/stderr -> ${run_log}"
echo "Progress checkpoints -> logs/task_k_gold/progress.log"

conda run -n liar_bar python -m liars_game_engine.analysis.task_k_gold_runner "$@" 2>&1 | tee "$run_log"
