#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs/task_k_gold
timestamp="$(date +%Y%m%d-%H%M%S)"
run_dir="logs/task_k_gold/${timestamp}"
mkdir -p "${run_dir}"
run_log="${run_dir}/run.log"

echo "Starting Task K gold run. stdout/stderr -> ${run_log}"
echo "Output dir -> ${run_dir}"
echo "Progress checkpoints -> ${run_dir}/progress.log"

conda run -n liar_bar python -m liars_game_engine.analysis.task_k_gold_runner --output-dir "${run_dir}" "$@" 2>&1 | tee "$run_log"
