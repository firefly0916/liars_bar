#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${CONDA_ENV_NAME:-liar_bar}"
RUN_MODE="${RUN_MODE:-task-k}"
CURRENT_BRANCH="${GIT_BRANCH:-$(git -C "${ROOT_DIR}" rev-parse --abbrev-ref HEAD)}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"

cd "${ROOT_DIR}"

mkdir -p logs logs/task_k_gold
git pull --ff-only origin "${CURRENT_BRANCH}"

case "${RUN_MODE}" in
  task-k)
    RUN_DIR="logs/task_k_gold/${TIMESTAMP}"
    mkdir -p "${RUN_DIR}"
    RUN_LOG="${RUN_DIR}/run.log"
    nohup conda run -n "${ENV_NAME}" python -m liars_game_engine.analysis.task_k_gold_runner --output-dir "${RUN_DIR}" > "${RUN_LOG}" 2>&1 &
    echo "Started Task K gold pipeline in background."
    echo "Output dir: ${RUN_DIR}"
    echo "Run log: ${RUN_LOG}"
    echo "Progress log: ${RUN_DIR}/progress.log"
    ;;
  llm)
    export OPENAI_API_KEY="${OPENAI_API_KEY:-${VLLM_API_KEY:-EMPTY}}"
    export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://${VLLM_HOST:-127.0.0.1}:${VLLM_PORT:-8000}/v1}"
    RUN_LOG="logs/server-llm-${TIMESTAMP}.log"
    nohup conda run -n "${ENV_NAME}" python -m liars_game_engine.main > "${RUN_LOG}" 2>&1 &
    echo "Started LLM game run in background."
    echo "Run log: ${RUN_LOG}"
    echo "Using base URL: ${OPENAI_BASE_URL}"
    ;;
  *)
    echo "Unsupported RUN_MODE=${RUN_MODE}. Use task-k or llm." >&2
    exit 1
    ;;
esac
