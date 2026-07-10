#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${CONDA_ENV_NAME:-liar_bar}"
CONFIG_PATH="${CONFIG_PATH:-config/experiment.yaml}"
GAMES="${GAMES:-5}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/logs/task_m_llm_drill/$(date +%Y%m%d-%H%M%S)}"
OPENAI_API_KEY_VALUE="${OPENAI_API_KEY:-${VLLM_API_KEY:-EMPTY}}"
OPENAI_BASE_URL_VALUE="${OPENAI_BASE_URL:-http://${VLLM_HOST:-127.0.0.1}:${VLLM_PORT:-8000}/v1}"
OPENAI_TIMEOUT_VALUE="${OPENAI_TIMEOUT_SECONDS:-120}"
TAIL_PROGRESS_CMD='tail -f "$OUT_DIR/progress.log"'
TAIL_STDOUT_CMD='tail -f "$OUT_DIR/stdout.log"'

cd "${ROOT_DIR}"
mkdir -p "${OUT_DIR}"
: > "${OUT_DIR}/stdout.log"
: > "${OUT_DIR}/progress.log"

nohup conda run -n "${ENV_NAME}" env \
  OPENAI_API_KEY="${OPENAI_API_KEY_VALUE}" \
  OPENAI_BASE_URL="${OPENAI_BASE_URL_VALUE}" \
  OPENAI_TIMEOUT_SECONDS="${OPENAI_TIMEOUT_VALUE}" \
  python scripts/run_llm_drill.py \
  --config "${CONFIG_PATH}" \
  --games "${GAMES}" \
  --log-dir "${OUT_DIR}" > "${OUT_DIR}/stdout.log" 2>&1 &

echo "Started Task M drill in background."
echo "Output dir: ${OUT_DIR}"
echo "Stdout log: ${OUT_DIR}/stdout.log"
echo "Progress log: ${OUT_DIR}/progress.log"
echo "Tail progress: ${TAIL_PROGRESS_CMD}"
echo "Tail stdout: ${TAIL_STDOUT_CMD}"
