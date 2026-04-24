#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${CONDA_ENV_NAME:-liar_bar}"
SESSION_NAME="${VLLM_TMUX_SESSION:-vllm_service}"
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
HF_HOME_DIR="${HF_HOME:-${ROOT_DIR}/models/huggingface}"
LOG_DIR="${ROOT_DIR}/logs/vllm"
LOG_FILE="${LOG_DIR}/vllm-${VLLM_PORT}.log"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda command not found. Install Miniconda or Anaconda first." >&2
  exit 1
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux command not found. Install tmux before starting the vLLM service." >&2
  exit 1
fi

mkdir -p "${HF_HOME_DIR}" "${LOG_DIR}"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session ${SESSION_NAME} already exists. Attach with: tmux attach -t ${SESSION_NAME}"
  exit 0
fi

tmux new-session -d -s "${SESSION_NAME}" \
  "cd '${ROOT_DIR}' && export HF_HOME='${HF_HOME_DIR}' && conda run -n '${ENV_NAME}' vllm serve '${VLLM_MODEL}' --host '${VLLM_HOST}' --port '${VLLM_PORT}' --api-key '${VLLM_API_KEY}' --gpu-memory-utilization '${VLLM_GPU_MEMORY_UTILIZATION}' >> '${LOG_FILE}' 2>&1"

echo "vLLM service started in tmux session ${SESSION_NAME}."
echo "Model: ${VLLM_MODEL}"
echo "Endpoint: http://${VLLM_HOST}:${VLLM_PORT}/v1"
echo "Logs: ${LOG_FILE}"
echo "Attach with: tmux attach -t ${SESSION_NAME}"
