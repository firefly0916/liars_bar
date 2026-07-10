#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-liar_bar}"
if [[ -f "/root/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "/root/miniconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
fi

EVAL_REPO="${EVAL_REPO:-${ROOT_DIR}}"
AUDIT_REPO="${AUDIT_REPO:-/root/liars_bar_dev_proxy_refine}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/logs/internal_vanilla/$(date +%Y%m%d-%H%M%S)}"

TASKM_CONFIG="${TASKM_CONFIG:-${EVAL_REPO}/config/experiment.yaml}"
AUDIT_CONFIG="${AUDIT_CONFIG:-${AUDIT_REPO}/config/experiment.yaml}"
PROXY_MODEL_PATH="${PROXY_MODEL_PATH:-/root/liars_bar_feat_grpo/models/proxy/value_proxy_mlp_distill.pt}"

POLICY_MODEL_PATH="${POLICY_MODEL_PATH:-/root/autodl-tmp/models/huggingface/Qwen/Qwen2.5-7B-Instruct}"
EVAL_MODEL_NAME="${EVAL_MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
LOCAL_LLM_MODEL_PATH="${LOCAL_LLM_MODEL_PATH:-${POLICY_MODEL_PATH}}"
LOCAL_LLM_DEVICE="${LOCAL_LLM_DEVICE:-cuda}"
LOCAL_LLM_MAX_NEW_TOKENS="${LOCAL_LLM_MAX_NEW_TOKENS:-192}"

GAMES="${GAMES:-100}"
RANDOM_SEED="${RANDOM_SEED:-42}"
LLM_PLAYER_ID="${LLM_PLAYER_ID:-p1}"
PHI_THRESHOLD="${PHI_THRESHOLD:--0.1}"
POTENTIAL_POINT_THRESHOLD="${POTENTIAL_POINT_THRESHOLD:-0.15}"

TASKM_OUT="${RUN_ROOT}/task_m"
AUDIT_OUT="${RUN_ROOT}/task_1_1"
SCORECARD_OUT="${RUN_ROOT}/scorecard.md"

echo "===== START INTERNAL VANILLA EVAL $(date) ====="
echo "RUN_ROOT=${RUN_ROOT}"
echo "EVAL_REPO=${EVAL_REPO}"
echo "AUDIT_REPO=${AUDIT_REPO}"
echo "POLICY_MODEL_PATH=${POLICY_MODEL_PATH}"
echo "PROXY_MODEL_PATH=${PROXY_MODEL_PATH}"
echo "GAMES=${GAMES}"
echo "RANDOM_SEED=${RANDOM_SEED}"

test -f "${EVAL_REPO}/scripts/run_llm_drill.py"
test -f "${EVAL_REPO}/scripts/build_eval_scorecard.py"
test -f "${AUDIT_REPO}/scripts/audit_llm_behavior.py"
test -f "${TASKM_CONFIG}"
test -f "${AUDIT_CONFIG}"
test -f "${PROXY_MODEL_PATH}"
test -f "${POLICY_MODEL_PATH}/config.json"

mkdir -p "${TASKM_OUT}" "${AUDIT_OUT}"

echo "===== TASK M VANILLA BASE MODEL $(date) ====="
cd "${EVAL_REPO}"
env \
  -u LOCAL_LLM_ADAPTER_PATH \
  LOCAL_LLM_DEVICE="${LOCAL_LLM_DEVICE}" \
  LOCAL_LLM_LOCAL_FILES_ONLY=1 \
  LOCAL_LLM_MAX_NEW_TOKENS="${LOCAL_LLM_MAX_NEW_TOKENS}" \
  LOCAL_LLM_MODEL_PATH="${LOCAL_LLM_MODEL_PATH}" \
  python scripts/run_llm_drill.py \
    --config "${TASKM_CONFIG}" \
    --expected-llm-model "${EVAL_MODEL_NAME}" \
    --games "${GAMES}" \
    --random-seed "${RANDOM_SEED}" \
    --log-dir "${TASKM_OUT}" \
    > "${RUN_ROOT}/task_m_stdout.log" 2>&1

echo "===== AUDIT VANILLA $(date) ====="
cd "${AUDIT_REPO}"
python scripts/audit_llm_behavior.py \
  "${TASKM_OUT}" \
  --model-path "${PROXY_MODEL_PATH}" \
  --output-dir "${AUDIT_OUT}" \
  --phi-threshold "${PHI_THRESHOLD}" \
  --potential-point-threshold "${POTENTIAL_POINT_THRESHOLD}" \
  --llm-player-id "${LLM_PLAYER_ID}" \
  --config-file "${AUDIT_CONFIG}" \
  --summary-path "${AUDIT_OUT}/summary.json" \
  > "${RUN_ROOT}/task_1_1_stdout.log" 2>&1

echo "===== SCORECARD VANILLA $(date) ====="
cd "${EVAL_REPO}"
python scripts/build_eval_scorecard.py \
  "${RUN_ROOT}" \
  --format markdown \
  --output "${SCORECARD_OUT}"

echo "===== DONE INTERNAL VANILLA EVAL $(date) ====="
echo "Task M summary: ${TASKM_OUT}/summary.json"
echo "Audit summary: ${AUDIT_OUT}/summary.json"
echo "Scorecard: ${SCORECARD_OUT}"
