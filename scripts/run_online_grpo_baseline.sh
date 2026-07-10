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
TRAIN_REPO="${TRAIN_REPO:-${ROOT_DIR}}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/experiments/$(date +%Y%m%d)-online-grpo-baseline}"

TRAIN_ROOT="${RUN_ROOT}/train"
FORMAL_ROOT="${RUN_ROOT}/formal_eval/final"
TASKM_OUT="${FORMAL_ROOT}/task_m"
AUDIT_OUT="${FORMAL_ROOT}/task_1_1"
SCORECARD_OUT="${RUN_ROOT}/scorecard.md"

ONLINE_GRPO_TRAIN_SCRIPT="${ONLINE_GRPO_TRAIN_SCRIPT:-${TRAIN_REPO}/scripts/train_online_grpo_baseline.py}"
ONLINE_GRPO_ADAPTER_PATH="${ONLINE_GRPO_ADAPTER_PATH:-${TRAIN_ROOT}/checkpoints/final}"
POLICY_MODEL_PATH="${POLICY_MODEL_PATH:-/root/autodl-tmp/models/huggingface/Qwen/Qwen2.5-7B-Instruct}"
EVAL_MODEL_NAME="${EVAL_MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
LOCAL_LLM_MODEL_PATH="${LOCAL_LLM_MODEL_PATH:-${POLICY_MODEL_PATH}}"
PROXY_MODEL_PATH="${PROXY_MODEL_PATH:-/root/liars_bar_feat_grpo/models/proxy/value_proxy_mlp_distill.pt}"

TASKM_CONFIG="${TASKM_CONFIG:-${EVAL_REPO}/config/experiment.yaml}"
AUDIT_CONFIG="${AUDIT_CONFIG:-${AUDIT_REPO}/config/experiment.yaml}"

GRPO_UPDATES="${GRPO_UPDATES:-20}"
ROLLOUT_GAMES_PER_UPDATE="${ROLLOUT_GAMES_PER_UPDATE:-2}"
TRAIN_MAX_TURNS="${TRAIN_MAX_TURNS:-0}"
GRPO_GROUP_SIZE="${GRPO_GROUP_SIZE:-4}"
GRPO_EPOCHS="${GRPO_EPOCHS:-1}"
GRPO_MINI_BATCH_SIZE="${GRPO_MINI_BATCH_SIZE:-1}"
GRPO_LEARNING_RATE="${GRPO_LEARNING_RATE:-1e-6}"
GRPO_RANDOM_SEED="${GRPO_RANDOM_SEED:-42}"
GRPO_EXTRA_ARGS="${GRPO_EXTRA_ARGS:---no-load-in-4bit --torch-dtype fp16 --max-new-tokens 192 --max-seq-len 1024 --lora-r 4 --lora-alpha 8 --temperature 0.7 --top-p 0.9}"

FORMAL_GAMES="${FORMAL_GAMES:-200}"
FORMAL_RANDOM_SEED="${FORMAL_RANDOM_SEED:-42}"
LLM_PLAYER_ID="${LLM_PLAYER_ID:-p1}"
LOCAL_LLM_DEVICE="${LOCAL_LLM_DEVICE:-cuda}"
LOCAL_LLM_MAX_NEW_TOKENS="${LOCAL_LLM_MAX_NEW_TOKENS:-192}"
PHI_THRESHOLD="${PHI_THRESHOLD:--0.1}"
POTENTIAL_POINT_THRESHOLD="${POTENTIAL_POINT_THRESHOLD:-0.15}"

echo "===== START ONLINE GRPO BASELINE $(date) ====="
echo "RUN_ROOT=${RUN_ROOT}"
echo "TRAIN_REPO=${TRAIN_REPO}"
echo "EVAL_REPO=${EVAL_REPO}"
echo "AUDIT_REPO=${AUDIT_REPO}"
echo "ONLINE_GRPO_TRAIN_SCRIPT=${ONLINE_GRPO_TRAIN_SCRIPT}"
echo "ONLINE_GRPO_ADAPTER_PATH=${ONLINE_GRPO_ADAPTER_PATH}"
echo "POLICY_MODEL_PATH=${POLICY_MODEL_PATH}"
echo "PROXY_MODEL_PATH=${PROXY_MODEL_PATH}"
echo "GRPO_UPDATES=${GRPO_UPDATES}"
echo "ROLLOUT_GAMES_PER_UPDATE=${ROLLOUT_GAMES_PER_UPDATE}"
echo "TRAIN_MAX_TURNS=${TRAIN_MAX_TURNS}"
echo "GRPO_GROUP_SIZE=${GRPO_GROUP_SIZE}"
echo "FORMAL_GAMES=${FORMAL_GAMES}"

test -f "${ONLINE_GRPO_TRAIN_SCRIPT}"
test -f "${EVAL_REPO}/scripts/run_llm_drill.py"
test -f "${EVAL_REPO}/scripts/build_eval_scorecard.py"
test -f "${AUDIT_REPO}/scripts/audit_llm_behavior.py"
test -f "${TASKM_CONFIG}"
test -f "${AUDIT_CONFIG}"
test -f "${PROXY_MODEL_PATH}"
test -f "${POLICY_MODEL_PATH}/config.json"

if [[ "${ALLOW_EXISTING_RUN_ROOT:-0}" != "1" ]]; then
  if [[ -e "${TRAIN_ROOT}/train_summary.json" || -e "${AUDIT_OUT}/summary.json" || -e "${SCORECARD_OUT}" ]]; then
    echo "ERROR: RUN_ROOT already contains online GRPO baseline outputs: ${RUN_ROOT}" >&2
    exit 2
  fi
fi

mkdir -p "${TRAIN_ROOT}" "${TASKM_OUT}" "${AUDIT_OUT}"
read -r -a GRPO_EXTRA_ARGS_ARRAY <<< "${GRPO_EXTRA_ARGS}"

echo "===== ONLINE GRPO TRAIN $(date) ====="
cd "${TRAIN_REPO}"
env \
  -u LOCAL_LLM_ADAPTER_PATH \
  python "${ONLINE_GRPO_TRAIN_SCRIPT}" \
    --config "${TASKM_CONFIG}" \
    --policy-model-path "${POLICY_MODEL_PATH}" \
    --output-dir "${TRAIN_ROOT}" \
    --checkpoint-dir "${TRAIN_ROOT}/checkpoints" \
    --updates "${GRPO_UPDATES}" \
    --rollout-games-per-update "${ROLLOUT_GAMES_PER_UPDATE}" \
    --train-max-turns "${TRAIN_MAX_TURNS}" \
    --group-size "${GRPO_GROUP_SIZE}" \
    --grpo-epochs "${GRPO_EPOCHS}" \
    --mini-batch-size "${GRPO_MINI_BATCH_SIZE}" \
    --learning-rate "${GRPO_LEARNING_RATE}" \
    --random-seed "${GRPO_RANDOM_SEED}" \
    --llm-player-id "${LLM_PLAYER_ID}" \
    --save-final-adapter \
    "${GRPO_EXTRA_ARGS_ARRAY[@]}" \
    > "${TRAIN_ROOT}/stdout.log" 2>&1

test -d "${ONLINE_GRPO_ADAPTER_PATH}"

echo "===== ONLINE GRPO FORMAL TASK M $(date) ====="
cd "${EVAL_REPO}"
LOCAL_LLM_DEVICE="${LOCAL_LLM_DEVICE}" \
LOCAL_LLM_LOCAL_FILES_ONLY=1 \
LOCAL_LLM_MAX_NEW_TOKENS="${LOCAL_LLM_MAX_NEW_TOKENS}" \
LOCAL_LLM_MODEL_PATH="${LOCAL_LLM_MODEL_PATH}" \
LOCAL_LLM_ADAPTER_PATH="${ONLINE_GRPO_ADAPTER_PATH}" \
python scripts/run_llm_drill.py \
  --config "${TASKM_CONFIG}" \
  --expected-llm-model "${EVAL_MODEL_NAME}" \
  --games "${FORMAL_GAMES}" \
  --random-seed "${FORMAL_RANDOM_SEED}" \
  --log-dir "${TASKM_OUT}" \
  > "${FORMAL_ROOT}/task_m_stdout.log" 2>&1

echo "===== ONLINE GRPO FORMAL AUDIT $(date) ====="
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
  > "${FORMAL_ROOT}/task_1_1_stdout.log" 2>&1

echo "===== ONLINE GRPO SCORECARD $(date) ====="
cd "${EVAL_REPO}"
python scripts/build_eval_scorecard.py \
  "${FORMAL_ROOT}" \
  --format markdown \
  --output "${SCORECARD_OUT}"

echo "===== DONE ONLINE GRPO BASELINE $(date) ====="
echo "Train summary: ${TRAIN_ROOT}/train_summary.json"
echo "Online GRPO adapter: ${ONLINE_GRPO_ADAPTER_PATH}"
echo "Task M summary: ${TASKM_OUT}/summary.json"
echo "Audit summary: ${AUDIT_OUT}/summary.json"
echo "Scorecard: ${SCORECARD_OUT}"
