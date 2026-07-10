#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-liar_bar}"
if [[ -f "/root/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "/root/miniconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
fi

TRAIN_REPO="${TRAIN_REPO:-${ROOT_DIR}}"
EVAL_REPO="${EVAL_REPO:-/root/liars_bar}"
AUDIT_REPO="${AUDIT_REPO:-/root/liars_bar_dev_proxy_refine}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/experiments/$(date +%Y%m%d)-crc-gra-ablation-suite}"

ABLATION_TRAIN_SCRIPT="${ABLATION_TRAIN_SCRIPT:-${TRAIN_REPO}/scripts/train_crc_gra_ablation.py}"
DATASET_PATH="${DATASET_PATH:-/root/liars_bar_dev_proxy_refine/logs/task_n_hicra_preprocessed/20260504-task2_1-english/savi_alignment_train.jsonl}"
POLICY_MODEL_PATH="${POLICY_MODEL_PATH:-/root/autodl-tmp/models/huggingface/Qwen/Qwen2.5-7B-Instruct}"
EVAL_MODEL_NAME="${EVAL_MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
LOCAL_LLM_MODEL_PATH="${LOCAL_LLM_MODEL_PATH:-${POLICY_MODEL_PATH}}"
PROXY_MODEL_PATH="${PROXY_MODEL_PATH:-/root/liars_bar_feat_grpo/models/proxy/value_proxy_mlp_distill.pt}"

TASKM_CONFIG="${TASKM_CONFIG:-${EVAL_REPO}/config/experiment.yaml}"
AUDIT_CONFIG="${AUDIT_CONFIG:-${AUDIT_REPO}/config/experiment.yaml}"

ABLATION_VARIANTS="${ABLATION_VARIANTS:-action_only_proxy no_token_localization proxy_target_only logged_only random_target heuristic_target}"
ABLATION_STEPS="${ABLATION_STEPS:-200}"
ABLATION_GROUP_SIZE="${ABLATION_GROUP_SIZE:-8}"
ABLATION_LEARNING_RATE="${ABLATION_LEARNING_RATE:-1e-4}"
ABLATION_KL_BETA="${ABLATION_KL_BETA:-0.05}"
ABLATION_HICRA_GAMMA="${ABLATION_HICRA_GAMMA:-1.0}"
ABLATION_MAX_GRAD_NORM="${ABLATION_MAX_GRAD_NORM:-1.0}"
ABLATION_MAX_RECORDS="${ABLATION_MAX_RECORDS:-}"
ABLATION_ANCHOR_RATIO="${ABLATION_ANCHOR_RATIO:-0.7}"
ABLATION_ANCHOR_ALPHA="${ABLATION_ANCHOR_ALPHA:-1.0}"
ABLATION_EV_GAP_THRESHOLD="${ABLATION_EV_GAP_THRESHOLD:-0.15}"
ABLATION_ACTION_MATCH_REWARD_WEIGHT="${ABLATION_ACTION_MATCH_REWARD_WEIGHT:-0.25}"
ABLATION_PROTOCOL_WARMUP_START="${ABLATION_PROTOCOL_WARMUP_START:-60}"
ABLATION_PROTOCOL_WARMUP_END="${ABLATION_PROTOCOL_WARMUP_END:-120}"
ABLATION_EXTRA_ARGS="${ABLATION_EXTRA_ARGS:---torch-dtype fp16 --load-in-4bit --lora-r 8 --lora-alpha 16 --lora-dropout 0.05 --max-seq-len 1024 --skip-optimizer-state}"

FORMAL_GAMES="${FORMAL_GAMES:-200}"
FORMAL_RANDOM_SEED="${FORMAL_RANDOM_SEED:-42}"
LLM_PLAYER_ID="${LLM_PLAYER_ID:-p1}"
LOCAL_LLM_DEVICE="${LOCAL_LLM_DEVICE:-cuda}"
LOCAL_LLM_MAX_NEW_TOKENS="${LOCAL_LLM_MAX_NEW_TOKENS:-192}"
PHI_THRESHOLD="${PHI_THRESHOLD:--0.1}"
POTENTIAL_POINT_THRESHOLD="${POTENTIAL_POINT_THRESHOLD:-0.15}"

echo "===== START CRC-GRA ABLATION SUITE $(date) ====="
echo "RUN_ROOT=${RUN_ROOT}"
echo "TRAIN_REPO=${TRAIN_REPO}"
echo "EVAL_REPO=${EVAL_REPO}"
echo "AUDIT_REPO=${AUDIT_REPO}"
echo "DATASET_PATH=${DATASET_PATH}"
echo "POLICY_MODEL_PATH=${POLICY_MODEL_PATH}"
echo "PROXY_MODEL_PATH=${PROXY_MODEL_PATH}"
echo "ABLATION_VARIANTS=${ABLATION_VARIANTS}"
echo "ABLATION_STEPS=${ABLATION_STEPS}"
echo "FORMAL_GAMES=${FORMAL_GAMES}"

test -f "${ABLATION_TRAIN_SCRIPT}"
test -f "${DATASET_PATH}"
test -f "${PROXY_MODEL_PATH}"
test -f "${POLICY_MODEL_PATH}/config.json"
test -f "${EVAL_REPO}/scripts/run_llm_drill.py"
test -f "${EVAL_REPO}/scripts/build_eval_scorecard.py"
test -f "${AUDIT_REPO}/scripts/audit_llm_behavior.py"
test -f "${TASKM_CONFIG}"
test -f "${AUDIT_CONFIG}"

if [[ "${ALLOW_EXISTING_RUN_ROOT:-0}" != "1" && -e "${RUN_ROOT}/suite_scorecard.md" ]]; then
  echo "ERROR: RUN_ROOT already contains ablation suite outputs: ${RUN_ROOT}" >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}"
read -r -a VARIANT_ARRAY <<< "${ABLATION_VARIANTS}"
read -r -a ABLATION_EXTRA_ARGS_ARRAY <<< "${ABLATION_EXTRA_ARGS}"

SCORECARD_LIST=()
for variant in "${VARIANT_ARRAY[@]}"; do
  VARIANT_ROOT="${RUN_ROOT}/${variant}"
  TRAIN_ROOT="${VARIANT_ROOT}/train"
  FORMAL_ROOT="${VARIANT_ROOT}/formal_eval/final"
  TASKM_OUT="${FORMAL_ROOT}/task_m"
  AUDIT_OUT="${FORMAL_ROOT}/task_1_1"
  SCORECARD_OUT="${VARIANT_ROOT}/scorecard.md"
  ADAPTER_PATH="${TRAIN_ROOT}/checkpoints/final"

  echo "===== ABLATION ${variant} TRAIN $(date) ====="
  mkdir -p "${TRAIN_ROOT}" "${TASKM_OUT}" "${AUDIT_OUT}"
  cd "${TRAIN_REPO}"
  TRAIN_ARGS=(
    "${DATASET_PATH}"
    --policy-model-path "${POLICY_MODEL_PATH}"
    --model-path "${PROXY_MODEL_PATH}"
    --ablation-variant "${variant}"
    --group-size "${ABLATION_GROUP_SIZE}"
    --steps "${ABLATION_STEPS}"
    --learning-rate "${ABLATION_LEARNING_RATE}"
    --kl-beta "${ABLATION_KL_BETA}"
    --hicra-gamma "${ABLATION_HICRA_GAMMA}"
    --max-grad-norm "${ABLATION_MAX_GRAD_NORM}"
    --anchor-ratio "${ABLATION_ANCHOR_RATIO}"
    --anchor-alpha "${ABLATION_ANCHOR_ALPHA}"
    --ev-gap-threshold "${ABLATION_EV_GAP_THRESHOLD}"
    --action-match-reward-weight "${ABLATION_ACTION_MATCH_REWARD_WEIGHT}"
    --protocol-penalty-warmup-start "${ABLATION_PROTOCOL_WARMUP_START}"
    --protocol-penalty-warmup-end "${ABLATION_PROTOCOL_WARMUP_END}"
    --checkpoint-dir "${TRAIN_ROOT}/checkpoints"
    --save-final-adapter
  )
  if [[ -n "${ABLATION_MAX_RECORDS}" ]]; then
    TRAIN_ARGS+=(--max-records "${ABLATION_MAX_RECORDS}")
  fi
  env -u LOCAL_LLM_ADAPTER_PATH python "${ABLATION_TRAIN_SCRIPT}" \
    "${TRAIN_ARGS[@]}" \
    "${ABLATION_EXTRA_ARGS_ARRAY[@]}" \
    --output-path "${TRAIN_ROOT}/train_summary.json" \
    > "${TRAIN_ROOT}/stdout.log" 2>&1

  test -d "${ADAPTER_PATH}"

  echo "===== ABLATION ${variant} FORMAL TASK M $(date) ====="
  cd "${EVAL_REPO}"
  LOCAL_LLM_DEVICE="${LOCAL_LLM_DEVICE}" \
  LOCAL_LLM_LOCAL_FILES_ONLY=1 \
  LOCAL_LLM_MAX_NEW_TOKENS="${LOCAL_LLM_MAX_NEW_TOKENS}" \
  LOCAL_LLM_MODEL_PATH="${LOCAL_LLM_MODEL_PATH}" \
  LOCAL_LLM_ADAPTER_PATH="${ADAPTER_PATH}" \
  python scripts/run_llm_drill.py \
    --config "${TASKM_CONFIG}" \
    --expected-llm-model "${EVAL_MODEL_NAME}" \
    --games "${FORMAL_GAMES}" \
    --random-seed "${FORMAL_RANDOM_SEED}" \
    --log-dir "${TASKM_OUT}" \
    > "${FORMAL_ROOT}/task_m_stdout.log" 2>&1

  echo "===== ABLATION ${variant} AUDIT $(date) ====="
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

  echo "===== ABLATION ${variant} SCORECARD $(date) ====="
  cd "${EVAL_REPO}"
  python scripts/build_eval_scorecard.py \
    "${FORMAL_ROOT}" \
    --format markdown \
    --output "${SCORECARD_OUT}"
  SCORECARD_LIST+=("${SCORECARD_OUT}")
done

{
  echo "| variant | scorecard |"
  echo "| --- | --- |"
  for path in "${SCORECARD_LIST[@]}"; do
    variant="$(basename "$(dirname "${path}")")"
    echo "| ${variant} | ${path} |"
  done
} > "${RUN_ROOT}/suite_scorecard.md"

echo "===== DONE CRC-GRA ABLATION SUITE $(date) ====="
echo "Suite root: ${RUN_ROOT}"
echo "Suite scorecard: ${RUN_ROOT}/suite_scorecard.md"
