#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-/home/hjt/workspace/liars_bar_migration_20260617/experiments/20260620-action-json-eval-schema-repair/formal_200g}"
EVAL_REPO="${EVAL_REPO:-/home/hjt/workspace/liars_bar_migration_20260617/code/liars_bar}"
AUDIT_REPO="${AUDIT_REPO:-/home/hjt/workspace/liars_bar_migration_20260617/code/liars_bar_dev_proxy_refine}"
MODEL_PATH="${MODEL_PATH:-/home/hjt/workspace/liars_bar_migration_20260617/models/huggingface/Qwen/Qwen2.5-7B-Instruct}"
ADAPTER_PATH="${ADAPTER_PATH:-/home/hjt/workspace/liars_bar_migration_20260617/experiments/20260618-conservative-action-json/formal_200g/conservative_expanded_action_json/train/checkpoints/final}"
PROXY_MODEL_PATH="${PROXY_MODEL_PATH:-/home/hjt/workspace/liars_bar_migration_20260617/models/proxy/value_proxy_mlp_distill.pt}"
VARIANT_LABEL="${VARIANT_LABEL:-conservative_expanded_action_json_eval_repair}"
GAMES="${GAMES:-200}"
RANDOM_SEED="${RANDOM_SEED:-42}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"

VARIANT_ROOT="${RUN_ROOT}/${VARIANT_LABEL}"
FORMAL_ROOT="${VARIANT_ROOT}/formal_eval/final"
TASKM_OUT="${FORMAL_ROOT}/task_m"
AUDIT_OUT="${FORMAL_ROOT}/task_1_1"

mkdir -p "${TASKM_OUT}" "${AUDIT_OUT}"

echo "===== ACTION JSON EVAL REPAIR START $(date) ====="
echo "RUN_ROOT=${RUN_ROOT}"
echo "EVAL_REPO=${EVAL_REPO}"
echo "AUDIT_REPO=${AUDIT_REPO}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "ADAPTER_PATH=${ADAPTER_PATH}"
echo "PROXY_MODEL_PATH=${PROXY_MODEL_PATH}"
echo "GAMES=${GAMES}"
echo "RANDOM_SEED=${RANDOM_SEED}"
echo "CUDA_DEVICE=${CUDA_DEVICE}"

cd "${EVAL_REPO}"
CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
LOCAL_LLM_DEVICE=cuda \
LOCAL_LLM_LOCAL_FILES_ONLY=1 \
LOCAL_LLM_MAX_NEW_TOKENS=96 \
LOCAL_LLM_MODEL_PATH="${MODEL_PATH}" \
LOCAL_LLM_ADAPTER_PATH="${ADAPTER_PATH}" \
/home/hjt/miniconda3/envs/liars_eval/bin/python scripts/run_llm_drill.py \
  --config config/experiment_action_only.yaml \
  --expected-llm-model Qwen/Qwen2.5-7B-Instruct \
  --games "${GAMES}" \
  --random-seed "${RANDOM_SEED}" \
  --log-dir "${TASKM_OUT}" \
  > "${FORMAL_ROOT}/task_m_stdout.log" 2>&1

echo "===== ACTION JSON EVAL REPAIR AUDIT $(date) ====="
cd "${AUDIT_REPO}"
/home/hjt/miniconda3/envs/liars_eval/bin/python scripts/audit_llm_behavior.py \
  "${TASKM_OUT}" \
  --model-path "${PROXY_MODEL_PATH}" \
  --output-dir "${AUDIT_OUT}" \
  --phi-threshold -0.1 \
  --potential-point-threshold 0.15 \
  --llm-player-id p1 \
  --config-file config/experiment.yaml \
  --summary-path "${AUDIT_OUT}/summary.json" \
  > "${FORMAL_ROOT}/task_1_1_stdout.log" 2>&1

echo "===== ACTION JSON EVAL REPAIR SCORECARD $(date) ====="
cd "${EVAL_REPO}"
/home/hjt/miniconda3/envs/liars_eval/bin/python scripts/build_eval_scorecard.py \
  "${FORMAL_ROOT}" \
  --format markdown \
  --output "${VARIANT_ROOT}/scorecard.md"

{
  echo "| variant | scorecard |"
  echo "| --- | --- |"
  echo "| ${VARIANT_LABEL} | ${VARIANT_ROOT}/scorecard.md |"
} > "${RUN_ROOT}/suite_scorecard.md"

echo "===== ACTION JSON EVAL REPAIR DONE $(date) ====="
