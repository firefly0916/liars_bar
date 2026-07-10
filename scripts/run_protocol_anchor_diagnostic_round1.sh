#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate liar_bar

TRAIN_REPO="${TRAIN_REPO:-/root/liars_bar_feat_grpo}"
EVAL_REPO="${EVAL_REPO:-/root/liars_bar}"
AUDIT_REPO="${AUDIT_REPO:-/root/liars_bar_dev_proxy_refine}"

RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/experiments/20260508-protocol-diagnostic-r1}"
TRAIN_ROOT="$RUN_ROOT/train"
SCREEN_ROOT="$RUN_ROOT/checkpoint_screening"
REPORT_ROOT="$RUN_ROOT/reports"
FORMAL_ROOT="$RUN_ROOT/formal_eval_top3"
SCORECARD_ROOT="$RUN_ROOT/scorecards"

DATASET_PATH="${DATASET_PATH:-/root/liars_bar_dev_proxy_refine/logs/task_n_hicra_preprocessed/20260504-task2_1-english/savi_alignment_train.jsonl}"
POLICY_MODEL_PATH="${POLICY_MODEL_PATH:-/root/autodl-tmp/models/huggingface/Qwen/Qwen2.5-7B-Instruct}"
EVAL_MODEL_NAME="${EVAL_MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
LOCAL_LLM_MODEL_PATH="${LOCAL_LLM_MODEL_PATH:-$POLICY_MODEL_PATH}"
PROXY_MODEL_PATH="${PROXY_MODEL_PATH:-/root/liars_bar_feat_grpo/models/proxy/value_proxy_mlp_distill.pt}"

TASKM_CONFIG="${TASKM_CONFIG:-/root/liars_bar/config/experiment.yaml}"
AUDIT_CONFIG="${AUDIT_CONFIG:-/root/liars_bar_dev_proxy_refine/config/experiment.yaml}"

TRAIN_STEPS="${TRAIN_STEPS:-200}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-5}"
DENSE_UNTIL_STEP="${DENSE_UNTIL_STEP:-100}"
DENSE_INTERVAL="${DENSE_INTERVAL:-5}"
SPARSE_INTERVAL="${SPARSE_INTERVAL:-10}"
SCREEN_GAMES="${SCREEN_GAMES:-10}"
SCREEN_RANDOM_SEED="${SCREEN_RANDOM_SEED:-4242}"
SHORTLIST_TOP_K="${SHORTLIST_TOP_K:-5}"
SHORTLIST_TARGET_LLM_TURN_COUNT="${SHORTLIST_TARGET_LLM_TURN_COUNT:-100}"
SHORTLIST_MAX_CONFLICT_COUNT="${SHORTLIST_MAX_CONFLICT_COUNT:-1}"
FORMAL_TOP_K="${FORMAL_TOP_K:-3}"
FORMAL_GAMES="${FORMAL_GAMES:-100}"
LOCAL_LLM_MAX_NEW_TOKENS="${LOCAL_LLM_MAX_NEW_TOKENS:-192}"

ACTION_MATCH_REWARD_WEIGHT="${ACTION_MATCH_REWARD_WEIGHT:-0.10}"
PARSE_ERROR_PENALTY="${PARSE_ERROR_PENALTY:--0.75}"
FALLBACK_USED_PENALTY="${FALLBACK_USED_PENALTY:--0.35}"
RESOLUTION_REPAIR_PENALTY="${RESOLUTION_REPAIR_PENALTY:--0.15}"
PROTOCOL_PENALTY_WARMUP_START="${PROTOCOL_PENALTY_WARMUP_START:-0}"
PROTOCOL_PENALTY_WARMUP_END="${PROTOCOL_PENALTY_WARMUP_END:-0}"

timestamp_now() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

log_done() {
  local stage="$1"
  local summary="$2"
  echo "===== ${stage} DONE $(timestamp_now) :: ${summary} ====="
}

echo "===== START DIAGNOSTIC ROUND1 $(date) ====="

test -f "$TRAIN_REPO/scripts/train_savi_alignment.py"
test -f "$EVAL_REPO/scripts/run_llm_drill.py"
test -f "$EVAL_REPO/scripts/list_diagnostic_checkpoints.py"
test -f "$EVAL_REPO/scripts/select_screening_candidates.py"
test -f "$EVAL_REPO/scripts/build_diagnostic_trajectory_report.py"
test -f "$EVAL_REPO/scripts/run_selected_formal_eval.py"
test -f "$EVAL_REPO/scripts/build_eval_scorecard.py"
test -f "$AUDIT_REPO/scripts/audit_llm_behavior.py"
test -f "$DATASET_PATH"
test -f "$PROXY_MODEL_PATH"
test -f "$POLICY_MODEL_PATH/config.json"
mkdir -p "$RUN_ROOT" "$TRAIN_ROOT" "$SCREEN_ROOT" "$REPORT_ROOT" "$FORMAL_ROOT" "$SCORECARD_ROOT"

echo "===== TRAIN ${TRAIN_STEPS}-STEP $(date) ====="
cd "$TRAIN_REPO"
python scripts/train_savi_alignment.py \
  "$DATASET_PATH" \
  --policy-model-path "$POLICY_MODEL_PATH" \
  --model-path "$PROXY_MODEL_PATH" \
  --device cuda \
  --torch-dtype bf16 \
  --load-in-4bit \
  --group-size 8 \
  --steps "$TRAIN_STEPS" \
  --action-match-reward-weight "$ACTION_MATCH_REWARD_WEIGHT" \
  --parse-error-penalty "$PARSE_ERROR_PENALTY" \
  --fallback-used-penalty "$FALLBACK_USED_PENALTY" \
  --resolution-repair-penalty "$RESOLUTION_REPAIR_PENALTY" \
  --protocol-penalty-warmup-start "$PROTOCOL_PENALTY_WARMUP_START" \
  --protocol-penalty-warmup-end "$PROTOCOL_PENALTY_WARMUP_END" \
  --checkpoint-dir "$TRAIN_ROOT/checkpoints" \
  --save-every-steps "$SAVE_EVERY_STEPS" \
  --save-final-adapter \
  --skip-optimizer-state \
  --output-path "$TRAIN_ROOT/train_summary.json" \
  > "$TRAIN_ROOT/stdout.log" 2>&1
log_done "TRAIN" "summary=$TRAIN_ROOT/train_summary.json final_adapter=$TRAIN_ROOT/checkpoints/final"

echo "===== BUILD CHECKPOINT LIST $(date) ====="
cd "$EVAL_REPO"
python scripts/list_diagnostic_checkpoints.py \
  --max-step "$TRAIN_STEPS" \
  --dense-until-step "$DENSE_UNTIL_STEP" \
  --dense-interval "$DENSE_INTERVAL" \
  --sparse-interval "$SPARSE_INTERVAL" \
  > "$RUN_ROOT/checkpoint_tags.txt"

while IFS= read -r TAG
do
  test -n "$TAG" || continue
  CKPT="$TRAIN_ROOT/checkpoints/$TAG"
  TASKM_OUT="$SCREEN_ROOT/$TAG/task_m"
  AUDIT_OUT="$SCREEN_ROOT/$TAG/task_1_1"
  mkdir -p "$TASKM_OUT" "$AUDIT_OUT"

  echo "===== SCREEN TASK_M $TAG $(date) ====="
  cd "$EVAL_REPO"
  LOCAL_LLM_DEVICE=cuda \
  LOCAL_LLM_LOCAL_FILES_ONLY=1 \
  LOCAL_LLM_MAX_NEW_TOKENS="$LOCAL_LLM_MAX_NEW_TOKENS" \
  LOCAL_LLM_MODEL_PATH="$LOCAL_LLM_MODEL_PATH" \
  LOCAL_LLM_ADAPTER_PATH="$CKPT" \
  python scripts/run_llm_drill.py \
    --config "$TASKM_CONFIG" \
    --expected-llm-model "$EVAL_MODEL_NAME" \
    --games "$SCREEN_GAMES" \
    --random-seed "$SCREEN_RANDOM_SEED" \
    --log-dir "$TASKM_OUT" \
    > "$SCREEN_ROOT/$TAG/task_m_stdout.log" 2>&1
  log_done "SCREEN TASK_M $TAG" "log=$SCREEN_ROOT/$TAG/task_m_stdout.log summary=$TASKM_OUT/summary.json"

  echo "===== SCREEN AUDIT $TAG $(date) ====="
  cd "$AUDIT_REPO"
  python scripts/audit_llm_behavior.py \
    "$TASKM_OUT" \
    --model-path "$PROXY_MODEL_PATH" \
    --output-dir "$AUDIT_OUT" \
    --phi-threshold -0.1 \
    --potential-point-threshold 0.15 \
    --llm-player-id p1 \
    --config-file "$AUDIT_CONFIG" \
    --summary-path "$AUDIT_OUT/summary.json" \
    > "$SCREEN_ROOT/$TAG/task_1_1_stdout.log" 2>&1
  log_done "SCREEN AUDIT $TAG" "log=$SCREEN_ROOT/$TAG/task_1_1_stdout.log summary=$AUDIT_OUT/summary.json"
done < "$RUN_ROOT/checkpoint_tags.txt"

echo "===== BUILD TRAJECTORY REPORT $(date) ====="
cd "$EVAL_REPO"
python scripts/build_diagnostic_trajectory_report.py \
  "$SCREEN_ROOT" \
  --format markdown \
  --output "$REPORT_ROOT/trajectory_report.md"

python scripts/build_diagnostic_trajectory_report.py \
  "$SCREEN_ROOT" \
  --format json \
  --output "$REPORT_ROOT/trajectory_report.json"
log_done "TRAJECTORY REPORT" "markdown=$REPORT_ROOT/trajectory_report.md json=$REPORT_ROOT/trajectory_report.json"

echo "===== BUILD SHORTLIST $(date) ====="
python scripts/select_screening_candidates.py \
  "$SCREEN_ROOT" \
  --top-k "$SHORTLIST_TOP_K" \
  --max-conflict-count "$SHORTLIST_MAX_CONFLICT_COUNT" \
  --target-llm-turn-count "$SHORTLIST_TARGET_LLM_TURN_COUNT" \
  --format json \
  --output "$REPORT_ROOT/shortlist_selection.json"

python scripts/select_screening_candidates.py \
  "$SCREEN_ROOT" \
  --top-k "$SHORTLIST_TOP_K" \
  --max-conflict-count "$SHORTLIST_MAX_CONFLICT_COUNT" \
  --target-llm-turn-count "$SHORTLIST_TARGET_LLM_TURN_COUNT" \
  --format markdown \
  --output "$REPORT_ROOT/shortlist_selection.md"
log_done "SHORTLIST" "json=$REPORT_ROOT/shortlist_selection.json markdown=$REPORT_ROOT/shortlist_selection.md top_k=$SHORTLIST_TOP_K"

echo "===== FORMAL TOP${FORMAL_TOP_K} $(date) ====="
cd "$EVAL_REPO"
python scripts/run_selected_formal_eval.py \
  "$RUN_ROOT" \
  --selection-json "$REPORT_ROOT/shortlist_selection.json" \
  --max-tags "$FORMAL_TOP_K" \
  --games "$FORMAL_GAMES" \
  --output-dir-name "$(basename "$FORMAL_ROOT")" \
  --eval-repo "$EVAL_REPO" \
  --audit-repo "$AUDIT_REPO" \
  --taskm-config "$TASKM_CONFIG" \
  --audit-config "$AUDIT_CONFIG" \
  --proxy-model-path "$PROXY_MODEL_PATH" \
  --expected-llm-model "$EVAL_MODEL_NAME" \
  --local-llm-device cuda \
  --local-llm-model-path "$LOCAL_LLM_MODEL_PATH" \
  --local-llm-max-new-tokens "$LOCAL_LLM_MAX_NEW_TOKENS" \
  > "$FORMAL_ROOT/master.log" 2>&1
log_done "FORMAL TOP${FORMAL_TOP_K}" "master_log=$FORMAL_ROOT/master.log root=$FORMAL_ROOT"

echo "===== BUILD SCORECARD $(date) ====="
python scripts/build_eval_scorecard.py \
  "$FORMAL_ROOT"/* \
  --format markdown \
  --output "$SCORECARD_ROOT/final_scorecard.md"
python scripts/build_eval_scorecard.py \
  "$FORMAL_ROOT"/* \
  --format json \
  --output "$SCORECARD_ROOT/final_scorecard.json"
log_done "SCORECARD" "markdown=$SCORECARD_ROOT/final_scorecard.md json=$SCORECARD_ROOT/final_scorecard.json"

echo "===== DIAGNOSTIC ROUND1 DONE $(date) ====="
