#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate liar_bar

TRAIN_REPO="${TRAIN_REPO:-/root/liars_bar_feat_grpo}"
EVAL_REPO="${EVAL_REPO:-/root/liars_bar}"
AUDIT_REPO="${AUDIT_REPO:-/root/liars_bar_dev_proxy_refine}"

RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/experiments/20260507-protocol-anchor-pipeline}"
TRAIN_ROOT="$RUN_ROOT/train"
SCREEN_ROOT="$RUN_ROOT/checkpoint_screening"
FORMAL_ROOT="$RUN_ROOT/formal_eval_topk"
SCORECARD_ROOT="$RUN_ROOT/scorecards"

DATASET_PATH="${DATASET_PATH:-/root/liars_bar_dev_proxy_refine/logs/task_n_hicra_preprocessed/20260504-task2_1-english/savi_alignment_train.jsonl}"
POLICY_MODEL_PATH="${POLICY_MODEL_PATH:-/root/autodl-tmp/models/huggingface/Qwen/Qwen2.5-7B-Instruct}"
EVAL_MODEL_NAME="${EVAL_MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
PROXY_MODEL_PATH="${PROXY_MODEL_PATH:-/root/liars_bar_feat_grpo/models/proxy/value_proxy_mlp_distill.pt}"

TASKM_CONFIG="${TASKM_CONFIG:-/root/liars_bar/config/experiment.yaml}"
AUDIT_CONFIG="${AUDIT_CONFIG:-/root/liars_bar_dev_proxy_refine/config/experiment.yaml}"
BASELINE_ROOT="${BASELINE_ROOT:-/root/autodl-tmp/experiments/20260505-baseline-final-100g}"
REFERENCE_ROOT="${REFERENCE_ROOT:-/root/autodl-tmp/experiments/20260506-conditional-best-100g}"

TRAIN_STEPS="${TRAIN_STEPS:-50}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-10}"
SCREEN_GAMES="${SCREEN_GAMES:-40}"
FORMAL_GAMES="${FORMAL_GAMES:-100}"
TOP_K="${TOP_K:-3}"
LOCAL_LLM_MAX_NEW_TOKENS="${LOCAL_LLM_MAX_NEW_TOKENS:-192}"
TARGET_LLM_TURN_COUNT="${TARGET_LLM_TURN_COUNT:-400}"
SCREEN_MAX_CONFLICT_COUNT="${SCREEN_MAX_CONFLICT_COUNT:-1}"

ACTION_MATCH_REWARD_WEIGHT="${ACTION_MATCH_REWARD_WEIGHT:-0.10}"
PARSE_ERROR_PENALTY="${PARSE_ERROR_PENALTY:--0.75}"
FALLBACK_USED_PENALTY="${FALLBACK_USED_PENALTY:--0.35}"
RESOLUTION_REPAIR_PENALTY="${RESOLUTION_REPAIR_PENALTY:--0.15}"

echo "===== START PIPELINE $(date) ====="

test -f "$TRAIN_REPO/scripts/train_savi_alignment.py"
test -f "$EVAL_REPO/scripts/run_llm_drill.py"
test -f "$EVAL_REPO/scripts/select_screening_candidates.py"
test -f "$EVAL_REPO/scripts/build_eval_scorecard.py"
test -f "$AUDIT_REPO/scripts/audit_llm_behavior.py"
test -f "$DATASET_PATH"
test -f "$PROXY_MODEL_PATH"
test -f "$POLICY_MODEL_PATH/config.json"
mkdir -p "$RUN_ROOT" "$TRAIN_ROOT" "$SCREEN_ROOT" "$FORMAL_ROOT" "$SCORECARD_ROOT"

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
  --checkpoint-dir "$TRAIN_ROOT/checkpoints" \
  --save-every-steps "$SAVE_EVERY_STEPS" \
  --save-final-adapter \
  --skip-optimizer-state \
  --output-path "$TRAIN_ROOT/train_summary.json" \
  > "$TRAIN_ROOT/stdout.log" 2>&1

for CKPT in \
  "$TRAIN_ROOT/checkpoints/step-000010" \
  "$TRAIN_ROOT/checkpoints/step-000020" \
  "$TRAIN_ROOT/checkpoints/step-000030" \
  "$TRAIN_ROOT/checkpoints/step-000040" \
  "$TRAIN_ROOT/checkpoints/step-000050" \
  "$TRAIN_ROOT/checkpoints/final"
do
  TAG=$(basename "$CKPT")
  TASKM_OUT="$SCREEN_ROOT/$TAG/task_m"
  AUDIT_OUT="$SCREEN_ROOT/$TAG/task_1_1"
  mkdir -p "$TASKM_OUT" "$AUDIT_OUT"

  echo "===== SCREEN TASK_M $TAG $(date) ====="
  cd "$EVAL_REPO"
  LOCAL_LLM_DEVICE=cuda \
  LOCAL_LLM_LOCAL_FILES_ONLY=1 \
  LOCAL_LLM_MAX_NEW_TOKENS="$LOCAL_LLM_MAX_NEW_TOKENS" \
  LOCAL_LLM_ADAPTER_PATH="$CKPT" \
  python scripts/run_llm_drill.py \
    --config "$TASKM_CONFIG" \
    --expected-llm-model "$EVAL_MODEL_NAME" \
    --games "$SCREEN_GAMES" \
    --log-dir "$TASKM_OUT" \
    > "$SCREEN_ROOT/$TAG/task_m_stdout.log" 2>&1

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
done

echo "===== SELECT TOP-$TOP_K $(date) ====="
cd "$EVAL_REPO"
python scripts/select_screening_candidates.py \
  "$SCREEN_ROOT" \
  --top-k "$TOP_K" \
  --max-conflict-count "$SCREEN_MAX_CONFLICT_COUNT" \
  --target-llm-turn-count "$TARGET_LLM_TURN_COUNT" \
  --always-include-tag final \
  --output "$RUN_ROOT/screening_selection.json"

TOP_TAGS=$(python - <<PY
import json
data = json.load(open("$RUN_ROOT/screening_selection.json", "r", encoding="utf-8"))
print(" ".join(item["tag"] for item in data["selected"]))
PY
)

echo "===== SELECTED: $TOP_TAGS ====="

for TAG in $TOP_TAGS
do
  CKPT="$TRAIN_ROOT/checkpoints/$TAG"
  ROOT="$FORMAL_ROOT/$TAG"
  TASKM_OUT="$ROOT/task_m"
  AUDIT_OUT="$ROOT/task_1_1"
  mkdir -p "$TASKM_OUT" "$AUDIT_OUT"

  echo "===== FORMAL 100G $TAG $(date) ====="
  cd "$EVAL_REPO"
  LOCAL_LLM_DEVICE=cuda \
  LOCAL_LLM_LOCAL_FILES_ONLY=1 \
  LOCAL_LLM_MAX_NEW_TOKENS="$LOCAL_LLM_MAX_NEW_TOKENS" \
  LOCAL_LLM_ADAPTER_PATH="$CKPT" \
  python scripts/run_llm_drill.py \
    --config "$TASKM_CONFIG" \
    --expected-llm-model "$EVAL_MODEL_NAME" \
    --games "$FORMAL_GAMES" \
    --log-dir "$TASKM_OUT" \
    > "$ROOT/task_m_stdout.log" 2>&1

  echo "===== AUDIT $TAG $(date) ====="
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
    > "$ROOT/task_1_1_stdout.log" 2>&1
done

echo "===== BUILD SCORECARD $(date) ====="
cd "$EVAL_REPO"
python scripts/build_eval_scorecard.py \
  "$BASELINE_ROOT" \
  "$REFERENCE_ROOT" \
  $(python - <<PY
import json
data = json.load(open("$RUN_ROOT/screening_selection.json", "r", encoding="utf-8"))
for item in data["selected"]:
    print(f"$FORMAL_ROOT/{item['tag']}")
PY
) \
  --format markdown \
  --output "$SCORECARD_ROOT/final_scorecard.md"

echo "===== ALL DONE $(date) ====="
