# Data Provenance Report For Post-Proxy CRC-GRA Refinement

## Purpose

This report addresses whether the proxy model, alignment training data, and formal evaluation data are separated clearly enough for the next isolated experiments. It is a provenance report, not a model-training change.

## Known Artifacts

| role | artifact / path | evidence | status |
| --- | --- | --- | --- |
| Base policy | `/root/autodl-tmp/models/huggingface/Qwen/Qwen2.5-7B-Instruct` | training summaries and run scripts | known |
| Frozen proxy model | `/root/liars_bar_feat_grpo/models/proxy/value_proxy_mlp_distill.pt` | run scripts and train summaries | known |
| Alignment dataset used by HICRA / DPO / ablation variants | `/root/liars_bar_dev_proxy_refine/logs/task_n_hicra_preprocessed/20260504-task2_1-english/savi_alignment_train.jsonl` | train summaries; default script path | known |
| Full-state alignment dataset for candidate expansion | `/root/liars_bar_dev_proxy_refine/logs/task_n_hicra_preprocessed/20260504-task2_3-english/savi_alignment_full_v3.jsonl` | server record inspection; contains `observation.private_hand` and `observation.legal_actions` | known |
| Alignment dataset size in recent variants | `60` records | train summaries report `records_used: 60` | known |
| Alignment record game IDs | `llm-drill-*` with `20260504-*` timestamps | train summaries step records | known |
| Action-only adapter | `/root/autodl-tmp/experiments/20260523-crc-gra-ablation-200g/action_only_proxy/train/checkpoints/final` | controlled eval script | known |
| Best-main adapter | `/root/autodl-tmp/experiments/20260510-protocol-mainline-r2-delay-rerun3/train/checkpoints/final` | controlled eval script | known |
| Controlled 200-game formal evaluation | `/root/autodl-tmp/experiments/20260524-controlled-best-action-proxy-200g` | backed-up scorecards and formal logs | known |
| Formal evaluation game IDs | `llm-drill-*` with `20260524-*` timestamps | backed-up `task_m/games/*.jsonl` | known |

## Current Split Assessment

The available evidence supports this separation:

- alignment training records are from `20260504`;
- controlled formal evaluation records are from `20260524`;
- recent HICRA rescue and DPO variants use the same 60-record alignment dataset, which makes their differences attributable to training variants rather than data changes;
- the controlled 200-game action-only and best-main comparison used a separate run root and fixed evaluation settings.

The available evidence does not fully prove this separation:

- the original CRC rollout records used to train `value_proxy_mlp_distill.pt` are not fully documented in the current backed-up server archive;
- exact proxy train/validation splits and seeds need to be recovered from proxy training artifacts or server history before final paper writing;
- the `20260504` alignment set may have been generated from logs that are downstream of the same project pipeline, but the current local archive does not prove whether it overlaps with proxy training data at the decision-point level.

## Decision For M3 Candidate Expansion

For the isolated candidate-expansion experiment, the compact prior-ablation dataset is not sufficient:

`/root/liars_bar_dev_proxy_refine/logs/task_n_hicra_preprocessed/20260504-task2_1-english/savi_alignment_train.jsonl`

It should not be used for concrete candidate enumeration because records do not preserve enough state for `truthful_play`, `bluff_play`, or `legal_challenge` expansion. A server dry-run using that compact dataset produced only `logged_action` and `proxy_target` candidates.

Use the full-state 20260504 alignment artifact instead:

`/root/liars_bar_dev_proxy_refine/logs/task_n_hicra_preprocessed/20260504-task2_3-english/savi_alignment_full_v3.jsonl`

Rationale:

- candidate expansion requires `observation.private_hand`, `observation.legal_actions`, and `observation.table_type`;
- the full-state dataset is still a 20260504 alignment artifact, so it avoids introducing fresh policy data into M3;
- using the same 60-record alignment size keeps the comparison against `action_only_proxy` more attributable;
- changing both data collection and candidate construction in one experiment would make failures hard to interpret.

## Paper-Quality Follow-Up

Before final paper claims, add one of the following:

1. Recover exact proxy train/validation provenance and document non-overlap with formal evaluation.
2. If exact provenance cannot be recovered, collect a fresh post-proxy alignment split with a new run root and seeds, then treat it as the paper-facing alignment dataset.

Recommended final split narrative:

- Split A: CRC rollout/proxy training.
- Split B: proxy validation/calibration.
- Split C: frozen-proxy alignment construction.
- Split D: held-out formal evaluation.

## Current Risk Level

For internal controlled ablations: **acceptable**, because variants share the same alignment data and fixed formal evaluation.

For final reviewer-facing claims: **needs documentation**, because proxy training provenance is not yet fully backed by the local archive.
