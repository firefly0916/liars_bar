# CRC-GRA Post-Proxy Modification Log

This is the tracked copy of the living log for post-proxy CRC-GRA refinements. A working copy also exists at `docs/tmp/pipeline/2026-05-26-crc-gra-post-proxy-modification-log.md`.

## Ground Rules

- Do not modify `best-main` checkpoints, scripts, or historical result directories.
- Follow OpenSpec before code changes.
- Prefer one mechanism per experiment.
- Use fixed 200-game formal eval for controlled comparisons unless explicitly stated otherwise.
- Treat HICRA/token signals as offline audit evidence, not mainline reward/loss/advantage/DPO supervision.
- Record server run roots and exact variant names before interpreting results.

## Current Mainline Interpretation

- Main framework name: `CRC-GRA` (`Counterfactual Rollout Credit Guided Reasoning Alignment`).
- Reliable performance source: action-level CRC/proxy-guided candidate alignment.
- Current strongest controlled result: `action_only_proxy_rerun_200g`, win rate about `0.615`, parse error near zero, illegal action `0`.
- HICRA status: failed or unstable as training signal; useful as offline audit/explanation.

## Modification Queue

| id | status | priority | change | motivation | affected area | risk | verification | server run root | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M0 | done | P0 | Create OpenSpec and modification log | Prevent forgetting or mixing changes | docs / openspec | low | file review | n/a | keep |
| M1 | done | P0 | Offline reasoning-action audit module | Separate action/proxy disagreement from true semantic mismatch; support case studies | `hicra_offline_auditor.py`, `run_hicra_offline_audit.py` | low | `python -m unittest tests.test_hicra_offline_auditor -v`; local action-only 200g audit | local: `docs/tmp/pipeline/2026-05-26-hicra-offline-audit-action-only-200g` | keep |
| M2 | done | P0 | Data provenance and split audit | Clarify whether proxy training data, alignment data, and formal eval data are separated; decide if fresh post-proxy collection is needed | `data_provenance_report.md` | low | report review against train summaries and run scripts | n/a | use same 20260504 alignment data for M3; document proxy provenance gap for paper |
| M3 | done | P1 | Candidate expansion variant | Current groups are often logged vs proxy-best only; richer concrete candidates may improve group-relative learning | `scripts/train_crc_gra_ablation.py`, `candidate_expansion.py` | medium | unit tests + dry run + smoke train + 200g eval | `/root/autodl-tmp/experiments/20260526-expanded-action-proxy-200g` | failed as training improvement; keep only as negative ablation evidence |
| M4 | pending | P2 | Protocol-quality filtering/repair path | Protocol variables may be empty and should be separated from strategy learning | new filter/report path | medium | audit-driven decision + 200g eval if used | TBD | TBD |
| M5 | pending | P3 | Action-conditioned reasoning regeneration | Current synthetic candidates may reuse original reasoning with changed actions | new dataset generation path | high | separate OpenSpec before code | TBD | defer |
| M6 | done | P1 | Action-only without action-match bonus | Test whether action-only stability depends on explicit proxy-target match reward or only proxy dense reward | `train_crc_gra_ablation.py` variant settings only | low | unit test + dry run + 200g eval | `/root/autodl-tmp/experiments/20260526-action-only-no-match-bonus-200g` | failed as improvement; shows action-match anchor is important |
| M7 | done | P1 | Action-only with Action-field-only loss | Test whether current action-only performance requires weak reasoning-token training | `train_crc_gra_ablation.py` label scope only | low | unit test + 5-step smoke + 200g eval | `/root/autodl-tmp/experiments/20260527-action-loss-only-200g` | failed as improvement; action-only needs protocol/full-response supervision |

## Field Corrections To Preserve In Writing

- Existing `reasoning_action_mismatch` in the current dataset means high-EVGap action/proxy disagreement, not necessarily semantic contradiction in natural language.
- Use `action_proxy_disagreement` for `chosen_action != proxy_target_action`.
- Use `high_ev_gap_decision_error` for `ev_gap > threshold`.
- Use `semantic_reasoning_action_mismatch` only after a text-intent rule or classifier actually examines the reasoning text.
- Use `reasoning_action_conflict` as a high-level audit label, not as a raw training variable.

## Coverage Map For User-Raised Issues

| user issue | covered by | current status |
| --- | --- | --- |
| Whether reasoning tokens are truly trained or only whole-response trained | M5, plus paper wording in field corrections | Not solved by current action-only training; must not be overclaimed |
| How the new scheme judges reasoning-action mismatch | M1 | Must split legacy action/proxy disagreement from semantic text-action mismatch |
| Whether all old game records are audited after proxy and whether fresh post-proxy games are needed | M2 | Previously under-specified; now tracked as a required data-provenance audit |
| How to separate proxy-training data, alignment data, and formal eval data | M2 | Needs report before new major training variants |
| How to avoid illegal actions/format errors when protocol variables may be empty | M4 | Needs audit-driven protocol handling, not assumed solved |
| Whether candidate groups should be richer than logged vs proxy-best | M3 | Candidate expansion variant will test this independently |

## Next Planned Step

Finish M3 smoke training with the full-state alignment dataset, then launch one controlled 200-game formal eval only if the smoke summary confirms expanded candidate roles remain active.

## M1 Result Notes

- Added offline audit module and CLI without modifying training code or best-main artifacts.
- The CLI supports both premerged JSONL records and `ev_gap_distribution.csv + task_m/games` merging.
- Local verification on `action_only_proxy_rerun_200g` produced `2262` audit records.
- Local action-only audit summary:
  - `clean_aligned`: `1752`
  - `proxy_disagreement`: `234`
  - `reasoning_action_conflict`: `157`
  - `strategic_overchallenge`: `27`
  - `strategic_overplay`: `92`
  - `protocol_failure`: `0`
- Interpretation: this controlled action-only run has no protocol failures in the audit records, so immediate follow-up should prioritize data provenance (`M2`) and candidate expansion (`M3`) before protocol repair (`M4`).

## M2 Result Notes

- Added `data_provenance_report.md`.
- Current controlled ablation data lineage:
  - alignment training data: `20260504-task2_1-english/savi_alignment_train.jsonl`, 60 records in recent train summaries;
  - controlled formal evaluation: `20260524-controlled-best-action-proxy-200g`, 200 games;
  - proxy model: `/root/liars_bar_feat_grpo/models/proxy/value_proxy_mlp_distill.pt`.
- Initial decision for M3 was to use the same compact 20260504 alignment dataset to isolate candidate expansion from data collection changes.
- Paper-facing risk: exact proxy train/validation source split is not fully documented in the local archive and should be recovered or replaced by a fresh split before final claims.

## M3 Result Notes

- Added isolated `expanded_action_proxy` training variant and `candidate_expansion.py`.
- The first server dry-run on compact `20260504-task2_1-english/savi_alignment_train.jsonl` did not actually expand candidate groups:
  - selected roles were only `logged_action` and `proxy_target`;
  - candidate groups had only 1-2 members;
  - reason: compact records do not preserve `observation.private_hand` and `observation.legal_actions`.
- Server inspection found full-state data at:
  `/root/liars_bar_dev_proxy_refine/logs/task_n_hicra_preprocessed/20260504-task2_3-english/savi_alignment_full_v3.jsonl`
- Server dry-run on `savi_alignment_full_v3.jsonl` produced real candidate expansion:
  - candidate count distribution: `{1: 2, 2: 14, 3: 28, 4: 16}`;
  - selected candidate roles: `logged_action: 60`, `bluff_play: 55`, `truthful_play: 31`, `legal_challenge: 14`, `proxy_target: 18`;
  - candidate-pool roles: `logged_action: 60`, `truthful_play: 53`, `bluff_play: 59`, `legal_challenge: 36`, `proxy_target: 18`;
  - mean advantage span: `0.2530`, max span: `1.3813`.
- Decision: use `savi_alignment_full_v3.jsonl` for M3 because it is still a 20260504 alignment artifact but contains the state fields required for legal concrete candidates.
- Server smoke training on 10 records and 5 steps completed under:
  `/root/autodl-tmp/experiments/20260526-expanded-action-proxy-smoke/train_smoke`
  - completed steps: `5/5`;
  - nonfinite loss steps: `0`;
  - idle steps: `0`;
  - signalless steps: `0`;
  - final adapter: `/root/autodl-tmp/experiments/20260526-expanded-action-proxy-smoke/train_smoke/checkpoints/final`.
- Controlled M3 run launched under:
  `/root/autodl-tmp/experiments/20260526-expanded-action-proxy-200g`
  - variant: `expanded_action_proxy`;
  - dataset: `savi_alignment_full_v3.jsonl`;
  - training steps: `200`;
  - group size: `4`;
  - formal eval games: `200`;
  - random seed: `42`;
  - pid file: `/root/autodl-tmp/experiments/20260526-expanded-action-proxy-200g/pid.txt`;
  - suite log: `/root/autodl-tmp/experiments/20260526-expanded-action-proxy-200g/suite_stdout.log`.
- Controlled M3 run completed:
  - final scorecard: `/root/autodl-tmp/experiments/20260526-expanded-action-proxy-200g/expanded_action_proxy/scorecard.md`;
  - win rate: `0.215`;
  - parse error rate: `0.000000`;
  - illegal chosen turns: `0`;
  - avg EVGap: `0.036456`;
  - high EVGap turns: `183`;
  - challenge rate: `0.243421`;
  - challenge accuracy: `0.340956`;
  - bluff efficiency: `0.558779`.
- Comparison against controlled `action_only_proxy_rerun_200g`:
  - action-only win rate: `0.615`;
  - action-only avg EVGap: `0.032470`;
  - action-only challenge rate: `0.115827`;
  - action-only challenge accuracy: `0.660305`.
- Interpretation:
  - M3 is not a useful improvement over action-only proxy alignment.
  - The failure is not caused by protocol collapse; parse and illegal action metrics are stable.
  - The main behavioral failure is over-challenge and low challenge precision.
  - Expanded synthetic candidates reuse the original reasoning while changing the action, so they can create inconsistent reasoning-action supervision. Treat this as evidence for deferring richer candidate groups until action-conditioned reasoning regeneration or stricter candidate filtering is specified.

## Candidate Follow-Up Options After M3

These options should be run one at a time from the `action_only_proxy` baseline:

1. `action_only_proxy_no_match_bonus`: keep logged/proxy candidate structure and disable only the explicit action-match reward. Purpose: test whether action-only performance comes from proxy dense credit alone or from the proxy-target match anchor.
2. `action_only_proxy_adv_clip`: keep action-only training but clip group-relative advantages. Purpose: test whether large proxy gaps cause over-updates while keeping the action-only design.
3. `action_only_proxy_high_kl`: keep action-only training but increase KL regularization. Purpose: test whether LoRA drift from the base policy explains instability.
4. `action_only_proxy_protocol_only_filter`: keep training objective unchanged but drop protocol-broken records only if audit shows protocol failures. Current action-only audit had `protocol_failure: 0`, so this is lower priority.
5. `action_conditioned_reasoning_regen`: regenerate reasoning for synthetic proxy-target actions. Purpose: solve the reasoning-action inconsistency seen in M3. This is higher risk and should get a separate OpenSpec before implementation.

## M6 Result Notes

- Added `action_only_proxy_no_match_bonus` as a minimal variant.
- Only intended behavior change:
  - `use_hicra_reward = False`;
  - `hicra_gamma = 0.0`;
  - `reward_component_weights.action_match_reward = 0.0`;
  - candidate mode remains `full`, so compact data still produces only logged/proxy candidates.
- Server test passed:
  `python -m unittest tests.test_crc_gra_ablation.CRCGRAAblationTest.test_action_only_no_match_bonus_removes_only_action_match_reward -v`
- Server dry-run on compact 20260504 alignment data:
  - records: `60`;
  - candidate count distribution: `{1: 42, 2: 18}`;
  - roles: `logged_action: 60`, `proxy_target: 18`;
  - nonzero weighted action-match rewards: `0`.
- Controlled 200-game result:
  - win rate: `0.195`;
  - parse error rate: `0.000000`;
  - illegal chosen turns: `0`;
  - avg EVGap: `0.037761`;
  - max EVGap: `0.977091`;
  - high EVGap turns: `169`;
  - challenge rate: `0.251354`;
  - challenge accuracy: `0.334052`;
  - bluff efficiency: `0.562410`.
- Comparison:
  - `action_only_proxy_rerun_200g`: win `0.615`, challenge rate `0.115827`, challenge accuracy `0.660305`, avg EVGap `0.032470`.
  - `expanded_action_proxy`: win `0.215`, challenge rate `0.243421`, challenge accuracy `0.340956`, avg EVGap `0.036456`.
- Interpretation:
  - Removing the explicit action-match bonus collapses performance into the same failure regime as expanded candidates.
  - The action-match anchor appears to be a key stabilizer for action-only proxy alignment.
  - Future variants should preserve the action-match anchor and instead test smaller changes such as advantage clipping or stronger KL regularization.

## M7 Result Notes

- Added `action_only_proxy_action_loss_only` as a minimal variant.
- Intended behavior:
  - keeps action-only candidate structure and rewards;
  - keeps explicit proxy-target action-match anchor;
  - keeps HICRA/token signals disabled;
  - changes only `label_scope` from full assistant response to `Action` field.
- Purpose:
  - test whether action-only performance depends on weak training of original reasoning tokens;
  - avoid training reused or inconsistent reasoning in proxy-target candidates.
- Server test passed:
  `python -m unittest tests.test_crc_gra_ablation.CRCGRAAblationTest.test_action_loss_only_variant_trains_only_action_span -v`
- Server smoke training completed under:
  `/root/autodl-tmp/experiments/20260527-action-loss-only-prep/smoke`
  - records used: `10`;
  - completed steps: `5/5`;
  - label scope: `action`;
  - nonfinite loss steps: `0`;
  - idle steps: `1`;
  - signalless steps: `1`;
  - candidate roles: `logged_action: 4`, `proxy_target: 4`;
  - active label count: mean `18.5`, min `18`, max `19`.
- Controlled 200-game result:
  - win rate: `0.325`;
  - parse error rate: `0.081789`;
  - illegal chosen turns: `34`;
  - avg EVGap: `0.033265`;
  - max EVGap: `1.395330`;
  - high EVGap turns: `171`;
  - challenge rate: `0.131948`;
  - challenge accuracy: `0.585271`;
  - bluff efficiency: `0.538462`.
- Comparison against `action_only_proxy_rerun_200g`:
  - action-only win rate: `0.615`;
  - parse error rate: `0.000442`;
  - illegal chosen turns: `0`;
  - challenge rate: `0.115827`;
  - challenge accuracy: `0.660305`.
- Interpretation:
  - Training only the `Action` field is not a viable replacement for full assistant-response training.
  - The failure mode is protocol/format degradation, not just weaker strategy: parse errors and illegal chosen turns increase sharply.
  - The result suggests that action-only proxy alignment benefits from weak full-response supervision at least for preserving the required JSON protocol and response structure.
  - Next single-factor variant should test `format_plus_action_loss`: train JSON/protocol wrapper and Action while masking only the free-form Reasoning content.
