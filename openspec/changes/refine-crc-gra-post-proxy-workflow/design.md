# Design: CRC-GRA Post-Proxy Refinement

## Guiding Principle

Each modification must be isolated enough that a scorecard change can be attributed to one mechanism. The current best artifacts remain read-only. New training and evaluation runs write to new `RUN_ROOT` directories and use explicit variant names.

## Stage 0: Change Log And Terminology

Create and maintain a modification log under `docs/tmp/pipeline/`. The log tracks the problem, proposed fix, affected code, expected risk, verification, server run root, and final decision.

Terminology should distinguish:

- `action_proxy_disagreement`: logged action differs from proxy-target action;
- `high_ev_gap_decision_error`: EVGap exceeds the audit threshold;
- `semantic_reasoning_action_mismatch`: reasoning text implies an intent inconsistent with the chosen action;
- `reasoning_action_conflict`: high-severity label when semantic mismatch and/or high-EVGap disagreement are both present.

The existing `reasoning_action_mismatch` field should be treated as a legacy high-EVGap action/proxy disagreement signal unless a new semantic classifier is explicitly used.

## Stage 1: Offline Audit, No Training Change

Add an isolated offline auditor that reads formal eval logs and proxy audit outputs, then emits:

- `hicra_audit_records.jsonl`;
- `hicra_audit_summary.json`;
- `hicra_case_studies.md`;
- `hicra_audit_scorecard.md`.

This stage does not change policy training. It supports data quality, failure attribution, and paper case studies.

## Stage 2: Data Provenance And Split Audit

Before launching new training variants, record which run roots and seeds were used for:

- CRC rollout/proxy training;
- proxy validation/calibration;
- post-proxy alignment data construction;
- formal held-out evaluation.

If the current data lineage is ambiguous, add a data-provenance report and prefer collecting a fresh post-proxy alignment run with seeds/run roots separated from proxy training and final evaluation. This does not change model training by itself, but it prevents later results from being hard to defend.

## Stage 3: Candidate Expansion Variant

Add a new isolated training variant that expands candidate groups beyond logged action and proxy target:

- logged action;
- proxy-best action;
- proxy second-best action when distinct;
- proxy worst action as a negative contrast;
- explicit challenge when legal;
- concrete truthful play candidate;
- concrete bluff play candidate.

The trainer should proxy-score concrete legal candidates and keep a bounded group size, initially 4 or 6, so group-relative advantages are meaningful without adding too much noise.

## Stage 4: Optional Protocol Data Handling

If offline audit shows protocol failures are materially present, add a separate protocol-quality path:

- filter protocol-broken samples out of strategy training;
- optionally build format-repair SFT examples;
- do not mix protocol repair with strategic proxy reward in the same conclusion.

## Stage 5: Optional Reasoning Regeneration

Only after candidate expansion is stable, test whether proxy-target actions need action-conditioned reasoning regeneration. This should be its own experiment because changing candidate reasoning text changes the learning target, not just action ranking.

Possible variants:

- keep original reasoning for all candidates, current behavior;
- replace proxy-target reasoning with teacher-generated action-conditioned reasoning;
- train reasoning only on clean-aligned samples.
