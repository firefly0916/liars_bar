## Context

The online PPO baseline established the reusable pattern for safe baseline comparison: train a LoRA policy from the base model under an isolated run root, evaluate the final adapter with Task M, audit the logs, and build a scorecard. GRPO should reuse that operational shape but replace the PPO value-baseline update with group-normalized relative advantages.

## Goals / Non-Goals

**Goals:**
- Train only player `p1` as a Qwen LoRA policy against fixed mock opponents.
- Collect on-policy trajectories from the current policy.
- Compute rewards from terminal outcome and protocol penalties only.
- Group decisions by rollout game/update and center rewards within each group.
- Optimize LoRA parameters with a clipped sequence-level policy objective.
- Evaluate the final adapter with 200 formal games by default.

**Non-Goals:**
- Do not implement full multi-agent self-play GRPO.
- Do not use CRC/proxy/HICRA reward signals.
- Do not train a value head.
- Do not change the current CRC-GRA best pipeline.

## Decisions

1. Use a dedicated `train_online_grpo_baseline.py`.
   - Rationale: GRPO has a distinct objective and should not overload PPO scripts.

2. Use group-normalized reward advantages.
   - Rationale: This captures GRPO's key baseline-free idea while keeping the implementation small and auditable.

3. Keep formal evaluation identical to PPO and vanilla scale.
   - Rationale: 200 games with Task M + audit + scorecard is the current internal comparison standard.

4. Prefer no 4-bit training for server runs.
   - Rationale: PPO diagnostics showed 4-bit generation can corrupt JSON outputs in the training path.

## Risks / Trade-offs

- Sparse terminal rewards remain noisy, so GRPO may be weak under small budgets.
- Group sampling increases generation cost.
- Sequence-level GRPO is less granular than token-level RLHF-style training, but it is sufficient as a reviewer-facing baseline.
