# Proposal: Refine CRC-GRA Post-Proxy Workflow

## Motivation

Recent controlled experiments show that the reliable performance source is action-level CRC/proxy-guided alignment, while direct HICRA-style token intervention is unstable as reward shaping, loss weighting, advantage reshaping, or DPO preference supervision. The project now needs a staged refinement path that keeps `best-main` untouched, improves the post-proxy workflow incrementally, and preserves experiment attribution.

The key risks to address are:

- reasoning tokens are currently trained only through whole-response likelihood, not through a trustworthy reasoning-level credit signal;
- `reasoning_action_mismatch` currently means high-EVGap action/proxy disagreement, not true semantic reasoning-action contradiction;
- proxy training data, alignment data, and formal evaluation data need clearer split boundaries;
- protocol failure variables can be empty in clean logs and should not be treated as a complete format-stability solution;
- candidate groups are often close to logged-action versus proxy-target only, which weakens group-relative comparison.

## Change

Add a staged, isolated refinement track for the post-proxy CRC-GRA workflow:

- maintain a living modification log for each proposed change, status, experiment root, and outcome;
- add an offline reasoning-action audit layer that renames and separates action/proxy disagreement from semantic reasoning-action mismatch;
- add conservative data-quality labels for filtering and case-study selection without feeding HICRA token penalties into training;
- add a candidate-expansion variant that builds concrete legal action candidates and evaluates top/bottom proxy-ranked alternatives;
- optionally add independent protocol-format repair data handling after audit results justify it.

## Non-Goals

- Do not modify `best-main` checkpoints, historical result directories, or existing best-result scripts.
- Do not reintroduce HICRA as scalar reward, token loss weighting, advantage reshaping, or DPO preference supervision in the mainline.
- Do not claim full reasoning-token credit assignment unless a later isolated experiment explicitly validates it.
- Do not run multiple conceptual changes in one experiment.

