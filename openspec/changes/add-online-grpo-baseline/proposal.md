## Why

The project now has vanilla and online PPO baselines. A lightweight online GRPO-style baseline is useful as an additional RL comparison because GRPO removes the PPO value-head dependency while still using on-policy environment interaction. This comparison should remain independent from CRC-GRA and must not modify the current best-result workflow.

## What Changes

- Add an isolated online GRPO-style LoRA trainer that collects live Liar's Bar rollouts against fixed mock opponents.
- Use group-normalized advantages computed from terminal outcome plus protocol penalties only.
- Add a separate runner that trains the GRPO adapter and evaluates it with 200 formal games by default.
- Keep CRC, proxy-best actions, EVGap, HICRA penalties, and current best scripts out of the baseline.

## Capabilities

### New Capabilities
- `online-grpo-baseline-workflow`: Trains and evaluates an isolated online GRPO-style LoRA baseline.

### Modified Capabilities
- None.

## Impact

- Adds a new GRPO trainer script under `scripts/`.
- Adds a new GRPO server runner under `scripts/`.
- Adds OpenSpec artifacts and tests for the GRPO baseline contract.
- Does not modify `run_protocol_anchor_*` scripts or the CRC-GRA best workflow.
