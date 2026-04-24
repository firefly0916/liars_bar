# Task K And Task L Runner Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a server-ready physical-rollout runner for Task K and a local negative-sampling/refinement pipeline for Task L, without requiring access to the 4090 server during development.

**Architecture:** Keep Task K strictly on the physical-rollout path and expose its configuration through a dedicated runner that can later be executed on the remote 4090 environment. For Task L, add a configurable poor-play data generator, mix those logs with existing probe logs, retrain the proxy on the mixed dataset, and emit an alignment comparison artifact against the current elite-data baseline.

**Tech Stack:** Python 3.12, unittest, existing orchestrator/logger/analyzer stack, JSON/CSV artifacts.

---

### Task 1: Lock Task K/Task L runner APIs with tests

**Files:**
- Modify: `tests/test_task_d_axiomatic_runner.py`
- Create: `tests/test_task_k_task_l_runners.py`
- Reference: `liars_game_engine/analysis/task_c_runner.py`

**Step 1: Write the failing tests**

Add tests covering:
- Task K runner builds physical-rollout config and exports `credit_report_final.csv` path metadata.
- Task L runner can select mixed log roots and emits comparison/report paths.
- Poor-play settings increase probe/random behavior through configuration rather than hardcoded edits.

**Step 2: Run test to verify it fails**

Run: `conda run -n liar_bar python -m unittest tests.test_task_k_task_l_runners -v`

Expected: FAIL because the new runner modules/APIs do not exist yet.

**Step 3: Write minimal implementation**

Add the smallest task-runner interfaces needed to satisfy those tests.

**Step 4: Run test to verify it passes**

Run: `conda run -n liar_bar python -m unittest tests.test_task_k_task_l_runners -v`

Expected: PASS.

### Task 2: Implement configurable poor-play data generation

**Files:**
- Modify: `liars_game_engine/config/schema.py`
- Modify: `config/experiment.yaml`
- Modify: `liars_game_engine/agents/mock_agent.py`
- Modify: `liars_game_engine/agents/factory.py`
- Modify: `tests/test_mock_agent_pipeline.py`

**Step 1: Write the failing test**

Add a test proving the mock agent respects a configurable high probe/poor-play probability when enabled.

**Step 2: Run test to verify it fails**

Run: `conda run -n liar_bar python -m unittest tests.test_mock_agent_pipeline -v`

Expected: FAIL because the probability remains hardcoded.

**Step 3: Write minimal implementation**

Introduce a runtime-configurable probability and thread it from settings into agent construction.

**Step 4: Run test to verify it passes**

Run: `conda run -n liar_bar python -m unittest tests.test_mock_agent_pipeline -v`

Expected: PASS.

### Task 3: Add Task K and Task L runner modules

**Files:**
- Create: `liars_game_engine/analysis/task_k_gold_runner.py`
- Create: `liars_game_engine/analysis/task_l_proxy_refine_runner.py`
- Modify: `liars_game_engine/analysis/train_value_proxy.py`

**Step 1: Write the failing test**

Add tests for:
- Task K summary includes `credit_report_final.csv`, physical rollout settings, and timing metadata.
- Task L summary includes negative-log directory, mixed-training metrics, and elite-vs-mixed alignment comparison.

**Step 2: Run test to verify it fails**

Run: `conda run -n liar_bar python -m unittest tests.test_task_k_task_l_runners -v`

Expected: FAIL until the runners exist.

**Step 3: Write minimal implementation**

Implement:
- Task K physical runner wiring only, suitable for remote execution later.
- Task L poor-play log generation, mixed-log temporary training root creation, retraining, and comparison export.

**Step 4: Run test to verify it passes**

Run: `conda run -n liar_bar python -m unittest tests.test_task_k_task_l_runners -v`

Expected: PASS.

### Task 4: Verify local L pipeline and non-remote K wiring

**Files:**
- Output: `logs/task_l_proxy_refine/*`
- Modify: `PROJECT_MEMORY.md`
- Modify: `logs/CHANGELOG_DAILY.md`
- Modify: `logs/CHANGELOG_DETAILED.md`

**Step 1: Run focused tests**

Run:
- `conda run -n liar_bar python -m unittest tests.test_task_k_task_l_runners -v`
- `conda run -n liar_bar python -m unittest tests.test_mock_agent_pipeline -v`

**Step 2: Run Task L locally**

Run the new Task L runner and capture:
- mixed training metrics
- elite vs mixed Pearson comparison
- output model path (for example `value_proxy_mlp_v2.pt`)

**Step 3: Do not run large remote K job**

Only verify that Task K runner starts and returns correct wiring/config paths; do not claim remote 4090 execution happened.

**Step 4: Run full regression suite**

Run: `conda run -n liar_bar python -m unittest discover tests -v`

Expected: PASS.
