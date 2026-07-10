# Task M Progress Monitoring Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add per-turn progress monitoring for Task M drill runs and provide a standard background server launcher with tail-friendly logs.

**Architecture:** Extend the drill runner so `GameOrchestrator` can emit a turn-level callback after each committed turn. `run_llm_drill` will aggregate those callbacks into a total-turn progress log with percentage, ETA, and ASCII bar, while a small shell wrapper will launch the drill in the background and print the exact files to tail.

**Tech Stack:** Python 3.10, unittest, bash, existing `liars_game_engine` drill runner.

---

### Task 1: Add failing tests for per-turn drill progress logging

**Files:**
- Modify: `tests/test_orchestrator_logging.py`
- Create: `tests/test_llm_drill.py`

**Step 1: Write the failing test**

Add a drill test that runs a tiny 2-game mock drill and asserts:
- `progress.log` exists before final summary
- progress lines contain `completed_turns`, `total_turn_budget`, `percent`, and `progress_bar`
- final progress line reaches the full budget

Add an orchestrator test that injects a progress callback and asserts the callback is called once per committed turn with `turns_played` and `max_turns`.

**Step 2: Run test to verify it fails**

Run: `conda run -n liar_bar python -m unittest tests.test_llm_drill tests.test_orchestrator_logging`

Expected: FAIL because no progress callback exists and drill progress is only logged once per game.

**Step 3: Write minimal implementation**

Add callback plumbing to `GameOrchestrator` and total-turn progress aggregation to `run_llm_drill`.

**Step 4: Run test to verify it passes**

Run: `conda run -n liar_bar python -m unittest tests.test_llm_drill tests.test_orchestrator_logging`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_llm_drill.py tests/test_orchestrator_logging.py liars_game_engine/experiment/orchestrator.py liars_game_engine/experiment/llm_drill.py
git commit -m "feat: add per-turn task m progress logging"
```

### Task 2: Add background launcher for Task M drill

**Files:**
- Create: `scripts/run_task_m_drill_server.sh`
- Test: `tests/test_task_m_server_script.py`

**Step 1: Write the failing test**

Add a lightweight script-content test that asserts the launcher:
- defaults to `liar_bar`
- uses `nohup conda run -n ... python scripts/run_llm_drill.py`
- writes `stdout.log`
- prints `tail -f` guidance for `progress.log` and `stdout.log`

**Step 2: Run test to verify it fails**

Run: `conda run -n liar_bar python -m unittest tests.test_task_m_server_script`

Expected: FAIL because the script does not exist yet.

**Step 3: Write minimal implementation**

Create the script with configurable `OUT_DIR`, `CONFIG_PATH`, and `GAMES`, and make it create the output directory before launching.

**Step 4: Run test to verify it passes**

Run: `conda run -n liar_bar python -m unittest tests.test_task_m_server_script`

Expected: PASS

**Step 5: Commit**

```bash
git add scripts/run_task_m_drill_server.sh tests/test_task_m_server_script.py
git commit -m "feat: add task m server launcher"
```

### Task 3: Verify end-to-end behavior

**Files:**
- Modify: `scripts/run_task_m_drill_server.sh`
- Modify: `liars_game_engine/experiment/llm_drill.py`

**Step 1: Run targeted tests**

Run:

```bash
conda run -n liar_bar python -m unittest tests.test_llm_drill tests.test_orchestrator_logging tests.test_task_m_server_script
```

Expected: PASS

**Step 2: Run broader regression slice**

Run:

```bash
conda run -n liar_bar python -m unittest tests.test_llm_agent tests.test_local_backend tests.test_action_parser
```

Expected: PASS

**Step 3: Document server usage**

Provide exact commands for:
- starting vLLM with `--served-model-name`
- launching Task M in background
- tailing `progress.log` and `stdout.log`

**Step 4: Commit**

```bash
git add docs/plans/2026-04-29-task-m-progress-monitoring.md
git commit -m "docs: record task m progress monitoring plan"
```
