# Task K Server Deployment And 8D Retrain Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `task_k_gold_runner` server-ready with 500-game progress logging and explicit environment requirements on `main`, then restore `dev-proxy-refine` to the 8D milestone and retrain after filtering out sub-3-turn games.

**Architecture:** Keep the production entrypoint as `python -m liars_game_engine.analysis.task_k_gold_runner`, but make the runner operate in batches so it can append progress after every 500 games without an external orchestrator. On `dev-proxy-refine`, discard the 9D line entirely by returning to `d2a3bc7`, then add one narrow data-quality filter in the training loader so 8D proxy retraining ignores games with fewer than 3 turns.

**Tech Stack:** Python 3.10+, `unittest`, conda env `liar_bar`, git branches, bash launcher script.

---

### Task 1: Add failing tests for Task K server progress behavior

**Files:**
- Modify: `tests/test_task_k_task_l_runners.py`

**Step 1: Write a failing progress-log test**

Assert that:
- `run_task_k_gold_pipeline(..., game_count=1000, progress_interval_games=500)` processes two batches
- `logs/task_k_gold/progress.log` is created
- one line is appended per 500-game checkpoint with cumulative counts

**Step 2: Write a failing launcher/summary test if needed**

Assert that:
- the runner summary exposes the progress log path
- defaults still match `game_count=2000`, `rollout_samples=200`

**Step 3: Run the targeted test and verify RED**

Run: `conda run -n liar_bar python -m unittest tests.test_task_k_task_l_runners -v`
Expected: FAIL because progress batching/logging does not exist yet

### Task 2: Implement Task K batching, progress logging, and server launcher

**Files:**
- Modify: `liars_game_engine/analysis/task_k_gold_runner.py`
- Modify: `pyproject.toml`
- Create: `scripts/run_task_k_gold_server.sh`

**Step 1: Add batched execution to the runner**

Implement:
- `progress_interval_games=500`
- repeated baseline-log generation in batches
- progress append after each batch to `logs/task_k_gold/progress.log`

**Step 2: Keep final report semantics stable**

Ensure:
- final `credit_report_final.csv` path is unchanged
- summary includes `progress_log`
- default execution still means 2000 games and rollout_samples 200

**Step 3: Add missing runtime dependency declaration**

Add the analysis/runtime dependency required for server execution (`torch`) to project metadata so the repo documents what the server env must have.

**Step 4: Add the thin launcher script**

Script should:
- `mkdir -p logs/task_k_gold`
- run `conda run -n liar_bar python -m liars_game_engine.analysis.task_k_gold_runner`
- write stdout/stderr to a timestamped log under `logs/task_k_gold/`

### Task 3: Verify Task K production path on main

**Files:**
- None beyond Task 2 outputs

**Step 1: Run targeted tests and verify GREEN**

Run: `conda run -n liar_bar python -m unittest tests.test_task_k_task_l_runners -v`
Expected: PASS

**Step 2: Sanity-check the launcher**

Run: `bash scripts/run_task_k_gold_server.sh --help` only if the script accepts passthrough args, otherwise inspect script content and confirm it executes the required command literally.

### Task 4: Restore dev-proxy-refine to the 8D milestone

**Files:**
- Modify: git branch state on `dev-proxy-refine`

**Step 1: Switch to `dev-proxy-refine`**

Run: `git switch dev-proxy-refine`

**Step 2: Roll back to `d2a3bc7`**

Use a branch reset because the human explicitly requested abandoning the 9D line.

### Task 5: Add failing tests for “ignore games under 3 turns”

**Files:**
- Modify: `tests/test_shapley_analyzer.py` or add a focused training-loader test

**Step 1: Write a failing test for sample loading**

Assert that:
- logs from a 2-turn game are ignored entirely
- logs from a 3-turn game still contribute samples

**Step 2: Run the targeted test and verify RED**

Run: `conda run -n liar_bar python -m unittest <targeted test module> -v`
Expected: FAIL because the loader currently accepts every non-empty game log

### Task 6: Implement the 8D short-game filter and retrain

**Files:**
- Modify: `liars_game_engine/analysis/train_value_proxy.py`
- Modify: `PROJECT_MEMORY.md`
- Modify: `logs/CHANGELOG_DAILY.md`
- Modify: `logs/CHANGELOG_DETAILED.md`
- Update: `logs/task_l_proxy_refine/` artifacts if retraining is run

**Step 1: Filter sub-3-turn games in the loader**

Keep the change minimal:
- discard a log before sample extraction when its valid turn count is less than 3

**Step 2: Run targeted tests and verify GREEN**

Run: `conda run -n liar_bar python -m unittest <targeted module list> -v`
Expected: PASS

**Step 3: Run the real 8D retraining/alignment**

Use the restored 8D pipeline and capture the new metrics after filtering.

### Task 7: Summarize final branch states and blockers

**Files:**
- None required beyond updated docs/logs

**Step 1: Report exact commits**

Include:
- `main` commit containing Task K serverization
- `dev-proxy-refine` commit/state after rollback and filtered retrain

**Step 2: Report server execution command**

Provide the exact server command and progress log path.
