# Unified Evaluation Layer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a unified scorecard that aggregates Task M, Task 1.1, and per-turn game logs into one comparable evaluation report.

**Architecture:** Add a small analysis module that reads `task_m/summary.json`, `task_1_1/summary.json`, and `task_m/games/*.jsonl`, then emits tiered metrics grouped as access, quality, stability, behavior, and auxiliary. Expose it through a lightweight CLI so baseline and SAVI checkpoints can be compared with one command.

**Tech Stack:** Python standard library, existing `liars_game_engine` analysis utilities, `unittest`.

---

### Task 1: Add failing scorecard tests

**Files:**
- Create: `tests/test_eval_scorecard.py`
- Test: `tests/test_eval_scorecard.py`

**Step 1: Write the failing test**

Cover:
- `challenge_accuracy`
- `bluff_efficiency`
- `conflict_rate`
- access gate metrics passthrough
- auxiliary `win_rate`

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_eval_scorecard -v`
Expected: FAIL because scorecard module does not exist yet.

### Task 2: Implement scorecard module

**Files:**
- Create: `liars_game_engine/analysis/eval_scorecard.py`
- Test: `tests/test_eval_scorecard.py`

**Step 1: Write minimal implementation**

Add:
- experiment loader
- behavior metric extraction from JSONL logs
- unified scorecard builder

**Step 2: Run test to verify it passes**

Run: `python -m unittest tests.test_eval_scorecard -v`
Expected: PASS

### Task 3: Add CLI entrypoint

**Files:**
- Create: `scripts/build_eval_scorecard.py`
- Test: `tests/test_eval_scorecard.py`

**Step 1: Write CLI test or script smoke test**

Ensure `--help` works and the script can render one scorecard.

**Step 2: Run targeted tests**

Run: `python -m unittest tests.test_eval_scorecard -v`
Expected: PASS

### Task 4: Verify broader stability

**Files:**
- Test: `tests/test_llm_drill.py`

**Step 1: Run focused regression tests**

Run:
- `python -m unittest tests.test_eval_scorecard -v`
- `python -m unittest tests.test_llm_drill -v`

**Step 2: Keep output green**

No unexpected failures.
