# Proxy Quality Diagnosis Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reconstruct the 20-point proxy alignment sample, identify the worst outliers, verify feature-order consistency, and record whether an 8-feature proxy requires new training data collection.

**Architecture:** Reuse the existing `task_i_proxy_runner` log-selection rule and `ShapleyAnalyzer` alignment sampling rule so the forensic analysis targets the exact same 20 points. Compute rollout/proxy attribution side by side, rank by absolute error and sign conflict, then summarize the findings in `PROJECT_MEMORY.md` without changing the production proxy path.

**Tech Stack:** Python 3.12, existing analyzer/predictor modules, JSON artifacts, unittest-free verification via deterministic script execution.

---

### Task 1: Reconstruct the alignment sample

**Files:**
- Reference: `liars_game_engine/analysis/task_i_proxy_runner.py`
- Reference: `liars_game_engine/analysis/shapley_analyzer.py`
- Output: `logs/task_d_probe/proxy_forensics_report.json`

**Step 1: Run a deterministic forensic script**

Use the existing latest-run selection and alignment sample seed to reproduce the same 20 decision points used in Task I.

**Step 2: Verify output shape**

Confirm the artifact includes:
- sampled game/turn/player keys
- rollout/proxy phi
- absolute error
- sign mismatch flag
- raw 6-dim state features

### Task 2: Analyze the top 3 outliers

**Files:**
- Output: `logs/task_d_probe/proxy_forensics_report.json`
- Modify: `PROJECT_MEMORY.md`

**Step 1: Rank the 20 points**

Sort by absolute error descending and keep the top 3 points, especially those with opposite rollout/proxy signs.

**Step 2: Summarize the game context**

For each point, explain the likely failure mode in terms of risk, claim context, hand pressure, and missing logic features.

### Task 3: Verify feature consistency

**Files:**
- Reference: `liars_game_engine/analysis/shapley_analyzer.py`
- Reference: `liars_game_engine/analysis/train_value_proxy.py`

**Step 1: Check the feature path**

Verify the proxy path calls the exact training feature encoder rather than duplicating feature ordering locally.

**Step 2: Record the conclusion**

State clearly whether the negative correlation is caused by feature-order drift or by missing information in the feature set.

### Task 4: Record data-collection guidance

**Files:**
- Modify: `PROJECT_MEMORY.md`
- Modify: `logs/CHANGELOG_DAILY.md`
- Modify: `logs/CHANGELOG_DETAILED.md`

**Step 1: Record the forensic finding**

Document the main outlier pattern and the decision to keep physical rollout as the paper-data source for EXP1.

**Step 2: Evaluate the 6D -> 8D feature expansion**

Answer whether new raw logs are required, or whether existing logs can be re-featurized for retraining.
