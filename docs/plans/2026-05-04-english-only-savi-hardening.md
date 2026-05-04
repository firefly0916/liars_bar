# English-Only SAVI Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enforce English-only runtime prompts and SAVI training inputs before continuing Task O.3 checkpoint work and post-train evaluation.

**Architecture:** Add fail-fast English guards at the prompt-generation and training-data entry points, then regenerate downstream datasets from fresh English Task M logs instead of reusing old Chinese samples. Keep the current reward/group design unchanged so future O.3 comparisons remain comparable to the existing run1-run5 results.

**Tech Stack:** Python, unittest, JSONL datasets, local prompt builders, LoRA smoke-training utilities

---

### Task 1: Harden Task M Prompt Output To English

**Files:**
- Modify: `liars_game_engine/agents/prompts.py`
- Modify: `tests/test_llm_agent.py`
- Modify: `.worktrees/dev-proxy-refine/liars_game_engine/agents/prompts.py`

**Step 1: Write the failing test**

Update `tests/test_llm_agent.py` so the prompt assertions require English-only status labels such as `honesty_reference`, `persona_stability`, and `roulette_death_probability`.

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_llm_agent.py -q`
Expected: FAIL because the current prompt still emits Chinese labels.

**Step 3: Write minimal implementation**

Replace Chinese-facing runtime prompt labels and qualitative descriptors in both prompt builders with English-only equivalents while preserving structure and token budget.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_llm_agent.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_llm_agent.py liars_game_engine/agents/prompts.py .worktrees/dev-proxy-refine/liars_game_engine/agents/prompts.py
git commit -m "fix: enforce english-only task m prompt text"
```

### Task 2: Harden SAVI Dataset Builders To Reject Chinese Runtime Content

**Files:**
- Modify: `.worktrees/dev-proxy-refine/liars_game_engine/analysis/hicra_preprocessor.py`
- Modify: `.worktrees/dev-proxy-refine/tests/test_hicra_preprocessor.py`
- Modify: `.worktrees/dev-proxy-refine/scripts/build_savi_alignment_dataset.py`
- Modify: `.worktrees/dev-proxy-refine/scripts/build_savi_alignment_full_dataset.py`

**Step 1: Write the failing test**

Add tests that:
- require holographic prompt fields to use English labels only
- reject records whose `thought` contains CJK characters

**Step 2: Run test to verify it fails**

Run: `python -m pytest .worktrees/dev-proxy-refine/tests/test_hicra_preprocessor.py -q`
Expected: FAIL because Chinese thoughts and prompt fields are still accepted.

**Step 3: Write minimal implementation**

Remove Chinese token regex branches from `TOKEN_PATTERNS` and add fail-fast validators for `thought`, `base_prompt`, `rendered_prompt`, and chat message content.

**Step 4: Run test to verify it passes**

Run: `python -m pytest .worktrees/dev-proxy-refine/tests/test_hicra_preprocessor.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add .worktrees/dev-proxy-refine/liars_game_engine/analysis/hicra_preprocessor.py .worktrees/dev-proxy-refine/tests/test_hicra_preprocessor.py .worktrees/dev-proxy-refine/scripts/build_savi_alignment_dataset.py .worktrees/dev-proxy-refine/scripts/build_savi_alignment_full_dataset.py
git commit -m "fix: reject non-english savi dataset content"
```

### Task 3: Harden Training Entry Points To Refuse Old Chinese Datasets

**Files:**
- Modify: `.worktrees/feat-savi-grpo/liars_game_engine/analysis/hicra_preprocessor.py`
- Modify: `.worktrees/feat-savi-grpo/scripts/train_savi_alignment.py`
- Modify: `.worktrees/feat-savi-grpo/tests/test_hicra_preprocessor.py`
- Modify: `.worktrees/feat-savi-grpo/tests/test_train_savi_alignment.py`

**Step 1: Write the failing test**

Add a loader-level test that feeds a Chinese-containing JSONL sample into `load_alignment_records()` and expects a validation error.

**Step 2: Run test to verify it fails**

Run: `python -m pytest .worktrees/feat-savi-grpo/tests/test_train_savi_alignment.py -q`
Expected: FAIL because the loader currently accepts Chinese dataset content.

**Step 3: Write minimal implementation**

Mirror the English-only token/record validation in the training branch and validate loaded JSONL records before smoke training or dry-run execution.

**Step 4: Run test to verify it passes**

Run: `python -m pytest .worktrees/feat-savi-grpo/tests/test_train_savi_alignment.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add .worktrees/feat-savi-grpo/liars_game_engine/analysis/hicra_preprocessor.py .worktrees/feat-savi-grpo/scripts/train_savi_alignment.py .worktrees/feat-savi-grpo/tests/test_hicra_preprocessor.py .worktrees/feat-savi-grpo/tests/test_train_savi_alignment.py
git commit -m "fix: block non-english savi training inputs"
```

### Task 4: Verify Hardening And Record The Required Rerun Sequence

**Files:**
- Modify: `docs/tmp/2026-04-28-project-handoff.md`

**Step 1: Write the failing test**

No code test for this step.

**Step 2: Run test to verify it fails**

Not applicable.

**Step 3: Write minimal implementation**

Append a concise note that old Chinese Task M / Task 1.1 / Task 2.x artifacts are now legacy-only and must be replaced by a fresh English rerun before Task O.3 and post-train evaluation.

**Step 4: Run test to verify it passes**

Run:
- `python -m pytest tests/test_llm_agent.py -q`
- `python -m pytest .worktrees/dev-proxy-refine/tests/test_hicra_preprocessor.py -q`
- `python -m pytest .worktrees/feat-savi-grpo/tests/test_train_savi_alignment.py -q`

Expected: PASS

**Step 5: Commit**

```bash
git add docs/tmp/2026-04-28-project-handoff.md
git commit -m "docs: record english-only savi rerun requirement"
```
