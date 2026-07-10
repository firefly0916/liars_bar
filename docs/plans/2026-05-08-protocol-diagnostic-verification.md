# Protocol Diagnostic Verification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a verification-first reporting layer that turns protocol diagnostic run artifacts into a single explicit report covering training signal health, shortlist/gameplay mismatch, and version-drift hints.

**Architecture:** Reuse existing `train_summary.json`, trajectory reports, shortlist selections, and optional proxy-rollout calibration summaries. Build one analysis module plus one CLI wrapper so server-side experiment outputs can be summarized deterministically without relying on manual log inspection.

**Tech Stack:** Python, stdlib JSON/pathlib/subprocess/hashlib, existing screening selector logic, `unittest`.

---

### Task 1: Lock training-signal diagnostics in tests

**Files:**
- Create: `tests/test_protocol_diagnostic_report.py`
- Test: `tests/test_protocol_diagnostic_report.py`

**Step 1: Write the failing test**

Cover:
- zero mask activation warning
- idle/signalless step counting
- mismatch-rate aggregation

**Step 2: Run test to verify it fails**

Run: `conda run -n liar_bar python -m unittest tests.test_protocol_diagnostic_report -v`

Expected: FAIL because the report module does not exist yet.

### Task 2: Lock shortlist-mismatch and legacy-selector detection in tests

**Files:**
- Modify: `tests/test_protocol_diagnostic_report.py`
- Test: `tests/test_protocol_diagnostic_report.py`

**Step 1: Write the failing test**

Cover:
- recomputed gameplay-first ranking from selection JSON rows
- mismatch flag when selected tags differ from gameplay-first top-k
- legacy selector hint when selection metadata is missing

**Step 2: Run test to verify it fails**

Run: `conda run -n liar_bar python -m unittest tests.test_protocol_diagnostic_report -v`

Expected: FAIL because the report module does not exist yet.

### Task 3: Implement protocol diagnostic reporting module

**Files:**
- Create: `liars_game_engine/analysis/protocol_diagnostic_report.py`
- Test: `tests/test_protocol_diagnostic_report.py`

**Step 1: Add training summary aggregation**

Implement:
- mask activity counts
- idle/signalless counts
- warning flags

**Step 2: Add shortlist/gameplay comparison**

Implement:
- robust row normalization from both legacy and current selection JSON
- gameplay-first re-ranking from artifact rows
- overlap/mismatch reporting

**Step 3: Add best-effort metadata capture**

Implement:
- git branch / head / dirty-state hint
- optional proxy model SHA256 when path is readable

**Step 4: Run focused tests**

Run: `conda run -n liar_bar python -m unittest tests.test_protocol_diagnostic_report -v`

Expected: PASS.

### Task 4: Add CLI wrapper

**Files:**
- Create: `scripts/build_protocol_diagnostic_report.py`
- Test: `tests/test_protocol_diagnostic_report.py`

**Step 1: Add CLI loader/writer**

Allow:
- positional `run_root`
- optional calibration summary override
- JSON or Markdown output

**Step 2: Verify module-level tests still pass**

Run: `conda run -n liar_bar python -m unittest tests.test_protocol_diagnostic_report -v`

Expected: PASS.

### Task 5: Verify minimal mainline regression set

**Files:**
- Test: `tests/test_protocol_diagnostic_report.py`
- Test: `tests/test_screening_selector.py`
- Test: `tests/test_proxy_rollout_calibration.py`
- Test: `tests/test_task_k_phi_distill_runner.py`

**Step 1: Run focused suite**

Run:
- `conda run -n liar_bar python -m unittest tests.test_protocol_diagnostic_report tests.test_screening_selector tests.test_proxy_rollout_calibration tests.test_task_k_phi_distill_runner -v`

**Step 2: Keep output green**

No failures, no import regressions.
