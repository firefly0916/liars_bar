# Task K Observability Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make Task K runs leave enough evidence to distinguish normal completion, Python exceptions, and abrupt termination/resource pressure.

**Architecture:** Extend the Task K runner to emit structured diagnostics into each run directory, including startup metadata, per-batch progress snapshots, and explicit failure records. Update the server launch scripts to run Python unbuffered so `run.log` reflects live stdout/stderr instead of being lost behind `conda run` buffering.

**Tech Stack:** Python 3.10, `unittest`, shell scripts, JSONL/plain-text logs.

---

### Task 1: Lock observability behavior with tests

**Files:**
- Modify: `tests/test_task_k_task_l_runners.py`
- Modify: `tests/test_server_scripts.py`

**Step 1: Write the failing tests**

- Add a Task K test that expects a `diagnostics.jsonl` path in the summary and verifies at least `pipeline_start`, `batch_completed`, and `pipeline_finished` events are written.
- Add a Task K test that forces `analyze_logs` to raise and verifies `failure.log` plus a `pipeline_failed` diagnostics event are emitted.
- Extend server script tests to require `conda run --no-capture-output` and `python -u`.

**Step 2: Run tests to verify they fail**

Run:

```bash
conda run -n liar_bar python -m unittest \
  tests.test_task_k_task_l_runners \
  tests.test_server_scripts -v
```

Expected: failures because diagnostics/failure logs and unbuffered launch are not implemented yet.

### Task 2: Implement runner diagnostics

**Files:**
- Modify: `liars_game_engine/analysis/task_k_gold_runner.py`

**Step 1: Add structured diagnostics helpers**

- Create helpers to append JSONL diagnostic events into the run directory.
- Include runtime snapshot fields that help reason about CPU and memory pressure: PID/PPID, `os.getloadavg`, CPU count, process CPU time, child CPU time, resident memory when available.

**Step 2: Integrate lifecycle logging**

- Emit `pipeline_start` before work begins.
- Emit `batch_completed` after each progress checkpoint.
- Emit `pipeline_finished` with summary on success.
- Wrap execution so any exception appends `pipeline_failed` and writes `failure.log` with traceback before re-raising.

### Task 3: Implement unbuffered launch and verify

**Files:**
- Modify: `scripts/run_task_k_gold_server.sh`
- Modify: `scripts/remote_run.sh`

**Step 1: Update launch commands**

- Use `conda run --no-capture-output`.
- Use `python -u`.

**Step 2: Run tests to verify green**

Run:

```bash
conda run -n liar_bar python -m unittest \
  tests.test_task_k_task_l_runners \
  tests.test_server_scripts -v
```

Expected: all tests pass.
