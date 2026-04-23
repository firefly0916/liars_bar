# Proxy-Based Attribution Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a proxy-based attribution path to `ShapleyAnalyzer` that uses the trained value model for fast per-step scoring, emits alignment metrics against physical rollouts, and generates a proxy credit report for the Task D probe logs.

**Architecture:** Keep the physical rollout path intact and add a parallel proxy path inside `liars_game_engine/analysis/shapley_analyzer.py`. Reuse the feature encoding from `train_value_proxy.py` so the `.pt` model sees identical normalized inputs. For proxy attribution, score the immediate successor state of the logged action and the immediate successor states of legal alternatives, then compare against the existing rollout path on a random 20-step sample to produce an alignment report.

**Tech Stack:** Python 3.12, unittest, PyTorch, existing game environment checkpoint loader.

---

### Task 1: Lock the proxy API with tests

**Files:**
- Modify: `tests/test_shapley_analyzer.py`
- Reference: `liars_game_engine/analysis/shapley_analyzer.py`
- Reference: `liars_game_engine/analysis/train_value_proxy.py`

**Step 1: Write the failing test**

Add tests for:
- `ProxyValuePredictor` using the same feature encoding helper as training.
- proxy attribution producing a non-null `ShapleyAttribution` from a logged turn.
- alignment report generation returning `pearson_correlation`, `mae`, and `speedup_ratio`.

**Step 2: Run test to verify it fails**

Run: `conda run -n liar_bar python -m unittest tests.test_shapley_analyzer -v`

Expected: FAIL because the proxy predictor and proxy attribution/report APIs do not exist yet.

**Step 3: Write minimal implementation**

Add only the minimal classes/methods needed to satisfy the test names and signatures.

**Step 4: Run test to verify it passes**

Run: `conda run -n liar_bar python -m unittest tests.test_shapley_analyzer -v`

Expected: PASS for the new proxy-focused tests.

### Task 2: Implement proxy scoring and fast attribution

**Files:**
- Modify: `liars_game_engine/analysis/shapley_analyzer.py`
- Reference: `liars_game_engine/analysis/train_value_proxy.py`

**Step 1: Write the failing test**

Add a test that the proxy path uses successor-state scoring for the logged action and legal alternatives, and that the resulting `phi` matches the difference between action score and legal-action average.

**Step 2: Run test to verify it fails**

Run: `conda run -n liar_bar python -m unittest tests.test_shapley_analyzer.ShapleyAnalyzerTest.test_proxy_attribution_uses_legal_action_average -v`

Expected: FAIL because proxy attribution math is not implemented yet.

**Step 3: Write minimal implementation**

Implement:
- `ProxyValuePredictor`
- helper(s) to derive successor observations/state features from checkpoints
- `ShapleyAnalyzer.attribute_step_proxy()`
- `ShapleyAnalyzer.analyze_logs_proxy()`

**Step 4: Run test to verify it passes**

Run: `conda run -n liar_bar python -m unittest tests.test_shapley_analyzer.ShapleyAnalyzerTest.test_proxy_attribution_uses_legal_action_average -v`

Expected: PASS.

### Task 3: Implement alignment experiment and artifact export

**Files:**
- Modify: `liars_game_engine/analysis/shapley_analyzer.py`
- Modify: `tests/test_shapley_analyzer.py`

**Step 1: Write the failing test**

Add a test that the alignment workflow emits a JSON-serializable report with:
- `pearson_correlation`
- `mae`
- `speedup_ratio`
- `sample_size`
- `alignment_passed`

**Step 2: Run test to verify it fails**

Run: `conda run -n liar_bar python -m unittest tests.test_shapley_analyzer.ShapleyAnalyzerTest.test_proxy_alignment_report_contains_required_metrics -v`

Expected: FAIL because the alignment workflow/report export does not exist yet.

**Step 3: Write minimal implementation**

Implement:
- random sampling of 20 decision points
- proxy vs rollout timing
- Pearson/MAE computation
- JSON export helper

**Step 4: Run test to verify it passes**

Run: `conda run -n liar_bar python -m unittest tests.test_shapley_analyzer.ShapleyAnalyzerTest.test_proxy_alignment_report_contains_required_metrics -v`

Expected: PASS.

### Task 4: Run Task I verification and produce artifacts

**Files:**
- Output: `logs/task_d_probe/proxy_alignment_report.json`
- Output: `logs/task_d_probe/credit_report_proxy.csv`
- Modify: `PROJECT_MEMORY.md`
- Modify: `logs/CHANGELOG_DAILY.md`
- Modify: `logs/CHANGELOG_DETAILED.md`

**Step 1: Run targeted tests**

Run: `conda run -n liar_bar python -m unittest tests.test_shapley_analyzer -v`

Expected: PASS.

**Step 2: Run the Task I verification script/entrypoint**

Run the new proxy-analysis entrypoint against `logs/task_d_probe/probe_logs` and export both artifacts.

Expected:
- `proxy_alignment_report.json` exists
- `credit_report_proxy.csv` exists
- report includes whether correlation passed the `> 0.75` gate

**Step 3: Run the broader regression suite**

Run: `conda run -n liar_bar python -m unittest discover tests -v`

Expected: PASS.

**Step 4: Update memory/changelog files**

Document the proxy attribution path, alignment gate, artifact locations, and any important caveats about the current 6-feature proxy.
