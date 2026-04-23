# Task C Shapley Sampling Validation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在现有 Mock 引擎与技能日志基础上，完成 50 局基线日志生成、反事实回溯采样（N=50）与 `credit_report.csv` 聚合输出。

**Architecture:** 复用 `GameOrchestrator` 生成结构化 JSONL，并在 `ShapleyAnalyzer` 内实现“原动作 vs 非原合法动作均值”的双分支回放。新增任务 C 运行入口负责批量对局与报表导出，采样回放使用 `ProcessPoolExecutor` 并在 100 步上限触发平局 0.5 防卡死。

**Tech Stack:** Python 3.12, asyncio, dataclasses, unittest, ProcessPoolExecutor.

---

### Task 1: Add failing tests for Task C requirements

**Files:**
- Modify: `tests/test_orchestrator_logging.py`
- Modify: `tests/test_shapley_analyzer.py`

**Step 1: Write failing tests**
- 在 orchestrator 日志测试中新增 `state_features` 断言，要求存在 6 个核心状态特征。
- 在 shapley 分析测试中新增断言：
  - 反事实采样不使用原动作；
  - 回放超步数时按 0.5 平局处理；
  - 可导出 `credit_report.csv` 并包含指定列。

**Step 2: Run tests to verify failures**

Run: `python -m unittest tests.test_orchestrator_logging tests.test_shapley_analyzer -v`
Expected: FAIL on new Task C assertions.

### Task 2: Implement counterfactual sampling and report export

**Files:**
- Modify: `liars_game_engine/experiment/orchestrator.py`
- Modify: `liars_game_engine/analysis/shapley_analyzer.py`

**Step 1: Minimal logging enhancement**
- 为每回合日志增加 `state_features`，包含 6 个核心特征（用于后续聚合和追踪）。

**Step 2: Minimal analyzer behavior changes**
- 将反事实 baseline 改为“除原动作外的合法动作平均胜率”。
- rollout 固定 100 步防护；超限或无法结束记 0.5。
- 保持 `phi = V_actual - V_baseline`。
- 并行回放时 `max_workers` 支持 CPU 核心数。

**Step 3: Add report export helper**
- 输出 `credit_report.csv`：`skill_name`, `death_prob_bucket`, `avg_shapley_value`, `sample_count`。

### Task 3: Add task-runner pipeline for 50-game execution

**Files:**
- Add: `liars_game_engine/analysis/task_c_runner.py`

**Step 1: Build baseline log generation**
- 基于配置构建 4 Mock Agent，运行 50 局并收集 `.jsonl`。

**Step 2: Run analyzer and export CSV**
- 运行 `ShapleyAnalyzer`（`rollout_samples=50`）并导出 `credit_report.csv`。

### Task 4: Verify and document changes

**Files:**
- Modify: `PROJECT_MEMORY.md`
- Modify: `logs/CHANGELOG_DAILY.md`
- Modify: `logs/CHANGELOG_DETAILED.md`

**Step 1: Run full tests**

Run: `python -m unittest discover tests -v`
Expected: PASS (29/29 或以上，且新增测试通过)。

**Step 2: Run Task C data pipeline**

Run: `python -m liars_game_engine.analysis.task_c_runner`
Expected: 生成 50 局日志与 `credit_report.csv`，无异常退出。
