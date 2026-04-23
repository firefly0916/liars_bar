# Null Probe Skill Injection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 通过配置开关 `enable_null_player_probe` 注入 `Null_Probe_Skill`，并确保日志与 Shapley 分析可对该探测技能进行专门统计。

**Architecture:** 在 `RuntimeSettings` 增加开关字段并沿 `build_agents -> Agent -> LiarPlanner` 管线传递。Planner 注册探测技能并在执行器实现“最小代价动作”逻辑；Orchestrator 日志增加 `skill_category` 与 `log_version`。ShapleyAnalyzer 增加 `Null_Probe_Skill` 专项统计聚合函数。

**Tech Stack:** Python 3.12, dataclasses, unittest.

---

### Task 1: Add failing tests for config and planner registration

**Files:**
- Modify: `tests/test_config_loader.py`
- Modify: `tests/test_mock_agent_pipeline.py`
- Modify: `tests/test_liar_planner.py`

**Step 1: Write failing tests**
- 断言配置可读取 `runtime.enable_null_player_probe`。
- 断言 `build_agents` 创建的 `MockAgent` 在开关开启时其 Planner 注册 `Null_Probe_Skill`。
- 断言 `LiarPlanner` 解析 `Null_Probe_Skill` 时会打上 Probe 参数并产出合法动作。

**Step 2: Run tests to verify failures**

Run: `python -m unittest tests.test_config_loader tests.test_mock_agent_pipeline tests.test_liar_planner -v`
Expected: FAIL on new assertions.

### Task 2: Implement config injection and skill execution

**Files:**
- Modify: `liars_game_engine/config/schema.py`
- Modify: `config/experiment.yaml`
- Modify: `liars_game_engine/agents/factory.py`
- Modify: `liars_game_engine/agents/mock_agent.py`
- Modify: `liars_game_engine/agents/langchain_agent.py`
- Modify: `liars_game_engine/agents/liar_planner.py`
- Modify: `liars_game_engine/agents/parsers.py`

**Step 1: Minimal config plumbing**
- 增加 `enable_null_player_probe: bool = False` 并沿工厂注入 Agent/Planner。

**Step 2: Minimal skill registration and behavior**
- 注册 `Null_Probe_Skill`。
- 实现“最小代价动作”：优先 1 张真牌，否则 1 张假牌，若仅可 challenge/pass 则遵守合法动作。
- 输出参数强制包含 `probe_type=Probe`。

**Step 3: Keep parser/prompt compatibility**
- Planner 输出校验支持 `Null_Probe_Skill`。
- LangChain 提示中的 Skill 枚举与实际注册保持一致。

### Task 3: Add logging mark and analyzer probe stats

**Files:**
- Modify: `liars_game_engine/experiment/orchestrator.py`
- Modify: `tests/test_orchestrator_logging.py`
- Modify: `liars_game_engine/analysis/shapley_analyzer.py`
- Modify: `tests/test_shapley_analyzer.py`

**Step 1: Add log alignment fields**
- 增加 `log_version`。
- 增加 `skill_category`，当 `skill_name==Null_Probe_Skill` 时标记为 `Probe`。

**Step 2: Add analyzer probe summary helper**
- 提供 `Null_Probe_Skill` 的 `phi_avg/phi_sum/sample_count` 汇总接口。

### Task 4: Verify and sync docs/changelog

**Files:**
- Modify: `README.md`
- Modify: `PROJECT_MEMORY.md`
- Modify: `logs/CHANGELOG_DAILY.md`
- Modify: `logs/CHANGELOG_DETAILED.md`

**Step 1: Run relevant tests**

Run: `python -m unittest tests.test_config_loader tests.test_mock_agent_pipeline tests.test_liar_planner tests.test_orchestrator_logging tests.test_shapley_analyzer -v`
Expected: PASS.

**Step 2: Run full regression**

Run: `python -m unittest discover tests -v`
Expected: PASS.
