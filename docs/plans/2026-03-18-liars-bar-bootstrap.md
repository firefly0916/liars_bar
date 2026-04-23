# Liar's Bar Bootstrap Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a runnable async Liar's Bar project skeleton with configurable multi-agent setup, phase-based engine, tolerant action parsing, and required project memory/changelog files.

**Architecture:** Use a pure Python async orchestrator that mediates between a deterministic environment state machine and pluggable agents. Keep game logic in `engine/`, agent behavior in `agents/`, and experiment I/O in `experiment/`. Use `.env` for secrets and one `config/experiment.yaml` file for all tunable experiment parameters.

**Tech Stack:** Python 3.10+, asyncio, pydantic v2, PyYAML, python-dotenv, pytest, pytest-asyncio.

---

### Task 1: Project scaffolding and config models

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `config/experiment.yaml`
- Create: `liars_game_engine/config/schema.py`
- Create: `liars_game_engine/config/loader.py`
- Test: `tests/test_config_loader.py`

**Step 1: Write the failing test**

```python
def test_load_settings_merges_env_and_yaml(...):
    settings = load_settings(...)
    assert settings.api.openrouter_api_key == "test-key"
    assert settings.players[1].model == "openai/gpt-4o-mini"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_config_loader.py -v`
Expected: FAIL with import/module missing.

**Step 3: Write minimal implementation**

Implement Pydantic settings schema and loader merging `.env` + YAML.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_config_loader.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add pyproject.toml .env.example config/experiment.yaml liars_game_engine/config tests/test_config_loader.py
git commit -m "feat: add config schema and loader"
```

### Task 2: Action schema and tolerant parser

**Files:**
- Create: `liars_game_engine/engine/game_state.py`
- Create: `liars_game_engine/agents/parsers.py`
- Test: `tests/test_action_parser.py`

**Step 1: Write failing tests**

```python
def test_parser_accepts_markdown_json_block():
    ...

def test_parser_returns_structured_error_on_invalid_output():
    ...
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_action_parser.py -v`
Expected: FAIL due missing parser.

**Step 3: Write minimal implementation**

Implement action schema + parser fallback layers + error codes.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_action_parser.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add liars_game_engine/engine/game_state.py liars_game_engine/agents/parsers.py tests/test_action_parser.py
git commit -m "feat: add tolerant action parser with error codes"
```

### Task 3: Phase pipeline environment and rules

**Files:**
- Create: `liars_game_engine/engine/rules/base.py`
- Create: `liars_game_engine/engine/rules/declare_rule.py`
- Create: `liars_game_engine/engine/rules/challenge_rule.py`
- Create: `liars_game_engine/engine/rules/roulette_rule.py`
- Create: `liars_game_engine/engine/environment.py`
- Test: `tests/test_environment_pipeline.py`

**Step 1: Write failing tests**

```python
def test_declare_moves_phase_to_response_window():
    ...

def test_challenge_penalty_eliminates_loser_when_shot_fires():
    ...
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_environment_pipeline.py -v`
Expected: FAIL due missing environment/rules.

**Step 3: Write minimal implementation**

Implement phase transitions and challenge + roulette penalty resolution.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_environment_pipeline.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add liars_game_engine/engine liars_game_engine/engine/rules tests/test_environment_pipeline.py
git commit -m "feat: add phase-based environment and core rules"
```

### Task 4: Agents, orchestrator, and experiment logging

**Files:**
- Create: `liars_game_engine/agents/base_agent.py`
- Create: `liars_game_engine/agents/mock_agent.py`
- Create: `liars_game_engine/agents/langchain_agent.py`
- Create: `liars_game_engine/agents/factory.py`
- Create: `liars_game_engine/agents/prompts.py`
- Create: `liars_game_engine/experiment/logger.py`
- Create: `liars_game_engine/experiment/orchestrator.py`
- Create: `liars_game_engine/main.py`
- Create: `prompts/profiles/baseline.yaml`
- Test: `tests/test_orchestrator_logging.py`

**Step 1: Write failing test**

```python
@pytest.mark.asyncio
async def test_orchestrator_records_turn_logs(...):
    ...
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_orchestrator_logging.py -v`
Expected: FAIL due missing orchestrator/logger.

**Step 3: Write minimal implementation**

Implement async loop, mock agent, optional LangChain skeleton, JSONL logger.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_orchestrator_logging.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add liars_game_engine/agents liars_game_engine/experiment liars_game_engine/main.py prompts/profiles/baseline.yaml tests/test_orchestrator_logging.py
git commit -m "feat: add orchestrator, agents, and experiment logging"
```

### Task 5: Project memory and changelog policy files

**Files:**
- Create: `PROJECT_MEMORY.md`
- Create: `logs/CHANGELOG_DAILY.md`
- Create: `logs/CHANGELOG_DETAILED.md`
- Create: `README.md`

**Step 1: Write policy verification test (optional lightweight)**

```python
def test_required_memory_files_exist():
    ...
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_project_policy_files.py -v`
Expected: FAIL when files are absent.

**Step 3: Write minimal implementation**

Create policy files with required maintenance instructions and first entries.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_project_policy_files.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add PROJECT_MEMORY.md logs/CHANGELOG_DAILY.md logs/CHANGELOG_DETAILED.md README.md tests/test_project_policy_files.py
git commit -m "docs: add project memory and mandatory changelog files"
```

