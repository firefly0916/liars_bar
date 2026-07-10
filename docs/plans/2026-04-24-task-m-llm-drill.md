# Task M LLM Drill Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build and run a lightweight 1 LLM vs 3 Mock drill pipeline on `dev-llm-agent`, with decision logs under `logs/llm_drill/`, representative excerpts, and parser hardening for mildly messy JSON output.

**Architecture:** Keep `build_agents()` as the single construction path, then add one local-first inference path inside `LlmAgent`: OpenAI-compatible HTTP remains supported, and `local://hf` enables in-process Hugging Face generation for real local drills without a separate HTTP server. Add a dedicated drill runner that loads settings, runs multiple games through `GameOrchestrator`, writes per-game JSONL logs, and emits a compact summary with three representative LLM decisions.

**Tech Stack:** Python 3.10, `unittest`, existing game engine/orchestrator/logger, `openai`, optional local `transformers` + `torch`.

---

### Task 1: Add parser coverage for prose-wrapped JSON

**Files:**
- Modify: `tests/test_action_parser.py`
- Modify: `liars_game_engine/agents/parsers.py`

**Step 1: Write the failing test**

Add a test that passes when the model returns explanatory prose before and after a valid JSON object, and still extracts `Reasoning` plus `Action`.

**Step 2: Run test to verify it fails**

Run: `conda run -n liar_bar python -m unittest tests.test_action_parser.ActionParserTest.test_parse_agent_output_extracts_embedded_json_payload -v`

Expected: FAIL because the parser only tries the whole string or fenced code blocks.

**Step 3: Write minimal implementation**

Teach `_extract_candidates()` to also try balanced JSON object slices from raw text, while preserving existing code-fence behavior.

**Step 4: Run test to verify it passes**

Run the same command and confirm PASS.

**Step 5: Commit**

Commit together with the next parser-related changes once all parser tests are green.

### Task 2: Add local Hugging Face backend support to `LlmAgent`

**Files:**
- Create: `liars_game_engine/agents/local_backend.py`
- Modify: `liars_game_engine/agents/llm_agent.py`
- Modify: `tests/test_llm_agent.py`
- Modify: `pyproject.toml`

**Step 1: Write the failing tests**

Add one test proving `LlmAgent` chooses a local backend when `base_url` starts with `local://hf`, and one test proving it still falls back to `challenge` with a structured `ParseError` if local dependencies are unavailable or local output is invalid.

**Step 2: Run tests to verify they fail**

Run: `conda run -n liar_bar python -m unittest tests.test_llm_agent.LlmAgentTest -v`

Expected: FAIL because `LlmAgent` currently only knows the OpenAI-compatible HTTP path.

**Step 3: Write minimal implementation**

Add a small cached helper that uses `transformers` locally, feeds the existing prompt message list into a chat-style prompt, generates short text, and returns a raw JSON-like string for the existing parser.

**Step 4: Run tests to verify they pass**

Run the same `tests.test_llm_agent` command and confirm PASS.

**Step 5: Commit**

Include `pyproject.toml` only if an optional dependency entry is needed for the local drill path.

### Task 3: Add a drill runner and drill summary output

**Files:**
- Create: `liars_game_engine/experiment/llm_drill.py`
- Create: `scripts/run_llm_drill.py`
- Create: `tests/test_llm_drill.py`

**Step 1: Write the failing tests**

Add tests for:
- building a 4-player drill from settings via `build_agents()`
- running multiple games and writing per-game JSONL logs under `logs/llm_drill/`
- computing a drill summary that includes parse-error counts and representative LLM decisions

**Step 2: Run tests to verify they fail**

Run: `conda run -n liar_bar python -m unittest tests.test_llm_drill -v`

Expected: FAIL because the drill module and script do not exist yet.

**Step 3: Write minimal implementation**

Create a thin drill module that loops over games, uses `GameEnvironment` + `ExperimentLogger` + `GameOrchestrator`, then extracts LLM turns from JSONL logs into a summary JSON with:
- total games
- llm turn count
- parse-error count / rate
- top three representative decisions, preferring high death risk and low honesty observations

**Step 4: Run tests to verify they pass**

Run the same `tests.test_llm_drill` command and confirm PASS.

**Step 5: Commit**

Commit with the runner and tests once targeted verification is green.

### Task 4: Configure a drill-ready experiment profile

**Files:**
- Modify: `config/experiment.yaml`
- Modify: `tests/test_config_loader.py`

**Step 1: Write the failing test**

Add a config-loader test that expects a valid 4-player drill profile with one `llm` player and three `mock` players, while still allowing env vars to override API/base URL.

**Step 2: Run test to verify it fails**

Run: `conda run -n liar_bar python -m unittest tests.test_config_loader -v`

Expected: FAIL because the current YAML only has two players and no drill-oriented local base URL.

**Step 3: Write minimal implementation**

Update `config/experiment.yaml` to a drill-friendly default:
- `p1` as `llm`
- `p2/p3/p4` as `mock`
- local-first `api.base_url: local://hf`
- `api.api_key: LOCAL`
- a small model such as `Qwen/Qwen2.5-0.5B-Instruct`

**Step 4: Run test to verify it passes**

Run the same `tests.test_config_loader` command and confirm PASS.

**Step 5: Commit**

Commit together with runner/config changes after targeted tests stay green.

### Task 5: Execute the drill and capture evidence

**Files:**
- Output only: `logs/llm_drill/`

**Step 1: Run targeted code verification**

Run:
- `conda run -n liar_bar python -m py_compile liars_game_engine/agents/local_backend.py liars_game_engine/agents/llm_agent.py liars_game_engine/agents/parsers.py liars_game_engine/experiment/llm_drill.py scripts/run_llm_drill.py`
- `conda run -n liar_bar python -m unittest tests.test_action_parser tests.test_llm_agent tests.test_llm_drill tests.test_config_loader tests.test_mock_agent_pipeline -v`

Expected: PASS.

**Step 2: Run the actual drill**

Run: `conda run -n liar_bar python scripts/run_llm_drill.py --games 5`

Expected: logs and summary files under `logs/llm_drill/`.

**Step 3: Review outputs**

Confirm:
- representative excerpts include 8D-style observation context, `Reasoning`, and `Action`
- parse-error rate is explicit
- high-risk / high-bluff-suspicion examples are surfaced if present in logs

**Step 4: Commit**

Commit only source/test/config changes, not generated drill logs unless explicitly desired.
