# Liar's Bar Basic Rules Alignment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Align the current local game engine behavior with the provided basic Liar's Bar rules while keeping the existing orchestrator/agent interfaces stable.

**Architecture:** Keep the current `ActionModel` (`play_claim` / `challenge` / `pass`) and orchestrator contract, but rework the environment round lifecycle. Add round state (table type, first-turn gate, next-round starter assignment), deck generation (Liar deck + table card), and revolver-card penalties. Use TDD-first tests to pin expected rule behavior before implementation.

**Tech Stack:** Python 3.12, dataclasses, asyncio, unittest.

---

### Task 1: Add failing environment tests for basic-rule deltas

**Files:**
- Modify: `tests/test_environment_pipeline.py`

**Step 1: Write failing tests**
- Add tests for first-turn challenge rejection.
- Add tests for 1-3 cards play limit.
- Add tests for innocence evaluation based on `table_type` + Joker.
- Add tests for next-round starter assignment after LIAR resolution.

**Step 2: Run tests to verify failures**

Run: `python -m unittest tests.test_environment_pipeline -v`
Expected: FAIL on new rule-alignment assertions.

### Task 2: Implement round/deck/revolver rule alignment in engine

**Files:**
- Modify: `liars_game_engine/engine/game_state.py`
- Modify: `liars_game_engine/engine/environment.py`
- Modify: `liars_game_engine/engine/rules/declare_rule.py`
- Modify: `liars_game_engine/engine/rules/challenge_rule.py`
- Modify: `liars_game_engine/engine/rules/roulette_rule.py`

**Step 1: Add minimal new state fields**
- Add round metadata (`round_index`, `table_type`, `first_turn_of_round`).
- Track per-player revolver deck state.

**Step 2: Implement minimal behavior changes**
- Enforce 2-4 players.
- Start each round by selecting table type and dealing 5 cards from standard deck composition.
- Enforce challenge gate (not first turn) and play count limit (1-3 cards).
- Resolve LIAR by innocence set (`table_type`, `JOKER`) and apply revolver-card penalty.
- Assign next-round starter per rules (caught liar or incorrect caller; fallback to next alive).

**Step 3: Run tests to verify pass**

Run: `python -m unittest tests.test_environment_pipeline -v`
Expected: PASS.

### Task 3: Keep agent behavior compatible with updated observation/rules

**Files:**
- Modify: `liars_game_engine/agents/mock_agent.py`

**Step 1: Update mock decision policy minimally**
- Prefer challenge when forced by environment (`must_call_liar`).
- Keep simple randomized play of 1-3 cards.

**Step 2: Verify related tests**

Run: `python -m unittest tests.test_orchestrator_logging -v`
Expected: PASS.

### Task 4: Update default config and project docs/memory logs

**Files:**
- Modify: `config/experiment.yaml`
- Modify: `PROJECT_MEMORY.md`
- Modify: `logs/CHANGELOG_DAILY.md`
- Modify: `logs/CHANGELOG_DETAILED.md`

**Step 1: Align default config values**
- Use default deck ranks consistent with basic rules and joker representation.

**Step 2: Record memory/changelog updates**
- Document rule alignment and affected file paths.

### Task 5: Full verification

**Files:**
- Modify (if needed): `tests/test_config_loader.py`
- Modify (if needed): `tests/test_orchestrator_logging.py`

**Step 1: Run full test suite**

Run: `python -m unittest discover tests -v`
Expected: PASS.

**Step 2: If failures are unrelated**
- Report them separately without broad, unrelated refactors.
