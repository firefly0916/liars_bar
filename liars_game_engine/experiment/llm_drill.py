from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from liars_game_engine.agents.factory import build_agents
from liars_game_engine.config.schema import AppSettings
from liars_game_engine.engine.environment import GameEnvironment
from liars_game_engine.engine.game_state import JOKER_RANK
from liars_game_engine.experiment.logger import ExperimentLogger
from liars_game_engine.experiment.orchestrator import GameOrchestrator


def _build_game_id(index: int) -> str:
    return f"llm-drill-{index:02d}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def _estimate_honesty_ratio(observation: dict[str, object]) -> float:
    private_hand = observation.get("private_hand", [])
    table_type = str(observation.get("table_type", ""))
    if not isinstance(private_hand, list) or not private_hand:
        return 0.0
    truthful_cards = sum(1 for card in private_hand if str(card) in {table_type, JOKER_RANK})
    return truthful_cards / len(private_hand)


def _extract_llm_turns(log_file: Path, llm_player_id: str) -> list[dict[str, object]]:
    turns: list[dict[str, object]] = []
    for line in log_file.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("player_id") != llm_player_id:
            continue

        observation = record.get("observation", {})
        death_probability = 0.0
        if isinstance(observation, dict):
            player_states = observation.get("player_states", {})
            player_state = player_states.get(llm_player_id, {}) if isinstance(player_states, dict) else {}
            if isinstance(player_state, dict):
                death_probability = float(player_state.get("death_probability", 0.0) or 0.0)

        turns.append(
            {
                "game_id": log_file.stem,
                "turn": int(record.get("turn", 0) or 0),
                "thought": str(record.get("thought", "")),
                "action": record.get("action", {}),
                "raw_output": str(record.get("raw_output", "")),
                "observation": observation,
                "death_probability": death_probability,
                "honesty_ratio": _estimate_honesty_ratio(observation if isinstance(observation, dict) else {}),
                "parse_error": record.get("parser_error"),
                "log_file": str(log_file),
            }
        )
    return turns


def select_representative_decisions(decisions: list[dict[str, object]], limit: int = 3) -> list[dict[str, object]]:
    ranked = sorted(
        decisions,
        key=lambda item: (
            -(float(item.get("death_probability", 0.0) or 0.0) + (1.0 - float(item.get("honesty_ratio", 0.0) or 0.0))),
            int(item.get("turn", 0) or 0),
        ),
    )
    return ranked[:limit]


async def run_llm_drill(
    settings: AppSettings,
    games: int = 5,
    log_dir: str | Path = "logs/llm_drill",
) -> dict[str, object]:
    log_dir_path = Path(log_dir)
    games_dir = log_dir_path / "games"
    games_dir.mkdir(parents=True, exist_ok=True)

    llm_player = next(player for player in settings.players if player.agent_type == "llm")
    game_summaries: list[dict[str, object]] = []
    llm_turns: list[dict[str, object]] = []

    for game_index in range(1, games + 1):
        env = GameEnvironment(settings)
        agents = build_agents(settings)
        logger = ExperimentLogger(base_dir=games_dir, game_id=_build_game_id(game_index))
        orchestrator = GameOrchestrator(
            env=env,
            agents=agents,
            logger=logger,
            fallback_action=settings.runtime.fallback_action,
            max_turns=settings.runtime.max_turns,
        )
        summary = await orchestrator.run_game_loop()
        game_summaries.append(summary)
        llm_turns.extend(_extract_llm_turns(logger.log_file, llm_player.player_id))

    parse_error_count = sum(1 for turn in llm_turns if turn.get("parse_error"))
    llm_turn_count = len(llm_turns)
    summary = {
        "total_games": games,
        "llm_player_id": llm_player.player_id,
        "llm_turn_count": llm_turn_count,
        "parse_error_count": parse_error_count,
        "parse_error_rate": (parse_error_count / llm_turn_count) if llm_turn_count else 0.0,
        "game_summaries": game_summaries,
        "representative_decisions": select_representative_decisions(llm_turns, limit=3),
    }

    summary_path = log_dir_path / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
