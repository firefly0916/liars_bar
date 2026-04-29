from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import time

from liars_game_engine.agents.factory import build_agents
from liars_game_engine.config.schema import AppSettings
from liars_game_engine.engine.environment import GameEnvironment
from liars_game_engine.engine.game_state import JOKER_RANK
from liars_game_engine.experiment.logger import ExperimentLogger
from liars_game_engine.experiment.orchestrator import GameOrchestrator


def _build_game_id(index: int) -> str:
    return f"llm-drill-{index:02d}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def _append_progress(progress_log: Path, **fields: object) -> None:
    progress_log.parent.mkdir(parents=True, exist_ok=True)
    line = " ".join(f"{key}={value}" for key, value in fields.items())
    with progress_log.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def _render_progress_bar(percent: float, width: int = 20) -> str:
    bounded = max(0.0, min(1.0, percent))
    filled = min(width, int(round(bounded * width)))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _estimate_eta_seconds(start_time: float, completed_turns: int, total_turn_budget: int) -> float:
    if completed_turns <= 0 or total_turn_budget <= completed_turns:
        return 0.0
    elapsed = max(0.0, time.perf_counter() - start_time)
    seconds_per_turn = elapsed / completed_turns
    return max(0.0, seconds_per_turn * (total_turn_budget - completed_turns))


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
                "llm_intent": record.get("llm_intent"),
                "resolution_reason": record.get("resolution_reason"),
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


def select_high_risk_reasoning_snippets(
    decisions: list[dict[str, object]],
    limit: int = 3,
    risk_threshold: float = 1.0 / 3.0,
) -> list[dict[str, object]]:
    filtered = [
        item for item in decisions if float(item.get("death_probability", 0.0) or 0.0) > float(risk_threshold)
    ]
    ranked = sorted(
        filtered,
        key=lambda item: (
            -float(item.get("death_probability", 0.0) or 0.0),
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
    progress_log = log_dir_path / "progress.log"
    start_time = time.perf_counter()

    llm_player = next(player for player in settings.players if player.agent_type == "llm")
    game_summaries: list[dict[str, object]] = []
    llm_turns: list[dict[str, object]] = []
    total_turn_budget = max(1, int(games) * int(settings.runtime.max_turns))
    completed_turns = 0

    _append_progress(
        progress_log,
        status="starting",
        completed_turns=0,
        total_turn_budget=total_turn_budget,
        percent="0.0%",
        progress_bar=_render_progress_bar(0.0),
        eta_seconds=f"{float(0.0):.6f}",
    )

    for game_index in range(1, games + 1):
        env = GameEnvironment(settings)
        agents = build_agents(settings)
        logger = ExperimentLogger(base_dir=games_dir, game_id=_build_game_id(game_index))

        def _handle_turn_progress(payload: dict[str, object]) -> None:
            nonlocal completed_turns
            completed_turns += 1
            percent = completed_turns / total_turn_budget if total_turn_budget else 1.0
            _append_progress(
                progress_log,
                status="running",
                game=f"{game_index}/{games}",
                current_turn=f"{payload.get('turns_played', 0)}/{payload.get('max_turns', settings.runtime.max_turns)}",
                completed_turns=completed_turns,
                total_turn_budget=total_turn_budget,
                percent=f"{percent * 100:.1f}%",
                progress_bar=_render_progress_bar(percent),
                eta_seconds=f"{_estimate_eta_seconds(start_time, completed_turns, total_turn_budget):.6f}",
                game_id=payload.get("game_id"),
            )

        orchestrator = GameOrchestrator(
            env=env,
            agents=agents,
            logger=logger,
            fallback_action=settings.runtime.fallback_action,
            max_turns=settings.runtime.max_turns,
            progress_callback=_handle_turn_progress,
        )
        summary = await orchestrator.run_game_loop()
        game_summaries.append(summary)
        extracted_turns = _extract_llm_turns(logger.log_file, llm_player.player_id)
        llm_turns.extend(extracted_turns)

    parse_error_count = sum(1 for turn in llm_turns if turn.get("parse_error"))
    resolution_adjustment_turns = [turn for turn in llm_turns if turn.get("resolution_reason")]
    llm_turn_count = len(llm_turns)
    high_risk_reasoning_snippets = select_high_risk_reasoning_snippets(llm_turns, limit=3)
    high_risk_reasoning_path = log_dir_path / "high_risk_reasoning.json"
    high_risk_reasoning_path.write_text(
        json.dumps(high_risk_reasoning_snippets, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "total_games": games,
        "llm_player_id": llm_player.player_id,
        "llm_turn_count": llm_turn_count,
        "parse_error_count": parse_error_count,
        "parse_error_rate": (parse_error_count / llm_turn_count) if llm_turn_count else 0.0,
        "resolution_adjustment_count": len(resolution_adjustment_turns),
        "resolution_adjustment_rate": (len(resolution_adjustment_turns) / llm_turn_count) if llm_turn_count else 0.0,
        "resolution_adjustment_examples": resolution_adjustment_turns[:3],
        "game_summaries": game_summaries,
        "representative_decisions": select_representative_decisions(llm_turns, limit=3),
        "high_risk_reasoning_snippets": high_risk_reasoning_snippets,
        "high_risk_reasoning_path": str(high_risk_reasoning_path),
        "progress_log": str(progress_log),
    }

    summary_path = log_dir_path / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _append_progress(
        progress_log,
        status="completed",
        run_complete="true",
        completed_games=f"{games}/{games}",
        completed_turns=completed_turns,
        total_turn_budget=total_turn_budget,
        percent="100.0%",
        progress_bar=_render_progress_bar(1.0),
        eta_seconds=f"{float(0.0):.6f}",
        total_llm_turns=llm_turn_count,
        parse_error_rate=summary["parse_error_rate"],
        resolution_adjustment_rate=summary["resolution_adjustment_rate"],
        high_risk_snippet_count=len(high_risk_reasoning_snippets),
    )
    return summary
