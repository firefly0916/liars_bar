from __future__ import annotations

import json
from pathlib import Path

from liars_game_engine.engine.game_state import JOKER_RANK


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_game_records(games_dir: Path) -> list[list[dict[str, object]]]:
    games: list[list[dict[str, object]]] = []
    for log_path in sorted(games_dir.glob("*.jsonl")):
        records = [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        games.append(records)
    return games


def _truthful_card_count(cards: list[object], table_type: str) -> int:
    truthful_ranks = {str(table_type), JOKER_RANK}
    return sum(1 for card in cards if str(card) in truthful_ranks)


def _challenge_outcome(events: object) -> bool | None:
    if not isinstance(events, list):
        return None
    for event in events:
        if not isinstance(event, str):
            continue
        if "At least one revealed card is a Liar." in event:
            return True
        if "All revealed cards are Innocent." in event:
            return False
    return None


def _extract_pending_signature(record: dict[str, object]) -> tuple[str, str, int] | None:
    action = record.get("action", {})
    observation = record.get("observation", {})
    if not isinstance(action, dict) or not isinstance(observation, dict):
        return None
    if str(action.get("type", "")) != "play_claim":
        return None
    cards = action.get("cards", [])
    if not isinstance(cards, list):
        return None
    claim_rank = str(action.get("claim_rank") or observation.get("table_type") or "")
    return str(record.get("player_id", "")), claim_rank, len(cards)


def _is_bluff_attempt(record: dict[str, object]) -> bool:
    if bool(record.get("fallback_used", False)):
        return False
    action = record.get("action", {})
    observation = record.get("observation", {})
    if not isinstance(action, dict) or not isinstance(observation, dict):
        return False
    if str(action.get("type", "")) != "play_claim":
        return False
    cards = action.get("cards", [])
    if not isinstance(cards, list) or not cards:
        return False
    table_type = str(observation.get("table_type", ""))
    truthful_count = _truthful_card_count(cards, table_type)
    return truthful_count < len(cards)


def _bluff_survived_immediate_response(records: list[dict[str, object]], start_index: int) -> bool:
    current = records[start_index]
    signature = _extract_pending_signature(current)
    if signature is None:
        return False

    current_events = current.get("step_result", {})
    if isinstance(current_events, dict):
        own_events = current_events.get("events", [])
        if isinstance(own_events, list) and any(
            isinstance(event, str) and "No player with cards can respond; starting a new round." in event
            for event in own_events
        ):
            return True

    actor_id, claim_rank, declared_count = signature
    for next_record in records[start_index + 1 :]:
        next_observation = next_record.get("observation", {})
        if not isinstance(next_observation, dict):
            continue
        pending_claim = next_observation.get("pending_claim")
        if not isinstance(pending_claim, dict):
            continue

        if (
            str(pending_claim.get("actor_id", "")) != actor_id
            or str(pending_claim.get("claim_rank", "")) != claim_rank
            or int(pending_claim.get("declared_count", 0) or 0) != declared_count
        ):
            continue

        next_action = next_record.get("action", {})
        if not isinstance(next_action, dict):
            return False

        if str(next_action.get("type", "")) == "challenge":
            outcome = _challenge_outcome(
                next_record.get("step_result", {}).get("events", [])
                if isinstance(next_record.get("step_result", {}), dict)
                else []
            )
            return outcome is not True

        return True

    return False


def _extract_behavior_metrics(games_dir: Path, llm_player_id: str) -> dict[str, object]:
    challenge_attempt_count = 0
    correct_challenge_count = 0
    bluff_attempt_count = 0
    successful_bluff_count = 0

    for records in _iter_game_records(games_dir):
        for index, record in enumerate(records):
            if str(record.get("player_id", "")) != llm_player_id:
                continue
            if bool(record.get("fallback_used", False)):
                continue

            action = record.get("action", {})
            if not isinstance(action, dict):
                continue

            action_type = str(action.get("type", ""))
            if action_type == "challenge":
                outcome = _challenge_outcome(
                    record.get("step_result", {}).get("events", [])
                    if isinstance(record.get("step_result", {}), dict)
                    else []
                )
                if outcome is None:
                    continue
                challenge_attempt_count += 1
                if outcome:
                    correct_challenge_count += 1
                continue

            if _is_bluff_attempt(record):
                bluff_attempt_count += 1
                if _bluff_survived_immediate_response(records, index):
                    successful_bluff_count += 1

    return {
        "challenge_attempt_count": challenge_attempt_count,
        "correct_challenge_count": correct_challenge_count,
        "challenge_accuracy": (
            correct_challenge_count / challenge_attempt_count if challenge_attempt_count else 0.0
        ),
        "bluff_attempt_count": bluff_attempt_count,
        "successful_bluff_count": successful_bluff_count,
        "bluff_efficiency": (
            successful_bluff_count / bluff_attempt_count if bluff_attempt_count else 0.0
        ),
    }


def build_experiment_scorecard(experiment_root: str | Path, label: str | None = None) -> dict[str, object]:
    root = Path(experiment_root)
    task_m_dir = root / "task_m"
    task_1_1_dir = root / "task_1_1"
    task_m_summary = _load_json(task_m_dir / "summary.json")
    task_1_1_summary = _load_json(task_1_1_dir / "summary.json")

    llm_player_id = str(task_m_summary.get("llm_player_id", "p1"))
    behavior = _extract_behavior_metrics(task_m_dir / "games", llm_player_id=llm_player_id)
    total_games = int(task_m_summary.get("total_games", 0) or 0)
    game_summaries = task_m_summary.get("game_summaries", [])
    win_count = 0
    if isinstance(game_summaries, list):
        win_count = sum(
            1
            for game_summary in game_summaries
            if isinstance(game_summary, dict) and str(game_summary.get("winner", "")) == llm_player_id
        )

    access = {
        "parse_error_rate": float(task_m_summary.get("parse_error_rate", 0.0) or 0.0),
        "parse_error_count": int(task_m_summary.get("parse_error_count", 0) or 0),
        "illegal_chosen_turn_count": int(task_1_1_summary.get("illegal_chosen_turn_count", 0) or 0),
    }
    access["access_ok"] = (
        float(access["parse_error_rate"]) == 0.0 and int(access["illegal_chosen_turn_count"]) == 0
    )

    quality = {
        "avg_ev_gap": float(task_1_1_summary.get("avg_ev_gap", 0.0) or 0.0),
        "conflict_count": int(task_1_1_summary.get("conflict_count", 0) or 0),
        "conflict_rate": (
            float(task_1_1_summary.get("conflict_count", 0) or 0)
            / float(task_1_1_summary.get("audited_turn_count", 0) or 1)
            if float(task_1_1_summary.get("audited_turn_count", 0) or 0) > 0
            else 0.0
        ),
        "resolution_adjustment_rate": float(task_m_summary.get("resolution_adjustment_rate", 0.0) or 0.0),
        "resolution_adjustment_count": int(task_m_summary.get("resolution_adjustment_count", 0) or 0),
    }

    stability = {
        "audited_turn_count": int(task_1_1_summary.get("audited_turn_count", 0) or 0),
        "negative_phi_turn_count": int(task_1_1_summary.get("negative_phi_turn_count", 0) or 0),
        "max_ev_gap": float(task_1_1_summary.get("max_ev_gap", 0.0) or 0.0),
        "high_ev_gap_turn_count": int(task_1_1_summary.get("high_ev_gap_turn_count", 0) or 0),
        "challenge_rate": float(task_1_1_summary.get("challenge_rate", 0.0) or 0.0),
        "play_claim_rate": float(task_1_1_summary.get("play_claim_rate", 0.0) or 0.0),
        "pass_rate": float(task_1_1_summary.get("pass_rate", 0.0) or 0.0),
    }

    auxiliary = {
        "total_games": total_games,
        "llm_turn_count": int(task_m_summary.get("llm_turn_count", 0) or 0),
        "win_count": win_count,
        "win_rate": (win_count / total_games) if total_games else 0.0,
    }

    return {
        "label": label or root.name,
        "experiment_root": str(root),
        "llm_player_id": llm_player_id,
        "sources": {
            "task_m_summary": str(task_m_dir / "summary.json"),
            "task_1_1_summary": str(task_1_1_dir / "summary.json"),
            "games_dir": str(task_m_dir / "games"),
        },
        "access": access,
        "quality": quality,
        "stability": stability,
        "behavior": behavior,
        "auxiliary": auxiliary,
    }


def render_scorecards_markdown(scorecards: list[dict[str, object]]) -> str:
    headers = [
        "label",
        "access_ok",
        "parse_error_rate",
        "illegal_chosen_turn_count",
        "avg_ev_gap",
        "conflict_count",
        "conflict_rate",
        "resolution_adjustment_rate",
        "max_ev_gap",
        "high_ev_gap_turn_count",
        "negative_phi_turn_count",
        "challenge_rate",
        "play_claim_rate",
        "pass_rate",
        "challenge_accuracy",
        "bluff_efficiency",
        "win_rate",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for card in scorecards:
        row = [
            str(card["label"]),
            str(card["access"]["access_ok"]),
            f"{float(card['access']['parse_error_rate']):.6f}",
            str(card["access"]["illegal_chosen_turn_count"]),
            f"{float(card['quality']['avg_ev_gap']):.6f}",
            str(card["quality"]["conflict_count"]),
            f"{float(card['quality']['conflict_rate']):.6f}",
            f"{float(card['quality']['resolution_adjustment_rate']):.6f}",
            f"{float(card['stability']['max_ev_gap']):.6f}",
            str(card["stability"]["high_ev_gap_turn_count"]),
            str(card["stability"]["negative_phi_turn_count"]),
            f"{float(card['stability']['challenge_rate']):.6f}",
            f"{float(card['stability']['play_claim_rate']):.6f}",
            f"{float(card['stability']['pass_rate']):.6f}",
            f"{float(card['behavior']['challenge_accuracy']):.6f}",
            f"{float(card['behavior']['bluff_efficiency']):.6f}",
            f"{float(card['auxiliary']['win_rate']):.6f}",
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
