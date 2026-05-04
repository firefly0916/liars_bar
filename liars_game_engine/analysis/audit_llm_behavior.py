from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from liars_game_engine.agents.parsers import parse_agent_output
from liars_game_engine.analysis.shapley_analyzer import ProxyValuePredictor
from liars_game_engine.analysis.train_value_proxy import (
    VALUE_PROXY_TARGET_PHI,
    build_value_proxy_feature_context,
)


STRONG_SIGNALS = (
    "definitely",
    "certainly",
    "clearly",
    "must",
    "immediately",
    "optimal",
)
HEDGE_SIGNALS = (
    "maybe",
    "perhaps",
    "might",
    "uncertain",
    "not certain",
    "conservative",
)


def classify_reasoning_confidence(reasoning: str) -> dict[str, object]:
    normalized = str(reasoning or "").strip()
    lowered = normalized.lower()
    strong_hits = [signal for signal in STRONG_SIGNALS if signal in normalized or signal in lowered]
    hedge_hits = [signal for signal in HEDGE_SIGNALS if signal in normalized or signal in lowered]
    score = len(strong_hits) - len(hedge_hits)

    if strong_hits and not hedge_hits:
        label = "strong"
    elif hedge_hits and not strong_hits:
        label = "hedged"
    elif strong_hits and hedge_hits:
        label = "mixed"
    else:
        label = "neutral"

    return {
        "label": label,
        "score": score,
        "strong_count": len(strong_hits),
        "hedge_count": len(hedge_hits),
        "strong_signals": strong_hits,
        "hedge_signals": hedge_hits,
    }


def _resolve_summary_path(log_root: Path, summary_path: Path | str | None = None) -> Path | None:
    if summary_path is not None:
        path = Path(summary_path)
        return path if path.exists() else None

    candidates = [log_root / "summary.json"]
    if log_root.name == "games":
        candidates.append(log_root.parent / "summary.json")
    else:
        candidates.append(log_root / "games" / "summary.json")
        candidates.append(log_root.parent / "summary.json")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_llm_player_id(
    log_root: Path,
    llm_player_id: str | None = None,
    summary_path: Path | str | None = None,
) -> str:
    if llm_player_id:
        return llm_player_id

    resolved_summary = _resolve_summary_path(log_root=log_root, summary_path=summary_path)
    if resolved_summary is None:
        raise RuntimeError("Unable to infer llm_player_id: summary.json not found and --llm-player-id not provided")

    payload = json.loads(resolved_summary.read_text(encoding="utf-8"))
    inferred = str(payload.get("llm_player_id", "")).strip()
    if not inferred:
        raise RuntimeError(f"summary.json missing llm_player_id: {resolved_summary}")
    return inferred


def _resolve_drill_log_paths(log_root: Path | str) -> list[Path]:
    root = Path(log_root)
    if root.is_file():
        return [root]

    games_dir = root / "games"
    if games_dir.is_dir():
        return sorted(games_dir.glob("*.jsonl"))
    return sorted(root.glob("*.jsonl"))


def _resolve_reasoning_text(record: dict[str, object]) -> str:
    thought = str(record.get("thought", "") or "").strip()
    if thought:
        return thought

    raw_output = str(record.get("raw_output", "") or "").strip()
    if not raw_output:
        return ""

    parsed = parse_agent_output(raw_output)
    if parsed.ok:
        return parsed.thought
    return ""


def _extract_death_probability(observation: dict[str, object], player_id: str) -> float:
    player_states = observation.get("player_states", {})
    if not isinstance(player_states, dict):
        return 0.0
    player_state = player_states.get(player_id, {})
    if not isinstance(player_state, dict):
        return 0.0
    try:
        return float(player_state.get("death_probability", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def run_llm_behavior_audit(
    log_root: Path | str,
    model_path: Path | str,
    output_dir: Path | str = "logs/task_n_llm_behavior_audit",
    phi_threshold: float = -0.1,
    llm_player_id: str | None = None,
    summary_path: Path | str | None = None,
) -> dict[str, object]:
    resolved_log_root = Path(log_root)
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    resolved_llm_player_id = _resolve_llm_player_id(
        log_root=resolved_log_root,
        llm_player_id=llm_player_id,
        summary_path=summary_path,
    )
    log_paths = _resolve_drill_log_paths(resolved_log_root)
    if not log_paths:
        raise RuntimeError(f"No drill JSONL files found under {resolved_log_root}")

    predictor = ProxyValuePredictor(model_path=model_path, output_mode=VALUE_PROXY_TARGET_PHI)

    audited_turns: list[dict[str, object]] = []
    conflict_cases: list[dict[str, object]] = []

    for log_path in log_paths:
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                continue
            if str(record.get("player_id", "")) != resolved_llm_player_id:
                continue

            observation = record.get("observation")
            action = record.get("action")
            if not isinstance(observation, dict) or not isinstance(action, dict):
                continue

            reasoning = _resolve_reasoning_text(record)
            feature_context = build_value_proxy_feature_context(
                observation=observation,
                player_id=resolved_llm_player_id,
                action=action,
            )
            phi_pred = float(predictor.predict_state_features(feature_context))
            reasoning_confidence = classify_reasoning_confidence(reasoning)

            audited = {
                "game_id": log_path.stem,
                "turn": int(record.get("turn", 0) or 0),
                "player_id": resolved_llm_player_id,
                "thought": reasoning,
                "raw_output": str(record.get("raw_output", "") or ""),
                "action": dict(action),
                "phi_pred": phi_pred,
                "death_probability": _extract_death_probability(observation, resolved_llm_player_id),
                "reasoning_confidence": reasoning_confidence,
                "log_file": str(log_path),
            }
            audited_turns.append(audited)

            if phi_pred < float(phi_threshold) and reasoning_confidence["label"] == "strong":
                conflict_cases.append(audited)

    conflict_cases_path = resolved_output_dir / "conflict_cases.jsonl"
    conflict_cases_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in conflict_cases) + ("\n" if conflict_cases else ""),
        encoding="utf-8",
    )

    summary = {
        "log_root": str(resolved_log_root),
        "model_path": str(model_path),
        "llm_player_id": resolved_llm_player_id,
        "phi_threshold": float(phi_threshold),
        "audited_turn_count": len(audited_turns),
        "negative_phi_turn_count": sum(1 for item in audited_turns if float(item["phi_pred"]) < float(phi_threshold)),
        "conflict_count": len(conflict_cases),
        "conflict_cases_path": str(conflict_cases_path),
        "summary_path": str(resolved_output_dir / "summary.json"),
    }
    Path(summary["summary_path"]).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
