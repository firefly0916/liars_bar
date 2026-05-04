from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from liars_game_engine.agents.parsers import parse_agent_output
from liars_game_engine.analysis.shapley_analyzer import (
    LogIterator,
    ProxyValuePredictor,
    ShapleyAnalyzer,
    TurnTrajectory,
    _action_to_payload,
)
from liars_game_engine.analysis.train_value_proxy import (
    VALUE_PROXY_TARGET_PHI,
    build_value_proxy_feature_context,
)
from liars_game_engine.config.loader import load_settings


DEFAULT_PROXY_MODEL_PATH = Path("models/proxy/value_proxy_mlp_distill.pt")
DEFAULT_OUTPUT_DIR = Path("logs/task_n_ev_gap_audit")
TASK_1_1_EV_GAP_THRESHOLD = 0.15
TASK_1_1_REPORT_FILENAME = "task_1.1_ev_gap_report.json"
EV_GAP_HEATMAP_FILENAME = "ev_gap_heatmap.svg"
EV_GAP_FIELDNAMES = (
    "game_id",
    "turn",
    "player_id",
    "action_type",
    "action_claim_rank",
    "action_cards",
    "chosen_action_is_legal",
    "phi_chosen",
    "best_action_type",
    "best_action_claim_rank",
    "best_action_cards",
    "phi_best",
    "ev_gap",
    "legal_action_count",
    "reasoning_confidence",
    "death_probability",
    "is_potential_point",
    "log_file",
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


def _resolve_model_path(model_path: Path | str | None = None) -> Path:
    if model_path is not None:
        resolved = Path(model_path)
        if not resolved.exists():
            raise RuntimeError(f"Proxy model path does not exist: {resolved}")
        return resolved

    if not DEFAULT_PROXY_MODEL_PATH.exists():
        raise RuntimeError(
            "Proxy model path not provided and default model is missing: "
            f"{DEFAULT_PROXY_MODEL_PATH}"
        )
    return DEFAULT_PROXY_MODEL_PATH


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


def _load_llm_record_index(
    log_paths: list[Path],
    llm_player_id: str,
) -> dict[tuple[str, int, str], dict[str, object]]:
    indexed: dict[tuple[str, int, str], dict[str, object]] = {}
    for log_path in log_paths:
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                continue
            if str(record.get("player_id", "")) != llm_player_id:
                continue
            key = (log_path.stem, int(record.get("turn", 0) or 0), llm_player_id)
            indexed[key] = record
    return indexed


def _score_action(
    predictor: ProxyValuePredictor | object,
    observation: dict[str, object],
    player_id: str,
    action_payload: dict[str, object],
) -> float:
    feature_context = build_value_proxy_feature_context(
        observation=observation,
        player_id=player_id,
        action=action_payload,
    )
    return float(predictor.predict_state_features(feature_context))


def _serialize_action(action_payload: dict[str, object]) -> tuple[str, str, str]:
    action_type = str(action_payload.get("type", "") or "")
    claim_rank = str(action_payload.get("claim_rank", "") or "")
    cards = [str(card) for card in action_payload.get("cards", [])] if isinstance(action_payload.get("cards", []), list) else []
    return action_type, claim_rank, "|".join(cards)


def _build_ev_gap_row(
    *,
    trajectory: TurnTrajectory,
    record: dict[str, object],
    chosen_score: float,
    best_action_payload: dict[str, object],
    best_score: float,
    legal_actions: list[object],
    legal_action_count: int,
    player_id: str,
    potential_point_threshold: float,
) -> dict[str, object]:
    chosen_type, chosen_rank, chosen_cards = _serialize_action(dict(record.get("action", {})))
    best_type, best_rank, best_cards = _serialize_action(best_action_payload)
    reasoning = _resolve_reasoning_text(record)
    confidence = classify_reasoning_confidence(reasoning)
    death_probability = (
        _extract_death_probability(trajectory.observation, player_id)
        if isinstance(trajectory.observation, dict)
        else 0.0
    )
    ev_gap = float(best_score - chosen_score)
    chosen_payload = dict(record.get("action", {}))
    chosen_action_is_legal = int(any(_action_to_payload(action) == chosen_payload for action in legal_actions))
    return {
        "game_id": trajectory.game_id,
        "turn": trajectory.turn,
        "player_id": player_id,
        "action_type": chosen_type,
        "action_claim_rank": chosen_rank,
        "action_cards": chosen_cards,
        "chosen_action_is_legal": chosen_action_is_legal,
        "phi_chosen": chosen_score,
        "best_action_type": best_type,
        "best_action_claim_rank": best_rank,
        "best_action_cards": best_cards,
        "phi_best": best_score,
        "ev_gap": ev_gap,
        "legal_action_count": legal_action_count,
        "reasoning_confidence": confidence["label"],
        "death_probability": death_probability,
        "is_potential_point": int(ev_gap > potential_point_threshold),
        "log_file": str(record.get("_log_file", "")),
    }


def _write_ev_gap_distribution(rows: list[dict[str, object]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EV_GAP_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _interpolate_heatmap_color(value: float) -> str:
    clamped = max(0.0, min(1.0, value))
    red = int(38 + (208 * clamped))
    green = int(83 + (104 * (1.0 - clamped)))
    blue = int(135 + (72 * (1.0 - clamped)))
    return f"#{red:02x}{green:02x}{blue:02x}"


def _write_ev_gap_heatmap(
    rows: list[dict[str, object]],
    output_path: Path,
    potential_point_threshold: float,
) -> None:
    ev_gaps = [float(row.get("ev_gap", 0.0) or 0.0) for row in rows]
    max_gap = max(ev_gaps) if ev_gaps else 0.0
    columns = min(20, max(1, len(rows)))
    cell_size = 20
    padding = 24
    title_height = 36
    legend_height = 48
    row_count = (len(rows) + columns - 1) // columns if rows else 1
    width = padding * 2 + columns * cell_size
    height = padding * 2 + title_height + row_count * cell_size + legend_height

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text { font-family: monospace; fill: #1f2933; }</style>',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#f8fafc" />',
        f'<text x="{padding}" y="{padding}" font-size="14">Task 1.1 EV Gap Heatmap</text>',
        (
            f'<text x="{padding}" y="{padding + 18}" font-size="11">'
            f'Potential point threshold: {potential_point_threshold:.2f} | Max EV gap: {max_gap:.4f}'
            "</text>"
        ),
    ]

    grid_top = padding + title_height
    if not rows:
        svg_lines.append(f'<text x="{padding}" y="{grid_top + 16}" font-size="12">No audited turns.</text>')
    else:
        for index, row in enumerate(rows):
            x = padding + (index % columns) * cell_size
            y = grid_top + (index // columns) * cell_size
            ev_gap = float(row.get("ev_gap", 0.0) or 0.0)
            normalized = (ev_gap / max_gap) if max_gap > 0.0 else 0.0
            fill = _interpolate_heatmap_color(normalized)
            stroke = "#111827" if ev_gap > potential_point_threshold else "#d1d5db"
            stroke_width = 2 if ev_gap > potential_point_threshold else 1
            tooltip = (
                f'{row.get("game_id", "")} turn {row.get("turn", "")} | '
                f'ev_gap={ev_gap:.4f} | action={row.get("action_type", "")}'
            )
            svg_lines.append(
                f'<rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}">'
                f"<title>{tooltip}</title></rect>"
            )

    legend_y = grid_top + row_count * cell_size + 24
    gradient_width = min(220, width - padding * 2)
    gradient_id = "evGapGradient"
    svg_lines.extend(
        [
            f'<defs><linearGradient id="{gradient_id}" x1="0%" y1="0%" x2="100%" y2="0%">'
            f'<stop offset="0%" stop-color="{_interpolate_heatmap_color(0.0)}" />'
            f'<stop offset="100%" stop-color="{_interpolate_heatmap_color(1.0)}" />'
            "</linearGradient></defs>",
            f'<rect x="{padding}" y="{legend_y}" width="{gradient_width}" height="12" fill="url(#{gradient_id})" stroke="#cbd5e1" />',
            f'<text x="{padding}" y="{legend_y + 28}" font-size="11">0.0</text>',
            f'<text x="{padding + gradient_width - 52}" y="{legend_y + 28}" font-size="11">max {max_gap:.4f}</text>',
        ]
    )
    svg_lines.append("</svg>")
    output_path.write_text("\n".join(svg_lines) + "\n", encoding="utf-8")


def run_llm_behavior_audit(
    log_root: Path | str,
    model_path: Path | str | None = None,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    phi_threshold: float = -0.1,
    llm_player_id: str | None = None,
    summary_path: Path | str | None = None,
    config_file: Path | str = "config/experiment.yaml",
    potential_point_threshold: float = TASK_1_1_EV_GAP_THRESHOLD,
) -> dict[str, object]:
    resolved_log_root = Path(log_root)
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    resolved_model_path = _resolve_model_path(model_path)

    resolved_llm_player_id = _resolve_llm_player_id(
        log_root=resolved_log_root,
        llm_player_id=llm_player_id,
        summary_path=summary_path,
    )
    log_paths = _resolve_drill_log_paths(resolved_log_root)
    if not log_paths:
        raise RuntimeError(f"No drill JSONL files found under {resolved_log_root}")

    predictor = ProxyValuePredictor(model_path=resolved_model_path, output_mode=VALUE_PROXY_TARGET_PHI)
    analyzer = ShapleyAnalyzer(
        settings=load_settings(config_file=config_file),
        rollout_samples=1,
        rollout_policy="random",
        max_workers=1,
    )

    record_index = _load_llm_record_index(log_paths=log_paths, llm_player_id=resolved_llm_player_id)
    audited_turns: list[dict[str, object]] = []
    conflict_cases: list[dict[str, object]] = []
    ev_gap_rows: list[dict[str, object]] = []
    action_counts = {"pass": 0, "challenge": 0, "play_claim": 0}

    for game in LogIterator(log_paths).iter_games():
        for trajectory in game.turns:
            if trajectory.player_id != resolved_llm_player_id:
                continue

            record_key = (trajectory.game_id, trajectory.turn, resolved_llm_player_id)
            record = record_index.get(record_key)
            if record is None:
                continue
            record = {**record, "_log_file": str(next((path for path in log_paths if path.stem == trajectory.game_id), ""))}

            observation = trajectory.observation if isinstance(trajectory.observation, dict) else {}
            chosen_payload = _action_to_payload(trajectory.action)
            phi_pred = _score_action(
                predictor=predictor,
                observation=observation,
                player_id=resolved_llm_player_id,
                action_payload=chosen_payload,
            )
            reasoning = _resolve_reasoning_text(record)
            reasoning_confidence = classify_reasoning_confidence(reasoning)

            audited = {
                "game_id": trajectory.game_id,
                "turn": trajectory.turn,
                "player_id": resolved_llm_player_id,
                "thought": reasoning,
                "raw_output": str(record.get("raw_output", "") or ""),
                "action": dict(record.get("action", chosen_payload)),
                "phi_pred": phi_pred,
                "death_probability": _extract_death_probability(observation, resolved_llm_player_id),
                "reasoning_confidence": reasoning_confidence,
                "log_file": str(record.get("_log_file", "")),
            }
            audited_turns.append(audited)

            action_type = str(chosen_payload.get("type", "") or "")
            if action_type in action_counts:
                action_counts[action_type] += 1

            if phi_pred < float(phi_threshold) and reasoning_confidence["label"] == "strong":
                conflict_cases.append(audited)

            legal_actions = analyzer._build_proxy_legal_actions(trajectory)
            if not legal_actions:
                legal_actions = [trajectory.action]

            legal_scores: list[tuple[dict[str, object], float]] = []
            for legal_action in legal_actions:
                legal_payload = _action_to_payload(legal_action)
                legal_scores.append(
                    (
                        legal_payload,
                        _score_action(
                            predictor=predictor,
                            observation=observation,
                            player_id=resolved_llm_player_id,
                            action_payload=legal_payload,
                        ),
                    )
                )

            best_action_payload, best_score = max(legal_scores, key=lambda item: item[1])
            ev_gap_rows.append(
                _build_ev_gap_row(
                    trajectory=trajectory,
                    record=record,
                    chosen_score=phi_pred,
                    best_action_payload=best_action_payload,
                    best_score=best_score,
                    legal_actions=legal_actions,
                    legal_action_count=len(legal_scores),
                    player_id=resolved_llm_player_id,
                    potential_point_threshold=float(potential_point_threshold),
                )
            )

    conflict_cases_path = resolved_output_dir / "conflict_cases.jsonl"
    conflict_cases_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in conflict_cases) + ("\n" if conflict_cases else ""),
        encoding="utf-8",
    )
    ev_gap_distribution_path = resolved_output_dir / "ev_gap_distribution.csv"
    _write_ev_gap_distribution(ev_gap_rows, ev_gap_distribution_path)
    ev_gap_heatmap_path = resolved_output_dir / EV_GAP_HEATMAP_FILENAME
    _write_ev_gap_heatmap(
        rows=ev_gap_rows,
        output_path=ev_gap_heatmap_path,
        potential_point_threshold=float(potential_point_threshold),
    )

    total_audited = len(audited_turns)
    ev_gaps = [float(item["ev_gap"]) for item in ev_gap_rows]
    summary_path_obj = resolved_output_dir / "summary.json"
    task_1_1_report_path = resolved_output_dir / TASK_1_1_REPORT_FILENAME
    summary = {
        "log_root": str(resolved_log_root),
        "model_path": str(resolved_model_path),
        "llm_player_id": resolved_llm_player_id,
        "phi_threshold": float(phi_threshold),
        "potential_point_ev_gap_threshold": float(potential_point_threshold),
        "audited_turn_count": total_audited,
        "negative_phi_turn_count": sum(1 for item in audited_turns if float(item["phi_pred"]) < float(phi_threshold)),
        "conflict_count": len(conflict_cases),
        "conflict_cases_path": str(conflict_cases_path),
        "ev_gap_distribution_path": str(ev_gap_distribution_path),
        "ev_gap_heatmap_path": str(ev_gap_heatmap_path),
        "avg_ev_gap": (sum(ev_gaps) / len(ev_gaps)) if ev_gaps else 0.0,
        "max_ev_gap": max(ev_gaps) if ev_gaps else 0.0,
        "high_ev_gap_turn_count": sum(1 for value in ev_gaps if value > float(potential_point_threshold)),
        "illegal_chosen_turn_count": sum(1 for row in ev_gap_rows if int(row["chosen_action_is_legal"]) == 0),
        "pass_rate": (action_counts["pass"] / total_audited) if total_audited else 0.0,
        "challenge_rate": (action_counts["challenge"] / total_audited) if total_audited else 0.0,
        "play_claim_rate": (action_counts["play_claim"] / total_audited) if total_audited else 0.0,
        "summary_path": str(summary_path_obj),
        "task_1_1_ev_gap_report_path": str(task_1_1_report_path),
    }
    summary_path_obj.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    task_1_1_report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
