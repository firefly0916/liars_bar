from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable

from liars_game_engine.analysis.implicit_skill_mapper import map_implicit_skill
from liars_game_engine.analysis.token_alignment import align_sample_to_tokens
from liars_game_engine.analysis.train_value_proxy import build_value_proxy_feature_context


TOKEN_PATTERNS: dict[str, tuple[str, float]] = {
    "risk": (r"\b(risk|risky|danger|dangerous|hazard)\b", 1.5),
    "bluff": (r"\b(bluff|lie|lying|deceive|deception)\b", 1.75),
    "game": (r"\b(game|gambit|tempo|pressure|mind game)\b", 1.25),
    "skepticism": (r"\b(doubt|doubtful|suspect|suspicious|challenge)\b", 1.4),
}

DEFAULT_ALIGNMENT_SAMPLE_TYPE = "Standard"
REASONING_ACTION_MISMATCH = "Reasoning-Action Mismatch"
_CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def _ensure_english_only_text(value: object, *, field_name: str) -> str:
    text = str(value or "")
    if _CJK_PATTERN.search(text):
        raise ValueError(f"English-only validation failed for {field_name}: detected CJK content")
    return text


def extract_strategic_tokens(reasoning: str) -> list[dict[str, object]]:
    """Extract strategy-related tokens and their weights from reasoning text."""
    text = str(reasoning or "")
    matches: list[dict[str, object]] = []
    for label, (pattern, weight) in TOKEN_PATTERNS.items():
        for matched in re.finditer(pattern, text, flags=re.IGNORECASE):
            matches.append(
                {
                    "label": label,
                    "token": matched.group(0),
                    "start": matched.start(),
                    "end": matched.end(),
                    "weight": float(weight),
                }
            )
    matches.sort(key=lambda item: (int(item["start"]), str(item["label"])))
    return matches


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_cards_field(raw_cards: object) -> list[str]:
    if isinstance(raw_cards, list):
        return [str(card) for card in raw_cards]
    text = str(raw_cards or "").strip()
    if not text:
        return []
    return [card for card in text.split("|") if card]


def _normalize_action(action: dict[str, object] | None = None) -> dict[str, object]:
    payload = dict(action or {})
    payload["type"] = str(payload.get("type", "") or "")
    payload["claim_rank"] = str(payload.get("claim_rank", "") or "")
    payload["cards"] = _parse_cards_field(payload.get("cards", []))
    return payload


def _build_action_from_audit_row(row: dict[str, object], prefix: str) -> dict[str, object]:
    return _normalize_action(
        {
            "type": row.get(f"{prefix}_type", ""),
            "claim_rank": row.get(f"{prefix}_claim_rank", ""),
            "cards": _parse_cards_field(row.get(f"{prefix}_cards", "")),
        }
    )


def _resolve_state_features(record: dict[str, object]) -> dict[str, object]:
    state_features = record.get("state_features")
    if isinstance(state_features, dict) and state_features:
        return dict(state_features)

    observation = record.get("observation")
    action = record.get("action")
    if isinstance(observation, dict):
        return build_value_proxy_feature_context(
            state_features=None,
            observation=observation,
            player_id=str(record.get("player_id", observation.get("player_id", ""))),
            action=action if isinstance(action, dict) else None,
        )
    return {}


def build_hicra_sample(
    record: dict[str, object],
    implicit_skill_label: str | None = None,
) -> dict[str, object]:
    """Build one HICRA sample with implicit-skill and token-weight metadata."""
    state_features = _resolve_state_features(record)
    action = record.get("action", {})
    if not isinstance(action, dict):
        action = {}
    reasoning = _ensure_english_only_text(record.get("thought", ""), field_name="thought")
    resolved_skill = implicit_skill_label or map_implicit_skill(state_features=state_features, action=action)
    strategic_tokens = extract_strategic_tokens(reasoning)
    strategic_token_weight = max([1.0, *[float(item["weight"]) for item in strategic_tokens]])
    return {
        "game_id": str(record.get("game_id", "") or record.get("trace_id", "")),
        "turn": int(record.get("turn", 0) or 0),
        "player_id": str(record.get("player_id", "") or ""),
        "thought": reasoning,
        "action": dict(action),
        "state_features": state_features,
        "implicit_skill_label": resolved_skill,
        "strategic_tokens": strategic_tokens,
        "strategic_token_weight": strategic_token_weight,
    }


def build_hicra_dataset(records: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    return [build_hicra_sample(record) for record in records]


def export_hicra_dataset(records: Iterable[dict[str, object]], output_path: Path | str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = build_hicra_dataset(records)
    path.write_text(
        "\n".join(json.dumps(sample, ensure_ascii=False) for sample in samples) + ("\n" if samples else ""),
        encoding="utf-8",
    )
    return path


def build_alignment_sample(
    record: dict[str, object],
    audit_row: dict[str, object],
    potential_point_threshold: float = 0.15,
) -> dict[str, object]:
    """Build one SAVI alignment sample from raw logs plus EV-gap audit output."""
    base_sample = build_hicra_sample(record)
    chosen_action = _normalize_action(record.get("action") if isinstance(record.get("action"), dict) else {})
    proxy_target_action = _build_action_from_audit_row(audit_row, prefix="best_action")
    chosen_from_audit = _build_action_from_audit_row(audit_row, prefix="action")
    if chosen_action["type"] == "" and chosen_from_audit["type"]:
        chosen_action = chosen_from_audit

    state_features = dict(base_sample.get("state_features", {}))
    ev_gap = _safe_float(audit_row.get("ev_gap", 0.0), default=0.0)
    phi_chosen = _safe_float(audit_row.get("phi_chosen", 0.0), default=0.0)
    phi_best = _safe_float(audit_row.get("phi_best", 0.0), default=0.0)
    is_potential_point = ev_gap > float(potential_point_threshold)
    reasoning_action_mismatch = is_potential_point and chosen_action != proxy_target_action
    token_penalty = -ev_gap

    proxy_target_skill = map_implicit_skill(state_features=state_features, action=proxy_target_action)
    strategic_tokens: list[dict[str, object]] = []
    for item in base_sample.get("strategic_tokens", []):
        token = dict(item)
        token["penalty_signal"] = token_penalty
        token["proxy_guidance_action_type"] = proxy_target_action.get("type", "")
        token["proxy_guidance_claim_rank"] = proxy_target_action.get("claim_rank", "")
        token["proxy_guidance_weight"] = phi_best
        strategic_tokens.append(token)

    return {
        **base_sample,
        "game_id": str(record.get("game_id", "") or audit_row.get("game_id", "") or base_sample.get("game_id", "")),
        "turn": _safe_int(record.get("turn", audit_row.get("turn", 0)), default=0),
        "player_id": str(record.get("player_id", audit_row.get("player_id", "")) or ""),
        "action": chosen_action,
        "chosen_implicit_skill_label": base_sample.get("implicit_skill_label", ""),
        "proxy_target_action": proxy_target_action,
        "proxy_target_implicit_skill_label": proxy_target_skill,
        "phi_chosen": phi_chosen,
        "phi_best": phi_best,
        "ev_gap": ev_gap,
        "token_penalty": token_penalty,
        "sample_type": REASONING_ACTION_MISMATCH if reasoning_action_mismatch else DEFAULT_ALIGNMENT_SAMPLE_TYPE,
        "reasoning_action_mismatch": bool(reasoning_action_mismatch),
        "is_potential_point": bool(is_potential_point),
        "potential_point_threshold": float(potential_point_threshold),
        "best_action_delta": phi_best - phi_chosen,
        "strategic_tokens": strategic_tokens,
    }


def load_task_m_record_index(log_root: Path | str) -> dict[tuple[str, int, str], dict[str, object]]:
    root = Path(log_root)
    games_dir = root / "games" if (root / "games").is_dir() else root
    indexed: dict[tuple[str, int, str], dict[str, object]] = {}
    for log_path in sorted(games_dir.glob("*.jsonl")):
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                continue
            game_id = str(record.get("game_id", "") or log_path.stem)
            turn = _safe_int(record.get("turn", 0), default=0)
            player_id = str(record.get("player_id", "") or "")
            enriched = dict(record)
            enriched["game_id"] = game_id
            indexed[(game_id, turn, player_id)] = enriched
    return indexed


def build_savi_alignment_dataset(
    *,
    report_path: Path | str,
    ev_gap_csv_path: Path | str,
    log_root: Path | str,
) -> list[dict[str, object]]:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    potential_point_threshold = _safe_float(report.get("potential_point_ev_gap_threshold", 0.15), default=0.15)
    record_index = load_task_m_record_index(log_root)

    samples: list[dict[str, object]] = []
    with Path(ev_gap_csv_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            game_id = str(row.get("game_id", "") or "")
            turn = _safe_int(row.get("turn", 0), default=0)
            player_id = str(row.get("player_id", "") or "")
            record = record_index.get((game_id, turn, player_id))
            if record is None:
                continue
            samples.append(
                build_alignment_sample(
                    record=record,
                    audit_row=row,
                    potential_point_threshold=potential_point_threshold,
                )
            )
    return samples


def export_savi_alignment_dataset(samples: Iterable[dict[str, object]], output_path: Path | str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = [dict(sample) for sample in samples]
    path.write_text(
        "\n".join(json.dumps(sample, ensure_ascii=False) for sample in serialized) + ("\n" if serialized else ""),
        encoding="utf-8",
    )
    return path


def attach_token_alignment(
    sample: dict[str, object],
    tokenizer: object,
    messages: list[dict[str, str]] | None = None,
    assistant_text: str | None = None,
    weight_distribution_strategy: str = "equal",
) -> dict[str, object]:
    alignment_payload = align_sample_to_tokens(
        sample=sample,
        tokenizer=tokenizer,
        messages=messages,
        assistant_text=assistant_text,
        weight_distribution_strategy=weight_distribution_strategy,
    )
    return {
        **dict(sample),
        "alignment_metadata": alignment_payload["alignment_metadata"],
        "token_weight_mask": alignment_payload["token_weight_mask"],
    }
