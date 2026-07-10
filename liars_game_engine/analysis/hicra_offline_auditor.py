from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from liars_game_engine.analysis.hicra_preprocessor import extract_strategic_tokens


DEFAULT_EV_GAP_THRESHOLD = 0.15
AUDIT_LABELS = (
    "clean_aligned",
    "proxy_disagreement",
    "reasoning_action_conflict",
    "strategic_overchallenge",
    "strategic_overplay",
    "protocol_failure",
)

_CHALLENGE_INTENT_PATTERN = re.compile(
    r"\b(challenge|call liar|call|suspect|suspicious|doubt|bluff|lying|lie)\b",
    flags=re.IGNORECASE,
)
_PLAY_SAFE_INTENT_PATTERN = re.compile(
    r"\b(play safe|safe card|play safely|avoid risk|avoid roulette|true card|truthful|conservative)\b",
    flags=re.IGNORECASE,
)


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


def _normalize_cards(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(card) for card in value]
    if isinstance(value, str) and value.strip():
        return [card for card in value.split("|") if card]
    return []


def normalize_action(action: object) -> dict[str, object]:
    payload = action if isinstance(action, dict) else {}
    return {
        "type": str(payload.get("type", "") or ""),
        "claim_rank": str(payload.get("claim_rank", "") or ""),
        "cards": _normalize_cards(payload.get("cards", [])),
    }


def _actions_equal(left: object, right: object) -> bool:
    return normalize_action(left) == normalize_action(right)


def _has_protocol_failure(record: dict[str, object]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    parse_error = record.get("parse_error")
    if isinstance(parse_error, dict) and str(parse_error.get("code", "") or "").strip():
        reasons.append("parse_error")
    elif isinstance(parse_error, str) and parse_error.strip():
        reasons.append("parse_error")

    if bool(record.get("fallback_used", False)):
        reasons.append("fallback_used")

    resolution_reason = str(record.get("resolution_reason", "") or "")
    if resolution_reason:
        repair_tokens = (
            "illegal",
            "repair",
            "redirected_to=",
            "claim_rank_forced_to_table_type",
        )
        if any(token in resolution_reason for token in repair_tokens):
            reasons.append("resolution_repair")

    step_result = record.get("step_result")
    if isinstance(step_result, dict):
        success = step_result.get("success")
        error_code = str(step_result.get("error_code", "") or "")
        if success is False or error_code:
            reasons.append("step_error")

    return bool(reasons), sorted(set(reasons))


def infer_semantic_reasoning_action_mismatch(
    *,
    thought: str,
    chosen_action: dict[str, object],
) -> bool:
    action_type = str(chosen_action.get("type", "") or "")
    text = str(thought or "")
    if not text.strip():
        return False
    challenge_intent = bool(_CHALLENGE_INTENT_PATTERN.search(text))
    play_safe_intent = bool(_PLAY_SAFE_INTENT_PATTERN.search(text))
    if action_type == "challenge" and play_safe_intent:
        return True
    if action_type == "play_claim" and challenge_intent and not play_safe_intent:
        return True
    return False


def _resolve_label(
    *,
    protocol_failure: bool,
    semantic_mismatch: bool,
    high_ev_gap: bool,
    action_proxy_disagreement: bool,
    chosen_action: dict[str, object],
    proxy_target_action: dict[str, object],
) -> str:
    chosen_type = str(chosen_action.get("type", "") or "")
    proxy_type = str(proxy_target_action.get("type", "") or "")
    if protocol_failure:
        return "protocol_failure"
    if semantic_mismatch and (high_ev_gap or action_proxy_disagreement):
        return "reasoning_action_conflict"
    if action_proxy_disagreement and high_ev_gap and chosen_type == "challenge":
        return "strategic_overchallenge"
    if action_proxy_disagreement and high_ev_gap and chosen_type == "play_claim" and proxy_type == "challenge":
        return "strategic_overplay"
    if action_proxy_disagreement:
        return "proxy_disagreement"
    return "clean_aligned"


def _resolve_severity(label: str, *, high_ev_gap: bool, protocol_failure: bool) -> str:
    if protocol_failure or label in {"reasoning_action_conflict", "strategic_overchallenge", "strategic_overplay"}:
        return "high" if high_ev_gap or protocol_failure else "medium"
    if label == "proxy_disagreement":
        return "low"
    return "none"


def _build_explanation(
    *,
    label: str,
    action_proxy_disagreement: bool,
    high_ev_gap: bool,
    semantic_mismatch: bool,
    chosen_action: dict[str, object],
    proxy_target_action: dict[str, object],
) -> str:
    if label == "protocol_failure":
        return "Protocol failure is present, so format or execution instability takes precedence over strategy interpretation."
    if label == "reasoning_action_conflict":
        return "The reasoning text appears inconsistent with the selected action under a high-value or proxy-disagreement decision point."
    if label == "strategic_overchallenge":
        return "The selected action is challenge while the proxy prefers a different legal action with higher estimated CRC."
    if label == "strategic_overplay":
        return "The selected action is play_claim while the proxy prefers challenge with higher estimated CRC."
    if label == "proxy_disagreement":
        return "The selected action differs from the proxy target, but the EV gap is below the high-severity threshold."
    if action_proxy_disagreement or high_ev_gap or semantic_mismatch:
        return "The record has audit signals, but they do not cross the configured label threshold."
    return "The selected action, proxy target, and reasoning audit signals are aligned under the configured thresholds."


def audit_decision_record(
    record: dict[str, object],
    *,
    ev_gap_threshold: float = DEFAULT_EV_GAP_THRESHOLD,
) -> dict[str, object]:
    chosen_action = normalize_action(record.get("action", {}))
    proxy_target_action = normalize_action(record.get("proxy_target_action", {}))
    ev_gap = _safe_float(record.get("ev_gap", 0.0))
    phi_chosen = _safe_float(record.get("phi_chosen", 0.0))
    phi_best = _safe_float(record.get("phi_best", 0.0))
    action_proxy_disagreement = not _actions_equal(chosen_action, proxy_target_action)
    high_ev_gap = ev_gap > float(ev_gap_threshold)
    legacy_mismatch = bool(record.get("reasoning_action_mismatch", False))
    thought = str(record.get("thought", "") or "")
    semantic_mismatch = infer_semantic_reasoning_action_mismatch(
        thought=thought,
        chosen_action=chosen_action,
    )
    protocol_failure, protocol_reasons = _has_protocol_failure(record)

    strategic_tokens = record.get("strategic_tokens", [])
    if not isinstance(strategic_tokens, list):
        strategic_tokens = extract_strategic_tokens(thought)
    elif not strategic_tokens and thought:
        strategic_tokens = extract_strategic_tokens(thought)

    label = _resolve_label(
        protocol_failure=protocol_failure,
        semantic_mismatch=semantic_mismatch,
        high_ev_gap=high_ev_gap,
        action_proxy_disagreement=action_proxy_disagreement,
        chosen_action=chosen_action,
        proxy_target_action=proxy_target_action,
    )
    severity = _resolve_severity(label, high_ev_gap=high_ev_gap, protocol_failure=protocol_failure)

    return {
        "game_id": str(record.get("game_id", "") or record.get("trace_id", "") or ""),
        "turn": _safe_int(record.get("turn", 0)),
        "player_id": str(record.get("player_id", "") or ""),
        "thought": thought,
        "chosen_action": chosen_action,
        "proxy_target_action": proxy_target_action,
        "phi_chosen": float(phi_chosen),
        "phi_best": float(phi_best),
        "ev_gap": float(ev_gap),
        "ev_gap_threshold": float(ev_gap_threshold),
        "action_proxy_disagreement": bool(action_proxy_disagreement),
        "high_ev_gap_decision_error": bool(high_ev_gap),
        "legacy_reasoning_action_mismatch": bool(legacy_mismatch),
        "semantic_reasoning_action_mismatch": bool(semantic_mismatch),
        "protocol_failure": bool(protocol_failure),
        "protocol_failure_reasons": protocol_reasons,
        "strategic_tokens": strategic_tokens,
        "audit_label": label,
        "severity": severity,
        "explanation": _build_explanation(
            label=label,
            action_proxy_disagreement=action_proxy_disagreement,
            high_ev_gap=high_ev_gap,
            semantic_mismatch=semantic_mismatch,
            chosen_action=chosen_action,
            proxy_target_action=proxy_target_action,
        ),
    }


def summarize_audit_records(records: Iterable[dict[str, object]]) -> dict[str, object]:
    materialized = [dict(record) for record in records]
    label_counter = Counter(str(record.get("audit_label", "")) for record in materialized)
    severity_counter = Counter(str(record.get("severity", "")) for record in materialized)
    label_counts = {label: int(label_counter.get(label, 0)) for label in AUDIT_LABELS}
    severity_counts = {severity: int(severity_counter.get(severity, 0)) for severity in ("none", "low", "medium", "high")}
    total = len(materialized)
    high_ev_gap_count = sum(1 for record in materialized if bool(record.get("high_ev_gap_decision_error", False)))
    semantic_count = sum(1 for record in materialized if bool(record.get("semantic_reasoning_action_mismatch", False)))
    protocol_count = sum(1 for record in materialized if bool(record.get("protocol_failure", False)))
    disagreement_count = sum(1 for record in materialized if bool(record.get("action_proxy_disagreement", False)))
    average_ev_gap = (
        sum(_safe_float(record.get("ev_gap", 0.0)) for record in materialized) / total
        if total
        else 0.0
    )
    return {
        "total_records": total,
        "label_counts": label_counts,
        "severity_counts": severity_counts,
        "action_proxy_disagreement_count": int(disagreement_count),
        "high_ev_gap_decision_error_count": int(high_ev_gap_count),
        "semantic_reasoning_action_mismatch_count": int(semantic_count),
        "protocol_failure_count": int(protocol_count),
        "average_ev_gap": float(average_ev_gap),
    }


def _case_sort_key(record: dict[str, object]) -> tuple[int, float, str, int]:
    severity_order = {"high": 0, "medium": 1, "low": 2, "none": 3}
    return (
        severity_order.get(str(record.get("severity", "")), 4),
        -_safe_float(record.get("ev_gap", 0.0)),
        str(record.get("game_id", "")),
        _safe_int(record.get("turn", 0)),
    )


def render_case_studies_markdown(records: Iterable[dict[str, object]], *, limit: int = 8) -> str:
    selected = sorted([dict(record) for record in records], key=_case_sort_key)[: max(1, int(limit))]
    lines = ["# HICRA Offline Audit Case Studies", ""]
    if not selected:
        lines.append("No audit records available.")
        return "\n".join(lines)

    for index, record in enumerate(selected, start=1):
        lines.append(f"## Case {index}: {record.get('audit_label', '')}")
        lines.append("")
        lines.append(f"- game_id: `{record.get('game_id', '')}`")
        lines.append(f"- turn: `{record.get('turn', 0)}`")
        lines.append(f"- player_id: `{record.get('player_id', '')}`")
        lines.append(f"- severity: `{record.get('severity', '')}`")
        lines.append(f"- ev_gap: `{_safe_float(record.get('ev_gap', 0.0)):.6f}`")
        lines.append(f"- chosen_action: `{json.dumps(record.get('chosen_action', {}), ensure_ascii=True)}`")
        lines.append(f"- proxy_target_action: `{json.dumps(record.get('proxy_target_action', {}), ensure_ascii=True)}`")
        lines.append(f"- explanation: {record.get('explanation', '')}")
        thought = str(record.get("thought", "") or "")
        if thought:
            lines.append("")
            lines.append("```text")
            lines.append(thought[:1200])
            lines.append("```")
        lines.append("")
    return "\n".join(lines)


def render_scorecard_markdown(summary: dict[str, object]) -> str:
    lines = ["# HICRA Offline Audit Scorecard", ""]
    lines.append("## Summary")
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    for key in [
        "total_records",
        "action_proxy_disagreement_count",
        "high_ev_gap_decision_error_count",
        "semantic_reasoning_action_mismatch_count",
        "protocol_failure_count",
        "average_ev_gap",
    ]:
        value = summary.get(key, 0)
        rendered = f"{value:.6f}" if isinstance(value, float) else str(value)
        lines.append(f"| {key} | {rendered} |")
    lines.append("")
    lines.append("## Labels")
    lines.append("| label | count |")
    lines.append("| --- | --- |")
    label_counts = summary.get("label_counts", {})
    if isinstance(label_counts, dict):
        for label in AUDIT_LABELS:
            lines.append(f"| {label} | {int(label_counts.get(label, 0) or 0)} |")
    return "\n".join(lines)


def write_audit_outputs(
    records: Iterable[dict[str, object]],
    output_dir: Path | str,
) -> dict[str, str]:
    materialized = [dict(record) for record in records]
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    records_path = root / "hicra_audit_records.jsonl"
    summary_path = root / "hicra_audit_summary.json"
    case_studies_path = root / "hicra_case_studies.md"
    scorecard_path = root / "hicra_audit_scorecard.md"

    records_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=True) for record in materialized)
        + ("\n" if materialized else ""),
        encoding="utf-8",
    )
    summary = summarize_audit_records(materialized)
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    case_studies_path.write_text(render_case_studies_markdown(materialized), encoding="utf-8")
    scorecard_path.write_text(render_scorecard_markdown(summary), encoding="utf-8")
    return {
        "records_path": str(records_path),
        "summary_path": str(summary_path),
        "case_studies_path": str(case_studies_path),
        "scorecard_path": str(scorecard_path),
    }
