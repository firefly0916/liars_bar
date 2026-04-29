from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any

from liars_game_engine.engine.game_state import ActionModel, ParseError, ParseResult, VALID_ACTION_TYPES


JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
VALID_SKILLS = {
    "Truthful_Action",
    "Calculated_Bluff",
    "Aggressive_Deception",
    "Logical_Skepticism",
    "Strategic_Drain",
    "Null_Probe_Skill",
}


@dataclass
class PlannerParseResult:
    ok: bool
    thought: str = ""
    selected_skill: str = ""
    skill_parameters: dict[str, object] = field(default_factory=dict)
    error: ParseError | None = None


@dataclass
class ActionIntent:
    type: str
    play_count: int | None = None
    true_card_count: int | None = None
    claim_rank: str | None = None
    cards: list[str] = field(default_factory=list)


def _extract_candidates(raw_output: str) -> list[str]:
    """作用: 从原始文本中提取可能的 JSON 候选片段。

    输入:
    - raw_output: 模型原始输出文本。

    返回:
    - list[str]: 待尝试解析的候选字符串列表。
    """
    candidates = [raw_output.strip()]
    for matched in JSON_BLOCK_PATTERN.findall(raw_output):
        candidate = matched.strip()
        if candidate:
            candidates.append(candidate)
    candidates.extend(_extract_braced_json_candidates(raw_output))
    return candidates


def _extract_braced_json_candidates(raw_output: str) -> list[str]:
    """作用: 从普通文本中提取花括号包裹的 JSON 片段。"""
    candidates: list[str] = []
    depth = 0
    start_index: int | None = None

    for index, char in enumerate(raw_output):
        if char == "{":
            if depth == 0:
                start_index = index
            depth += 1
        elif char == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start_index is not None:
                candidate = raw_output[start_index : index + 1].strip()
                if candidate:
                    candidates.append(candidate)
                start_index = None

    return candidates


def _normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """作用: 归一化别名字段，提升解析兼容性。

    输入:
    - payload: 已 JSON 解码的字典。

    返回:
    - dict[str, Any]: 规范化后的字段结构。
    """
    normalized = _lowercase_dict_keys(payload)

    if "action" not in normalized and "act" in normalized:
        normalized["action"] = normalized["act"]

    if "thought" not in normalized:
        if "reasoning" in normalized:
            normalized["thought"] = normalized["reasoning"]
        elif "analysis" in normalized:
            normalized["thought"] = normalized["analysis"]

    if isinstance(normalized.get("action"), dict):
        action_payload = dict(normalized["action"])
        if "claim_rank" not in action_payload and "claimrank" in action_payload:
            action_payload["claim_rank"] = action_payload["claimrank"]
        normalized["action"] = action_payload

    return normalized


def _lowercase_dict_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key).lower(): _lowercase_dict_keys(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_lowercase_dict_keys(item) for item in value]
    return value


def _validate_action(action_payload: Any, raw_output: str) -> ParseResult:
    """作用: 校验 action 字段并转换为 ActionModel。

    输入:
    - action_payload: 待校验的 action 子结构。
    - raw_output: 原始输出文本，用于报错回溯。

    返回:
    - ParseResult: 成功时携带 action，失败时携带结构化错误。
    """
    if not isinstance(action_payload, dict):
        return ParseResult(
            ok=False,
            error=ParseError(
                code="E_ACTION_SCHEMA_MISSING",
                message="action field must be a JSON object",
                raw_output=raw_output,
            ),
        )

    action_type = str(action_payload.get("type", "")).strip().lower()
    if action_type not in VALID_ACTION_TYPES:
        return ParseResult(
            ok=False,
            error=ParseError(
                code="E_ACTION_SCHEMA_MISSING",
                message="action.type is missing or unsupported",
                raw_output=raw_output,
            ),
        )

    claim_rank = action_payload.get("claim_rank")
    cards = action_payload.get("cards")
    play_count = action_payload.get("play_count")
    true_card_count = action_payload.get("true_card_count")

    if action_type in {"challenge", "pass"}:
        return ParseResult(
            ok=True,
            action=ActionModel(type=action_type, claim_rank=claim_rank, cards=[]),
            action_intent=ActionIntent(type=action_type),
        )

    if isinstance(cards, list):
        normalized_cards = [str(card) for card in cards]
        inferred_true_count = action_payload.get("true_card_count")
        try:
            normalized_true_count = int(inferred_true_count) if inferred_true_count is not None else None
        except (TypeError, ValueError):
            normalized_true_count = None
        return ParseResult(
            ok=True,
            action=ActionModel(type=action_type, claim_rank=claim_rank, cards=normalized_cards),
            action_intent=ActionIntent(
                type=action_type,
                play_count=len(normalized_cards),
                true_card_count=normalized_true_count,
                claim_rank=str(claim_rank) if claim_rank is not None else None,
                cards=normalized_cards,
            ),
        )

    if play_count is not None:
        try:
            normalized_play_count = int(play_count)
        except (TypeError, ValueError):
            normalized_play_count = -1
        try:
            normalized_true_count = int(true_card_count) if true_card_count is not None else 0
        except (TypeError, ValueError):
            normalized_true_count = 0
        if normalized_play_count < 1:
            return ParseResult(
                ok=False,
                error=ParseError(
                    code="E_ACTION_SCHEMA_MISSING",
                    message="play_count must be a positive integer for play_claim",
                    raw_output=raw_output,
                ),
            )
        return ParseResult(
            ok=True,
            action=None,
            action_intent=ActionIntent(
                type=action_type,
                play_count=normalized_play_count,
                true_card_count=max(0, normalized_true_count),
                claim_rank=str(claim_rank) if claim_rank is not None else None,
                cards=[],
            ),
        )

    return ParseResult(
        ok=False,
        error=ParseError(
            code="E_ACTION_SCHEMA_MISSING",
            message="play_claim requires either cards list or play_count intent fields",
            raw_output=raw_output,
        ),
    )


def parse_agent_output(raw_output: str) -> ParseResult:
    """作用: 将模型输出文本解析为统一动作结构。

    输入:
    - raw_output: 模型输出原文。

    返回:
    - ParseResult: 解析成功或失败的统一结果对象。
    """
    for candidate in _extract_candidates(raw_output):
        try:
            decoded = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        if not isinstance(decoded, dict):
            continue

        normalized = _normalize_payload(decoded)
        parsed = _validate_action(normalized.get("action"), raw_output)
        if parsed.ok:
            parsed.thought = str(normalized.get("thought", ""))
            return parsed

        if parsed.error and parsed.error.code == "E_ACTION_SCHEMA_MISSING":
            return parsed

    return ParseResult(
        ok=False,
        error=ParseError(
            code="E_AGENT_FORMAT_INVALID",
            message="Unable to parse a valid JSON action payload",
            raw_output=raw_output,
        ),
    )


def parse_planner_output(raw_output: str) -> PlannerParseResult:
    """作用: 将模型输出解析为 Skill 选择 JSON。

    输入:
    - raw_output: 模型原始输出。

    返回:
    - PlannerParseResult: 包含 thought/selected_skill/skill_parameters 或错误。
    """
    for candidate in _extract_candidates(raw_output):
        try:
            decoded = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        if not isinstance(decoded, dict):
            continue

        normalized = _normalize_payload(decoded)
        thought = str(normalized.get("thought", "")).strip()
        selected_skill = str(normalized.get("selected_skill", "")).strip()

        skill_parameters = normalized.get("skill_parameters")
        if skill_parameters is None and "parameter" in normalized:
            skill_parameters = normalized.get("parameter")
        if skill_parameters is None:
            skill_parameters = {}

        if not thought:
            return PlannerParseResult(
                ok=False,
                error=ParseError(
                    code="E_ACTION_SCHEMA_MISSING",
                    message="thought is required",
                    raw_output=raw_output,
                ),
            )

        if selected_skill not in VALID_SKILLS:
            return PlannerParseResult(
                ok=False,
                error=ParseError(
                    code="E_ACTION_SCHEMA_MISSING",
                    message="selected_skill is missing or unsupported",
                    raw_output=raw_output,
                ),
            )

        if not isinstance(skill_parameters, dict):
            return PlannerParseResult(
                ok=False,
                error=ParseError(
                    code="E_ACTION_SCHEMA_MISSING",
                    message="skill_parameters must be a JSON object",
                    raw_output=raw_output,
                ),
            )

        return PlannerParseResult(
            ok=True,
            thought=thought,
            selected_skill=selected_skill,
            skill_parameters=skill_parameters,
        )

    return PlannerParseResult(
        ok=False,
        error=ParseError(
            code="E_AGENT_FORMAT_INVALID",
            message="Unable to parse a valid planner JSON payload",
            raw_output=raw_output,
        ),
    )
