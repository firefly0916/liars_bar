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
    return candidates


def _normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """作用: 归一化别名字段，提升解析兼容性。

    输入:
    - payload: 已 JSON 解码的字典。

    返回:
    - dict[str, Any]: 规范化后的字段结构。
    """
    normalized = dict(payload)

    if "action" not in normalized and "act" in normalized:
        normalized["action"] = normalized["act"]

    if "thought" not in normalized:
        if "reasoning" in normalized:
            normalized["thought"] = normalized["reasoning"]
        elif "analysis" in normalized:
            normalized["thought"] = normalized["analysis"]

    return normalized


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

    if cards is None:
        cards = []
    if not isinstance(cards, list):
        cards = [str(cards)]

    action = ActionModel(type=action_type, claim_rank=claim_rank, cards=[str(card) for card in cards])
    return ParseResult(ok=True, action=action)


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
