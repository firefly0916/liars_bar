from __future__ import annotations

from pathlib import Path

import yaml


DEFAULT_PROFILE = {
    "system": "You are a strategic Liar's Bar player.",
    "instruction": (
        "Return JSON with keys thought and action. "
        "action.type must be one of play_claim, challenge, pass."
    ),
    "output_schema": {
        "thought": "string",
        "action": {
            "type": "play_claim|challenge|pass",
            "claim_rank": "string|null",
            "cards": ["string"],
        },
    },
}


def load_prompt_profile(profile_name: str, profiles_dir: Path | str = "prompts/profiles") -> dict[str, object]:
    """作用: 加载指定 profile，不存在时回落到默认模板。

    输入:
    - profile_name: profile 文件名（不含扩展名）。
    - profiles_dir: profile 目录路径。

    返回:
    - dict[str, object]: 合并默认值后的 profile 配置。
    """
    base_dir = Path(profiles_dir)
    profile_path = base_dir / f"{profile_name}.yaml"
    if not profile_path.exists():
        return DEFAULT_PROFILE

    with profile_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}

    merged = dict(DEFAULT_PROFILE)
    merged.update(loaded)
    return merged


def build_prompt(profile: dict[str, object], observation: dict[str, object]) -> str:
    """作用: 将 profile 与 observation 拼装为提示词文本。

    输入:
    - profile: 包含 system/instruction 等字段的模板字典。
    - observation: 当前局面观测。

    返回:
    - str: 发送给模型的完整 prompt。
    """
    return (
        f"SYSTEM:\n{profile['system']}\n\n"
        f"INSTRUCTION:\n{profile['instruction']}\n\n"
        f"OBSERVATION:\n{observation}\n\n"
        "Respond in JSON only."
    )


def format_observation_for_llm(observation: dict[str, object]) -> str:
    """作用: 将核心博弈状态整理成更稳定的文本模板。"""
    player_id = str(observation.get("player_id", ""))
    phase = str(observation.get("phase", ""))
    table_type = str(observation.get("table_type", ""))
    private_hand = observation.get("private_hand", [])
    player_states = observation.get("player_states", {})
    pending_claim = observation.get("pending_claim")
    legal_actions = observation.get("legal_actions", [])

    if not isinstance(private_hand, list):
        private_hand = []
    if not isinstance(player_states, dict):
        player_states = {}
    if not isinstance(legal_actions, list):
        legal_actions = []

    current_state = player_states.get(player_id, {}) if isinstance(player_states.get(player_id, {}), dict) else {}
    death_probability = current_state.get("death_probability", 0.0)

    pending_claim_text = "none"
    if isinstance(pending_claim, dict) and pending_claim:
        pending_claim_text = (
            f"actor={pending_claim.get('actor_id', '')}, "
            f"rank={pending_claim.get('claim_rank', '')}, "
            f"declared_count={pending_claim.get('declared_count', 0)}"
        )

    legal_action_lines: list[str] = []
    for item in legal_actions:
        if not isinstance(item, dict):
            continue
        action_type = str(item.get("type", ""))
        if action_type == "play_claim":
            legal_action_lines.append(
                "play_claim"
                f"(claim_rank={item.get('claim_rank', table_type)}, "
                f"min_cards={item.get('min_cards', 1)}, max_cards={item.get('max_cards', 3)})"
            )
        elif action_type:
            legal_action_lines.append(action_type)

    return (
        f"PLAYER_ID: {player_id}\n"
        f"PHASE: {phase}\n"
        f"TABLE_TYPE: {table_type}\n"
        f"PRIVATE_HAND: {private_hand}\n"
        f"DEATH_PROBABILITY: {death_probability}\n"
        f"PENDING_CLAIM: {pending_claim_text}\n"
        f"MUST_CALL_LIAR: {bool(observation.get('must_call_liar', False))}\n"
        f"LEGAL_ACTIONS: {legal_action_lines}\n"
        f"RAW_OBSERVATION: {observation}"
    )


def build_openai_messages(profile: dict[str, object], observation: dict[str, object]) -> list[dict[str, str]]:
    """作用: 构造 OpenAI-compatible chat.completions 请求消息。

    输入:
    - profile: 包含 system/instruction 等字段的模板字典。
    - observation: 当前局面观测。

    返回:
    - list[dict[str, str]]: system/user 双消息结构。
    """
    return [
        {
            "role": "system",
            "content": str(profile["system"]),
        },
        {
            "role": "user",
            "content": (
                f"INSTRUCTION:\n{profile['instruction']}\n\n"
                f"STATE_TEMPLATE:\n{format_observation_for_llm(observation)}\n\n"
                "Respond in JSON only."
            ),
        },
    ]
