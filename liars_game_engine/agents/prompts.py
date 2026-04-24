from __future__ import annotations

from pathlib import Path

import yaml

from liars_game_engine.analysis.train_value_proxy import build_value_proxy_feature_context
from liars_game_engine.engine.game_state import JOKER_RANK


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


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _format_percent(value: float) -> str:
    return f"{round(_clamp_unit(value) * 100):d}%"


def _estimate_roulette_layout(death_probability: float, roulette_slots: int = 6) -> tuple[int, int]:
    if death_probability <= 0.0:
        return 0, roulette_slots
    lethal_slots = max(1, min(roulette_slots, round(_clamp_unit(death_probability) * roulette_slots)))
    return lethal_slots, max(0, roulette_slots - lethal_slots)


def _truthful_card_count(cards: list[str], table_type: str) -> int:
    truthful_cards = {table_type, JOKER_RANK}
    return sum(1 for card in cards if str(card) in truthful_cards)


def _describe_truth_ratio(ratio: float) -> str:
    if ratio >= 0.75:
        return "你手里的真牌储备很充足，继续诚实出牌的空间较大。"
    if ratio >= 0.4:
        return "你手里真假参半，还能维持一定可信度。"
    return "你手里大部分是假牌，继续诚实出牌会越来越吃紧。"


def _describe_persona_stability(score: float) -> str:
    if score >= 0.75:
        return "你的人设目前相当稳定，对手更容易相信你。"
    if score >= 0.4:
        return "你的人设还算稳，但已经不能随意透支。"
    return "你的人设比较脆弱，一旦被抓到破绽就容易崩盘。"


def _build_qualitative_context(
    hand_truth_ratio: float,
    action_consistency_score: float,
    death_probability: float,
    pending_claim_text: str,
) -> str:
    risk_text = "非常危险" if death_probability >= 1.0 / 3.0 else ("有明显风险" if death_probability > 1.0 / 6.0 else "相对可控")
    truth_text = _describe_truth_ratio(hand_truth_ratio)
    persona_text = _describe_persona_stability(action_consistency_score)
    pending_text = (
        f"上家当前的声明是：{pending_claim_text}。"
        if pending_claim_text != "none"
        else "当前没有待处理的上家声明，你可以主动塑造新一轮叙事。"
    )
    return (
        f"局势分析：你现在的处境{risk_text}。{persona_text}"
        f"{truth_text}{pending_text}"
        "请综合判断自己是该继续维持叙事，还是立刻挑战对手。"
    )


def _describe_legal_actions(observation: dict[str, object], table_type: str, truthful_cards_in_hand: int) -> list[str]:
    legal_actions = observation.get("legal_actions", [])
    if not isinstance(legal_actions, list):
        return []

    descriptions: list[str] = []
    for item in legal_actions:
        if not isinstance(item, dict):
            continue
        action_type = str(item.get("type", ""))
        if action_type == "challenge":
            descriptions.append(
                "challenge: 质疑上家。这不会要求你立刻出牌，适合在你怀疑对方或自身死亡风险过高时使用。"
            )
        elif action_type == "play_claim":
            min_cards = int(item.get("min_cards", 1))
            max_cards = int(item.get("max_cards", 3))
            descriptions.append(
                "play_claim: 继续报牌。"
                f" 你可以声明 {item.get('claim_rank', table_type)}，一次打出 {min_cards}-{max_cards} 张。"
                f" 你当前手里与桌面匹配的真牌/JOKER 共有 {truthful_cards_in_hand} 张。"
            )
        elif action_type:
            descriptions.append(f"{action_type}: 按系统给出的合法动作执行。")
    return descriptions


def format_observation_for_llm(observation: dict[str, object]) -> str:
    """作用: 将 8D 状态特征翻译为“三段式”自然语言 briefing。"""
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
    death_probability = _clamp_unit(float(current_state.get("death_probability", 0.0) or 0.0))
    feature_context = build_value_proxy_feature_context(
        state_features=observation.get("state_features") if isinstance(observation.get("state_features"), dict) else None,
        observation=observation,
        player_id=player_id,
        action=None,
    )
    private_hand_text = [str(card) for card in private_hand]
    truthful_cards_in_hand = _truthful_card_count(private_hand_text, table_type)
    hand_truth_ratio = truthful_cards_in_hand / len(private_hand_text) if private_hand_text else 0.0
    pending_claim_declared_count = int(feature_context.get("pending_claim_declared_count", 0) or 0)
    action_consistency_score = _clamp_unit((truthful_cards_in_hand + pending_claim_declared_count) / 8.0)
    lethal_slots, safe_slots = _estimate_roulette_layout(death_probability)

    pending_claim_text = "none"
    if isinstance(pending_claim, dict) and pending_claim:
        pending_claim_text = (
            f"actor={pending_claim.get('actor_id', '')}, "
            f"rank={pending_claim.get('claim_rank', '')}, "
            f"declared_count={pending_claim.get('declared_count', 0)}"
        )
    legal_action_lines = _describe_legal_actions(observation, table_type=table_type, truthful_cards_in_hand=truthful_cards_in_hand)
    qualitative_context = _build_qualitative_context(
        hand_truth_ratio=hand_truth_ratio,
        action_consistency_score=action_consistency_score,
        death_probability=death_probability,
        pending_claim_text=pending_claim_text,
    )

    return (
        "Status Report\n"
        f"- 当前阶段: {phase}\n"
        f"- 桌面牌型: {table_type}\n"
        f"- 手牌: {private_hand_text}\n"
        f"- 诚实度参考: {_format_percent(hand_truth_ratio)} ({_describe_truth_ratio(hand_truth_ratio)})\n"
        f"- 人设稳定性: {_format_percent(action_consistency_score)} ({_describe_persona_stability(action_consistency_score)})\n"
        f"- 轮盘死亡概率: {_format_percent(death_probability)} "
        f"(按 6 槽轮盘估算，约有 {lethal_slots} 个致命位、{safe_slots} 个安全位)\n"
        f"- 待回应声明: {pending_claim_text}\n"
        f"- 是否被迫质疑: {bool(observation.get('must_call_liar', False))}\n"
        f"- 合法动作空间: {legal_action_lines}\n\n"
        "Qualitative Context\n"
        f"{qualitative_context}\n\n"
        "Decision Request\n"
        "基于以上客观数据与局势分析，在合法动作范围内选择你本回合的最佳动作。"
        "不要手动编造任何额外分值、评分或启发式期望，也不要虚构系统未给出的隐藏信息。"
        "请先给出你的 Reasoning，再给出最终 Action。"
        "只返回 JSON，格式如下：\n"
        "{\n"
        '  "Reasoning": "string",\n'
        '  "Action": {\n'
        '    "type": "play_claim|challenge|pass",\n'
        '    "claim_rank": "A|K|Q|null",\n'
        '    "cards": ["string"]\n'
        "  }\n"
        "}"
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
                f"ROLE REMINDER:\n{profile['instruction']}\n\n"
                f"STATE_TEMPLATE:\n{format_observation_for_llm(observation)}\n\n"
                "Respond in JSON only."
            ),
        },
    ]
