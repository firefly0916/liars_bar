from __future__ import annotations

from pathlib import Path
import re

import yaml

from liars_game_engine.analysis.train_value_proxy import build_value_proxy_feature_context
from liars_game_engine.engine.game_state import JOKER_RANK


DEFAULT_PROFILE = {
    "system": "You are a strategic Liar's Bar player.",
    "instruction": (
        "Return JSON only with keys Reasoning and Action. "
        "Action.type must be one legal action type from the current Status Report. "
        "For play_claim, output play_count and true_card_count only. "
        "Do not output claim_rank or specific card names."
    ),
    "output_schema": {
        "Reasoning": "string",
        "Action": {
            "type": "legal_action_type",
            "play_count": "integer|null",
            "true_card_count": "integer|null",
        },
    },
}


def load_prompt_profile(profile_name: str, profiles_dir: Path | str = "prompts/profiles") -> dict[str, object]:
    """Load a named prompt profile and fall back to defaults when absent."""
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
    """Render a single-string prompt from a profile and observation payload."""
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
        return "truth-heavy"
    if ratio >= 0.4:
        return "mixed"
    return "truth-light"


def _describe_persona_stability(score: float) -> str:
    if score >= 0.75:
        return "stable"
    if score >= 0.4:
        return "adaptive"
    return "fragile"


def _build_qualitative_context(
    hand_truth_ratio: float,
    action_consistency_score: float,
    death_probability: float,
    pending_claim_text: str,
) -> str:
    risk_text = "high" if death_probability >= 1.0 / 3.0 else ("mid" if death_probability > 1.0 / 6.0 else "low")
    claim_text = pending_claim_text if pending_claim_text != "none" else "no pending claim"
    return (
        f"Risk={risk_text}; persona={_describe_persona_stability(action_consistency_score)}; "
        f"honesty={_describe_truth_ratio(hand_truth_ratio)}; pending={claim_text}."
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
            descriptions.append("challenge")
        elif action_type == "play_claim":
            min_cards = int(item.get("min_cards", 1))
            max_cards = int(item.get("max_cards", 3))
            descriptions.append(
                f"play_claim(rank={item.get('claim_rank', table_type)},play_count={min_cards}-{max_cards},true_card_count<= {truthful_cards_in_hand})"
            )
        elif action_type:
            descriptions.append(action_type)
    return descriptions


def _legal_action_types(observation: dict[str, object]) -> list[str]:
    legal_actions = observation.get("legal_actions", [])
    if not isinstance(legal_actions, list):
        return ["play_claim", "challenge"]

    ordered_types: list[str] = []
    for candidate in ("play_claim", "challenge", "pass"):
        if any(isinstance(item, dict) and str(item.get("type", "")) == candidate for item in legal_actions):
            ordered_types.append(candidate)
    return ordered_types or ["play_claim", "challenge"]


def _sanitize_instruction(instruction: str, allowed_action_types: list[str]) -> str:
    action_csv = ", ".join(allowed_action_types)
    action_union = "|".join(allowed_action_types)
    sanitized = instruction
    sanitized = sanitized.replace("play_claim, challenge, pass", action_csv)
    sanitized = sanitized.replace("play_claim|challenge|pass", action_union)
    sanitized = re.sub(
        r"Action\.type must be one of [^.]+\.",
        f"Action.type must be one of {action_csv}.",
        sanitized,
    )
    sanitized = re.sub(
        r"Action\.type must be one legal action type from the current Status Report\.",
        f"Action.type must be one of {action_csv}.",
        sanitized,
    )
    return sanitized


def format_observation_for_llm(observation: dict[str, object]) -> str:
    """Translate structured game state into the three-section LLM briefing."""
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
    legal_action_types = _legal_action_types(observation)
    qualitative_context = _build_qualitative_context(
        hand_truth_ratio=hand_truth_ratio,
        action_consistency_score=action_consistency_score,
        death_probability=death_probability,
        pending_claim_text=pending_claim_text,
    )

    return (
        "Status Report\n"
        f"- phase: {phase}\n"
        f"- table: {table_type}\n"
        f"- hand: {private_hand_text}\n"
        f"- honesty_reference: {_format_percent(hand_truth_ratio)}\n"
        f"- persona_stability: {_format_percent(action_consistency_score)}\n"
        f"- roulette_death_probability: {_format_percent(death_probability)} ({lethal_slots} lethal / {safe_slots} safe)\n"
        f"- pending: {pending_claim_text}\n"
        f"- must_challenge: {bool(observation.get('must_call_liar', False))}\n"
        f"- legal: {' | '.join(legal_action_lines)}\n\n"
        "Qualitative Context\n"
        f"{qualitative_context}\n\n"
        "Protocol Rules\n"
        f"- If you choose play_claim, the system will automatically set claim_rank to table={table_type}.\n"
        "- Output play_count as the number of face-down cards to play.\n"
        "- Output true_card_count as how many of those cards are truthful table cards or Jokers.\n"
        "- Never output card names, and never output claim_rank yourself.\n"
        "- Example: hand=['K','Q'], table=K, hidden bluff of two cards with one truthful card => "
        '{"type":"play_claim","play_count":2,"true_card_count":1}\n\n'
        "Decision Request\n"
        "Choose one legal action. No hidden info. Return JSON only:\n"
        "{\n"
        '  "Reasoning": "string",\n'
        '  "Action": {\n'
        f'    "type": "{"|".join(legal_action_types)}",\n'
        '    "play_count": "integer|null",\n'
        '    "true_card_count": "integer|null"\n'
        "  }\n"
        "}"
    )


def build_openai_messages(profile: dict[str, object], observation: dict[str, object]) -> list[dict[str, str]]:
    """Build OpenAI-compatible system/user chat messages for one decision."""
    allowed_action_types = _legal_action_types(observation)
    return [
        {
            "role": "system",
            "content": str(profile["system"]),
        },
        {
            "role": "user",
            "content": (
                f"{_sanitize_instruction(str(profile['instruction']), allowed_action_types)}\n\n"
                f"{format_observation_for_llm(observation)}"
            ),
        },
    ]
