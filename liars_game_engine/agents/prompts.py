from __future__ import annotations

from pathlib import Path
import re

import yaml

from liars_game_engine.engine.game_state import JOKER_RANK


_CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


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


def _legal_action_types(observation: dict[str, object]) -> list[str]:
    legal_actions = observation.get("legal_actions", [])
    if not isinstance(legal_actions, list):
        return ["play_claim", "challenge"]

    ordered_types: list[str] = []
    for candidate in ("play_claim", "challenge", "pass"):
        if any(isinstance(item, dict) and str(item.get("type", "")) == candidate for item in legal_actions):
            ordered_types.append(candidate)
    return ordered_types or ["play_claim", "challenge"]


def _truthful_card_count(cards: list[str], table_type: str) -> int:
    truthful_cards = {table_type, JOKER_RANK}
    return sum(1 for card in cards if str(card) in truthful_cards)


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


def _ensure_english_only_text(value: object, *, field_name: str) -> str:
    text = str(value or "")
    if _CJK_PATTERN.search(text):
        raise ValueError(f"English-only validation failed for {field_name}: detected CJK content")
    return text


def format_observation_for_llm(observation: dict[str, object]) -> str:
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
    death_probability = float(current_state.get("death_probability", 0.0) or 0.0)
    hand = [str(card) for card in private_hand]
    truthful_cards_in_hand = _truthful_card_count(hand, table_type)
    hand_truth_ratio = truthful_cards_in_hand / len(hand) if hand else 0.0

    pending_claim_text = "none"
    if isinstance(pending_claim, dict) and pending_claim:
        pending_claim_text = (
            f"actor={pending_claim.get('actor_id', '')}, "
            f"rank={pending_claim.get('claim_rank', '')}, "
            f"declared_count={pending_claim.get('declared_count', 0)}"
        )

    legal_lines: list[str] = []
    for item in legal_actions:
        if not isinstance(item, dict):
            continue
        action_type = str(item.get("type", ""))
        if action_type == "challenge":
            legal_lines.append("challenge")
        elif action_type == "play_claim":
            min_cards = int(item.get("min_cards", 1))
            max_cards = int(item.get("max_cards", 3))
            legal_lines.append(
                f"play_claim(rank={item.get('claim_rank', table_type)},play_count={min_cards}-{max_cards},true_card_count<= {truthful_cards_in_hand})"
            )
        elif action_type:
            legal_lines.append(action_type)

    action_types = _legal_action_types(observation)
    risk_text = "high" if death_probability >= 1.0 / 3.0 else ("mid" if death_probability > 1.0 / 6.0 else "low")
    honesty_text = "truth-heavy" if hand_truth_ratio >= 0.75 else ("mixed" if hand_truth_ratio >= 0.4 else "truth-light")

    return (
        "Status Report\n"
        f"- phase: {phase}\n"
        f"- table: {table_type}\n"
        f"- hand: {hand}\n"
        f"- honesty_reference: {round(max(0.0, min(1.0, hand_truth_ratio)) * 100):d}%\n"
        f"- roulette_death_probability: {round(max(0.0, min(1.0, death_probability)) * 100):d}%\n"
        f"- pending: {pending_claim_text}\n"
        f"- must_challenge: {bool(observation.get('must_call_liar', False))}\n"
        f"- legal: {' | '.join(legal_lines)}\n\n"
        "Qualitative Context\n"
        f"Risk={risk_text}; honesty={honesty_text}; pending={pending_claim_text if pending_claim_text != 'none' else 'no pending claim'}.\n\n"
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
        f'    "type": "{"|".join(action_types)}",\n'
        '    "play_count": "integer|null",\n'
        '    "true_card_count": "integer|null"\n'
        "  }\n"
        "}"
    )


def build_openai_messages(profile: dict[str, object], observation: dict[str, object]) -> list[dict[str, str]]:
    allowed_action_types = _legal_action_types(observation)
    messages = [
        {"role": "system", "content": str(profile["system"])},
        {
            "role": "user",
            "content": (
                f"{_sanitize_instruction(str(profile['instruction']), allowed_action_types)}\n\n"
                f"{format_observation_for_llm(observation)}"
            ),
        },
    ]
    for index, message in enumerate(messages):
        _ensure_english_only_text(message.get("content", ""), field_name=f"messages[{index}].content")
    return messages
