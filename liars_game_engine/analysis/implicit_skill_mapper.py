from __future__ import annotations

from typing import Any

from liars_game_engine.engine.game_state import JOKER_RANK


IMPLICIT_SKILL_LABELS = (
    "Truthful_Action",
    "Calculated_Bluff",
    "Aggressive_Deception",
    "Logical_Skepticism",
    "Strategic_Drain",
)


def _truthful_card_count(cards: list[str], table_type: str) -> int:
    truthful_cards = {table_type, JOKER_RANK}
    return sum(1 for card in cards if str(card) in truthful_cards)


def _normalize_action(action: dict[str, object] | None) -> dict[str, object]:
    payload = dict(action or {})
    cards = payload.get("cards", [])
    if not isinstance(cards, list):
        cards = []
    payload["cards"] = [str(card) for card in cards]
    payload["type"] = str(payload.get("type", "") or "")
    payload["claim_rank"] = str(payload.get("claim_rank", "") or "")
    return payload


def map_implicit_skill(
    state_features: dict[str, Any],
    action: dict[str, object] | None = None,
) -> str:
    """作用: 基于 8D 特征上下文与原子动作后验映射回五大策略 Skill。"""
    feature_context = dict(state_features or {})
    action_payload = _normalize_action(action)

    action_type = str(action_payload.get("type") or feature_context.get("action_type", "") or "")
    action_cards = action_payload.get("cards") or feature_context.get("action_cards", [])
    if not isinstance(action_cards, list):
        action_cards = []
    action_cards = [str(card) for card in action_cards]

    table_type = str(feature_context.get("table_type", action_payload.get("claim_rank", "A")) or "A")
    private_hand = feature_context.get("private_hand", [])
    if not isinstance(private_hand, list):
        private_hand = []
    private_hand = [str(card) for card in private_hand]

    pending_claim_declared_count = int(feature_context.get("pending_claim_declared_count", 0) or 0)
    death_probability = float(feature_context.get("death_probability", 0.0) or 0.0)

    if action_type == "challenge":
        return "Logical_Skepticism"
    if action_type == "pass":
        return "Strategic_Drain"

    truthful_cards_played = _truthful_card_count(action_cards, table_type)
    total_cards_played = len(action_cards)
    truthful_cards_in_hand = _truthful_card_count(private_hand, table_type)

    if total_cards_played <= 0:
        return "Strategic_Drain"

    if truthful_cards_played == total_cards_played:
        return "Truthful_Action"

    if truthful_cards_played == 0:
        if total_cards_played >= 2 or death_probability < 0.2:
            return "Aggressive_Deception"
        return "Calculated_Bluff"

    mixed_play = 0 < truthful_cards_played < total_cards_played
    if mixed_play:
        return "Strategic_Drain"

    if truthful_cards_in_hand == 0 and pending_claim_declared_count > 0:
        return "Logical_Skepticism"
    return "Calculated_Bluff"
