from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from liars_game_engine.engine.game_state import ActionModel, JOKER_RANK


@dataclass
class ResolvedAction:
    action: ActionModel
    resolution_reason: str | None = None


def _truthful_cards(private_hand: list[str], table_type: str) -> list[str]:
    return [card for card in private_hand if card in {table_type, JOKER_RANK}]


def _bluff_cards(private_hand: list[str], table_type: str) -> list[str]:
    return [card for card in private_hand if card not in {table_type, JOKER_RANK}]


def _find_play_claim_legal_window(observation: dict[str, object]) -> tuple[int, int]:
    legal_actions = observation.get("legal_actions", [])
    if not isinstance(legal_actions, list):
        return (1, 1)
    for item in legal_actions:
        if not isinstance(item, dict):
            continue
        if str(item.get("type", "")) != "play_claim":
            continue
        return (
            max(1, int(item.get("min_cards", 1) or 1)),
            max(1, int(item.get("max_cards", 1) or 1)),
        )
    return (1, 1)


def _has_legal_action_type(observation: dict[str, object], action_type: str) -> bool:
    legal_actions = observation.get("legal_actions", [])
    if not isinstance(legal_actions, list):
        return False
    return any(isinstance(item, dict) and str(item.get("type", "")) == action_type for item in legal_actions)


def _normalize_count(value: object, default: int) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return max(0, int(default))


def _serialize_reason(parts: list[str]) -> str | None:
    normalized = [part for part in parts if part]
    if not normalized:
        return None
    return "; ".join(normalized)


def _build_cards_from_counts(
    private_hand: list[str],
    table_type: str,
    play_count: int,
    true_card_count: int,
) -> list[str]:
    truthful_pool = _truthful_cards(private_hand, table_type)
    bluff_pool = _bluff_cards(private_hand, table_type)
    selected_truthful = truthful_pool[:true_card_count]
    selected_bluff = bluff_pool[: max(0, play_count - len(selected_truthful))]
    return selected_truthful + selected_bluff


def _redirect_illegal_pass(observation: dict[str, object]) -> ResolvedAction:
    private_hand_raw = observation.get("private_hand", [])
    private_hand = [str(card) for card in private_hand_raw] if isinstance(private_hand_raw, list) else []
    table_type = str(observation.get("table_type", "A"))

    if _has_legal_action_type(observation, "play_claim") and private_hand:
        min_cards, max_cards = _find_play_claim_legal_window(observation)
        play_count = max(1, min(min_cards, max_cards, len(private_hand)))
        truthful_pool = _truthful_cards(private_hand, table_type)
        truthful_count = min(len(truthful_pool), play_count)
        resolved_cards = _build_cards_from_counts(
            private_hand=private_hand,
            table_type=table_type,
            play_count=play_count,
            true_card_count=truthful_count,
        )
        return ResolvedAction(
            action=ActionModel(type="play_claim", claim_rank=table_type, cards=resolved_cards),
            resolution_reason=(
                f"illegal_pass_redirection: redirected_to=play_claim resolved_play={len(resolved_cards)} "
                f"resolved_true={truthful_count}"
            ),
        )

    if _has_legal_action_type(observation, "challenge"):
        return ResolvedAction(
            action=ActionModel(type="challenge"),
            resolution_reason="illegal_pass_redirection: redirected_to=challenge",
        )

    return ResolvedAction(
        action=ActionModel(type="pass"),
        resolution_reason="illegal_pass_redirection: no_legal_alternative_detected",
    )


def resolve_action_from_intent(
    observation: dict[str, object],
    action_type: str,
    play_count: object | None = None,
    true_card_count: object | None = None,
    cards: list[str] | None = None,
) -> ResolvedAction:
    """作用: 将 LLM 意图协议稳定降级为环境可执行动作。"""
    normalized_action_type = str(action_type or "").strip().lower()
    if normalized_action_type == "pass":
        if _has_legal_action_type(observation, "pass"):
            return ResolvedAction(action=ActionModel(type="pass"))
        return _redirect_illegal_pass(observation)
    if normalized_action_type == "challenge":
        return ResolvedAction(action=ActionModel(type=normalized_action_type))

    private_hand_raw = observation.get("private_hand", [])
    private_hand = [str(card) for card in private_hand_raw] if isinstance(private_hand_raw, list) else []
    table_type = str(observation.get("table_type", "A"))
    min_cards, max_cards = _find_play_claim_legal_window(observation)
    requested_play_count = _normalize_count(play_count, default=min_cards)
    requested_cards = [str(card) for card in cards] if isinstance(cards, list) else []
    requested_true_count = true_card_count
    reasons: list[str] = []

    if requested_cards:
        requested_play_count = len(requested_cards)
        truthful_count_from_cards = sum(1 for card in requested_cards if card in {table_type, JOKER_RANK})
        if requested_true_count is None:
            requested_true_count = truthful_count_from_cards
        request_counter = Counter(requested_cards)
        hand_counter = Counter(private_hand)
        requested_cards_supported = all(hand_counter.get(card, 0) >= count for card, count in request_counter.items())
        if requested_cards_supported and min_cards <= len(requested_cards) <= max_cards:
            resolution_reason = None
            if str(observation.get("table_type", "")) and table_type != "":
                resolution_reason = "claim_rank_forced_to_table_type"
            return ResolvedAction(
                action=ActionModel(type="play_claim", claim_rank=table_type, cards=requested_cards),
                resolution_reason=resolution_reason,
            )
        reasons.append("legacy_cards_invalid_rebuilt_from_intent")

    normalized_play_count = requested_play_count
    if requested_play_count < min_cards:
        normalized_play_count = min_cards
        reasons.append(f"play_count_clamped: requested_play={requested_play_count} resolved_play={normalized_play_count}")
    elif requested_play_count > max_cards:
        normalized_play_count = max_cards
        reasons.append(f"play_count_clamped: requested_play={requested_play_count} resolved_play={normalized_play_count}")

    normalized_play_count = min(normalized_play_count, len(private_hand))
    truthful_pool = _truthful_cards(private_hand, table_type)
    bluff_pool = _bluff_cards(private_hand, table_type)
    available_truthful = len(truthful_pool)

    normalized_true_count = _normalize_count(requested_true_count, default=0)
    if normalized_true_count > normalized_play_count:
        normalized_true_count = normalized_play_count
        reasons.append(
            f"intent_true_count_clamped_to_play_count: requested_true={requested_true_count} resolved_true={normalized_true_count}"
        )
    if normalized_true_count > available_truthful:
        reasons.append(
            f"intent_true_count_downgraded: requested_true={requested_true_count} resolved_true={available_truthful}"
        )
        normalized_true_count = available_truthful

    needed_bluff = max(0, normalized_play_count - normalized_true_count)
    if needed_bluff > len(bluff_pool):
        adjusted_play_count = normalized_true_count + len(bluff_pool)
        if adjusted_play_count < min_cards:
            adjusted_play_count = min(len(private_hand), max(1, min_cards))
            normalized_true_count = min(normalized_true_count, adjusted_play_count, available_truthful)
        reasons.append(
            f"play_count_downgraded_for_hand_support: requested_play={requested_play_count} resolved_play={adjusted_play_count}"
        )
        normalized_play_count = adjusted_play_count

    resolved_cards = _build_cards_from_counts(
        private_hand=private_hand,
        table_type=table_type,
        play_count=normalized_play_count,
        true_card_count=normalized_true_count,
    )

    if len(resolved_cards) < min_cards and private_hand:
        resolved_cards = private_hand[:min_cards]
        reasons.append("forced_minimum_legal_play")

    return ResolvedAction(
        action=ActionModel(type="play_claim", claim_rank=table_type, cards=resolved_cards),
        resolution_reason=_serialize_reason(reasons),
    )
