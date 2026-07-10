from __future__ import annotations

import json
from collections.abc import Callable

from liars_game_engine.engine.game_state import JOKER_RANK


def _parse_cards(raw_cards: object) -> list[str]:
    if isinstance(raw_cards, list):
        return [str(card) for card in raw_cards]
    if isinstance(raw_cards, str) and raw_cards.strip():
        return [card for card in raw_cards.split("|") if card]
    return []


def normalize_action(action: object) -> dict[str, object]:
    payload = action if isinstance(action, dict) else {}
    return {
        "type": str(payload.get("type", "") or ""),
        "claim_rank": str(payload.get("claim_rank", "") or ""),
        "cards": _parse_cards(payload.get("cards", [])),
    }


def action_key(action: object) -> tuple[str, str, tuple[str, ...]]:
    normalized = normalize_action(action)
    return (
        str(normalized.get("type", "")),
        str(normalized.get("claim_rank", "")),
        tuple(str(card) for card in normalized.get("cards", [])),
    )


def _record_copy(record: dict[str, object]) -> dict[str, object]:
    return json.loads(json.dumps(record, ensure_ascii=True))


def _legal_actions(record: dict[str, object]) -> list[dict[str, object]]:
    observation = record.get("observation", {})
    if not isinstance(observation, dict):
        return []
    legal_actions = observation.get("legal_actions", [])
    if not isinstance(legal_actions, list):
        return []
    return [action for action in legal_actions if isinstance(action, dict)]


def _private_hand(record: dict[str, object]) -> list[str]:
    observation = record.get("observation", {})
    if isinstance(observation, dict):
        private_hand = observation.get("private_hand", [])
        if isinstance(private_hand, list) and private_hand:
            return [str(card) for card in private_hand]
    state_features = record.get("state_features", {})
    if isinstance(state_features, dict):
        private_hand = state_features.get("private_hand", [])
        if isinstance(private_hand, list):
            return [str(card) for card in private_hand]
    return []


def _table_type(record: dict[str, object], legal_action: dict[str, object]) -> str:
    observation = record.get("observation", {})
    if isinstance(observation, dict) and observation.get("table_type"):
        return str(observation.get("table_type", ""))
    return str(legal_action.get("claim_rank", "") or "")


def _candidate_from_action(
    record: dict[str, object],
    *,
    role: str,
    action: dict[str, object],
) -> dict[str, object]:
    candidate = _record_copy(record)
    normalized_action = normalize_action(action)
    candidate["action"] = normalized_action
    candidate["candidate_role"] = str(role)
    state_features = candidate.get("state_features", {})
    if not isinstance(state_features, dict):
        state_features = {}
    state_features = dict(state_features)
    state_features["action_type"] = normalized_action["type"]
    state_features["action_claim_rank"] = normalized_action["claim_rank"]
    state_features["action_cards"] = list(normalized_action["cards"])
    candidate["state_features"] = state_features
    return candidate


def _add_candidate(
    candidates: list[dict[str, object]],
    record: dict[str, object],
    *,
    role: str,
    action: dict[str, object],
) -> None:
    normalized = normalize_action(action)
    if not normalized["type"]:
        return
    candidates.append(_candidate_from_action(record, role=role, action=normalized))


def build_expanded_candidate_pool(record: dict[str, object]) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    chosen_action = normalize_action(record.get("action", {}))
    proxy_target_action = normalize_action(record.get("proxy_target_action", {}))

    _add_candidate(candidates, record, role="logged_action", action=chosen_action)
    if proxy_target_action["type"] and action_key(proxy_target_action) != action_key(chosen_action):
        _add_candidate(candidates, record, role="proxy_target", action=proxy_target_action)

    for legal_action in _legal_actions(record):
        normalized_legal = normalize_action(legal_action)
        if normalized_legal["type"] == "challenge":
            _add_candidate(candidates, record, role="legal_challenge", action=normalized_legal)

    hand = _private_hand(record)
    for legal_action in _legal_actions(record):
        normalized_legal = normalize_action(legal_action)
        if normalized_legal["type"] != "play_claim":
            continue
        claim_rank = normalized_legal["claim_rank"] or _table_type(record, legal_action)
        table_type = _table_type(record, legal_action) or claim_rank
        truthful_cards = {table_type, JOKER_RANK}
        truthful_card = next((card for card in hand if card in truthful_cards), None)
        bluff_card = next((card for card in hand if card not in truthful_cards), None)
        if truthful_card is not None:
            _add_candidate(
                candidates,
                record,
                role="truthful_play",
                action={"type": "play_claim", "claim_rank": claim_rank, "cards": [truthful_card]},
            )
        if bluff_card is not None:
            _add_candidate(
                candidates,
                record,
                role="bluff_play",
                action={"type": "play_claim", "claim_rank": claim_rank, "cards": [bluff_card]},
            )
    return candidates


def _select_distinct_candidates(
    scored_pool: list[dict[str, object]],
    *,
    group_size: int,
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    selected_keys: set[tuple[str, str, tuple[str, ...]]] = set()

    def add(candidate: dict[str, object]) -> None:
        if len(selected) >= max(1, int(group_size)):
            return
        key = action_key(candidate.get("action", {}))
        if key in selected_keys:
            return
        selected_keys.add(key)
        selected.append(candidate)

    for role in ("logged_action", "proxy_target"):
        for candidate in scored_pool:
            if str(candidate.get("candidate_role", "")) == role:
                add(candidate)
                break

    ranked_desc = sorted(scored_pool, key=lambda item: float(item.get("proxy_score", 0.0)), reverse=True)
    ranked_asc = list(reversed(ranked_desc))
    for candidate in ranked_desc[:1]:
        add(candidate)
    for candidate in ranked_asc[:1]:
        add(candidate)
    for candidate in ranked_desc:
        add(candidate)
    return selected


def _select_conservative_candidates(
    scored_pool: list[dict[str, object]],
    *,
    group_size: int,
    challenge_margin: float = 0.05,
    challenge_top_k: int = 2,
) -> list[dict[str, object]]:
    ranked_desc = sorted(scored_pool, key=lambda item: float(item.get("proxy_score", 0.0)), reverse=True)
    logged_score = next(
        (
            float(candidate.get("proxy_score", 0.0))
            for candidate in scored_pool
            if str(candidate.get("candidate_role", "")) == "logged_action"
        ),
        0.0,
    )
    top_challenge_keys = {
        action_key(candidate.get("action", {}))
        for candidate in ranked_desc[: max(0, int(challenge_top_k))]
        if str(candidate.get("candidate_role", "")) == "legal_challenge"
    }

    def challenge_allowed(candidate: dict[str, object]) -> bool:
        if str(candidate.get("candidate_role", "")) != "legal_challenge":
            return True
        score = float(candidate.get("proxy_score", 0.0))
        return action_key(candidate.get("action", {})) in top_challenge_keys or score >= logged_score + float(challenge_margin)

    filtered_pool = [candidate for candidate in scored_pool if challenge_allowed(candidate)]
    return _select_distinct_candidates(filtered_pool, group_size=group_size)


def build_expanded_scored_candidates(
    *,
    score_candidate: Callable[[dict[str, object]], float],
    record: dict[str, object],
    group_size: int,
    selection_mode: str = "legacy",
) -> dict[str, object]:
    pool = build_expanded_candidate_pool(record)
    scored_pool: list[dict[str, object]] = []
    for index, candidate in enumerate(pool):
        scored = dict(candidate)
        scored["candidate_index"] = index
        scored["proxy_score"] = float(score_candidate(scored))
        scored_pool.append(scored)
    if str(selection_mode) == "conservative":
        selected = _select_conservative_candidates(scored_pool, group_size=group_size)
    else:
        selected = _select_distinct_candidates(scored_pool, group_size=group_size)
    return {
        "expanded_candidate_pool": scored_pool,
        "selected_candidates": selected,
        "selection_mode": str(selection_mode),
    }
