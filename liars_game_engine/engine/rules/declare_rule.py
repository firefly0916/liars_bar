from __future__ import annotations

from collections import Counter

from liars_game_engine.engine.game_state import ActionModel, GamePhase, RuntimeGameState
from liars_game_engine.engine.rules.base import RuleModule, RuleResult


class DeclareRule(RuleModule):
    name = "declare"

    def validate(self, state: RuntimeGameState, player_id: str, action: ActionModel) -> RuleResult:
        """作用: 校验 play_claim 动作是否合法。

        输入:
        - state: 当前运行态。
        - player_id: 发起动作的玩家 ID。
        - action: 出牌声明动作。

        返回:
        - RuleResult: 包含通过/失败及错误信息。
        """
        if state.phase not in {GamePhase.TURN_START, GamePhase.DECLARE, GamePhase.RESPONSE_WINDOW}:
            return RuleResult(
                ok=False,
                events=[],
                error_code="E_ENV_PHASE_MISMATCH",
                error_reason="play_claim is not allowed in current phase",
            )

        if not action.cards:
            return RuleResult(
                ok=False,
                events=[],
                error_code="E_ACTION_RULE_VIOLATION",
                error_reason="cards is required for play_claim",
            )

        if len(action.cards) < 1 or len(action.cards) > 3:
            return RuleResult(
                ok=False,
                events=[],
                error_code="E_ACTION_RULE_VIOLATION",
                error_reason="play_claim must contain 1 to 3 cards",
            )

        if action.claim_rank is not None and action.claim_rank != state.table_type:
            return RuleResult(
                ok=False,
                events=[],
                error_code="E_ACTION_RULE_VIOLATION",
                error_reason="claim_rank must match current table type",
            )

        player = state.players[player_id]
        in_hand = Counter(player.hand)
        requested = Counter(action.cards)
        missing_cards = [card for card, count in requested.items() if in_hand.get(card, 0) < count]
        if missing_cards:
            return RuleResult(
                ok=False,
                events=[],
                error_code="E_ACTION_RULE_VIOLATION",
                error_reason=f"player does not have cards: {missing_cards}",
            )

        return RuleResult(ok=True, events=[])
