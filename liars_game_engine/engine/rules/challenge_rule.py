from __future__ import annotations

from liars_game_engine.engine.game_state import ActionModel, GamePhase, JOKER_RANK, RuntimeGameState
from liars_game_engine.engine.rules.base import RuleModule, RuleResult


class ChallengeRule(RuleModule):
    name = "challenge"

    def validate(self, state: RuntimeGameState, player_id: str, action: ActionModel) -> RuleResult:
        """作用: 校验 challenge 在当前 phase 与上下文中是否可执行。

        输入:
        - state: 当前游戏状态。
        - player_id: 发起挑战的玩家 ID。
        - action: 待校验动作。

        返回:
        - RuleResult: challenge 校验结果。
        """
        if state.phase != GamePhase.RESPONSE_WINDOW:
            return RuleResult(
                ok=False,
                events=[],
                error_code="E_ENV_PHASE_MISMATCH",
                error_reason="challenge is only allowed in RESPONSE_WINDOW",
            )

        if state.pending_claim is None:
            return RuleResult(
                ok=False,
                events=[],
                error_code="E_ACTION_RULE_VIOLATION",
                error_reason="no pending claim to challenge",
            )

        return RuleResult(ok=True, events=[])

    @staticmethod
    def has_liar_cards(state: RuntimeGameState) -> bool:
        """作用: 判断待揭示牌中是否存在 Liar 牌。

        输入:
        - state: 当前状态，需包含 pending_claim 与 table_type。

        返回:
        - bool: True 表示至少有一张牌不属于 Innocent 集合。
        """
        claim = state.pending_claim
        if claim is None:
            return False

        innocent_cards = {state.table_type, JOKER_RANK}
        return any(card not in innocent_cards for card in claim.cards)

    @staticmethod
    def evaluate_truth(state: RuntimeGameState) -> bool:
        """作用: 判断被挑战声明是否真实（无 Liar）。

        输入:
        - state: 当前状态。

        返回:
        - bool: True 表示声明真实，False 表示存在 Liar。
        """
        return not ChallengeRule.has_liar_cards(state)
