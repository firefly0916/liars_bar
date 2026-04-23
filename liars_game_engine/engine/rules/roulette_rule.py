from __future__ import annotations

import random

from liars_game_engine.engine.game_state import PlayerRuntimeState, REVOLVER_BLANK, REVOLVER_LETHAL, RouletteOutcome


class RouletteRule:
    name = "roulette"

    def __init__(self, roulette_slots: int, rng: random.Random) -> None:
        """作用: 初始化轮盘规则组件。

        输入:
        - roulette_slots: 左轮牌堆总槽位数（1 张致命牌 + 若干空包）。
        - rng: 随机数生成器。

        返回:
        - 无。
        """
        self.roulette_slots = max(1, roulette_slots)
        self.rng = rng

    def build_revolver_deck(self) -> list[str]:
        """作用: 构建并洗牌单名玩家的左轮牌堆。

        输入:
        - 无。

        返回:
        - list[str]: 由 LETHAL/BLANK 组成的随机顺序牌堆。
        """
        blanks = max(0, self.roulette_slots - 1)
        deck = [REVOLVER_LETHAL] + [REVOLVER_BLANK] * blanks
        self.rng.shuffle(deck)
        return deck

    def apply_penalty(self, player: PlayerRuntimeState) -> RouletteOutcome:
        """作用: 对玩家执行一次俄罗斯轮盘惩罚。

        输入:
        - player: 受罚玩家运行态。

        返回:
        - RouletteOutcome: 是否击发、是否淘汰及事件文案。
        """
        if not player.revolver_deck:
            player.revolver_deck = self.build_revolver_deck()

        revealed = player.revolver_deck.pop(0)
        fired = revealed == REVOLVER_LETHAL

        if fired:
            player.eliminated = True
            return RouletteOutcome(
                fired=True,
                eliminated=True,
                event=f"Roulette revealed {REVOLVER_LETHAL} on {player.player_id}; player eliminated.",
            )

        return RouletteOutcome(
            fired=False,
            eliminated=False,
            event=f"Roulette revealed {REVOLVER_BLANK} on {player.player_id}; player survived.",
        )
