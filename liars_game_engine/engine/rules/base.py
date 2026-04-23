from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from liars_game_engine.engine.game_state import ActionModel, RuntimeGameState


@dataclass
class RuleResult:
    ok: bool
    events: list[str]
    error_code: str | None = None
    error_reason: str | None = None


class RuleModule(ABC):
    name: str

    @abstractmethod
    def validate(self, state: RuntimeGameState, player_id: str, action: ActionModel) -> RuleResult:
        """作用: 校验动作在当前状态下是否合法。

        输入:
        - state: 当前运行态。
        - player_id: 发起动作的玩家 ID。
        - action: 待校验动作。

        返回:
        - RuleResult: 校验结果与错误信息。
        """
        raise NotImplementedError
