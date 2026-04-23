from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from liars_game_engine.engine.game_state import ActionModel, ParseError


@dataclass
class AgentDecision:
    thought: str
    action: ActionModel
    raw_output: str = ""
    parse_error: ParseError | None = None
    selected_skill: str | None = None
    skill_parameters: dict[str, object] = field(default_factory=dict)
    decision_bias: str | None = None


class BaseAgent(ABC):
    def __init__(self, player_id: str, model: str, prompt_profile: str, temperature: float) -> None:
        """作用: 初始化智能体公共配置。

        输入:
        - player_id: 玩家唯一标识。
        - model: 模型名或策略标识。
        - prompt_profile: prompt 模板名称。
        - temperature: 采样温度。

        返回:
        - 无。
        """
        self.player_id = player_id
        self.model = model
        self.prompt_profile = prompt_profile
        self.temperature = temperature

    @abstractmethod
    async def act(self, observation: dict[str, object]) -> AgentDecision:
        """作用: 根据观测生成本回合决策。

        输入:
        - observation: 环境提供的当前可见状态。

        返回:
        - AgentDecision: 包含 thought、action、raw_output 等字段。
        """
        raise NotImplementedError
