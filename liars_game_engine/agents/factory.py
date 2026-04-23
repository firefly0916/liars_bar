from __future__ import annotations

from liars_game_engine.agents.base_agent import BaseAgent
from liars_game_engine.agents.langchain_agent import LangChainAgent
from liars_game_engine.agents.mock_agent import MockAgent
from liars_game_engine.config.schema import AppSettings


def build_agents(settings: AppSettings) -> dict[str, BaseAgent]:
    """作用: 按配置构建每个玩家对应的 Agent 实例。

    输入:
    - settings: 已加载的应用配置，包含 players、api、parser、runtime 等。

    返回:
    - dict[str, BaseAgent]: 键为 player_id，值为对应 Agent 对象。
    """
    agents: dict[str, BaseAgent] = {}

    for index, player in enumerate(settings.players):
        if player.agent_type == "langchain":
            agent = LangChainAgent(
                player_id=player.player_id,
                model=player.model,
                prompt_profile=player.prompt_profile,
                temperature=player.temperature,
                openrouter_api_key=settings.api.openrouter_api_key,
                openrouter_base_url=settings.api.openrouter_base_url,
                max_retries=settings.parser.max_retries,
                enable_null_player_probe=settings.runtime.enable_null_player_probe,
            )
        else:
            agent = MockAgent(
                player_id=player.player_id,
                model=player.model,
                prompt_profile=player.prompt_profile,
                temperature=player.temperature,
                seed=settings.runtime.random_seed + index,
                enable_null_player_probe=settings.runtime.enable_null_player_probe,
            )

        agents[player.player_id] = agent

    return agents
