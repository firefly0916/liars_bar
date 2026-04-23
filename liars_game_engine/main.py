from __future__ import annotations

import asyncio
from datetime import datetime

from liars_game_engine.agents.factory import build_agents
from liars_game_engine.config.loader import load_settings
from liars_game_engine.engine.environment import GameEnvironment
from liars_game_engine.experiment.logger import ExperimentLogger
from liars_game_engine.experiment.orchestrator import GameOrchestrator


async def run() -> dict[str, object]:
    """作用: 组装配置、环境、Agent 与编排器并运行一局游戏。

    输入:
    - 无。

    返回:
    - dict[str, object]: 对局摘要信息。
    """
    settings = load_settings()
    env = GameEnvironment(settings)
    agents = build_agents(settings)

    game_id = datetime.now().strftime("game-%Y%m%d-%H%M%S")
    logger = ExperimentLogger(base_dir=settings.logging.run_log_dir, game_id=game_id)

    orchestrator = GameOrchestrator(
        env=env,
        agents=agents,
        logger=logger,
        fallback_action=settings.runtime.fallback_action,
        max_turns=settings.runtime.max_turns,
    )
    return await orchestrator.run_game_loop()


def main() -> None:
    """作用: 命令行入口，执行对局并打印摘要。

    输入:
    - 无。

    返回:
    - 无。
    """
    summary = asyncio.run(run())
    print("Liar's Bar run finished:", summary)


if __name__ == "__main__":
    main()
