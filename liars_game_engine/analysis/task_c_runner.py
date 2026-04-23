from __future__ import annotations

import asyncio
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from liars_game_engine.agents.factory import build_agents
from liars_game_engine.analysis.shapley_analyzer import BASELINE_MODE_RANDOM_LEGAL_AGENT, ShapleyAnalyzer
from liars_game_engine.config.loader import load_settings
from liars_game_engine.config.schema import AppSettings
from liars_game_engine.engine.environment import GameEnvironment
from liars_game_engine.experiment.logger import ExperimentLogger
from liars_game_engine.experiment.orchestrator import GameOrchestrator


def _ensure_four_mock_players(settings: AppSettings) -> AppSettings:
    """作用: 确保任务 C 使用 4 个 Mock 玩家配置。

    输入:
    - settings: 原始加载配置。

    返回:
    - AppSettings: 补全/覆盖后的 4 玩家 mock 配置。
    """
    raw = asdict(settings)
    existing_players = raw.get("players", []) if isinstance(raw.get("players", []), list) else []

    normalized_players: list[dict[str, object]] = []
    for index in range(4):
        if index < len(existing_players) and isinstance(existing_players[index], dict):
            source = existing_players[index]
            normalized_players.append(
                {
                    "player_id": str(source.get("player_id", f"p{index + 1}")),
                    "name": str(source.get("name", f"Player{index + 1}")),
                    "agent_type": "mock",
                    "model": str(source.get("model", f"mock-model-{index + 1}")),
                    "prompt_profile": str(source.get("prompt_profile", "baseline")),
                    "temperature": float(source.get("temperature", 0.2)),
                }
            )
            continue

        normalized_players.append(
            {
                "player_id": f"p{index + 1}",
                "name": f"Player{index + 1}",
                "agent_type": "mock",
                "model": f"mock-model-{index + 1}",
                "prompt_profile": "baseline",
                "temperature": 0.2,
            }
        )

    raw["players"] = normalized_players
    return AppSettings.from_dict(raw)


async def _run_single_game(settings: AppSettings, log_dir: Path, game_id: str) -> Path:
    env = GameEnvironment(settings)
    agents = build_agents(settings)
    logger = ExperimentLogger(base_dir=log_dir, game_id=game_id)
    orchestrator = GameOrchestrator(
        env=env,
        agents=agents,
        logger=logger,
        fallback_action=settings.runtime.fallback_action,
        max_turns=settings.runtime.max_turns,
    )
    await orchestrator.run_game_loop()
    return logger.log_file


async def generate_baseline_logs(settings: AppSettings, game_count: int, log_dir: Path) -> list[Path]:
    """作用: 运行多局对战并输出 baseline JSONL 轨迹文件。

    输入:
    - settings: 任务 C 对局配置。
    - game_count: 对战局数。
    - log_dir: 日志输出目录。

    返回:
    - list[Path]: 每局 JSONL 文件路径。
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    log_files: list[Path] = []
    base_raw = asdict(settings)
    run_prefix = datetime.now().strftime("task-c-%Y%m%d-%H%M%S")

    for game_index in range(game_count):
        game_raw = dict(base_raw)
        runtime_raw = dict(base_raw.get("runtime", {}))
        runtime_raw["random_seed"] = int(settings.runtime.random_seed) + game_index
        game_raw["runtime"] = runtime_raw

        per_game_settings = AppSettings.from_dict(game_raw)
        game_id = f"{run_prefix}-{game_index + 1:03d}"
        log_file = await _run_single_game(settings=per_game_settings, log_dir=log_dir, game_id=game_id)
        log_files.append(log_file)

    return log_files


def run_task_c_pipeline(
    config_file: str | Path = "config/experiment.yaml",
    game_count: int = 50,
    rollout_samples: int = 50,
    output_dir: str | Path = "logs/task_c",
    max_workers: int | None = None,
) -> dict[str, object]:
    """作用: 执行任务 C 全流程（基线日志、反事实采样、信用报表）。

    输入:
    - config_file: 主配置文件路径。
    - game_count: 基线对战局数。
    - rollout_samples: 每个决策点每组采样次数。
    - output_dir: 输出目录（含 baseline 日志与 credit_report.csv）。
    - max_workers: 并行 worker 数，默认 CPU 核心数。

    返回:
    - dict[str, object]: 流水线执行摘要。
    """
    settings = load_settings(config_file=config_file)
    task_settings = _ensure_four_mock_players(settings)

    output_path = Path(output_dir)
    baseline_dir = output_path / "baseline_logs"
    baseline_logs = asyncio.run(generate_baseline_logs(task_settings, game_count=game_count, log_dir=baseline_dir))

    analyzer_workers = max_workers if max_workers is not None else (os.cpu_count() or 1)
    analyzer = ShapleyAnalyzer(
        settings=task_settings,
        rollout_samples=rollout_samples,
        rollout_policy="random",
        max_workers=max(1, analyzer_workers),
        baseline_mode=BASELINE_MODE_RANDOM_LEGAL_AGENT,
    )

    attributions, _ = analyzer.analyze_logs(baseline_logs)
    report_path = output_path / "credit_report.csv"
    ShapleyAnalyzer.export_credit_report(attributions=attributions, output_path=report_path)

    return {
        "baseline_game_count": len(baseline_logs),
        "baseline_dir": str(baseline_dir),
        "attribution_count": len(attributions),
        "credit_report": str(report_path),
        "rollout_samples": rollout_samples,
        "max_workers": max(1, analyzer_workers),
    }


def main() -> None:
    summary = run_task_c_pipeline()
    print("Task C pipeline finished:", summary)


if __name__ == "__main__":
    main()
