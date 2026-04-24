from __future__ import annotations

import asyncio
import inspect
import os
import time
from pathlib import Path

from liars_game_engine.analysis.shapley_analyzer import BASELINE_MODE_RANDOM_LEGAL_AGENT, ShapleyAnalyzer
from liars_game_engine.analysis.task_c_runner import _ensure_four_mock_players, generate_baseline_logs
from liars_game_engine.config.loader import load_settings


def _resolve_async_result(result: object) -> object:
    if inspect.isawaitable(result):
        return asyncio.runners.run(result)
    return result


def run_task_k_gold_pipeline(
    config_file: str | Path = "config/experiment.yaml",
    game_count: int = 2_000,
    rollout_samples: int = 200,
    output_dir: str | Path = "logs/task_k_gold",
    max_workers: int | None = 24,
) -> dict[str, object]:
    """作用: 执行 Task K 金标准物理 rollout 归因流水线。

    输入:
    - config_file: 主配置文件路径。
    - game_count: 生成的对局数量；本地默认只接线，不要求立刻跑满服务器规模。
    - rollout_samples: 每个决策点的物理采样次数。
    - output_dir: 输出目录。
    - max_workers: 并行 worker 数；为空时退回 CPU 核数。

    返回:
    - dict[str, object]: 基线日志、报表路径与平均归因耗时摘要。
    """
    settings = load_settings(config_file=config_file)
    task_settings = _ensure_four_mock_players(settings)

    output_path = Path(output_dir)
    baseline_dir = output_path / "baseline_logs"
    baseline_logs = _resolve_async_result(
        generate_baseline_logs(task_settings, game_count=game_count, log_dir=baseline_dir)
    )

    analyzer_workers = max_workers if max_workers is not None else (os.cpu_count() or 1)
    analyzer = ShapleyAnalyzer(
        settings=task_settings,
        rollout_samples=rollout_samples,
        rollout_policy="random",
        max_workers=max(1, analyzer_workers),
        baseline_mode=BASELINE_MODE_RANDOM_LEGAL_AGENT,
    )

    start = time.perf_counter()
    attributions, _ = analyzer.analyze_logs(baseline_logs)
    elapsed = max(0.0, time.perf_counter() - start)
    report_path = output_path / "credit_report_final.csv"
    analyzer.export_credit_report(attributions=attributions, output_path=report_path)

    return {
        "baseline_game_count": len(baseline_logs),
        "baseline_dir": str(baseline_dir),
        "attribution_count": len(attributions),
        "credit_report": str(report_path),
        "rollout_samples": int(rollout_samples),
        "max_workers": max(1, analyzer_workers),
        "attribution_elapsed_seconds": elapsed,
        "average_seconds_per_attribution": elapsed / max(1, len(attributions)),
    }


def main() -> None:
    summary = run_task_k_gold_pipeline()
    print("Task K gold pipeline finished:", summary)


if __name__ == "__main__":
    main()
