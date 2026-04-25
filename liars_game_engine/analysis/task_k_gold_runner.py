from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import time
from pathlib import Path

from liars_game_engine.analysis.shapley_analyzer import (
    BASELINE_MODE_RANDOM_LEGAL_AGENT,
    ShapleyAnalyzer,
    ShapleyAttribution,
)
from liars_game_engine.analysis.task_c_runner import _ensure_four_mock_players, generate_baseline_logs
from liars_game_engine.config.loader import load_settings


def _resolve_async_result(result: object) -> object:
    if inspect.isawaitable(result):
        return asyncio.runners.run(result)
    return result


def _export_attributed_logs(
    baseline_logs: list[Path],
    attributions: list[ShapleyAttribution],
    output_dir: Path | str,
) -> Path:
    attributed_dir = Path(output_dir)
    attributed_dir.mkdir(parents=True, exist_ok=True)

    attribution_by_key = {
        (item.game_id, int(item.turn), item.player_id): item
        for item in attributions
    }

    for log_path in baseline_logs:
        output_path = attributed_dir / Path(log_path).name
        enriched_lines: list[str] = []

        for line in Path(log_path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue

            record = json.loads(line)
            if not isinstance(record, dict):
                continue

            key = (Path(log_path).stem, int(record.get("turn", 0)), str(record.get("player_id", "")))
            attribution = attribution_by_key.get(key)
            if attribution is not None:
                record["shapley_value"] = float(attribution.phi)
                record["phi"] = float(attribution.phi)
                record["value_action"] = float(attribution.value_action)
                record["value_counterfactual"] = float(attribution.value_counterfactual)
                record["winner"] = attribution.winner
                record["rollout_samples"] = int(attribution.rollout_samples)
                record["death_prob_bucket"] = attribution.death_prob_bucket
                record["state_feature"] = attribution.state_feature

            enriched_lines.append(json.dumps(record, ensure_ascii=False))

        output_path.write_text("\n".join(enriched_lines) + "\n", encoding="utf-8")

    return attributed_dir


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
    attributed_dir = _export_attributed_logs(
        baseline_logs=baseline_logs,
        attributions=attributions,
        output_dir=output_path / "attributed_logs",
    )
    analyzer.export_credit_report(attributions=attributions, output_path=report_path)

    return {
        "baseline_game_count": len(baseline_logs),
        "baseline_dir": str(baseline_dir),
        "attributed_dir": str(attributed_dir),
        "attribution_count": len(attributions),
        "credit_report": str(report_path),
        "rollout_samples": int(rollout_samples),
        "max_workers": max(1, analyzer_workers),
        "attribution_elapsed_seconds": elapsed,
        "average_seconds_per_attribution": elapsed / max(1, len(attributions)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task K gold rollout attribution pipeline.")
    parser.add_argument("--game-count", type=int, default=2_000)
    parser.add_argument("--rollout-samples", type=int, default=200)
    parser.add_argument("--output-dir", default="logs/task_k_gold")
    parser.add_argument("--max-workers", type=int, default=24)
    args = parser.parse_args()

    summary = run_task_k_gold_pipeline(
        game_count=args.game_count,
        rollout_samples=args.rollout_samples,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
    )
    print("Task K gold pipeline finished:", summary)


if __name__ == "__main__":
    main()
