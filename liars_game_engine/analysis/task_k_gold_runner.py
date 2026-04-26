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


def _build_progress_bar(completed_games: int, total_games: int, width: int = 20) -> str:
    safe_total = max(1, int(total_games))
    ratio = max(0.0, min(1.0, completed_games / safe_total))
    filled = min(width, int(ratio * width))
    return f"[{'#' * filled}{'-' * (width - filled)}]"


def _resolve_async_result(result: object) -> object:
    if inspect.isawaitable(result):
        return asyncio.runners.run(result)
    return result


def _append_progress_log(
    progress_log_path: Path,
    *,
    completed_games: int,
    total_games: int,
    batch_index: int,
    batch_game_count: int,
    cumulative_attribution_count: int,
    batch_elapsed_seconds: float,
    total_elapsed_seconds: float,
) -> None:
    progress_log_path.parent.mkdir(parents=True, exist_ok=True)
    safe_total = max(1, int(total_games))
    percent_complete = max(0.0, min(100.0, (completed_games / safe_total) * 100.0))
    average_seconds_per_game = total_elapsed_seconds / max(1, completed_games)
    eta_seconds = average_seconds_per_game * max(0, safe_total - completed_games)
    line = (
        f"batch={batch_index} "
        f"completed_games={completed_games}/{total_games} "
        f"percent={percent_complete:.1f}% "
        f"progress_bar={_build_progress_bar(completed_games, safe_total)} "
        f"batch_game_count={batch_game_count} "
        f"cumulative_attributions={cumulative_attribution_count} "
        f"batch_elapsed_seconds={batch_elapsed_seconds:.6f} "
        f"total_elapsed_seconds={total_elapsed_seconds:.6f} "
        f"eta_seconds={eta_seconds:.6f}"
    )
    with progress_log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


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
        output_path = attributed_dir / log_path.name
        enriched_lines: list[str] = []

        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue

            record = json.loads(line)
            if not isinstance(record, dict):
                continue

            key = (log_path.stem, int(record.get("turn", 0)), str(record.get("player_id", "")))
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
    progress_interval_games: int = 50,
    existing_baseline_dir: str | Path | None = None,
) -> dict[str, object]:
    """作用: 执行 Task K 金标准物理 rollout 归因流水线。

    输入:
    - config_file: 主配置文件路径。
    - game_count: 生成的对局数量；本地默认只接线，不要求立刻跑满服务器规模。
    - rollout_samples: 每个决策点的物理采样次数。
    - output_dir: 输出目录。
    - max_workers: 并行 worker 数；为空时退回 CPU 核数。
    - progress_interval_games: 每处理多少局追加一条进度日志。
    - existing_baseline_dir: 若提供，则直接复用现有 baseline 日志目录，不再生成新对局。

    返回:
    - dict[str, object]: 基线日志、报表路径与平均归因耗时摘要。
    """
    settings = load_settings(config_file=config_file)
    task_settings = _ensure_four_mock_players(settings)

    output_path = Path(output_dir)
    progress_log_path = output_path / "progress.log"
    progress_log_path.parent.mkdir(parents=True, exist_ok=True)
    progress_log_path.write_text("", encoding="utf-8")

    existing_baseline_path = Path(existing_baseline_dir) if existing_baseline_dir is not None else None
    if existing_baseline_path is not None:
        baseline_dir = existing_baseline_path
        baseline_logs = sorted(path for path in baseline_dir.glob("*.jsonl") if path.is_file())
        if not baseline_logs:
            raise RuntimeError(f"No baseline logs found under existing_baseline_dir={baseline_dir}")
    else:
        baseline_dir = output_path / "baseline_logs"
        baseline_logs = []

    analyzer_workers = max_workers if max_workers is not None else (os.cpu_count() or 1)
    analyzer = ShapleyAnalyzer(
        settings=task_settings,
        rollout_samples=rollout_samples,
        rollout_policy="random",
        max_workers=max(1, analyzer_workers),
        baseline_mode=BASELINE_MODE_RANDOM_LEGAL_AGENT,
    )

    all_attributions = []
    attributed_dir = output_path / "attributed_logs"
    completed_games = 0
    elapsed = 0.0
    batch_index = 0
    batch_size = max(1, int(progress_interval_games))
    if existing_baseline_path is None:
        games_remaining = max(1, int(game_count))
        while completed_games < games_remaining:
            batch_index += 1
            current_batch_game_count = min(batch_size, games_remaining - completed_games)
            batch_logs = _resolve_async_result(
                generate_baseline_logs(task_settings, game_count=current_batch_game_count, log_dir=baseline_dir)
            )
            baseline_logs.extend(batch_logs)

            batch_start = time.perf_counter()
            batch_attributions, _ = analyzer.analyze_logs(batch_logs)
            batch_elapsed = max(0.0, time.perf_counter() - batch_start)
            elapsed += batch_elapsed
            all_attributions.extend(batch_attributions)
            _export_attributed_logs(
                baseline_logs=batch_logs,
                attributions=batch_attributions,
                output_dir=attributed_dir,
            )

            completed_games += len(batch_logs)
            _append_progress_log(
                progress_log_path,
                completed_games=completed_games,
                total_games=games_remaining,
                batch_index=batch_index,
                batch_game_count=len(batch_logs),
                cumulative_attribution_count=len(all_attributions),
                batch_elapsed_seconds=batch_elapsed,
                total_elapsed_seconds=elapsed,
            )
    else:
        games_remaining = len(baseline_logs)
        while completed_games < games_remaining:
            batch_index += 1
            batch_logs = baseline_logs[completed_games : completed_games + batch_size]

            batch_start = time.perf_counter()
            batch_attributions, _ = analyzer.analyze_logs(batch_logs)
            batch_elapsed = max(0.0, time.perf_counter() - batch_start)
            elapsed += batch_elapsed
            all_attributions.extend(batch_attributions)
            _export_attributed_logs(
                baseline_logs=batch_logs,
                attributions=batch_attributions,
                output_dir=attributed_dir,
            )

            completed_games += len(batch_logs)
            _append_progress_log(
                progress_log_path,
                completed_games=completed_games,
                total_games=games_remaining,
                batch_index=batch_index,
                batch_game_count=len(batch_logs),
                cumulative_attribution_count=len(all_attributions),
                batch_elapsed_seconds=batch_elapsed,
                total_elapsed_seconds=elapsed,
            )

    report_path = output_path / "credit_report_final.csv"
    analyzer.export_credit_report(attributions=all_attributions, output_path=report_path)

    return {
        "baseline_game_count": len(baseline_logs),
        "baseline_dir": str(baseline_dir),
        "attributed_dir": str(attributed_dir),
        "attribution_count": len(all_attributions),
        "credit_report": str(report_path),
        "progress_log": str(progress_log_path),
        "rollout_samples": int(rollout_samples),
        "max_workers": max(1, analyzer_workers),
        "attribution_elapsed_seconds": elapsed,
        "average_seconds_per_attribution": elapsed / max(1, len(all_attributions)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task K gold rollout attribution pipeline.")
    parser.add_argument("--config-file", default="config/experiment.yaml")
    parser.add_argument("--game-count", type=int, default=2_000)
    parser.add_argument("--rollout-samples", type=int, default=200)
    parser.add_argument("--output-dir", default="logs/task_k_gold")
    parser.add_argument("--max-workers", type=int, default=24)
    parser.add_argument("--progress-interval-games", type=int, default=50)
    parser.add_argument("--existing-baseline-dir", default=None)
    args = parser.parse_args()

    summary = run_task_k_gold_pipeline(
        config_file=args.config_file,
        game_count=args.game_count,
        rollout_samples=args.rollout_samples,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        progress_interval_games=args.progress_interval_games,
        existing_baseline_dir=args.existing_baseline_dir,
    )
    print("Task K gold pipeline finished:", summary)


if __name__ == "__main__":
    main()
