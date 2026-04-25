from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from liars_game_engine.analysis.shapley_analyzer import BASELINE_MODE_RANDOM_LEGAL_AGENT, ShapleyAnalyzer
from liars_game_engine.analysis.task_c_runner import _ensure_four_mock_players
from liars_game_engine.analysis.task_k_gold_runner import _export_attributed_logs
from liars_game_engine.config.loader import load_settings


def run_task_k_backfill_pipeline(
    log_root: Path | str,
    config_file: str | Path = "config/experiment.yaml",
    rollout_samples: int = 200,
    output_dir: Path | str = "logs/task_k_gold/attributed_logs",
    max_workers: int | None = 24,
) -> dict[str, object]:
    """作用: 对现有 Task K baseline 日志执行离线归因回填。"""
    root = Path(log_root)
    log_paths = sorted(root.rglob("*.jsonl"))
    if not log_paths:
        raise RuntimeError(f"No baseline Task K logs found under {root}")

    settings = load_settings(config_file=config_file)
    task_settings = _ensure_four_mock_players(settings)
    analyzer_workers = max_workers if max_workers is not None else (os.cpu_count() or 1)
    analyzer = ShapleyAnalyzer(
        settings=task_settings,
        rollout_samples=rollout_samples,
        rollout_policy="random",
        max_workers=max(1, analyzer_workers),
        baseline_mode=BASELINE_MODE_RANDOM_LEGAL_AGENT,
    )

    start = time.perf_counter()
    attributions, _ = analyzer.analyze_logs(log_paths)
    elapsed = max(0.0, time.perf_counter() - start)
    attributed_dir = _export_attributed_logs(
        baseline_logs=log_paths,
        attributions=attributions,
        output_dir=output_dir,
    )

    return {
        "log_root": str(root),
        "baseline_log_count": len(log_paths),
        "attributed_dir": str(attributed_dir),
        "attribution_count": len(attributions),
        "rollout_samples": int(rollout_samples),
        "max_workers": max(1, analyzer_workers),
        "backfill_elapsed_seconds": elapsed,
        "average_seconds_per_attribution": elapsed / max(1, len(attributions)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill shapley_value/phi labels into existing Task K baseline logs.")
    parser.add_argument("log_root", help="Directory containing existing Task K baseline_logs/*.jsonl")
    parser.add_argument("--config-file", default="config/experiment.yaml")
    parser.add_argument("--rollout-samples", type=int, default=200)
    parser.add_argument("--output-dir", default="logs/task_k_gold/attributed_logs")
    parser.add_argument("--max-workers", type=int, default=24)
    args = parser.parse_args()

    summary = run_task_k_backfill_pipeline(
        log_root=args.log_root,
        config_file=args.config_file,
        rollout_samples=args.rollout_samples,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
    )
    print(summary)


if __name__ == "__main__":
    main()
