from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from liars_game_engine.analysis.shapley_analyzer import BASELINE_MODE_RANDOM_LEGAL_AGENT, ShapleyAnalyzer
from liars_game_engine.analysis.task_c_runner import _ensure_four_mock_players
from liars_game_engine.analysis.task_k_gold_runner import _export_attributed_logs
from liars_game_engine.config.loader import load_settings


def _build_progress_bar(completed_logs: int, total_logs: int, width: int = 20) -> str:
    safe_total = max(1, int(total_logs))
    ratio = max(0.0, min(1.0, completed_logs / safe_total))
    filled = min(width, int(ratio * width))
    return f"[{'#' * filled}{'-' * (width - filled)}]"


def _append_progress_log(
    progress_log_path: Path,
    *,
    completed_logs: int,
    total_logs: int,
    batch_index: int,
    batch_log_count: int,
    cumulative_attribution_count: int,
    batch_elapsed_seconds: float,
    total_elapsed_seconds: float,
) -> None:
    progress_log_path.parent.mkdir(parents=True, exist_ok=True)
    safe_total = max(1, int(total_logs))
    percent_complete = max(0.0, min(100.0, (completed_logs / safe_total) * 100.0))
    average_seconds_per_log = total_elapsed_seconds / max(1, completed_logs)
    eta_seconds = average_seconds_per_log * max(0, safe_total - completed_logs)
    line = (
        f"batch={batch_index} "
        f"completed_logs={completed_logs}/{total_logs} "
        f"percent={percent_complete:.1f}% "
        f"progress_bar={_build_progress_bar(completed_logs, safe_total)} "
        f"batch_log_count={batch_log_count} "
        f"cumulative_attributions={cumulative_attribution_count} "
        f"batch_elapsed_seconds={batch_elapsed_seconds:.6f} "
        f"total_elapsed_seconds={total_elapsed_seconds:.6f} "
        f"eta_seconds={eta_seconds:.6f}"
    )
    with progress_log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def run_task_k_backfill_pipeline(
    log_root: Path | str,
    config_file: str | Path = "config/experiment.yaml",
    rollout_samples: int = 200,
    output_dir: Path | str = "logs/task_k_gold/attributed_logs",
    max_workers: int | None = 24,
    progress_interval_logs: int = 10,
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

    output_path = Path(output_dir)
    progress_log_path = output_path.parent / "progress.log"
    progress_log_path.parent.mkdir(parents=True, exist_ok=True)
    progress_log_path.write_text("", encoding="utf-8")

    completed_logs = 0
    batch_index = 0
    total_elapsed = 0.0
    all_attributions = []
    batch_size = max(1, int(progress_interval_logs))

    while completed_logs < len(log_paths):
        batch_index += 1
        batch_logs = log_paths[completed_logs : completed_logs + batch_size]
        batch_start = time.perf_counter()
        batch_attributions, _ = analyzer.analyze_logs(batch_logs)
        batch_elapsed = max(0.0, time.perf_counter() - batch_start)
        total_elapsed += batch_elapsed
        all_attributions.extend(batch_attributions)

        attributed_dir = _export_attributed_logs(
            baseline_logs=batch_logs,
            attributions=batch_attributions,
            output_dir=output_path,
        )
        completed_logs += len(batch_logs)
        _append_progress_log(
            progress_log_path,
            completed_logs=completed_logs,
            total_logs=len(log_paths),
            batch_index=batch_index,
            batch_log_count=len(batch_logs),
            cumulative_attribution_count=len(all_attributions),
            batch_elapsed_seconds=batch_elapsed,
            total_elapsed_seconds=total_elapsed,
        )

    return {
        "log_root": str(root),
        "baseline_log_count": len(log_paths),
        "attributed_dir": str(attributed_dir),
        "progress_log": str(progress_log_path),
        "attribution_count": len(all_attributions),
        "rollout_samples": int(rollout_samples),
        "max_workers": max(1, analyzer_workers),
        "progress_interval_logs": batch_size,
        "backfill_elapsed_seconds": total_elapsed,
        "average_seconds_per_attribution": total_elapsed / max(1, len(all_attributions)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill shapley_value/phi labels into existing Task K baseline logs.")
    parser.add_argument("log_root", help="Directory containing existing Task K baseline_logs/*.jsonl")
    parser.add_argument("--config-file", default="config/experiment.yaml")
    parser.add_argument("--rollout-samples", type=int, default=200)
    parser.add_argument("--output-dir", default="logs/task_k_gold/attributed_logs")
    parser.add_argument("--max-workers", type=int, default=24)
    parser.add_argument("--progress-interval-logs", type=int, default=10)
    args = parser.parse_args()

    summary = run_task_k_backfill_pipeline(
        log_root=args.log_root,
        config_file=args.config_file,
        rollout_samples=args.rollout_samples,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        progress_interval_logs=args.progress_interval_logs,
    )
    print(summary)


if __name__ == "__main__":
    main()
