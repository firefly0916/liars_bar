from __future__ import annotations

import os
import time
from pathlib import Path

from liars_game_engine.analysis.shapley_analyzer import (
    ProxyValuePredictor,
    ShapleyAnalyzer,
)
from liars_game_engine.analysis.train_value_proxy import DEFAULT_VALUE_PROXY_EPOCHS, train_value_proxy
from liars_game_engine.analysis.task_d_axiomatic_runner import _build_probe_experiment_settings
from liars_game_engine.config.loader import load_settings


def _select_latest_probe_logs(log_root: Path, game_count: int) -> tuple[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for log_path in sorted(log_root.glob("*.jsonl")):
        parts = log_path.stem.split("-")
        if len(parts) < 5:
            continue
        run_id = "-".join(parts[:4])
        grouped.setdefault(run_id, []).append(log_path)

    if not grouped:
        raise RuntimeError(f"No probe logs found under: {log_root}")

    latest_run_id = sorted(grouped)[-1]
    selected = sorted(grouped[latest_run_id])[: max(1, int(game_count))]
    return latest_run_id, selected


def run_task_i_proxy_pipeline(
    config_file: str | Path = "config/experiment.yaml",
    log_root: str | Path = "logs/task_d_probe/probe_logs",
    model_path: str | Path = "logs/task_d_probe/value_proxy_mlp.pt",
    output_dir: str | Path = "logs/task_d_probe",
    game_count: int = 100,
    alignment_sample_size: int = 20,
    alignment_sample_seed: int = 42,
    rollout_samples: int = 200,
    max_workers: int | None = None,
    training_epochs: int = DEFAULT_VALUE_PROXY_EPOCHS,
) -> dict[str, object]:
    settings = load_settings(config_file=config_file)
    task_settings = _build_probe_experiment_settings(settings)

    log_root_path = Path(log_root)
    output_dir_path = Path(output_dir)
    training_metrics = train_value_proxy(
        log_root=log_root_path,
        output_dir=output_dir_path,
        epochs=training_epochs,
    )
    trained_model_path = Path(training_metrics["model_path"])
    run_id, log_paths = _select_latest_probe_logs(log_root=log_root_path, game_count=game_count)

    analyzer_workers = max_workers if max_workers is not None else (os.cpu_count() or 1)
    analyzer = ShapleyAnalyzer(
        settings=task_settings,
        rollout_samples=rollout_samples,
        rollout_policy="random",
        max_workers=max(1, analyzer_workers),
    )
    predictor = ProxyValuePredictor(model_path=trained_model_path)

    alignment_report = analyzer.run_proxy_alignment(
        log_paths=log_paths,
        predictor=predictor,
        sample_size=alignment_sample_size,
        sample_seed=alignment_sample_seed,
    )
    alignment_report["selected_run_id"] = run_id
    alignment_report["log_count"] = len(log_paths)
    alignment_report_path = analyzer.export_alignment_report(
        report=alignment_report,
        output_path=output_dir_path / "proxy_alignment_report.json",
    )

    start_proxy = time.perf_counter()
    attributions, _ = analyzer.analyze_logs_proxy(log_paths=log_paths, predictor=predictor)
    proxy_elapsed = max(0.0, time.perf_counter() - start_proxy)
    report_path = analyzer.export_credit_report(
        attributions=attributions,
        output_path=output_dir_path / "credit_report_proxy.csv",
    )

    return {
        "selected_run_id": run_id,
        "log_count": len(log_paths),
        "attribution_count": len(attributions),
        "proxy_alignment_report": str(alignment_report_path),
        "credit_report_proxy": str(report_path),
        "proxy_analysis_seconds": proxy_elapsed,
        "alignment_passed": bool(alignment_report.get("alignment_passed", False)),
        "pearson_correlation": float(alignment_report.get("pearson_correlation", 0.0)),
        "mae": float(alignment_report.get("mae", 0.0)),
        "speedup_ratio": float(alignment_report.get("speedup_ratio", 0.0)),
        "model_path": str(trained_model_path if trained_model_path else model_path),
        "val_mse": float(training_metrics.get("val_mse", 0.0)),
        "best_val_mse": float(training_metrics.get("best_val_mse", 0.0)),
    }


def main() -> None:
    summary = run_task_i_proxy_pipeline()
    print("Task I proxy pipeline finished:", summary)


if __name__ == "__main__":
    main()
