from __future__ import annotations

import asyncio
import inspect
import json
import os
from dataclasses import asdict
from pathlib import Path

from liars_game_engine.analysis.shapley_analyzer import ProxyValuePredictor, ShapleyAnalyzer
from liars_game_engine.analysis.task_c_runner import generate_baseline_logs
from liars_game_engine.analysis.task_d_axiomatic_runner import _build_probe_experiment_settings
from liars_game_engine.analysis.task_i_proxy_runner import _select_latest_probe_logs
from liars_game_engine.analysis.train_value_proxy import (
    DEFAULT_VALUE_PROXY_EPOCHS,
    VALUE_PROXY_TARGET_PHI,
    VALUE_PROXY_TARGET_WINNER,
    train_value_proxy,
)
from liars_game_engine.config.loader import load_settings
from liars_game_engine.config.schema import AppSettings


def _resolve_async_result(result: object) -> object:
    if inspect.isawaitable(result):
        return asyncio.runners.run(result)
    return result


def _build_task_l_negative_settings(settings: AppSettings, probe_probability: float = 0.85) -> AppSettings:
    """作用: 构建 Task L 负采样设置，提升 Null Probe 的触发概率。"""
    probe_settings = _build_probe_experiment_settings(settings)
    raw = asdict(probe_settings)
    runtime_raw = dict(raw.get("runtime", {}))
    runtime_raw["enable_null_player_probe"] = True
    runtime_raw["null_probe_action_probability"] = max(0.0, min(1.0, float(probe_probability)))
    raw["runtime"] = runtime_raw
    return AppSettings.from_dict(raw)


def _count_log_records(log_paths: list[Path]) -> int:
    return sum(
        len([line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()])
        for log_path in log_paths
    )


def generate_negative_logs_until_records(
    settings: AppSettings,
    output_dir: Path,
    record_target: int,
    batch_game_count: int = 100,
) -> dict[str, object]:
    """作用: 持续生成高 Probe 轨迹，直到达到指定日志条目数量。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    log_paths: list[Path] = []
    record_count = 0
    while record_count < max(1, int(record_target)):
        batch_logs = asyncio.run(
            generate_baseline_logs(
                settings,
                game_count=max(1, int(batch_game_count)),
                log_dir=output_dir,
            )
        )
        batch_logs = _resolve_async_result(batch_logs)
        log_paths.extend(batch_logs)
        record_count = _count_log_records(log_paths)

    return {
        "negative_log_dir": str(output_dir),
        "record_count": record_count,
        "game_count": len(log_paths),
        "log_paths": log_paths,
    }


def run_proxy_alignment_for_model(
    settings: AppSettings,
    log_paths: list[Path],
    model_path: str | Path,
    target_mode: str = VALUE_PROXY_TARGET_WINNER,
    rollout_samples: int = 200,
    sample_size: int = 20,
    sample_seed: int = 42,
    max_workers: int | None = None,
) -> dict[str, object]:
    """作用: 对指定 proxy 模型执行双线对齐校验。"""
    analyzer_workers = max_workers if max_workers is not None else (os.cpu_count() or 1)
    analyzer = ShapleyAnalyzer(
        settings=settings,
        rollout_samples=rollout_samples,
        rollout_policy="random",
        max_workers=max(1, analyzer_workers),
    )
    predictor = ProxyValuePredictor(model_path=model_path, output_mode=target_mode)
    if target_mode == VALUE_PROXY_TARGET_PHI:
        return analyzer.run_direct_phi_alignment(
            log_paths=log_paths,
            predictor=predictor,
            sample_size=sample_size,
            sample_seed=sample_seed,
        )
    return analyzer.run_proxy_alignment(
        log_paths=log_paths,
        predictor=predictor,
        sample_size=sample_size,
        sample_seed=sample_seed,
    )


def run_task_l_proxy_refine_pipeline(
    config_file: str | Path = "config/experiment.yaml",
    elite_log_root: str | Path = "logs/task_d_probe/probe_logs",
    output_dir: str | Path = "logs/task_l_proxy_refine",
    negative_record_target: int = 10_000,
    negative_probe_probability: float = 0.85,
    negative_batch_game_count: int = 100,
    training_epochs: int = DEFAULT_VALUE_PROXY_EPOCHS,
    target_mode: str = VALUE_PROXY_TARGET_PHI,
    alignment_game_count: int = 100,
    alignment_sample_size: int = 20,
    alignment_sample_seed: int = 42,
    rollout_samples: int = 200,
    max_workers: int | None = None,
) -> dict[str, object]:
    """作用: 执行 Task L 负采样增强、混合重训与对齐对比。"""
    settings = load_settings(config_file=config_file)
    task_settings = _build_probe_experiment_settings(settings)
    negative_settings = _build_task_l_negative_settings(settings, probe_probability=negative_probe_probability)

    elite_log_root_path = Path(elite_log_root)
    output_path = Path(output_dir)
    negative_log_dir = output_path / "negative_logs"
    elite_model_dir = output_path / "elite_model"
    mixed_model_dir = output_path / "mixed_model"

    negative_summary = generate_negative_logs_until_records(
        settings=negative_settings,
        output_dir=negative_log_dir,
        record_target=negative_record_target,
        batch_game_count=negative_batch_game_count,
    )

    elite_metrics = train_value_proxy(
        log_root=elite_log_root_path,
        output_dir=elite_model_dir,
        target_mode=target_mode,
        epochs=training_epochs,
    )
    mixed_metrics = train_value_proxy(
        log_roots=[elite_log_root_path, Path(negative_summary["negative_log_dir"])],
        output_dir=mixed_model_dir,
        target_mode=target_mode,
        epochs=training_epochs,
        model_filename="value_proxy_mlp_v2.pt",
        metrics_filename="value_proxy_metrics_v2.json",
    )

    selected_run_id, alignment_logs = _select_latest_probe_logs(
        log_root=elite_log_root_path,
        game_count=alignment_game_count,
    )
    elite_alignment = run_proxy_alignment_for_model(
        settings=task_settings,
        log_paths=alignment_logs,
        model_path=elite_metrics["model_path"],
        target_mode=target_mode,
        rollout_samples=rollout_samples,
        sample_size=alignment_sample_size,
        sample_seed=alignment_sample_seed,
        max_workers=max_workers,
    )
    mixed_alignment = run_proxy_alignment_for_model(
        settings=task_settings,
        log_paths=alignment_logs,
        model_path=mixed_metrics["model_path"],
        target_mode=target_mode,
        rollout_samples=rollout_samples,
        sample_size=alignment_sample_size,
        sample_seed=alignment_sample_seed,
        max_workers=max_workers,
    )

    comparison = {
        "selected_run_id": selected_run_id,
        "alignment_log_count": len(alignment_logs),
        "negative_log_dir": str(negative_log_dir),
        "negative_record_count": int(negative_summary["record_count"]),
        "negative_game_count": int(negative_summary["game_count"]),
        "negative_probe_probability": float(negative_probe_probability),
        "target_mode": target_mode,
        "elite_model_path": str(elite_metrics["model_path"]),
        "mixed_model_path": str(mixed_metrics["model_path"]),
        "elite_val_mse": float(elite_metrics.get("val_mse", 0.0)),
        "mixed_val_mse": float(mixed_metrics.get("val_mse", 0.0)),
        "elite_alignment": elite_alignment,
        "mixed_alignment": mixed_alignment,
    }

    report_path = output_path / "proxy_refine_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        **comparison,
        "report_path": str(report_path),
    }


def main() -> None:
    summary = run_task_l_proxy_refine_pipeline()
    print("Task L proxy refine pipeline finished:", summary)


if __name__ == "__main__":
    main()
