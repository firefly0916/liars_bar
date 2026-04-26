from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from liars_game_engine.analysis.shapley_analyzer import ProxyValuePredictor
from liars_game_engine.analysis.task_l_proxy_refine_runner import run_proxy_alignment_for_model
from liars_game_engine.analysis.train_value_proxy import (
    DEFAULT_VALUE_PROXY_EPOCHS,
    VALUE_PROXY_TARGET_PHI,
    build_value_proxy_feature_context,
    encode_value_proxy_features,
    train_value_proxy,
)
from liars_game_engine.config.loader import load_settings


@dataclass
class TaskKPhiSample:
    game_id: str
    turn: int
    player_id: str
    skill_name: str
    action: dict[str, object]
    phi: float
    state_features: dict[str, object]
    normalized_features: list[float]
    source_log: str


def _append_progress(progress_log: Path, stage: str, **fields: object) -> None:
    progress_log.parent.mkdir(parents=True, exist_ok=True)
    line = " ".join([f"stage={stage}", *[f"{key}={value}" for key, value in fields.items()]])
    with progress_log.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def _normalize_log_root(log_root: Path | str) -> Path:
    root = Path(log_root)
    attributed_dir = root / "attributed_logs"
    if attributed_dir.is_dir():
        return attributed_dir
    return root


def _candidate_task_k_search_roots(workspace_root: Path | str | None = None) -> list[Path]:
    if workspace_root is None:
        root = Path.cwd()
    else:
        root = Path(workspace_root)

    candidates = [root]
    if root.parent.name == ".worktrees":
        candidates.append(root.parent.parent)

    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_candidates.append(resolved)
    return unique_candidates


def _discover_latest_task_k_attributed_logs(workspace_root: Path | str | None = None) -> Path:
    discovered: list[Path] = []
    for search_root in _candidate_task_k_search_roots(workspace_root):
        task_k_root = search_root / "logs" / "task_k_gold"
        if not task_k_root.is_dir():
            continue
        for attributed_dir in sorted(task_k_root.glob("*/attributed_logs")):
            if attributed_dir.is_dir() and any(attributed_dir.glob("*.jsonl")):
                discovered.append(attributed_dir)

    if not discovered:
        raise RuntimeError("No Task K attributed_logs directory found under logs/task_k_gold")
    return max(discovered, key=lambda path: path.parent.name)


def _resolve_task_k_log_root(
    log_root: Path | str | None,
    workspace_root: Path | str | None = None,
) -> Path:
    if log_root is None:
        return _discover_latest_task_k_attributed_logs(workspace_root=workspace_root)
    resolved = _normalize_log_root(log_root)
    if not resolved.is_dir():
        raise RuntimeError(f"Task K log root does not exist: {resolved}")
    if not any(resolved.rglob("*.jsonl")):
        raise RuntimeError(f"No Task K JSONL files found under {resolved}")
    return resolved


def _build_high_risk_challenge_audit(
    samples: list[TaskKPhiSample],
    predictor: ProxyValuePredictor,
    risk_threshold: float = 1.0 / 3.0,
) -> dict[str, object]:
    subset = [
        sample
        for sample in samples
        if str(sample.action.get("type", "")) == "challenge"
        and float(sample.state_features.get("death_probability", 0.0) or 0.0) > float(risk_threshold)
    ]
    if not subset:
        return {
            "sample_count": 0,
            "risk_threshold": float(risk_threshold),
            "actual_negative_rate": 0.0,
            "predicted_negative_rate": 0.0,
            "sign_agreement_rate": 0.0,
            "mean_actual_phi": 0.0,
            "mean_predicted_phi": 0.0,
            "mae": 0.0,
        }

    actual_phi = [float(sample.phi) for sample in subset]
    predicted_phi = [float(predictor.predict_state_features(sample.state_features)) for sample in subset]
    sign_matches = [
        (actual_value < 0.0) == (predicted_value < 0.0)
        for actual_value, predicted_value in zip(actual_phi, predicted_phi)
    ]

    return {
        "sample_count": len(subset),
        "risk_threshold": float(risk_threshold),
        "actual_negative_rate": sum(1 for value in actual_phi if value < 0.0) / len(actual_phi),
        "predicted_negative_rate": sum(1 for value in predicted_phi if value < 0.0) / len(predicted_phi),
        "sign_agreement_rate": sum(1 for matched in sign_matches if matched) / len(sign_matches),
        "mean_actual_phi": sum(actual_phi) / len(actual_phi),
        "mean_predicted_phi": sum(predicted_phi) / len(predicted_phi),
        "mae": sum(abs(left - right) for left, right in zip(actual_phi, predicted_phi)) / len(actual_phi),
    }


def _extract_phi_value(record: dict[str, object]) -> float | None:
    for key in ("shapley_value", "phi"):
        value = record.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def extract_task_k_phi_dataset(log_root: Path | str) -> list[TaskKPhiSample]:
    """作用: 从 Task K JSONL 中对齐提取 (State, Action, Phi) 蒸馏样本。"""
    root = Path(log_root)
    samples: list[TaskKPhiSample] = []

    for log_path in sorted(root.rglob("*.jsonl")):
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                continue

            phi = _extract_phi_value(record)
            player_id = str(record.get("player_id", ""))
            action = record.get("action")
            if phi is None or not player_id or not isinstance(action, dict):
                continue

            state_features = build_value_proxy_feature_context(
                state_features=record.get("state_features") if isinstance(record.get("state_features"), dict) else None,
                observation=record.get("observation") if isinstance(record.get("observation"), dict) else None,
                player_id=player_id,
                action=action,
            )
            samples.append(
                TaskKPhiSample(
                    game_id=log_path.stem,
                    turn=int(record.get("turn", 0)),
                    player_id=player_id,
                    skill_name=str(record.get("skill_name", "Unknown")),
                    action=dict(action),
                    phi=phi,
                    state_features=state_features,
                    normalized_features=encode_value_proxy_features(state_features),
                    source_log=str(log_path),
                )
            )

    return samples


def export_task_k_phi_dataset(samples: list[TaskKPhiSample], output_path: Path | str) -> Path:
    """作用: 将 Task K 蒸馏样本导出为可复查的 JSONL。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(asdict(sample), ensure_ascii=False) for sample in samples) + "\n", encoding="utf-8")
    return path


def run_task_k_phi_training_pipeline(
    log_root: Path | str | None = None,
    output_dir: Path | str = "logs/task_l_task_k_distill",
    config_file: Path | str = "config/experiment.yaml",
    workspace_root: Path | str | None = None,
    epochs: int = DEFAULT_VALUE_PROXY_EPOCHS,
    rollout_samples: int = 200,
    alignment_sample_size: int = 20,
    alignment_sample_seed: int = 42,
    max_workers: int | None = None,
    dataset_filename: str = "task_k_phi_dataset.jsonl",
    model_filename: str = "value_proxy_mlp_distill.pt",
    metrics_filename: str = "value_proxy_metrics_distill.json",
    progress_filename: str = "progress.log",
    diagnostics_filename: str = "diagnostics.jsonl",
    report_filename: str = "task_k_phi_audit_report.json",
) -> dict[str, object]:
    """作用: 直接读取 Task K shapley_value 日志并启动 phi 监督训练。"""
    output_path = Path(output_dir)
    progress_log = output_path / progress_filename
    diagnostics_log = output_path / diagnostics_filename
    root = _resolve_task_k_log_root(log_root=log_root, workspace_root=workspace_root)
    _append_progress(progress_log, "resolve_logs", log_root=root)

    samples = extract_task_k_phi_dataset(root)
    if not samples:
        raise RuntimeError(f"No Task K shapley_value/phi samples found under {root}")
    _append_progress(progress_log, "extract_dataset", sample_count=len(samples))

    dataset_path = export_task_k_phi_dataset(samples, output_path / dataset_filename)
    _append_progress(progress_log, "export_dataset", dataset_path=dataset_path)
    metrics = train_value_proxy(
        log_root=root,
        output_dir=output_path,
        target_mode=VALUE_PROXY_TARGET_PHI,
        epochs=epochs,
        model_filename=model_filename,
        metrics_filename=metrics_filename,
    )
    _append_progress(progress_log, "train_proxy", val_mse=metrics.get("val_mse", 0.0), model_path=metrics["model_path"])

    settings = load_settings(config_file=config_file)
    alignment = run_proxy_alignment_for_model(
        settings=settings,
        log_paths=sorted(root.rglob("*.jsonl")),
        model_path=metrics["model_path"],
        target_mode=VALUE_PROXY_TARGET_PHI,
        rollout_samples=rollout_samples,
        sample_size=alignment_sample_size,
        sample_seed=alignment_sample_seed,
        max_workers=max_workers,
    )
    _append_progress(
        progress_log,
        "alignment_audit",
        pearson_correlation=alignment.get("pearson_correlation", 0.0),
        mae=alignment.get("mae", 0.0),
    )

    predictor = ProxyValuePredictor(model_path=metrics["model_path"], output_mode=VALUE_PROXY_TARGET_PHI)
    high_risk_challenge_audit = _build_high_risk_challenge_audit(samples=samples, predictor=predictor)
    _append_progress(
        progress_log,
        "high_risk_challenge_audit",
        sample_count=high_risk_challenge_audit["sample_count"],
        predicted_negative_rate=high_risk_challenge_audit["predicted_negative_rate"],
    )

    diagnostics_log.parent.mkdir(parents=True, exist_ok=True)
    with diagnostics_log.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "alignment", **alignment}, ensure_ascii=False) + "\n")
        handle.write(
            json.dumps({"type": "high_risk_challenge_audit", **high_risk_challenge_audit}, ensure_ascii=False) + "\n"
        )

    report = {
        "log_root": str(root),
        "dataset_path": str(dataset_path),
        "target_mode": VALUE_PROXY_TARGET_PHI,
        "sample_count": len(samples),
        "training_metrics": metrics,
        "alignment": alignment,
        "high_risk_challenge_audit": high_risk_challenge_audit,
        "progress_log": str(progress_log),
        "diagnostics_log": str(diagnostics_log),
    }
    report_path = output_path / report_filename
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _append_progress(progress_log, "completed", report_path=report_path)

    return {
        "log_root": str(root),
        "sample_count": len(samples),
        "dataset_path": str(dataset_path),
        "target_mode": VALUE_PROXY_TARGET_PHI,
        "alignment": alignment,
        "high_risk_challenge_audit": high_risk_challenge_audit,
        "progress_log": str(progress_log),
        "diagnostics_log": str(diagnostics_log),
        "report_path": str(report_path),
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a direct-phi proxy from Task K JSONL logs.")
    parser.add_argument(
        "log_root",
        nargs="?",
        default=None,
        help="Directory containing Task K JSONL logs with shapley_value fields. Defaults to latest logs/task_k_gold/*/attributed_logs.",
    )
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--output-dir", default="logs/task_l_task_k_distill")
    parser.add_argument("--epochs", type=int, default=DEFAULT_VALUE_PROXY_EPOCHS)
    parser.add_argument("--rollout-samples", type=int, default=200)
    parser.add_argument("--alignment-sample-size", type=int, default=20)
    parser.add_argument("--alignment-sample-seed", type=int, default=42)
    parser.add_argument("--max-workers", type=int, default=None)
    args = parser.parse_args()

    summary = run_task_k_phi_training_pipeline(
        log_root=args.log_root,
        config_file=args.config,
        output_dir=args.output_dir,
        epochs=args.epochs,
        rollout_samples=args.rollout_samples,
        alignment_sample_size=args.alignment_sample_size,
        alignment_sample_seed=args.alignment_sample_seed,
        max_workers=args.max_workers,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
