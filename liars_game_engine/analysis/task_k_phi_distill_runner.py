from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from liars_game_engine.analysis.train_value_proxy import (
    DEFAULT_VALUE_PROXY_EPOCHS,
    VALUE_PROXY_TARGET_PHI,
    build_value_proxy_feature_context,
    encode_value_proxy_features,
    train_value_proxy,
)


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
    log_root: Path | str,
    output_dir: Path | str = "logs/task_l_task_k_distill",
    epochs: int = DEFAULT_VALUE_PROXY_EPOCHS,
    dataset_filename: str = "task_k_phi_dataset.jsonl",
    model_filename: str = "value_proxy_mlp_distill.pt",
    metrics_filename: str = "value_proxy_metrics_distill.json",
) -> dict[str, object]:
    """作用: 直接读取 Task K shapley_value 日志并启动 phi 监督训练。"""
    root = Path(log_root)
    output_path = Path(output_dir)
    samples = extract_task_k_phi_dataset(root)
    if not samples:
        raise RuntimeError(f"No Task K shapley_value/phi samples found under {root}")

    dataset_path = export_task_k_phi_dataset(samples, output_path / dataset_filename)
    metrics = train_value_proxy(
        log_root=root,
        output_dir=output_path,
        target_mode=VALUE_PROXY_TARGET_PHI,
        epochs=epochs,
        model_filename=model_filename,
        metrics_filename=metrics_filename,
    )

    return {
        "log_root": str(root),
        "sample_count": len(samples),
        "dataset_path": str(dataset_path),
        "target_mode": VALUE_PROXY_TARGET_PHI,
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a direct-phi proxy from Task K JSONL logs.")
    parser.add_argument("log_root", help="Directory containing Task K JSONL logs with shapley_value fields.")
    parser.add_argument("--output-dir", default="logs/task_l_task_k_distill")
    parser.add_argument("--epochs", type=int, default=DEFAULT_VALUE_PROXY_EPOCHS)
    args = parser.parse_args()

    summary = run_task_k_phi_training_pipeline(
        log_root=args.log_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
