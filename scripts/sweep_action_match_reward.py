from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_weight_values(raw: str) -> list[float]:
    text = str(raw).strip()
    if not text:
        raise ValueError("weight specification must not be empty")
    if ":" in text:
        parts = [part.strip() for part in text.split(":")]
        if len(parts) != 3:
            raise ValueError("range syntax must be start:end:step")
        start, end, step = (float(part) for part in parts)
        if step <= 0:
            raise ValueError("range step must be positive")
        values: list[float] = []
        current = start
        while current <= end + 1e-9:
            values.append(round(current, 6))
            current += step
        return values
    return [round(float(part.strip()), 6) for part in text.split(",") if part.strip()]


def build_run_slug(weight: float) -> str:
    normalized = int(round(float(weight) * 100))
    return f"amrw-{normalized:03d}"


def run_command(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    subprocess.run(command, cwd=str(cwd), env=merged_env, check=True)


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_sweep(
    *,
    weights: list[float],
    feat_repo_root: Path,
    task_m_repo_root: Path,
    dataset_path: Path,
    policy_model_path: Path,
    proxy_model_path: Path,
    task_m_config_path: Path,
    output_root: Path,
    task_m_games: int,
    smoke_steps: int,
    local_llm_max_new_tokens: int,
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, object]] = []

    for weight in weights:
        slug = build_run_slug(weight)
        train_root = output_root / slug / "train"
        task_m_root = output_root / slug / "task_m"
        train_summary_path = train_root / "summary.json"
        train_command = [
            sys.executable,
            "scripts/train_savi_alignment.py",
            str(dataset_path),
            "--policy-model-path",
            str(policy_model_path),
            "--model-path",
            str(proxy_model_path),
            "--device",
            "cuda",
            "--torch-dtype",
            "bf16",
            "--load-in-4bit",
            "--group-size",
            "8",
            "--steps",
            str(smoke_steps),
            "--action-match-reward-weight",
            str(weight),
            "--checkpoint-dir",
            str(train_root / "checkpoints"),
            "--save-every-steps",
            "5",
            "--save-final-adapter",
            "--output-path",
            str(train_summary_path),
        ]
        run_command(train_command, cwd=feat_repo_root)
        train_summary = _load_json(train_summary_path)

        adapter_path = Path(str(train_summary["final_adapter_path"]))
        task_m_command = [
            sys.executable,
            "scripts/run_llm_drill.py",
            "--config",
            str(task_m_config_path),
            "--games",
            str(task_m_games),
            "--log-dir",
            str(task_m_root),
        ]
        task_m_env = {
            "LOCAL_LLM_DEVICE": "cuda",
            "LOCAL_LLM_LOCAL_FILES_ONLY": "1",
            "LOCAL_LLM_MAX_NEW_TOKENS": str(local_llm_max_new_tokens),
            "LOCAL_LLM_ADAPTER_PATH": str(adapter_path),
        }
        run_command(task_m_command, cwd=task_m_repo_root, env=task_m_env)
        task_m_summary = _load_json(task_m_root / "summary.json")

        runs.append(
            {
                "action_match_reward_weight": float(weight),
                "slug": slug,
                "train_summary_path": str(train_summary_path),
                "task_m_summary_path": str(task_m_root / "summary.json"),
                "train": {
                    "completed_steps": train_summary.get("completed_steps"),
                    "effective_step_count": train_summary.get("step_metrics", {}).get("effective_step_count"),
                    "signal_density_rate": train_summary.get("step_metrics", {}).get("signal_density_rate"),
                    "final_adapter_path": str(adapter_path),
                },
                "task_m": {
                    "total_games": task_m_summary.get("total_games"),
                    "llm_turn_count": task_m_summary.get("llm_turn_count"),
                    "parse_error_rate": task_m_summary.get("parse_error_rate"),
                    "resolution_adjustment_rate": task_m_summary.get("resolution_adjustment_rate"),
                    "parse_error_count": task_m_summary.get("parse_error_count"),
                    "resolution_adjustment_count": task_m_summary.get("resolution_adjustment_count"),
                },
            }
        )

    summary = {
        "weights": [float(weight) for weight in weights],
        "smoke_steps": int(smoke_steps),
        "task_m_games": int(task_m_games),
        "runs": runs,
    }
    (output_root / "sweep_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep action-match reward weights for SAVI smoke training and Task M eval.")
    parser.add_argument("--weights", required=True, help="CSV weights like 0.4,0.45,0.5 or range start:end:step.")
    parser.add_argument("--feat-repo-root", default=".")
    parser.add_argument("--task-m-repo-root", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--policy-model-path", required=True)
    parser.add_argument("--proxy-model-path", required=True)
    parser.add_argument("--task-m-config-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--task-m-games", type=int, default=1)
    parser.add_argument("--smoke-steps", type=int, default=10)
    parser.add_argument("--local-llm-max-new-tokens", type=int, default=192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_sweep(
        weights=parse_weight_values(args.weights),
        feat_repo_root=Path(args.feat_repo_root).resolve(),
        task_m_repo_root=Path(args.task_m_repo_root).resolve(),
        dataset_path=Path(args.dataset_path).resolve(),
        policy_model_path=Path(args.policy_model_path).resolve(),
        proxy_model_path=Path(args.proxy_model_path).resolve(),
        task_m_config_path=Path(args.task_m_config_path).resolve(),
        output_root=Path(args.output_root).resolve(),
        task_m_games=args.task_m_games,
        smoke_steps=args.smoke_steps,
        local_llm_max_new_tokens=args.local_llm_max_new_tokens,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
