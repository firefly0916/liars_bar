from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from liars_game_engine.analysis.formal_eval_driver import (
    build_formal_eval_plan,
    formal_eval_plan_to_json,
    load_checkpoint_tags,
    load_selection_payload,
    reset_experiment_root,
    render_formal_eval_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run manual formal evaluation on selected checkpoint tags from a diagnostic or pipeline run."
    )
    parser.add_argument("run_root", help="Experiment root containing train/checkpoints/")
    parser.add_argument("--checkpoint-tag", action="append", default=[], help="Checkpoint tag to run. Repeatable.")
    parser.add_argument(
        "--selection-json",
        default=None,
        help="Optional screening selection JSON used when checkpoint tags are not passed explicitly.",
    )
    parser.add_argument("--games", type=int, default=100, help="Number of games per checkpoint.")
    parser.add_argument(
        "--output-dir-name",
        default="formal_eval_manual",
        help="Output directory name created under run_root.",
    )
    parser.add_argument(
        "--eval-repo",
        default=os.environ.get("EVAL_REPO", "/root/liars_bar"),
        help="Repository containing scripts/run_llm_drill.py",
    )
    parser.add_argument(
        "--audit-repo",
        default=os.environ.get("AUDIT_REPO", "/root/liars_bar_dev_proxy_refine"),
        help="Repository containing scripts/audit_llm_behavior.py",
    )
    parser.add_argument(
        "--taskm-config",
        default=os.environ.get("TASKM_CONFIG", "/root/liars_bar/config/experiment.yaml"),
        help="Task M config path.",
    )
    parser.add_argument(
        "--audit-config",
        default=os.environ.get("AUDIT_CONFIG", "/root/liars_bar_dev_proxy_refine/config/experiment.yaml"),
        help="Audit config path.",
    )
    parser.add_argument(
        "--proxy-model-path",
        default=os.environ.get("PROXY_MODEL_PATH", "/root/liars_bar_feat_grpo/models/proxy/value_proxy_mlp_distill.pt"),
        help="Proxy model path used by the audit step.",
    )
    parser.add_argument(
        "--expected-llm-model",
        default=os.environ.get("EVAL_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct"),
        help="Fail fast unless Task M config uses this exact llm model.",
    )
    parser.add_argument("--phi-threshold", type=float, default=-0.1)
    parser.add_argument("--potential-point-threshold", type=float, default=0.15)
    parser.add_argument("--llm-player-id", default="p1")
    parser.add_argument("--local-llm-device", default=os.environ.get("LOCAL_LLM_DEVICE", "cuda"))
    parser.add_argument("--local-llm-max-new-tokens", type=int, default=192)
    parser.add_argument("--dry-run", action="store_true", help="Only print the execution plan.")
    parser.add_argument("--format", choices=["json", "markdown"], default="markdown")
    return parser.parse_args()


def _write_log(log_path: Path, content: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(content, encoding="utf-8")


def _run_command(command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            handle.write(line)
            handle.flush()
        process.stdout.close()
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def main() -> int:
    args = parse_args()
    selection_payload = load_selection_payload(args.selection_json) if args.selection_json else None
    checkpoint_tags = load_checkpoint_tags(
        selection_payload=selection_payload,
        explicit_tags=list(args.checkpoint_tag),
    )
    if not checkpoint_tags:
        raise RuntimeError("No checkpoint tags were provided and selection JSON did not contain any selected tags.")

    plan = build_formal_eval_plan(
        run_root=Path(args.run_root),
        checkpoint_tags=checkpoint_tags,
        output_dir_name=args.output_dir_name,
        games=args.games,
    )
    rendered = render_formal_eval_markdown(plan) if args.format == "markdown" else formal_eval_plan_to_json(plan)
    if args.dry_run:
        print(rendered, flush=True)
        return 0

    print(rendered, flush=True)
    eval_repo = Path(args.eval_repo)
    audit_repo = Path(args.audit_repo)
    taskm_script = eval_repo / "scripts" / "run_llm_drill.py"
    audit_script = audit_repo / "scripts" / "audit_llm_behavior.py"

    for entry in plan["entries"]:
        if not isinstance(entry, dict):
            continue
        experiment_root = reset_experiment_root(Path(str(entry["experiment_root"])))
        task_m_root = experiment_root / "task_m"
        audit_root = experiment_root / "task_1_1"
        task_m_root.mkdir(parents=True, exist_ok=True)
        audit_root.mkdir(parents=True, exist_ok=True)

        llm_env = dict(os.environ)
        llm_env["LOCAL_LLM_DEVICE"] = args.local_llm_device
        llm_env["LOCAL_LLM_LOCAL_FILES_ONLY"] = "1"
        llm_env["LOCAL_LLM_MAX_NEW_TOKENS"] = str(args.local_llm_max_new_tokens)
        llm_env["LOCAL_LLM_ADAPTER_PATH"] = str(entry["checkpoint_path"])

        print(
            f"[formal-eval] start task_m tag={entry['tag']} games={args.games} log={entry['task_m_log_path']}",
            flush=True,
        )
        _run_command(
            [
                sys.executable,
                str(taskm_script),
                "--config",
                args.taskm_config,
                "--expected-llm-model",
                args.expected_llm_model,
                "--games",
                str(args.games),
                "--log-dir",
                str(task_m_root),
            ],
            cwd=eval_repo,
            env=llm_env,
            log_path=Path(str(entry["task_m_log_path"])),
        )
        print(f"[formal-eval] done task_m tag={entry['tag']}", flush=True)

        print(
            f"[formal-eval] start audit tag={entry['tag']} log={entry['audit_log_path']}",
            flush=True,
        )
        _run_command(
            [
                sys.executable,
                str(audit_script),
                str(task_m_root),
                "--model-path",
                args.proxy_model_path,
                "--output-dir",
                str(audit_root),
                "--phi-threshold",
                str(args.phi_threshold),
                "--potential-point-threshold",
                str(args.potential_point_threshold),
                "--llm-player-id",
                args.llm_player_id,
                "--config-file",
                args.audit_config,
                "--summary-path",
                str(audit_root / "summary.json"),
            ],
            cwd=audit_repo,
            env=dict(os.environ),
            log_path=Path(str(entry["audit_log_path"])),
        )
        print(f"[formal-eval] done audit tag={entry['tag']}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
