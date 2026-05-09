from __future__ import annotations

import json
from pathlib import Path
import shutil


def load_selection_payload(path: Path | str) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Selection payload must be a JSON object: {path}")
    return payload


def _dedupe_preserve_order(tags: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for tag in tags:
        normalized = str(tag).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def load_checkpoint_tags(
    *,
    selection_payload: dict[str, object] | None,
    explicit_tags: list[str] | None,
) -> list[str]:
    explicit = _dedupe_preserve_order([str(tag) for tag in (explicit_tags or [])])
    if explicit:
        return explicit

    selected = []
    payload = selection_payload or {}
    for item in payload.get("selected", []):
        if not isinstance(item, dict):
            continue
        tag = str(item.get("tag", "")).strip()
        if tag:
            selected.append(tag)
    return _dedupe_preserve_order(selected)


def _resolve_checkpoint_root(run_root: Path) -> Path:
    nested = run_root / "train" / "checkpoints"
    if nested.is_dir():
        return nested
    flat = run_root / "checkpoints"
    if flat.is_dir():
        return flat
    return nested


def reset_experiment_root(experiment_root: Path | str) -> Path:
    root = Path(experiment_root)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def build_formal_eval_plan(
    *,
    run_root: Path | str,
    checkpoint_tags: list[str],
    output_dir_name: str = "formal_eval_manual",
    games: int = 100,
) -> dict[str, object]:
    root = Path(run_root)
    checkpoint_root = _resolve_checkpoint_root(root)
    tags = _dedupe_preserve_order([str(tag) for tag in checkpoint_tags])
    formal_root = root / output_dir_name
    entries: list[dict[str, object]] = []
    for tag in tags:
        experiment_root = formal_root / tag
        task_m_root = experiment_root / "task_m"
        audit_root = experiment_root / "task_1_1"
        entries.append(
            {
                "tag": tag,
                "games": int(games),
                "checkpoint_path": str(checkpoint_root / tag),
                "experiment_root": str(experiment_root),
                "task_m_root": str(task_m_root),
                "audit_root": str(audit_root),
                "task_m_log_path": str(experiment_root / "task_m_stdout.log"),
                "audit_log_path": str(experiment_root / "task_1_1_stdout.log"),
            }
        )
    return {
        "run_root": str(root),
        "checkpoint_root": str(checkpoint_root),
        "checkpoint_tags": tags,
        "formal_root": str(formal_root),
        "games": int(games),
        "entries": entries,
    }


def render_formal_eval_markdown(plan: dict[str, object]) -> str:
    lines = ["# Formal Eval Plan", ""]
    lines.append(f"- run_root: {plan.get('run_root')}")
    lines.append(f"- checkpoint_root: {plan.get('checkpoint_root')}")
    lines.append(f"- formal_root: {plan.get('formal_root')}")
    lines.append(f"- games: {plan.get('games')}")
    lines.append("")
    lines.append("| tag | checkpoint_path | experiment_root |")
    lines.append("| --- | --- | --- |")
    for entry in plan.get("entries", []):
        if not isinstance(entry, dict):
            continue
        lines.append(
            f"| {entry.get('tag')} | {entry.get('checkpoint_path')} | {entry.get('experiment_root')} |"
        )
    lines.append("")
    return "\n".join(lines)


def formal_eval_plan_to_json(plan: dict[str, object]) -> str:
    return json.dumps(plan, ensure_ascii=False, indent=2)
