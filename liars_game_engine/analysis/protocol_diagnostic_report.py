from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_artifact_path(run_root: Path, nested_parts: tuple[str, ...], flat_name: str) -> Path:
    nested_path = run_root.joinpath(*nested_parts)
    if nested_path.is_file():
        return nested_path
    return run_root / flat_name


def _sha256_if_readable(path: Path | None) -> str | None:
    try:
        if path is None or not path.is_file():
            return None
    except (OSError, PermissionError):
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except (OSError, PermissionError):
        return None
    return digest.hexdigest()


def _git_metadata(repo_root: Path) -> dict[str, object]:
    def _run_git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    branch = _run_git("branch", "--show-current")
    head = _run_git("rev-parse", "HEAD")
    status = _run_git("status", "--short")
    return {
        "repo_root": str(repo_root),
        "branch": branch,
        "head": head,
        "dirty": bool(status),
    }


def summarize_training_signal(train_summary: dict[str, object]) -> dict[str, object]:
    step_metrics = train_summary.get("step_metrics", {})
    if not isinstance(step_metrics, dict):
        step_metrics = {}
    step_summaries = train_summary.get("step_summaries", [])
    if not isinstance(step_summaries, list):
        step_summaries = []

    mask_nonzero_step_count = 0
    mask_hit_step_count = 0
    reasoning_action_mismatch_count = 0
    for step in step_summaries:
        if not isinstance(step, dict):
            continue
        if bool(step.get("reasoning_action_mismatch", False)):
            reasoning_action_mismatch_count += 1
        mask_metrics = step.get("mask_metrics", {})
        if not isinstance(mask_metrics, dict):
            mask_metrics = {}
        if _safe_int(mask_metrics.get("non_zero_mask_count", 0)) > 0:
            mask_nonzero_step_count += 1
        if _safe_int(mask_metrics.get("mask_hit_count", 0)) > 0:
            mask_hit_step_count += 1

    warnings: list[str] = []
    if step_summaries and mask_nonzero_step_count == 0:
        warnings.append("mask_never_activated")
    if _safe_int(step_metrics.get("signalless_step_count", 0)) > 0:
        warnings.append("signalless_steps_present")
    if _safe_int(step_metrics.get("idle_step_count", 0)) > 0:
        warnings.append("idle_steps_present")

    total_steps = max(1, len(step_summaries))
    return {
        "requested_steps": _safe_int(train_summary.get("requested_steps", 0)),
        "completed_steps": _safe_int(train_summary.get("completed_steps", 0)),
        "effective_step_count": _safe_int(step_metrics.get("effective_step_count", 0)),
        "idle_step_count": _safe_int(step_metrics.get("idle_step_count", 0)),
        "signalless_step_count": _safe_int(step_metrics.get("signalless_step_count", 0)),
        "signal_density_rate": _safe_float(step_metrics.get("signal_density_rate", 0.0)),
        "average_hicra_mask_intensity": _safe_float(step_metrics.get("average_hicra_mask_intensity", 0.0)),
        "mask_nonzero_step_count": mask_nonzero_step_count,
        "mask_hit_step_count": mask_hit_step_count,
        "reasoning_action_mismatch_rate": reasoning_action_mismatch_count / total_steps,
        "warnings": warnings,
    }


def _selection_row_metric(row: dict[str, object], path: tuple[str, ...], default: float = 0.0) -> float:
    current: object = row
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return _safe_float(current, default=default)


def _normalize_selection_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "tag": str(row.get("tag", "")),
        "access_ok": bool(row.get("access_ok", False)),
        "risk_ok": bool(row.get("risk_ok", False)),
        "win_rate": (
            _safe_float(row.get("win_rate"), default=-1.0)
            if "win_rate" in row
            else _selection_row_metric(row, ("scorecard", "auxiliary", "win_rate"), default=0.0)
        ),
        "challenge_accuracy": (
            _safe_float(row.get("challenge_accuracy"), default=0.0)
            if "challenge_accuracy" in row
            else _selection_row_metric(row, ("scorecard", "behavior", "challenge_accuracy"), default=0.0)
        ),
        "bluff_efficiency": (
            _safe_float(row.get("bluff_efficiency"), default=0.0)
            if "bluff_efficiency" in row
            else _selection_row_metric(row, ("scorecard", "behavior", "bluff_efficiency"), default=0.0)
        ),
        "avg_ev_gap": _safe_float(row.get("avg_ev_gap"), default=0.0),
        "resolution_adjustment_rate": _safe_float(row.get("resolution_adjustment_rate"), default=0.0),
        "conflict_count": _safe_int(row.get("conflict_count", 0)),
        "turn_count_distance": _safe_int(row.get("turn_count_distance", 0)),
        "max_ev_gap": _safe_float(row.get("max_ev_gap"), default=0.0),
        "top1_match_rate": _safe_float(row.get("top1_match_rate"), default=0.0),
        "mean_spearman_rank_correlation": _safe_float(
            row.get("mean_spearman_rank_correlation"), default=0.0
        ),
        "mean_rollout_regret": _safe_float(row.get("mean_rollout_regret"), default=0.0),
        "mean_chosen_action_rollout_rank": _safe_float(
            row.get("mean_chosen_action_rollout_rank"), default=0.0
        ),
        "proxy_calibration_available": bool(row.get("proxy_calibration_available", False)),
    }


def _gameplay_sort_key(row: dict[str, object]) -> tuple[object, ...]:
    calibration_available = bool(row.get("proxy_calibration_available", False))
    top1_match_rate = _safe_float(row.get("top1_match_rate"), 0.0)
    mean_spearman = _safe_float(row.get("mean_spearman_rank_correlation"), 0.0)
    mean_rollout_regret = _safe_float(row.get("mean_rollout_regret"), 0.0)
    mean_chosen_rank = _safe_float(row.get("mean_chosen_action_rollout_rank"), 0.0)
    return (
        0 if bool(row["access_ok"]) else 1,
        0 if bool(row["risk_ok"]) else 1,
        -_safe_float(row["win_rate"], 0.0),
        0 if calibration_available else 1,
        -top1_match_rate,
        mean_rollout_regret if calibration_available else float("inf"),
        -mean_spearman if calibration_available else float("inf"),
        mean_chosen_rank if calibration_available else float("inf"),
        -_safe_float(row["challenge_accuracy"], 0.0),
        -_safe_float(row["bluff_efficiency"], 0.0),
        _safe_float(row["resolution_adjustment_rate"], 0.0),
        _safe_int(row["turn_count_distance"], 0),
        _safe_float(row["avg_ev_gap"], 0.0),
        str(row["tag"]),
    )


def summarize_shortlist_alignment(selection: dict[str, object]) -> dict[str, object]:
    all_rows_raw = selection.get("all_rows", [])
    selected_rows_raw = selection.get("selected", [])
    all_rows = [_normalize_selection_row(row) for row in all_rows_raw if isinstance(row, dict)]
    selected_tags = [
        str(row.get("tag", ""))
        for row in selected_rows_raw
        if isinstance(row, dict) and str(row.get("tag", "")).strip()
    ]
    top_k = max(1, _safe_int(selection.get("top_k", len(selected_tags) or 1), default=len(selected_tags) or 1))
    gameplay_top = sorted(all_rows, key=_gameplay_sort_key)[:top_k]
    gameplay_top_tags = [str(row["tag"]) for row in gameplay_top]
    selected_set = set(selected_tags)
    overlap_count = sum(1 for tag in gameplay_top_tags if tag in selected_set)

    warnings: list[str] = []
    mismatch_detected = selected_tags != gameplay_top_tags
    if mismatch_detected:
        warnings.append("shortlist_misaligned_with_gameplay")

    selected_lookup = {str(row["tag"]): row for row in all_rows}
    selected_best_win_rate = max(
        [_safe_float(selected_lookup[tag]["win_rate"], 0.0) for tag in selected_tags if tag in selected_lookup] or [0.0]
    )
    gameplay_best_win_rate = max([_safe_float(row["win_rate"], 0.0) for row in gameplay_top] or [0.0])
    if gameplay_best_win_rate > selected_best_win_rate:
        warnings.append("shortlist_missed_best_gameplay_checkpoint")

    version_hint = "current"
    if "selection_profile" not in selection or any("win_rate" not in row for row in all_rows_raw if isinstance(row, dict)):
        version_hint = "legacy"

    return {
        "selected_tags": selected_tags,
        "gameplay_top_tags": gameplay_top_tags,
        "overlap_count": overlap_count,
        "top_k": top_k,
        "mismatch_detected": mismatch_detected,
        "selected_best_win_rate": selected_best_win_rate,
        "gameplay_best_win_rate": gameplay_best_win_rate,
        "warnings": warnings,
        "selection_profile": str(selection.get("selection_profile", "")),
        "version_hint": version_hint,
    }


def _trajectory_leaderboard(rows: list[dict[str, object]], limit: int = 5) -> list[dict[str, object]]:
    normalized = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        normalized.append(
            {
                "tag": str(row.get("tag", "")),
                "access_ok": bool(row.get("access_ok", False)),
                "win_rate": _safe_float(row.get("win_rate", 0.0)),
                "avg_ev_gap": _safe_float(row.get("avg_ev_gap", 0.0)),
                "parse_error_rate": _safe_float(row.get("parse_error_rate", 0.0)),
            }
        )
    normalized.sort(key=lambda row: (0 if row["access_ok"] else 1, -row["win_rate"], row["avg_ev_gap"], row["tag"]))
    return normalized[: max(1, int(limit))]


def _leaderboard_from_selection(selection: dict[str, object], limit: int = 5) -> list[dict[str, object]]:
    rows = selection.get("all_rows", [])
    if not isinstance(rows, list):
        return []
    normalized = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        flattened = _normalize_selection_row(row)
        normalized.append(
            {
                "tag": flattened["tag"],
                "access_ok": flattened["access_ok"],
                "win_rate": flattened["win_rate"],
                "avg_ev_gap": flattened["avg_ev_gap"],
                "parse_error_rate": _safe_float(row.get("parse_error_rate", 0.0)),
            }
        )
    normalized.sort(key=lambda row: (0 if row["access_ok"] else 1, -row["win_rate"], row["avg_ev_gap"], row["tag"]))
    return normalized[: max(1, int(limit))]


def build_protocol_diagnostic_report(
    run_root: Path | str,
    *,
    calibration_summary_path: Path | str | None = None,
) -> dict[str, object]:
    root = Path(run_root)
    repo_root = Path(__file__).resolve().parents[2]

    train_summary_path = _resolve_artifact_path(root, ("train", "train_summary.json"), "train_summary.json")
    trajectory_report_path = _resolve_artifact_path(
        root,
        ("reports", "trajectory_report.json"),
        "trajectory_report.json",
    )
    shortlist_selection_path = _resolve_artifact_path(
        root,
        ("reports", "shortlist_selection.json"),
        "shortlist_selection.json",
    )

    train_summary = _load_json(train_summary_path) if train_summary_path.is_file() else {}
    if not isinstance(train_summary, dict):
        train_summary = {}
    training_signal = summarize_training_signal(train_summary) if train_summary else {}

    trajectory_rows = _load_json(trajectory_report_path) if trajectory_report_path.is_file() else []
    if not isinstance(trajectory_rows, list):
        trajectory_rows = []
    shortlist_selection = _load_json(shortlist_selection_path) if shortlist_selection_path.is_file() else {}
    if not isinstance(shortlist_selection, dict):
        shortlist_selection = {}

    shortlist_alignment = summarize_shortlist_alignment(shortlist_selection) if shortlist_selection else {}

    warnings: list[str] = []
    warnings.extend(training_signal.get("warnings", []) if isinstance(training_signal, dict) else [])
    warnings.extend(shortlist_alignment.get("warnings", []) if isinstance(shortlist_alignment, dict) else [])

    if trajectory_rows:
        final_rows = [row for row in trajectory_rows if isinstance(row, dict) and str(row.get("tag", "")) == "final"]
        if final_rows and not bool(final_rows[0].get("access_ok", False)):
            warnings.append("final_checkpoint_access_regression")

    proxy_model_path = None
    if train_summary:
        raw_proxy_path = train_summary.get("proxy_model_path")
        if raw_proxy_path is not None:
            proxy_model_path = Path(str(raw_proxy_path))

    calibration_summary = None
    if calibration_summary_path is not None:
        calibration_path = Path(calibration_summary_path)
        if calibration_path.is_file():
            calibration_summary = _load_json(calibration_path)

    deduped_warnings: list[str] = []
    seen_warnings: set[str] = set()
    for warning in warnings:
        if warning in seen_warnings:
            continue
        seen_warnings.add(warning)
        deduped_warnings.append(warning)

    return {
        "run_root": str(root),
        "paths": {
            "train_summary": str(train_summary_path),
            "trajectory_report": str(trajectory_report_path),
            "shortlist_selection": str(shortlist_selection_path),
            "calibration_summary": str(calibration_summary_path) if calibration_summary_path is not None else None,
        },
        "repo_metadata": _git_metadata(repo_root),
        "artifact_metadata": {
            "proxy_model_path": str(proxy_model_path) if proxy_model_path is not None else None,
            "proxy_model_sha256": _sha256_if_readable(proxy_model_path),
        },
        "training_signal": training_signal,
        "trajectory_leaderboard": (
            _trajectory_leaderboard(trajectory_rows)
            if trajectory_rows
            else _leaderboard_from_selection(shortlist_selection)
        ),
        "shortlist_alignment": shortlist_alignment,
        "selector_metadata": {
            "selection_profile": (
                str(shortlist_selection.get("selection_profile", "")) if shortlist_selection else ""
            ),
            "version_hint": (
                str(shortlist_alignment.get("version_hint", "")) if shortlist_alignment else ""
            ),
        },
        "calibration_summary": calibration_summary,
        "warnings": deduped_warnings,
    }


def render_protocol_diagnostic_markdown(report: dict[str, object]) -> str:
    lines = ["# Protocol Diagnostic Report", ""]
    warnings = report.get("warnings", [])
    if isinstance(warnings, list) and warnings:
        lines.append("## Warnings")
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")

    training_signal = report.get("training_signal", {})
    if isinstance(training_signal, dict) and training_signal:
        lines.append("## Training Signal")
        lines.append("| metric | value |")
        lines.append("| --- | --- |")
        for key in (
            "requested_steps",
            "completed_steps",
            "effective_step_count",
            "idle_step_count",
            "signalless_step_count",
            "signal_density_rate",
            "average_hicra_mask_intensity",
            "mask_nonzero_step_count",
            "mask_hit_step_count",
            "reasoning_action_mismatch_rate",
        ):
            lines.append(f"| {key} | {training_signal.get(key)} |")
        lines.append("")

    shortlist_alignment = report.get("shortlist_alignment", {})
    if isinstance(shortlist_alignment, dict) and shortlist_alignment:
        lines.append("## Shortlist Alignment")
        lines.append("| metric | value |")
        lines.append("| --- | --- |")
        for key in (
            "selected_tags",
            "gameplay_top_tags",
            "overlap_count",
            "top_k",
            "mismatch_detected",
            "selected_best_win_rate",
            "gameplay_best_win_rate",
            "version_hint",
        ):
            lines.append(f"| {key} | {shortlist_alignment.get(key)} |")
        lines.append("")

    leaderboard = report.get("trajectory_leaderboard", [])
    if isinstance(leaderboard, list) and leaderboard:
        lines.append("## Gameplay Leaders")
        lines.append("| tag | access_ok | win_rate | avg_ev_gap | parse_error_rate |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in leaderboard:
            if not isinstance(row, dict):
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("tag", "")),
                        str(row.get("access_ok", False)),
                        f"{_safe_float(row.get('win_rate', 0.0)):.6f}",
                        f"{_safe_float(row.get('avg_ev_gap', 0.0)):.6f}",
                        f"{_safe_float(row.get('parse_error_rate', 0.0)):.6f}",
                    ]
                )
                + " |"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
