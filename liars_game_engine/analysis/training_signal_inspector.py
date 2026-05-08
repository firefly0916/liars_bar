from __future__ import annotations

import json
from pathlib import Path

from liars_game_engine.analysis.protocol_diagnostic_report import (
    _load_json,
    _safe_float,
    _safe_int,
    summarize_training_signal,
)


def _step_row(step_summary: dict[str, object]) -> dict[str, object]:
    mask_metrics = step_summary.get("mask_metrics", {})
    if not isinstance(mask_metrics, dict):
        mask_metrics = {}
    return {
        "step": _safe_int(step_summary.get("step", 0)),
        "loss": _safe_float(step_summary.get("loss", 0.0)),
        "ev_gap": _safe_float(step_summary.get("ev_gap", 0.0)),
        "reward_span": _safe_float(step_summary.get("reward_span", 0.0)),
        "idle_step": bool(step_summary.get("idle_step", False)),
        "signalless_step": bool(step_summary.get("signalless_step", False)),
        "skip_update": bool(step_summary.get("skip_update", False)),
        "reasoning_action_mismatch": bool(step_summary.get("reasoning_action_mismatch", False)),
        "non_zero_mask_count": _safe_int(mask_metrics.get("non_zero_mask_count", 0)),
        "mask_hit_count": _safe_int(mask_metrics.get("mask_hit_count", 0)),
        "average_hicra_mask_intensity": _safe_float(mask_metrics.get("average_hicra_mask_intensity", 0.0)),
    }


def _top_rows(step_rows: list[dict[str, object]], key: str, limit: int = 5) -> list[dict[str, object]]:
    return sorted(
        step_rows,
        key=lambda row: (-_safe_float(row.get(key, 0.0)), _safe_int(row.get("step", 0))),
    )[: max(1, int(limit))]


def build_training_signal_report(train_summary: dict[str, object]) -> dict[str, object]:
    raw_steps = train_summary.get("step_summaries", [])
    if not isinstance(raw_steps, list):
        raw_steps = []
    step_rows = [_step_row(step) for step in raw_steps if isinstance(step, dict)]
    step_rows.sort(key=lambda row: _safe_int(row.get("step", 0)))

    idle_steps = [_safe_int(row["step"]) for row in step_rows if bool(row["idle_step"])]
    signalless_steps = [_safe_int(row["step"]) for row in step_rows if bool(row["signalless_step"])]
    mask_miss_steps = [
        _safe_int(row["step"])
        for row in step_rows
        if _safe_int(row["mask_hit_count"]) == 0
    ]
    mismatch_steps = [
        _safe_int(row["step"])
        for row in step_rows
        if bool(row["reasoning_action_mismatch"])
    ]

    return {
        "summary": summarize_training_signal(train_summary),
        "step_rows": step_rows,
        "idle_steps": idle_steps,
        "signalless_steps": signalless_steps,
        "mask_miss_steps": mask_miss_steps,
        "reasoning_action_mismatch_steps": mismatch_steps,
        "top_ev_gap_steps": _top_rows(step_rows, "ev_gap"),
        "top_loss_steps": _top_rows(step_rows, "loss"),
    }


def load_training_signal_report(train_summary_path: Path | str) -> dict[str, object]:
    payload = _load_json(Path(train_summary_path))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Training summary must be a JSON object: {train_summary_path}")
    return build_training_signal_report(payload)


def render_training_signal_markdown(report: dict[str, object]) -> str:
    summary = report.get("summary", {})
    step_rows = report.get("step_rows", [])
    lines = ["# Training Signal Report", ""]

    warnings = summary.get("warnings", [])
    lines.append("## Warnings")
    if isinstance(warnings, list) and warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- none")
    lines.append("")

    lines.append("## Summary")
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    for key in [
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
    ]:
        value = summary.get(key)
        rendered = f"{value:.6f}" if isinstance(value, float) else str(value)
        lines.append(f"| {key} | {rendered} |")
    lines.append("")

    def _append_step_list(title: str, steps: object) -> None:
        lines.append(f"## {title}")
        if isinstance(steps, list) and steps:
            lines.append(", ".join(str(step) for step in steps))
        else:
            lines.append("none")
        lines.append("")

    _append_step_list("Idle Steps", report.get("idle_steps"))
    _append_step_list("Signalless Steps", report.get("signalless_steps"))
    _append_step_list("Mask Miss Steps", report.get("mask_miss_steps"))

    lines.append("## Top EV Gap Steps")
    lines.extend(_render_step_table(report.get("top_ev_gap_steps", [])))
    lines.append("")
    lines.append("## Top Loss Steps")
    lines.extend(_render_step_table(report.get("top_loss_steps", [])))
    lines.append("")
    lines.append("## First Steps")
    lines.extend(_render_step_table(step_rows[:5] if isinstance(step_rows, list) else []))
    lines.append("")
    return "\n".join(lines)


def _render_step_table(step_rows: object) -> list[str]:
    rows = step_rows if isinstance(step_rows, list) else []
    lines = [
        "| step | loss | ev_gap | reward_span | idle | signalless | mismatch | non_zero_mask_count | mask_hit_count | average_hicra_mask_intensity |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        if not isinstance(row, dict):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(_safe_int(row.get("step", 0))),
                    f"{_safe_float(row.get('loss', 0.0)):.6f}",
                    f"{_safe_float(row.get('ev_gap', 0.0)):.6f}",
                    f"{_safe_float(row.get('reward_span', 0.0)):.6f}",
                    str(bool(row.get("idle_step", False))),
                    str(bool(row.get("signalless_step", False))),
                    str(bool(row.get("reasoning_action_mismatch", False))),
                    str(_safe_int(row.get("non_zero_mask_count", 0))),
                    str(_safe_int(row.get("mask_hit_count", 0))),
                    f"{_safe_float(row.get('average_hicra_mask_intensity', 0.0)):.6f}",
                ]
            )
            + " |"
        )
    return lines


def training_signal_report_to_json(report: dict[str, object]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2)
