from __future__ import annotations

import json
from pathlib import Path

from liars_game_engine.analysis.eval_scorecard import build_experiment_scorecard


def _discover_candidate_roots(screening_root: Path) -> list[Path]:
    roots: list[Path] = []
    for child in sorted(screening_root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "task_m" / "summary.json").is_file() and (child / "task_1_1" / "summary.json").is_file():
            roots.append(child)
    return roots


def _candidate_row(
    experiment_root: Path,
    *,
    max_conflict_count: int | None,
    target_llm_turn_count: int,
    calibration_index: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    scorecard = build_experiment_scorecard(experiment_root, label=experiment_root.name)
    conflict_count = int(scorecard["quality"]["conflict_count"])
    access_ok = bool(scorecard["access"]["access_ok"])
    risk_ok = bool(access_ok)
    if max_conflict_count is not None:
        risk_ok = bool(risk_ok and conflict_count <= int(max_conflict_count))
    llm_turn_count = int(scorecard["auxiliary"]["llm_turn_count"])
    row = {
        "tag": str(scorecard["label"]),
        "experiment_root": str(experiment_root),
        "scorecard": scorecard,
        "access_ok": access_ok,
        "risk_ok": risk_ok,
        "parse_error_rate": float(scorecard["access"]["parse_error_rate"]),
        "parse_error_count": int(scorecard["access"]["parse_error_count"]),
        "illegal_chosen_turn_count": int(scorecard["access"]["illegal_chosen_turn_count"]),
        "conflict_count": conflict_count,
        "conflict_rate": float(scorecard["quality"]["conflict_rate"]),
        "avg_ev_gap": float(scorecard["quality"]["avg_ev_gap"]),
        "resolution_adjustment_rate": float(scorecard["quality"]["resolution_adjustment_rate"]),
        "resolution_adjustment_count": int(scorecard["quality"]["resolution_adjustment_count"]),
        "max_ev_gap": float(scorecard["stability"]["max_ev_gap"]),
        "challenge_accuracy": float(scorecard["behavior"]["challenge_accuracy"]),
        "bluff_efficiency": float(scorecard["behavior"]["bluff_efficiency"]),
        "win_rate": float(scorecard["auxiliary"]["win_rate"]),
        "llm_turn_count": llm_turn_count,
        "turn_count_distance": abs(llm_turn_count - int(target_llm_turn_count)),
        "proxy_calibration_available": False,
        "top1_match_rate": None,
        "mean_spearman_rank_correlation": None,
        "mean_rollout_regret": None,
        "mean_chosen_action_rollout_rank": None,
    }
    calibration_row = (calibration_index or {}).get(str(scorecard["label"]))
    if isinstance(calibration_row, dict):
        row["proxy_calibration_available"] = True
        row["top1_match_rate"] = float(calibration_row.get("top1_match_rate", 0.0) or 0.0)
        row["mean_spearman_rank_correlation"] = float(
            calibration_row.get("mean_spearman_rank_correlation", 0.0) or 0.0
        )
        row["mean_rollout_regret"] = float(calibration_row.get("mean_rollout_regret", 0.0) or 0.0)
        row["mean_chosen_action_rollout_rank"] = float(
            calibration_row.get("mean_chosen_action_rollout_rank", 0.0) or 0.0
        )
    return row


def _load_calibration_index(calibration_summary_path: Path | str | None) -> dict[str, dict[str, object]]:
    if calibration_summary_path is None:
        return {}

    path = Path(calibration_summary_path)
    if path.is_dir():
        path = path / "summary.json"
    if not path.is_file():
        raise RuntimeError(f"Calibration summary path does not exist: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError(f"Calibration summary must be a JSON list: {path}")

    index: dict[str, dict[str, object]] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        label = str(item.get("checkpoint_label", "")).strip()
        if not label:
            continue
        index[label] = item
    return index


def _stability_sort_key(row: dict[str, object]) -> tuple[object, ...]:
    return (
        0 if bool(row["access_ok"]) else 1,
        0 if bool(row["risk_ok"]) else 1,
        float(row["resolution_adjustment_rate"]),
        int(row["conflict_count"]),
        float(row["avg_ev_gap"]),
        float(row["max_ev_gap"]),
        str(row["tag"]),
    )


def _gameplay_sort_key(row: dict[str, object]) -> tuple[object, ...]:
    calibration_available = bool(row.get("proxy_calibration_available"))
    top1_match_rate = float(row.get("top1_match_rate", 0.0) or 0.0)
    mean_spearman = float(row.get("mean_spearman_rank_correlation", 0.0) or 0.0)
    mean_rollout_regret = float(row.get("mean_rollout_regret", 0.0) or 0.0)
    mean_chosen_rank = float(row.get("mean_chosen_action_rollout_rank", 0.0) or 0.0)
    return (
        0 if bool(row["access_ok"]) else 1,
        0 if bool(row["risk_ok"]) else 1,
        -float(row["win_rate"]),
        0 if calibration_available else 1,
        -top1_match_rate,
        mean_rollout_regret if calibration_available else float("inf"),
        -mean_spearman if calibration_available else float("inf"),
        mean_chosen_rank if calibration_available else float("inf"),
        -float(row["challenge_accuracy"]),
        -float(row["bluff_efficiency"]),
        float(row["resolution_adjustment_rate"]),
        int(row["turn_count_distance"]),
        float(row["avg_ev_gap"]),
        str(row["tag"]),
    )


def _sort_key(row: dict[str, object], selection_profile: str) -> tuple[object, ...]:
    if selection_profile == "gameplay":
        return _gameplay_sort_key(row)
    return _stability_sort_key(row)


def select_screening_candidates(
    screening_root: Path | str,
    *,
    top_k: int = 3,
    max_conflict_count: int | None = None,
    target_llm_turn_count: int = 220,
    always_include_tags: list[str] | None = None,
    selection_profile: str = "gameplay",
    calibration_summary_path: Path | str | None = None,
) -> dict[str, object]:
    root = Path(screening_root)
    if selection_profile not in {"stability", "gameplay"}:
        raise RuntimeError(f"Unsupported selection_profile: {selection_profile}")
    calibration_index = _load_calibration_index(calibration_summary_path)
    rows = [
        _candidate_row(
            candidate_root,
            max_conflict_count=max_conflict_count,
            target_llm_turn_count=target_llm_turn_count,
            calibration_index=calibration_index,
        )
        for candidate_root in _discover_candidate_roots(root)
    ]
    rows = sorted(rows, key=lambda row: _sort_key(row, selection_profile))
    forced_tags = {str(tag) for tag in (always_include_tags or []) if str(tag).strip()}
    forced_rows = [row for row in rows if str(row["tag"]) in forced_tags and bool(row["access_ok"])]
    selected: list[dict[str, object]] = list(forced_rows)
    for row in rows:
        if any(str(existing["tag"]) == str(row["tag"]) for existing in selected):
            continue
        selected.append(row)
        if len(selected) >= max(1, int(top_k)):
            break
    return {
        "screening_root": str(root),
        "top_k": int(top_k),
        "max_conflict_count": (int(max_conflict_count) if max_conflict_count is not None else None),
        "target_llm_turn_count": int(target_llm_turn_count),
        "selection_profile": selection_profile,
        "calibration_summary_path": (
            str(Path(calibration_summary_path)) if calibration_summary_path is not None else None
        ),
        "always_include_tags": sorted(forced_tags),
        "all_rows": rows,
        "selected": selected[: max(1, int(top_k))],
    }


def render_selection_markdown(selection: dict[str, object]) -> str:
    headers = [
        "tag",
        "selected",
        "access_ok",
        "risk_ok",
        "parse_error_rate",
        "illegal_chosen_turn_count",
        "conflict_count",
        "win_rate",
        "challenge_accuracy",
        "bluff_efficiency",
        "avg_ev_gap",
        "resolution_adjustment_rate",
        "top1_match_rate",
        "mean_rollout_regret",
        "max_ev_gap",
        "llm_turn_count",
    ]
    selected_tags = {str(item["tag"]) for item in selection.get("selected", []) if isinstance(item, dict)}
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in selection.get("all_rows", []):
        if not isinstance(row, dict):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["tag"]),
                    str(str(row["tag"]) in selected_tags),
                    str(row["access_ok"]),
                    str(row["risk_ok"]),
                    f"{float(row['parse_error_rate']):.6f}",
                    str(row["illegal_chosen_turn_count"]),
                    str(row["conflict_count"]),
                    f"{float(row['win_rate']):.6f}",
                    f"{float(row['challenge_accuracy']):.6f}",
                    f"{float(row['bluff_efficiency']):.6f}",
                    f"{float(row['avg_ev_gap']):.6f}",
                    f"{float(row['resolution_adjustment_rate']):.6f}",
                    (
                        f"{float(row['top1_match_rate']):.6f}"
                        if row.get("top1_match_rate") is not None
                        else ""
                    ),
                    (
                        f"{float(row['mean_rollout_regret']):.6f}"
                        if row.get("mean_rollout_regret") is not None
                        else ""
                    ),
                    f"{float(row['max_ev_gap']):.6f}",
                    str(row["llm_turn_count"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def selection_to_json(selection: dict[str, object]) -> str:
    return json.dumps(selection, ensure_ascii=False, indent=2)
