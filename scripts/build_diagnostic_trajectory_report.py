from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from liars_game_engine.analysis.eval_scorecard import build_experiment_scorecard

_STEP_PATTERN = re.compile(r"^step-(\d{6})$")


def _sort_key(root: Path) -> tuple[int, int]:
    name = root.name
    if name == "final":
        return (1, 10**9)
    match = _STEP_PATTERN.match(name)
    if match:
        return (0, int(match.group(1)))
    return (0, 10**9 - 1)


def _discover(screening_root: Path) -> list[Path]:
    roots: list[Path] = []
    for child in sorted(screening_root.iterdir(), key=_sort_key):
        if not child.is_dir():
            continue
        if (child / "task_m" / "summary.json").is_file() and (child / "task_1_1" / "summary.json").is_file():
            roots.append(child)
    return roots


def _render_markdown(rows: list[dict[str, object]]) -> str:
    headers = [
        "tag",
        "access_ok",
        "parse_error_rate",
        "illegal",
        "avg_ev_gap",
        "conflict_count",
        "resolution_adjustment_rate",
        "max_ev_gap",
        "challenge_accuracy",
        "bluff_efficiency",
        "win_rate",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["tag"]),
                    str(row["access_ok"]),
                    f"{float(row['parse_error_rate']):.6f}",
                    str(row["illegal_chosen_turn_count"]),
                    f"{float(row['avg_ev_gap']):.6f}",
                    str(row["conflict_count"]),
                    f"{float(row['resolution_adjustment_rate']):.6f}",
                    f"{float(row['max_ev_gap']):.6f}",
                    f"{float(row['challenge_accuracy']):.6f}",
                    f"{float(row['bluff_efficiency']):.6f}",
                    f"{float(row['win_rate']):.6f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a trajectory report across checkpoint screening outputs.")
    parser.add_argument("screening_root")
    parser.add_argument("--format", choices=["json", "markdown"], default="markdown")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    screening_root = Path(args.screening_root)
    rows: list[dict[str, object]] = []
    for root in _discover(screening_root):
        scorecard = build_experiment_scorecard(root, label=root.name)
        rows.append(
            {
                "tag": root.name,
                "access_ok": bool(scorecard["access"]["access_ok"]),
                "parse_error_rate": float(scorecard["access"]["parse_error_rate"]),
                "illegal_chosen_turn_count": int(scorecard["access"]["illegal_chosen_turn_count"]),
                "avg_ev_gap": float(scorecard["quality"]["avg_ev_gap"]),
                "conflict_count": int(scorecard["quality"]["conflict_count"]),
                "resolution_adjustment_rate": float(scorecard["quality"]["resolution_adjustment_rate"]),
                "max_ev_gap": float(scorecard["stability"]["max_ev_gap"]),
                "challenge_accuracy": float(scorecard["behavior"]["challenge_accuracy"]),
                "bluff_efficiency": float(scorecard["behavior"]["bluff_efficiency"]),
                "win_rate": float(scorecard["auxiliary"]["win_rate"]),
            }
        )
    rendered = _render_markdown(rows) if args.format == "markdown" else json.dumps(rows, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + ("\n" if not rendered.endswith("\n") else ""), encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
