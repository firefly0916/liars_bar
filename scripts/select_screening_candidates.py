from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from liars_game_engine.analysis.screening_selector import (
    render_selection_markdown,
    select_screening_candidates,
    selection_to_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select top checkpoint candidates from screening Task M + Task 1.1 artifacts."
    )
    parser.add_argument("screening_root", help="Root directory containing per-checkpoint screening outputs.")
    parser.add_argument("--top-k", type=int, default=3, help="How many candidates to select.")
    parser.add_argument(
        "--max-conflict-count",
        type=int,
        default=None,
        help="Optional risk threshold; candidates above it are deprioritized.",
    )
    parser.add_argument(
        "--target-llm-turn-count",
        type=int,
        default=220,
        help="Tie-break target for llm_turn_count closeness.",
    )
    parser.add_argument(
        "--always-include-tag",
        action="append",
        default=[],
        help="Tag(s) to force into selection when access_ok is true. Repeatable.",
    )
    parser.add_argument(
        "--selection-profile",
        choices=["stability", "gameplay"],
        default="gameplay",
        help="Ranking profile to use when ordering candidates.",
    )
    parser.add_argument(
        "--calibration-summary",
        default=None,
        help="Optional proxy-rollout calibration summary.json used as gameplay tie-break input.",
    )
    parser.add_argument("--format", choices=["json", "markdown"], default="json")
    parser.add_argument("--output", help="Optional output file path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selection = select_screening_candidates(
        args.screening_root,
        top_k=args.top_k,
        max_conflict_count=args.max_conflict_count,
        target_llm_turn_count=args.target_llm_turn_count,
        always_include_tags=list(args.always_include_tag),
        selection_profile=args.selection_profile,
        calibration_summary_path=args.calibration_summary,
    )
    rendered = (
        render_selection_markdown(selection)
        if args.format == "markdown"
        else selection_to_json(selection)
    )
    if args.output:
        Path(args.output).write_text(rendered + ("\n" if not rendered.endswith("\n") else ""), encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
