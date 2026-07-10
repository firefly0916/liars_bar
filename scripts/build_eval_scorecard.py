from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from liars_game_engine.analysis.eval_scorecard import (
    build_experiment_scorecard,
    render_scorecards_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a unified evaluation scorecard from Task M and Task 1.1 artifacts."
    )
    parser.add_argument("experiment_roots", nargs="+", help="Experiment root(s) containing task_m and task_1_1.")
    parser.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="json",
        help="Output format.",
    )
    parser.add_argument("--output", help="Optional output file path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiment_roots = [Path(root) for root in args.experiment_roots if Path(root).is_dir()]
    if not experiment_roots:
        raise ValueError("No valid experiment directories were provided.")

    scorecards = [build_experiment_scorecard(root) for root in experiment_roots]
    if args.format == "markdown":
        rendered = render_scorecards_markdown(scorecards)
    else:
        rendered = json.dumps(scorecards if len(scorecards) > 1 else scorecards[0], ensure_ascii=False, indent=2)

    if args.output:
        Path(args.output).write_text(rendered + ("\n" if not rendered.endswith("\n") else ""), encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
