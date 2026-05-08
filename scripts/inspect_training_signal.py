from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from liars_game_engine.analysis.training_signal_inspector import (
    load_training_signal_report,
    render_training_signal_markdown,
    training_signal_report_to_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect per-step HICRA/mask training signal from train_summary.json.")
    parser.add_argument("train_summary_path", help="Path to train_summary.json")
    parser.add_argument("--format", choices=["json", "markdown"], default="markdown")
    parser.add_argument("--output", default=None, help="Optional output file path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = load_training_signal_report(args.train_summary_path)
    rendered = (
        render_training_signal_markdown(report)
        if args.format == "markdown"
        else training_signal_report_to_json(report)
    )
    if args.output:
        Path(args.output).write_text(rendered + ("" if rendered.endswith("\n") else "\n"), encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
