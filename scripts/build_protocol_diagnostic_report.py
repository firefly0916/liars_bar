from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from liars_game_engine.analysis.protocol_diagnostic_report import (
    build_protocol_diagnostic_report,
    render_protocol_diagnostic_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a unified protocol diagnostic report from run artifacts.")
    parser.add_argument("run_root", help="Experiment run root containing train/ and reports/ outputs.")
    parser.add_argument(
        "--calibration-summary",
        default=None,
        help="Optional proxy rollout calibration summary.json to embed into the report.",
    )
    parser.add_argument("--format", choices=["json", "markdown"], default="json")
    parser.add_argument("--output", default=None, help="Optional output file path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_protocol_diagnostic_report(
        run_root=Path(args.run_root),
        calibration_summary_path=(Path(args.calibration_summary) if args.calibration_summary else None),
    )
    rendered = (
        render_protocol_diagnostic_markdown(report)
        if args.format == "markdown"
        else json.dumps(report, ensure_ascii=False, indent=2)
    )
    if args.output:
        Path(args.output).write_text(rendered + ("" if rendered.endswith("\n") else "\n"), encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
