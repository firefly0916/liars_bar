from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from liars_game_engine.analysis.hicra_preprocessor import (  # noqa: E402
    build_savi_alignment_dataset,
    export_savi_alignment_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SAVI alignment training data from Task 1.1 audit outputs.")
    parser.add_argument("--report-path", required=True)
    parser.add_argument("--ev-gap-csv", required=True)
    parser.add_argument("--log-root", required=True)
    parser.add_argument("--output-path", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = build_savi_alignment_dataset(
        report_path=args.report_path,
        ev_gap_csv_path=args.ev_gap_csv,
        log_root=args.log_root,
    )
    output_path = export_savi_alignment_dataset(samples=samples, output_path=args.output_path)
    summary = {
        "sample_count": len(samples),
        "output_path": str(output_path),
        "mismatch_sample_count": sum(1 for sample in samples if bool(sample.get("reasoning_action_mismatch", False))),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
