from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from liars_game_engine.analysis.diagnostic_schedule import build_checkpoint_schedule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List checkpoint tags for dense-then-sparse diagnostic screening.")
    parser.add_argument("--max-step", type=int, default=200)
    parser.add_argument("--dense-until-step", type=int, default=100)
    parser.add_argument("--dense-interval", type=int, default=5)
    parser.add_argument("--sparse-interval", type=int, default=10)
    parser.add_argument("--exclude-final", action="store_true", default=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tags = build_checkpoint_schedule(
        max_step=args.max_step,
        dense_until_step=args.dense_until_step,
        dense_interval=args.dense_interval,
        sparse_interval=args.sparse_interval,
        include_final=not bool(args.exclude_final),
    )
    print("\n".join(tags))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
