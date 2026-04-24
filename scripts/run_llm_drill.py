from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from liars_game_engine.config.loader import load_settings
from liars_game_engine.experiment.llm_drill import run_llm_drill


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight LLM drill with 1 LLM and 3 Mock agents.")
    parser.add_argument("--config", default="config/experiment.yaml", help="Path to experiment config file.")
    parser.add_argument("--games", type=int, default=5, help="Number of games to run.")
    parser.add_argument("--log-dir", default="logs/llm_drill", help="Directory for drill logs and summary.")
    return parser.parse_args()


async def _run() -> dict[str, object]:
    args = parse_args()
    settings = load_settings(config_file=args.config)
    return await run_llm_drill(settings=settings, games=args.games, log_dir=args.log_dir)


def main() -> None:
    summary = asyncio.run(_run())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
