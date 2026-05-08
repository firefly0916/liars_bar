#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from liars_game_engine.analysis.proxy_rollout_calibration import (
    render_checkpoint_markdown,
    run_checkpoint_proxy_rollout_calibration,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run proxy-vs-rollout action-ranking calibration on selected checkpoint screening logs.",
    )
    parser.add_argument("screening_root", help="Root directory containing per-checkpoint screening outputs.")
    parser.add_argument(
        "--checkpoint-tag",
        action="append",
        dest="checkpoint_tags",
        required=True,
        help="Checkpoint tag to evaluate. Repeat this flag for multiple checkpoints.",
    )
    parser.add_argument("--model-path", required=True, help="Path to the distilled proxy value model.")
    parser.add_argument("--config", default="config/experiment.yaml", help="Experiment config used to rebuild env.")
    parser.add_argument("--output-dir", required=True, help="Directory for summary and per-checkpoint reports.")
    parser.add_argument(
        "--proxy-output-mode",
        choices=["phi", "winner"],
        default="phi",
        help="Interpret proxy outputs as phi-style scores or winner/rollout probabilities.",
    )
    parser.add_argument("--sample-size", type=int, default=40, help="Number of LLM turns to sample per checkpoint.")
    parser.add_argument("--sample-seed", type=int, default=4242, help="Fixed sampling seed.")
    parser.add_argument(
        "--rollout-samples",
        type=int,
        default=12,
        help="Number of continuation rollouts per candidate action.",
    )
    parser.add_argument("--max-workers", type=int, default=1, help="Parallel workers for rollout scoring.")
    parser.add_argument(
        "--risk-priority-fraction",
        type=float,
        default=0.5,
        help="Fraction of sampled turns taken from highest-death-probability states before random fill.",
    )
    parser.add_argument("--llm-player-id", default=None, help="Override LLM player id if needed.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    screening_root = Path(args.screening_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_summaries: list[dict[str, object]] = []
    for checkpoint_tag in args.checkpoint_tags:
        checkpoint_root = screening_root / checkpoint_tag
        result = run_checkpoint_proxy_rollout_calibration(
            checkpoint_root,
            model_path=args.model_path,
            config_file=args.config,
            sample_size=args.sample_size,
            sample_seed=args.sample_seed,
            rollout_samples=args.rollout_samples,
            max_workers=args.max_workers,
            risk_priority_fraction=args.risk_priority_fraction,
            llm_player_id=args.llm_player_id,
            proxy_output_mode=args.proxy_output_mode,
        )
        checkpoint_dir = output_dir / checkpoint_tag
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "turn_reports.json").write_text(
            json.dumps(result["turn_reports"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (checkpoint_dir / "summary.json").write_text(
            json.dumps(result["summary"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        checkpoint_summaries.append(result["summary"])

    checkpoint_summaries = sorted(checkpoint_summaries, key=lambda item: str(item["checkpoint_label"]))
    (output_dir / "summary.json").write_text(
        json.dumps(checkpoint_summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(
        render_checkpoint_markdown(checkpoint_summaries),
        encoding="utf-8",
    )
    print(json.dumps(checkpoint_summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
