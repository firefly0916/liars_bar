from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from liars_game_engine.analysis.audit_llm_behavior import run_llm_behavior_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit LLM drill behavior against a distilled phi proxy.")
    parser.add_argument("log_root", help="Task M drill log directory or JSONL file path.")
    parser.add_argument("--model-path", required=True, help="Path to the trained value_proxy_mlp_distill.pt file.")
    parser.add_argument("--output-dir", default="logs/task_n_llm_behavior_audit")
    parser.add_argument("--phi-threshold", type=float, default=-0.1)
    parser.add_argument("--llm-player-id", default=None)
    parser.add_argument("--summary-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_llm_behavior_audit(
        log_root=args.log_root,
        model_path=args.model_path,
        output_dir=args.output_dir,
        phi_threshold=args.phi_threshold,
        llm_player_id=args.llm_player_id,
        summary_path=args.summary_path,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
