from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from liars_game_engine.analysis.hicra_offline_auditor import (
    DEFAULT_EV_GAP_THRESHOLD,
    audit_decision_record,
    write_audit_outputs,
)
from liars_game_engine.analysis.hicra_preprocessor import load_task_m_record_index


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        records.append(payload)
    return records


def _parse_cards(raw_cards: object) -> list[str]:
    text = str(raw_cards or "").strip()
    if not text:
        return []
    return [card.strip() for card in text.split("|") if card.strip()] if "|" in text else [text]


def _action_from_row(row: dict[str, object], *, prefix: str) -> dict[str, object]:
    return {
        "type": str(row.get(f"{prefix}_type", "") or "").strip(),
        "claim_rank": str(row.get(f"{prefix}_claim_rank", "") or "").strip(),
        "cards": _parse_cards(row.get(f"{prefix}_cards", "")),
    }


def _load_ev_gap_records(ev_gap_csv: Path, log_root: Path) -> list[dict[str, object]]:
    indexed_logs = load_task_m_record_index(log_root)
    records: list[dict[str, object]] = []
    with ev_gap_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            game_id = str(row.get("game_id", "") or "").strip()
            turn = int(str(row.get("turn", 0) or 0).strip())
            player_id = str(row.get("player_id", "") or "").strip()
            base = dict(indexed_logs.get((game_id, turn, player_id), {}))
            base.update(
                {
                    "game_id": game_id,
                    "turn": turn,
                    "player_id": player_id,
                    "action": base.get("action") if isinstance(base.get("action"), dict) else _action_from_row(row, prefix="action"),
                    "proxy_target_action": _action_from_row(row, prefix="best_action"),
                    "phi_chosen": float(str(row.get("phi_chosen", 0.0) or 0.0).strip()),
                    "phi_best": float(str(row.get("phi_best", 0.0) or 0.0).strip()),
                    "ev_gap": float(str(row.get("ev_gap", 0.0) or 0.0).strip()),
                }
            )
            records.append(base)
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run offline reasoning-action audit over JSONL decision records.")
    parser.add_argument("--records-jsonl", help="Path to JSONL records with reasoning/action/proxy fields.")
    parser.add_argument("--ev-gap-csv", help="Path to task_1_1/ev_gap_distribution.csv.")
    parser.add_argument("--log-root", help="Path to task_m root containing games/*.jsonl.")
    parser.add_argument("--output-dir", required=True, help="Directory for HICRA offline audit outputs.")
    parser.add_argument("--ev-gap-threshold", type=float, default=DEFAULT_EV_GAP_THRESHOLD)
    args = parser.parse_args(argv)

    if args.records_jsonl:
        raw_records = _load_jsonl(Path(args.records_jsonl))
    elif args.ev_gap_csv and args.log_root:
        raw_records = _load_ev_gap_records(Path(args.ev_gap_csv), Path(args.log_root))
    else:
        parser.error("provide either --records-jsonl or both --ev-gap-csv and --log-root")

    audited = [
        audit_decision_record(record, ev_gap_threshold=float(args.ev_gap_threshold))
        for record in raw_records
    ]
    outputs = write_audit_outputs(audited, Path(args.output_dir))
    for key, value in outputs.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
