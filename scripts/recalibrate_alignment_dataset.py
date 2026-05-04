from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from liars_game_engine.analysis.token_alignment import recalibrate_alignment_dataset


def _load_records(dataset_path: Path | str) -> list[dict[str, object]]:
    path = Path(dataset_path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _export_records(records: list[dict[str, object]], output_path: Path | str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + ("\n" if records else ""),
        encoding="utf-8",
    )
    return path


def _load_tokenizer(tokenizer_path: str) -> object:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised on server once deps are installed.
        raise RuntimeError("transformers is required for alignment recalibration") from exc
    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild holographic token alignment with a real tokenizer.")
    parser.add_argument("dataset_path", help="Path to holographic alignment JSONL dataset.")
    parser.add_argument("--tokenizer-path", required=True, help="Tokenizer/model path for AutoTokenizer.")
    parser.add_argument("--output-path", required=True, help="Path to recalibrated JSONL output.")
    parser.add_argument("--weight-distribution-strategy", default="equal")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = _load_tokenizer(args.tokenizer_path)
    records = _load_records(args.dataset_path)
    recalibrated = recalibrate_alignment_dataset(
        samples=records,
        tokenizer=tokenizer,
        weight_distribution_strategy=args.weight_distribution_strategy,
    )
    output_path = _export_records(recalibrated, args.output_path)
    non_zero_mask_samples = sum(
        1 for record in recalibrated if any(float(weight) != 0.0 for weight in record.get("token_weight_mask", []))
    )
    aligned_samples = sum(
        1
        for record in recalibrated
        if isinstance(record.get("alignment_metadata"), dict)
        and bool(record["alignment_metadata"].get("strategic_token_alignments", []))
    )
    summary = {
        "input_path": str(args.dataset_path),
        "output_path": str(output_path),
        "sample_count": len(recalibrated),
        "aligned_sample_count": aligned_samples,
        "non_zero_mask_sample_count": non_zero_mask_samples,
        "tokenizer_path": args.tokenizer_path,
        "weight_distribution_strategy": args.weight_distribution_strategy,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
