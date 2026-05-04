from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from liars_game_engine.analysis.hicra_preprocessor import (  # noqa: E402
    build_savi_alignment_full_dataset,
    export_savi_alignment_full_dataset,
)


class _CharOffsetTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        rendered = ""
        for message in messages:
            rendered += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
        return rendered

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        payload = {"input_ids": list(range(len(text)))}
        if return_offsets_mapping:
            payload["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a holographic SAVI alignment dataset with prompt and token alignment fields.")
    parser.add_argument("--report-path", required=True)
    parser.add_argument("--ev-gap-csv", required=True)
    parser.add_argument("--log-root", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--prompt-profile", default="alignment_action")
    parser.add_argument("--weight-distribution-strategy", default="equal")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = _CharOffsetTokenizer()
    samples = build_savi_alignment_full_dataset(
        report_path=args.report_path,
        ev_gap_csv_path=args.ev_gap_csv,
        log_root=args.log_root,
        tokenizer=tokenizer,
        prompt_profile=args.prompt_profile,
        weight_distribution_strategy=args.weight_distribution_strategy,
    )
    output_path = export_savi_alignment_full_dataset(samples=samples, output_path=args.output_path)
    summary = {
        "sample_count": len(samples),
        "output_path": str(output_path),
        "samples_with_alignment_metadata": sum(1 for sample in samples if "alignment_metadata" in sample),
        "samples_with_messages": sum(1 for sample in samples if "messages" in sample),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
