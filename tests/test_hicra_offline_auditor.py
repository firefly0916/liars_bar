import json
import importlib.util
import tempfile
import unittest
from pathlib import Path

from liars_game_engine.analysis.hicra_offline_auditor import (
    audit_decision_record,
    summarize_audit_records,
    write_audit_outputs,
)


class HICRAOfflineAuditorTest(unittest.TestCase):
    def _base_record(self) -> dict[str, object]:
        return {
            "game_id": "g1",
            "turn": 7,
            "player_id": "p1",
            "thought": "I suspect a bluff and should challenge.",
            "action": {"type": "challenge", "claim_rank": "", "cards": []},
            "proxy_target_action": {"type": "challenge", "claim_rank": "", "cards": []},
            "phi_chosen": 0.12,
            "phi_best": 0.13,
            "ev_gap": 0.01,
            "reasoning_action_mismatch": False,
            "strategic_tokens": [{"label": "skepticism", "token": "challenge", "weight": 1.4}],
        }

    def test_protocol_failure_takes_label_precedence(self) -> None:
        record = self._base_record()
        record.update(
            {
                "parse_error": {"code": "invalid_json"},
                "fallback_used": True,
                "ev_gap": 0.42,
                "action": {"type": "challenge", "claim_rank": "", "cards": []},
                "proxy_target_action": {"type": "play_claim", "claim_rank": "A", "cards": ["A"]},
                "thought": "I should avoid risk and play safely.",
            }
        )

        audited = audit_decision_record(record)

        self.assertEqual(audited["audit_label"], "protocol_failure")
        self.assertEqual(audited["severity"], "high")
        self.assertTrue(audited["protocol_failure"])
        self.assertTrue(audited["action_proxy_disagreement"])
        self.assertTrue(audited["semantic_reasoning_action_mismatch"])

    def test_legacy_reasoning_action_mismatch_is_not_semantic_by_default(self) -> None:
        record = self._base_record()
        record.update(
            {
                "reasoning_action_mismatch": True,
                "ev_gap": 0.31,
                "phi_chosen": -0.12,
                "phi_best": 0.19,
                "thought": "I suspect a bluff and should challenge.",
                "action": {"type": "challenge", "claim_rank": "", "cards": []},
                "proxy_target_action": {"type": "play_claim", "claim_rank": "A", "cards": ["A"]},
            }
        )

        audited = audit_decision_record(record)

        self.assertTrue(audited["legacy_reasoning_action_mismatch"])
        self.assertTrue(audited["action_proxy_disagreement"])
        self.assertTrue(audited["high_ev_gap_decision_error"])
        self.assertFalse(audited["semantic_reasoning_action_mismatch"])
        self.assertEqual(audited["audit_label"], "strategic_overchallenge")

    def test_low_gap_action_proxy_disagreement_is_not_high_severity_conflict(self) -> None:
        record = self._base_record()
        record.update(
            {
                "ev_gap": 0.04,
                "phi_chosen": 0.10,
                "phi_best": 0.14,
                "thought": "I can play a safe card and avoid a challenge.",
                "action": {"type": "play_claim", "claim_rank": "A", "cards": ["A"]},
                "proxy_target_action": {"type": "challenge", "claim_rank": "", "cards": []},
            }
        )

        audited = audit_decision_record(record)

        self.assertTrue(audited["action_proxy_disagreement"])
        self.assertFalse(audited["high_ev_gap_decision_error"])
        self.assertEqual(audited["audit_label"], "proxy_disagreement")
        self.assertEqual(audited["severity"], "low")

    def test_summary_and_output_files_are_written(self) -> None:
        clean = audit_decision_record(self._base_record())
        conflict_record = self._base_record()
        conflict_record.update(
            {
                "ev_gap": 0.35,
                "phi_chosen": -0.20,
                "phi_best": 0.15,
                "thought": "I should avoid risk and play safely.",
                "action": {"type": "challenge", "claim_rank": "", "cards": []},
                "proxy_target_action": {"type": "play_claim", "claim_rank": "A", "cards": ["A"]},
            }
        )
        conflict = audit_decision_record(conflict_record)
        summary = summarize_audit_records([clean, conflict])

        self.assertEqual(summary["total_records"], 2)
        self.assertEqual(summary["label_counts"]["clean_aligned"], 1)
        self.assertEqual(summary["label_counts"]["protocol_failure"], 0)
        self.assertEqual(summary["semantic_reasoning_action_mismatch_count"], 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = write_audit_outputs([clean, conflict], Path(tmpdir))

            for key in [
                "records_path",
                "summary_path",
                "case_studies_path",
                "scorecard_path",
            ]:
                self.assertTrue(Path(outputs[key]).is_file())

            records = [
                json.loads(line)
                for line in Path(outputs["records_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(records), 2)
            self.assertIn("reasoning_action_conflict", Path(outputs["case_studies_path"]).read_text(encoding="utf-8"))

    def test_cli_audits_jsonl_records(self) -> None:
        script_path = Path("scripts/run_hicra_offline_audit.py")
        spec = importlib.util.spec_from_file_location("run_hicra_offline_audit", script_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        record = self._base_record()
        record.update(
            {
                "ev_gap": 0.22,
                "thought": "I should avoid risk and play safely.",
                "action": {"type": "challenge", "claim_rank": "", "cards": []},
                "proxy_target_action": {"type": "play_claim", "claim_rank": "A", "cards": ["A"]},
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "records.jsonl"
            output_dir = Path(tmpdir) / "audit"
            input_path.write_text(json.dumps(record, ensure_ascii=True) + "\n", encoding="utf-8")

            exit_code = module.main(
                [
                    "--records-jsonl",
                    str(input_path),
                    "--output-dir",
                    str(output_dir),
                    "--ev-gap-threshold",
                    "0.15",
                ]
            )

            self.assertEqual(exit_code, 0)
            summary = json.loads((output_dir / "hicra_audit_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["total_records"], 1)
            self.assertEqual(summary["label_counts"]["reasoning_action_conflict"], 1)

    def test_cli_merges_ev_gap_csv_with_task_m_logs(self) -> None:
        script_path = Path("scripts/run_hicra_offline_audit.py")
        spec = importlib.util.spec_from_file_location("run_hicra_offline_audit", script_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            log_root = root / "task_m" / "games"
            log_root.mkdir(parents=True)
            log_record = {
                "game_id": "game-1",
                "turn": 4,
                "player_id": "p1",
                "thought": "I should avoid risk and play safely.",
                "action": {"type": "challenge", "claim_rank": "", "cards": []},
                "parse_error": None,
                "fallback_used": False,
                "resolution_reason": "",
            }
            (log_root / "game-1.jsonl").write_text(json.dumps(log_record, ensure_ascii=True) + "\n", encoding="utf-8")
            csv_path = root / "ev_gap_distribution.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "game_id,turn,player_id,action_type,action_claim_rank,action_cards,phi_chosen,best_action_type,best_action_claim_rank,best_action_cards,phi_best,ev_gap",
                        "game-1,4,p1,challenge,,, -0.12,play_claim,A,A,0.19,0.31",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            output_dir = root / "audit"

            exit_code = module.main(
                [
                    "--ev-gap-csv",
                    str(csv_path),
                    "--log-root",
                    str(root / "task_m"),
                    "--output-dir",
                    str(output_dir),
                ]
            )

            self.assertEqual(exit_code, 0)
            records = [
                json.loads(line)
                for line in (output_dir / "hicra_audit_records.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(records[0]["proxy_target_action"]["type"], "play_claim")
            self.assertEqual(records[0]["audit_label"], "reasoning_action_conflict")


if __name__ == "__main__":
    unittest.main()
