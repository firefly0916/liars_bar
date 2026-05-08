import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class ProtocolDiagnosticReportTest(unittest.TestCase):
    def test_summarize_training_signal_flags_zero_mask_and_idle_steps(self) -> None:
        module = importlib.import_module("liars_game_engine.analysis.protocol_diagnostic_report")

        summary = module.summarize_training_signal(
            {
                "requested_steps": 4,
                "completed_steps": 4,
                "step_metrics": {
                    "effective_step_count": 2,
                    "idle_step_count": 2,
                    "signalless_step_count": 2,
                    "signal_density_rate": 0.5,
                    "average_hicra_mask_intensity": 0.0,
                },
                "step_summaries": [
                    {
                        "reasoning_action_mismatch": True,
                        "mask_metrics": {
                            "non_zero_mask_count": 0,
                            "mask_hit_count": 0,
                            "average_hicra_mask_intensity": 0.0,
                        },
                        "skip_update": True,
                        "idle_step": True,
                        "signalless_step": True,
                    },
                    {
                        "reasoning_action_mismatch": False,
                        "mask_metrics": {
                            "non_zero_mask_count": 0,
                            "mask_hit_count": 0,
                            "average_hicra_mask_intensity": 0.0,
                        },
                        "skip_update": False,
                        "idle_step": False,
                        "signalless_step": False,
                    },
                    {
                        "reasoning_action_mismatch": True,
                        "mask_metrics": {
                            "non_zero_mask_count": 0,
                            "mask_hit_count": 0,
                            "average_hicra_mask_intensity": 0.0,
                        },
                        "skip_update": True,
                        "idle_step": True,
                        "signalless_step": True,
                    },
                    {
                        "reasoning_action_mismatch": True,
                        "mask_metrics": {
                            "non_zero_mask_count": 0,
                            "mask_hit_count": 0,
                            "average_hicra_mask_intensity": 0.0,
                        },
                        "skip_update": False,
                        "idle_step": False,
                        "signalless_step": False,
                    },
                ],
            }
        )

        self.assertEqual(summary["requested_steps"], 4)
        self.assertEqual(summary["completed_steps"], 4)
        self.assertEqual(summary["effective_step_count"], 2)
        self.assertEqual(summary["idle_step_count"], 2)
        self.assertEqual(summary["signalless_step_count"], 2)
        self.assertEqual(summary["mask_nonzero_step_count"], 0)
        self.assertEqual(summary["mask_hit_step_count"], 0)
        self.assertAlmostEqual(summary["reasoning_action_mismatch_rate"], 0.75, places=6)
        self.assertIn("mask_never_activated", summary["warnings"])
        self.assertIn("signalless_steps_present", summary["warnings"])

    def test_build_protocol_diagnostic_report_flags_shortlist_mismatch_and_legacy_selector(self) -> None:
        module = importlib.import_module("liars_game_engine.analysis.protocol_diagnostic_report")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "train").mkdir(parents=True, exist_ok=True)
            (root / "reports").mkdir(parents=True, exist_ok=True)

            (root / "train" / "train_summary.json").write_text(
                json.dumps(
                    {
                        "requested_steps": 3,
                        "completed_steps": 3,
                        "step_metrics": {
                            "effective_step_count": 3,
                            "idle_step_count": 0,
                            "signalless_step_count": 0,
                            "signal_density_rate": 1.0,
                            "average_hicra_mask_intensity": 0.1,
                        },
                        "step_summaries": [
                            {
                                "reasoning_action_mismatch": True,
                                "mask_metrics": {
                                    "non_zero_mask_count": 1,
                                    "mask_hit_count": 1,
                                    "average_hicra_mask_intensity": 0.2,
                                },
                                "skip_update": False,
                                "idle_step": False,
                                "signalless_step": False,
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (root / "reports" / "trajectory_report.json").write_text(
                json.dumps(
                    [
                        {"tag": "step-000140", "access_ok": True, "win_rate": 0.5, "avg_ev_gap": 0.043},
                        {"tag": "step-000045", "access_ok": True, "win_rate": 0.4, "avg_ev_gap": 0.026},
                        {"tag": "step-000025", "access_ok": True, "win_rate": 0.2, "avg_ev_gap": 0.039},
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (root / "reports" / "shortlist_selection.json").write_text(
                json.dumps(
                    {
                        "screening_root": "/tmp/fake-screening",
                        "top_k": 2,
                        "max_conflict_count": 1,
                        "target_llm_turn_count": 100,
                        "all_rows": [
                            {
                                "tag": "step-000025",
                                "access_ok": True,
                                "risk_ok": True,
                                "avg_ev_gap": 0.039,
                                "parse_error_rate": 0.0,
                                "illegal_chosen_turn_count": 0,
                                "conflict_count": 0,
                                "resolution_adjustment_rate": 0.03,
                                "max_ev_gap": 0.43,
                                "llm_turn_count": 78,
                                "turn_count_distance": 22,
                                "scorecard": {
                                    "behavior": {"challenge_accuracy": 0.33, "bluff_efficiency": 0.52},
                                    "auxiliary": {"win_rate": 0.2},
                                },
                            },
                            {
                                "tag": "step-000140",
                                "access_ok": True,
                                "risk_ok": True,
                                "avg_ev_gap": 0.043,
                                "parse_error_rate": 0.0,
                                "illegal_chosen_turn_count": 0,
                                "conflict_count": 0,
                                "resolution_adjustment_rate": 0.07,
                                "max_ev_gap": 0.60,
                                "llm_turn_count": 83,
                                "turn_count_distance": 17,
                                "scorecard": {
                                    "behavior": {"challenge_accuracy": 0.42, "bluff_efficiency": 0.65},
                                    "auxiliary": {"win_rate": 0.5},
                                },
                            },
                            {
                                "tag": "step-000045",
                                "access_ok": True,
                                "risk_ok": True,
                                "avg_ev_gap": 0.026,
                                "parse_error_rate": 0.0,
                                "illegal_chosen_turn_count": 0,
                                "conflict_count": 1,
                                "resolution_adjustment_rate": 0.09,
                                "max_ev_gap": 0.41,
                                "llm_turn_count": 76,
                                "turn_count_distance": 24,
                                "scorecard": {
                                    "behavior": {"challenge_accuracy": 0.58, "bluff_efficiency": 0.55},
                                    "auxiliary": {"win_rate": 0.4},
                                },
                            },
                        ],
                        "selected": [
                            {"tag": "step-000025"},
                            {"tag": "step-000045"},
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            report = module.build_protocol_diagnostic_report(run_root=root)

        self.assertEqual(report["shortlist_alignment"]["selected_tags"], ["step-000025", "step-000045"])
        self.assertEqual(report["shortlist_alignment"]["gameplay_top_tags"], ["step-000140", "step-000045"])
        self.assertTrue(report["shortlist_alignment"]["mismatch_detected"])
        self.assertEqual(report["selector_metadata"]["version_hint"], "legacy")
        self.assertIn("shortlist_misaligned_with_gameplay", report["warnings"])

    def test_build_protocol_diagnostic_report_supports_flat_artifact_layout(self) -> None:
        module = importlib.import_module("liars_game_engine.analysis.protocol_diagnostic_report")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "train_summary.json").write_text(
                json.dumps(
                    {
                        "requested_steps": 2,
                        "completed_steps": 2,
                        "step_metrics": {
                            "effective_step_count": 2,
                            "idle_step_count": 0,
                            "signalless_step_count": 0,
                            "signal_density_rate": 1.0,
                            "average_hicra_mask_intensity": 0.3,
                        },
                        "step_summaries": [],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (root / "trajectory_report.json").write_text(
                json.dumps(
                    [{"tag": "step-000010", "access_ok": True, "win_rate": 0.3, "avg_ev_gap": 0.02}],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (root / "shortlist_selection.json").write_text(
                json.dumps(
                    {
                        "top_k": 1,
                        "selection_profile": "gameplay",
                        "all_rows": [
                            {
                                "tag": "step-000010",
                                "access_ok": True,
                                "risk_ok": True,
                                "win_rate": 0.3,
                                "challenge_accuracy": 0.4,
                                "bluff_efficiency": 0.5,
                                "avg_ev_gap": 0.02,
                                "resolution_adjustment_rate": 0.03,
                                "conflict_count": 0,
                                "turn_count_distance": 10,
                                "max_ev_gap": 0.5,
                            }
                        ],
                        "selected": [{"tag": "step-000010"}],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            report = module.build_protocol_diagnostic_report(run_root=root)

        self.assertEqual(report["training_signal"]["completed_steps"], 2)
        self.assertEqual(report["shortlist_alignment"]["selected_tags"], ["step-000010"])
        self.assertEqual(report["trajectory_leaderboard"][0]["tag"], "step-000010")

    def test_sha256_if_readable_returns_none_on_permission_error(self) -> None:
        module = importlib.import_module("liars_game_engine.analysis.protocol_diagnostic_report")

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "proxy.pt"
            path.write_bytes(b"stub")
            with patch.object(Path, "is_file", side_effect=PermissionError("denied")):
                self.assertIsNone(module._sha256_if_readable(path))

    def test_build_protocol_diagnostic_report_falls_back_to_selection_rows_for_leaderboard(self) -> None:
        module = importlib.import_module("liars_game_engine.analysis.protocol_diagnostic_report")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "shortlist_selection.json").write_text(
                json.dumps(
                    {
                        "top_k": 2,
                        "selection_profile": "gameplay",
                        "all_rows": [
                            {
                                "tag": "step-000050",
                                "access_ok": True,
                                "risk_ok": True,
                                "win_rate": 0.3,
                                "avg_ev_gap": 0.02,
                                "parse_error_rate": 0.0,
                                "challenge_accuracy": 0.5,
                                "bluff_efficiency": 0.6,
                                "resolution_adjustment_rate": 0.08,
                                "conflict_count": 0,
                                "turn_count_distance": 10,
                                "max_ev_gap": 0.4,
                            },
                            {
                                "tag": "step-000140",
                                "access_ok": True,
                                "risk_ok": True,
                                "win_rate": 0.5,
                                "avg_ev_gap": 0.04,
                                "parse_error_rate": 0.0,
                                "challenge_accuracy": 0.4,
                                "bluff_efficiency": 0.65,
                                "resolution_adjustment_rate": 0.07,
                                "conflict_count": 0,
                                "turn_count_distance": 17,
                                "max_ev_gap": 0.6,
                            },
                        ],
                        "selected": [{"tag": "step-000140"}, {"tag": "step-000050"}],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            report = module.build_protocol_diagnostic_report(run_root=root)

        self.assertEqual(report["trajectory_leaderboard"][0]["tag"], "step-000140")
        self.assertAlmostEqual(report["trajectory_leaderboard"][0]["win_rate"], 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
