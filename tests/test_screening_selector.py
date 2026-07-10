import importlib
import json
import tempfile
import unittest
from pathlib import Path


class ScreeningSelectorTest(unittest.TestCase):
    def test_select_screening_candidates_gameplay_profile_prefers_win_rate(self) -> None:
        module = importlib.import_module("liars_game_engine.analysis.screening_selector")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            self._write_candidate(
                root / "step-000010",
                parse_error_rate=0.0,
                parse_error_count=0,
                illegal_chosen_turn_count=0,
                conflict_count=0,
                avg_ev_gap=0.020,
                resolution_adjustment_rate=0.060,
                llm_turn_count=220,
                challenge_accuracy=0.30,
                bluff_efficiency=0.45,
                win_rate=0.10,
            )
            self._write_candidate(
                root / "step-000020",
                parse_error_rate=0.0,
                parse_error_count=0,
                illegal_chosen_turn_count=0,
                conflict_count=0,
                avg_ev_gap=0.050,
                resolution_adjustment_rate=0.090,
                llm_turn_count=220,
                challenge_accuracy=0.40,
                bluff_efficiency=0.55,
                win_rate=0.40,
            )

            selection = module.select_screening_candidates(
                root,
                top_k=2,
                max_conflict_count=1,
                target_llm_turn_count=220,
                selection_profile="gameplay",
            )

            selected_tags = [row["tag"] for row in selection["selected"]]
            self.assertEqual(selected_tags, ["step-000020", "step-000010"])
            self.assertEqual(selection["all_rows"][0]["tag"], "step-000020")

    def test_select_screening_candidates_gameplay_profile_uses_calibration_as_tiebreak(self) -> None:
        module = importlib.import_module("liars_game_engine.analysis.screening_selector")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            calibration_path = root / "calibration.json"

            self._write_candidate(
                root / "step-000010",
                parse_error_rate=0.0,
                parse_error_count=0,
                illegal_chosen_turn_count=0,
                conflict_count=0,
                avg_ev_gap=0.030,
                resolution_adjustment_rate=0.070,
                llm_turn_count=220,
                challenge_accuracy=0.35,
                bluff_efficiency=0.50,
                win_rate=0.30,
            )
            self._write_candidate(
                root / "step-000020",
                parse_error_rate=0.0,
                parse_error_count=0,
                illegal_chosen_turn_count=0,
                conflict_count=0,
                avg_ev_gap=0.030,
                resolution_adjustment_rate=0.070,
                llm_turn_count=220,
                challenge_accuracy=0.35,
                bluff_efficiency=0.50,
                win_rate=0.30,
            )

            calibration_path.write_text(
                json.dumps(
                    [
                        {
                            "checkpoint_label": "step-000010",
                            "top1_match_rate": 0.25,
                            "mean_spearman_rank_correlation": 0.10,
                            "mean_rollout_regret": 0.20,
                            "mean_chosen_action_rollout_rank": 6.0,
                        },
                        {
                            "checkpoint_label": "step-000020",
                            "top1_match_rate": 0.45,
                            "mean_spearman_rank_correlation": 0.40,
                            "mean_rollout_regret": 0.05,
                            "mean_chosen_action_rollout_rank": 3.0,
                        },
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            selection = module.select_screening_candidates(
                root,
                top_k=2,
                max_conflict_count=1,
                target_llm_turn_count=220,
                selection_profile="gameplay",
                calibration_summary_path=calibration_path,
            )

            self.assertEqual([row["tag"] for row in selection["selected"]], ["step-000020", "step-000010"])
            self.assertAlmostEqual(selection["selected"][0]["top1_match_rate"], 0.45, places=6)

    def test_select_screening_candidates_prefers_access_and_risk_clean_rows(self) -> None:
        module = importlib.import_module("liars_game_engine.analysis.screening_selector")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            self._write_candidate(
                root / "step-000010",
                parse_error_rate=0.0,
                parse_error_count=0,
                illegal_chosen_turn_count=0,
                conflict_count=1,
                avg_ev_gap=0.040,
                resolution_adjustment_rate=0.080,
                llm_turn_count=215,
            )
            self._write_candidate(
                root / "step-000020",
                parse_error_rate=0.0,
                parse_error_count=0,
                illegal_chosen_turn_count=0,
                conflict_count=2,
                avg_ev_gap=0.020,
                resolution_adjustment_rate=0.060,
                llm_turn_count=220,
            )
            self._write_candidate(
                root / "step-000030",
                parse_error_rate=0.0,
                parse_error_count=0,
                illegal_chosen_turn_count=0,
                conflict_count=0,
                avg_ev_gap=0.030,
                resolution_adjustment_rate=0.070,
                llm_turn_count=230,
            )
            self._write_candidate(
                root / "step-000040",
                parse_error_rate=0.001,
                parse_error_count=1,
                illegal_chosen_turn_count=1,
                conflict_count=0,
                avg_ev_gap=0.010,
                resolution_adjustment_rate=0.050,
                llm_turn_count=221,
            )

            selection = module.select_screening_candidates(
                root,
                top_k=3,
                max_conflict_count=1,
                target_llm_turn_count=220,
                selection_profile="stability",
            )

            selected_tags = [row["tag"] for row in selection["selected"]]
            self.assertEqual(selected_tags, ["step-000030", "step-000010", "step-000020"])
            self.assertTrue(selection["selected"][0]["access_ok"])
            self.assertTrue(selection["selected"][0]["risk_ok"])
            self.assertFalse(selection["selected"][2]["risk_ok"])
            self.assertFalse(selection["all_rows"][-1]["access_ok"])

            markdown = module.render_selection_markdown(selection)
            self.assertIn("| step-000030 | True | True | True |", markdown)

    def test_select_screening_candidates_prioritizes_stability_and_forces_final_when_access_clean(self) -> None:
        module = importlib.import_module("liars_game_engine.analysis.screening_selector")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            self._write_candidate(
                root / "final",
                parse_error_rate=0.0,
                parse_error_count=0,
                illegal_chosen_turn_count=0,
                conflict_count=1,
                avg_ev_gap=0.060,
                resolution_adjustment_rate=0.090,
                llm_turn_count=405,
                max_ev_gap=0.70,
            )
            self._write_candidate(
                root / "step-000020",
                parse_error_rate=0.0,
                parse_error_count=0,
                illegal_chosen_turn_count=0,
                conflict_count=0,
                avg_ev_gap=0.020,
                resolution_adjustment_rate=0.120,
                llm_turn_count=402,
                max_ev_gap=0.60,
            )
            self._write_candidate(
                root / "step-000030",
                parse_error_rate=0.0,
                parse_error_count=0,
                illegal_chosen_turn_count=0,
                conflict_count=1,
                avg_ev_gap=0.050,
                resolution_adjustment_rate=0.060,
                llm_turn_count=401,
                max_ev_gap=0.55,
            )
            self._write_candidate(
                root / "step-000040",
                parse_error_rate=0.0,
                parse_error_count=0,
                illegal_chosen_turn_count=0,
                conflict_count=0,
                avg_ev_gap=0.030,
                resolution_adjustment_rate=0.070,
                llm_turn_count=398,
                max_ev_gap=0.65,
            )

            selection = module.select_screening_candidates(
                root,
                top_k=3,
                max_conflict_count=1,
                target_llm_turn_count=400,
                always_include_tags=["final"],
                selection_profile="stability",
            )

            selected_tags = [row["tag"] for row in selection["selected"]]
            self.assertEqual(selected_tags[0], "final")
            self.assertEqual(selected_tags[1:], ["step-000030", "step-000040"])
            self.assertEqual(selection["all_rows"][0]["tag"], "step-000030")
            self.assertEqual(selection["all_rows"][1]["tag"], "step-000040")
            self.assertEqual(selection["all_rows"][2]["tag"], "final")

    @staticmethod
    def _write_candidate(
        root: Path,
        *,
        parse_error_rate: float,
        parse_error_count: int,
        illegal_chosen_turn_count: int,
        conflict_count: int,
        avg_ev_gap: float,
        resolution_adjustment_rate: float,
        llm_turn_count: int,
        max_ev_gap: float = 0.5,
        challenge_accuracy: float = 0.5,
        bluff_efficiency: float = 0.5,
        win_rate: float = 0.5,
    ) -> None:
        task_m_dir = root / "task_m"
        task_1_1_dir = root / "task_1_1"
        games_dir = task_m_dir / "games"
        games_dir.mkdir(parents=True)
        task_1_1_dir.mkdir(parents=True)

        (task_m_dir / "summary.json").write_text(
            json.dumps(
                {
                    "total_games": 20,
                    "llm_player_id": "p1",
                    "llm_turn_count": llm_turn_count,
                    "parse_error_rate": parse_error_rate,
                    "parse_error_count": parse_error_count,
                    "resolution_adjustment_rate": resolution_adjustment_rate,
                    "resolution_adjustment_count": round(resolution_adjustment_rate * llm_turn_count),
                    "game_summaries": [
                        {"winner": "p1"} for _ in range(round(20 * win_rate))
                    ]
                    + [
                        {"winner": "p2"} for _ in range(20 - round(20 * win_rate))
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (task_1_1_dir / "summary.json").write_text(
            json.dumps(
                {
                    "audited_turn_count": llm_turn_count,
                    "negative_phi_turn_count": 5,
                    "conflict_count": conflict_count,
                    "avg_ev_gap": avg_ev_gap,
                    "max_ev_gap": max_ev_gap,
                    "high_ev_gap_turn_count": 10,
                    "illegal_chosen_turn_count": illegal_chosen_turn_count,
                    "challenge_rate": 0.25,
                    "play_claim_rate": 0.75,
                    "pass_rate": 0.0,
                    "challenge_accuracy": challenge_accuracy,
                    "bluff_efficiency": bluff_efficiency,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
