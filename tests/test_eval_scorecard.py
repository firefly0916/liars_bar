import importlib
import json
import tempfile
import unittest
from pathlib import Path


class EvalScorecardTest(unittest.TestCase):
    def test_build_experiment_scorecard_aggregates_access_quality_behavior_and_auxiliary_metrics(self) -> None:
        module = importlib.import_module("liars_game_engine.analysis.eval_scorecard")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "baseline-100g"
            task_m_dir = root / "task_m"
            games_dir = task_m_dir / "games"
            task_1_1_dir = root / "task_1_1"
            games_dir.mkdir(parents=True)
            task_1_1_dir.mkdir(parents=True)

            (task_m_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "total_games": 2,
                        "llm_player_id": "p1",
                        "llm_turn_count": 10,
                        "parse_error_rate": 0.0,
                        "parse_error_count": 0,
                        "resolution_adjustment_rate": 0.2,
                        "resolution_adjustment_count": 2,
                        "game_summaries": [
                            {"winner": "p1", "turns_played": 12},
                            {"winner": "p2", "turns_played": 9},
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (task_1_1_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "audited_turn_count": 10,
                        "negative_phi_turn_count": 2,
                        "conflict_count": 1,
                        "avg_ev_gap": 0.125,
                        "max_ev_gap": 0.7,
                        "high_ev_gap_turn_count": 3,
                        "illegal_chosen_turn_count": 0,
                        "challenge_rate": 0.3,
                        "play_claim_rate": 0.7,
                        "pass_rate": 0.0,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            (games_dir / "game-01.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "turn": 1,
                                "player_id": "p1",
                                "observation": {"table_type": "A", "pending_claim": None},
                                "action": {"type": "play_claim", "claim_rank": "A", "cards": ["K"]},
                                "fallback_used": False,
                                "step_result": {
                                    "success": True,
                                    "events": [
                                        "p1 played 1 face-down card(s).",
                                        "Response window opened for p2.",
                                    ],
                                },
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "turn": 2,
                                "player_id": "p2",
                                "observation": {
                                    "table_type": "A",
                                    "pending_claim": {"actor_id": "p1", "claim_rank": "A", "declared_count": 1},
                                },
                                "action": {"type": "challenge", "claim_rank": None, "cards": []},
                                "fallback_used": False,
                                "step_result": {
                                    "success": True,
                                    "events": [
                                        "p2 called LIAR on p1.",
                                        "At least one revealed card is a Liar.",
                                    ],
                                },
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "turn": 3,
                                "player_id": "p1",
                                "observation": {
                                    "table_type": "K",
                                    "pending_claim": {"actor_id": "p2", "claim_rank": "K", "declared_count": 2},
                                },
                                "action": {"type": "challenge", "claim_rank": None, "cards": []},
                                "fallback_used": False,
                                "step_result": {
                                    "success": True,
                                    "events": [
                                        "p1 called LIAR on p2.",
                                        "At least one revealed card is a Liar.",
                                    ],
                                },
                            },
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            (games_dir / "game-02.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "turn": 1,
                                "player_id": "p1",
                                "observation": {"table_type": "K", "pending_claim": None},
                                "action": {"type": "play_claim", "claim_rank": "K", "cards": ["A"]},
                                "fallback_used": False,
                                "step_result": {
                                    "success": True,
                                    "events": [
                                        "p1 played 1 face-down card(s).",
                                        "Response window opened for p2.",
                                    ],
                                },
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "turn": 2,
                                "player_id": "p2",
                                "observation": {
                                    "table_type": "K",
                                    "pending_claim": {"actor_id": "p1", "claim_rank": "K", "declared_count": 1},
                                },
                                "action": {"type": "play_claim", "claim_rank": "K", "cards": ["K"]},
                                "fallback_used": False,
                                "step_result": {
                                    "success": True,
                                    "events": [
                                        "p2 played 1 face-down card(s).",
                                        "Response window opened for p3.",
                                    ],
                                },
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "turn": 3,
                                "player_id": "p1",
                                "observation": {
                                    "table_type": "Q",
                                    "pending_claim": {"actor_id": "p3", "claim_rank": "Q", "declared_count": 1},
                                },
                                "action": {"type": "challenge", "claim_rank": None, "cards": []},
                                "fallback_used": False,
                                "step_result": {
                                    "success": True,
                                    "events": [
                                        "p1 called LIAR on p3.",
                                        "All revealed cards are Innocent.",
                                    ],
                                },
                            },
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            scorecard = module.build_experiment_scorecard(root, label="baseline")

            self.assertEqual(scorecard["label"], "baseline")
            self.assertTrue(scorecard["access"]["access_ok"])
            self.assertEqual(scorecard["access"]["parse_error_rate"], 0.0)
            self.assertEqual(scorecard["access"]["illegal_chosen_turn_count"], 0)
            self.assertAlmostEqual(scorecard["quality"]["conflict_rate"], 0.1, places=6)
            self.assertAlmostEqual(scorecard["quality"]["avg_ev_gap"], 0.125, places=6)
            self.assertAlmostEqual(scorecard["quality"]["resolution_adjustment_rate"], 0.2, places=6)
            self.assertEqual(scorecard["stability"]["negative_phi_turn_count"], 2)
            self.assertAlmostEqual(scorecard["stability"]["max_ev_gap"], 0.7, places=6)
            self.assertEqual(scorecard["stability"]["high_ev_gap_turn_count"], 3)
            self.assertAlmostEqual(scorecard["stability"]["challenge_rate"], 0.3, places=6)
            self.assertAlmostEqual(scorecard["stability"]["play_claim_rate"], 0.7, places=6)
            self.assertAlmostEqual(scorecard["stability"]["pass_rate"], 0.0, places=6)
            self.assertEqual(scorecard["behavior"]["challenge_attempt_count"], 2)
            self.assertEqual(scorecard["behavior"]["correct_challenge_count"], 1)
            self.assertAlmostEqual(scorecard["behavior"]["challenge_accuracy"], 0.5, places=6)
            self.assertEqual(scorecard["behavior"]["bluff_attempt_count"], 2)
            self.assertEqual(scorecard["behavior"]["successful_bluff_count"], 1)
            self.assertAlmostEqual(scorecard["behavior"]["bluff_efficiency"], 0.5, places=6)
            self.assertEqual(scorecard["auxiliary"]["win_count"], 1)
            self.assertAlmostEqual(scorecard["auxiliary"]["win_rate"], 0.5, places=6)

            markdown = module.render_scorecards_markdown([scorecard])
            self.assertIn("max_ev_gap", markdown)
            self.assertIn("high_ev_gap_turn_count", markdown)
            self.assertIn("negative_phi_turn_count", markdown)
            self.assertIn("challenge_rate", markdown)
            self.assertIn("play_claim_rate", markdown)
            self.assertIn("pass_rate", markdown)
            self.assertIn("| baseline | True | 0.000000 | 0 | 0.125000 | 1 | 0.100000 | 0.200000 | 0.700000 | 3 | 2 | 0.300000 | 0.700000 | 0.000000 | 0.500000 | 0.500000 | 0.500000 |", markdown)


if __name__ == "__main__":
    unittest.main()
