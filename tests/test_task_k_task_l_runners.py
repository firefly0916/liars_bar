import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from liars_game_engine.analysis.shapley_analyzer import ShapleyAttribution
from liars_game_engine.analysis import task_k_gold_runner
from liars_game_engine.analysis.task_k_gold_runner import run_task_k_gold_pipeline
from liars_game_engine.analysis.task_l_proxy_refine_runner import (
    _build_task_l_negative_settings,
    run_task_l_proxy_refine_pipeline,
)
from liars_game_engine.config.schema import AppSettings


class TaskKTaskLRunnerTest(unittest.TestCase):
    def _make_settings(self) -> AppSettings:
        return AppSettings.from_dict(
            {
                "runtime": {
                    "max_turns": 20,
                    "random_seed": 11,
                    "fallback_action": "challenge",
                    "enable_null_player_probe": False,
                },
                "players": [
                    {
                        "player_id": "p1",
                        "name": "Alice",
                        "agent_type": "mock",
                        "model": "m1",
                        "prompt_profile": "baseline",
                    },
                    {
                        "player_id": "p2",
                        "name": "Bob",
                        "agent_type": "mock",
                        "model": "m2",
                        "prompt_profile": "baseline",
                    },
                ],
            }
        )

    def test_build_task_l_negative_settings_enables_probe_and_high_probability(self) -> None:
        settings = _build_task_l_negative_settings(self._make_settings(), probe_probability=0.85)

        self.assertTrue(settings.runtime.enable_null_player_probe)
        self.assertEqual(settings.runtime.null_probe_action_probability, 0.85)
        self.assertEqual(len(settings.players), 4)
        self.assertEqual({player.agent_type for player in settings.players}, {"mock"})

    def test_task_k_pipeline_returns_gold_report_paths_and_timing(self) -> None:
        fake_attr = ShapleyAttribution(
            game_id="g1",
            turn=1,
            player_id="p1",
            skill_name="Truthful_Action",
            state_feature="phase=turn_start|table=A|risk=0-1/6|must_call_liar=False",
            death_prob_bucket="0-1/6",
            winner="p1",
            value_action=0.8,
            value_counterfactual=0.4,
            phi=0.4,
            rollout_samples=200,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_log = Path(temp_dir) / "g1.jsonl"
            baseline_log.write_text(
                json.dumps(
                    {
                        "turn": 1,
                        "player_id": "p1",
                        "skill_name": "Truthful_Action",
                        "observation": {
                            "player_id": "p1",
                            "phase": "turn_start",
                            "table_type": "A",
                            "must_call_liar": False,
                            "alive_players": ["p1", "p2"],
                            "private_hand": ["A", "Q"],
                            "player_states": {"p1": {"death_probability": 0.0}},
                            "pending_claim": None,
                        },
                        "action": {"type": "play_claim", "claim_rank": "A", "cards": ["A"]},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch(
                "liars_game_engine.analysis.task_k_gold_runner.load_settings",
                return_value=self._make_settings(),
            ), patch(
                "liars_game_engine.analysis.task_k_gold_runner.generate_baseline_logs",
                return_value=[baseline_log],
            ), patch(
                "liars_game_engine.analysis.task_k_gold_runner.asyncio.run",
                side_effect=lambda result: result,
            ), patch(
                "liars_game_engine.analysis.task_k_gold_runner.ShapleyAnalyzer.analyze_logs",
                return_value=([fake_attr], object()),
            ), patch(
                "liars_game_engine.analysis.task_k_gold_runner.ShapleyAnalyzer.export_credit_report",
                side_effect=lambda attributions, output_path: Path(output_path),
            ):
                summary = run_task_k_gold_pipeline(
                    output_dir=temp_dir,
                    game_count=2,
                    rollout_samples=200,
                    max_workers=24,
                )

        self.assertEqual(summary["rollout_samples"], 200)
        self.assertEqual(summary["max_workers"], 24)
        self.assertTrue(str(summary["credit_report"]).endswith("credit_report_final.csv"))
        self.assertIn("average_seconds_per_attribution", summary)

    def test_task_k_pipeline_exports_attributed_logs_with_shapley_labels(self) -> None:
        fake_attr = ShapleyAttribution(
            game_id="g1",
            turn=1,
            player_id="p1",
            skill_name="Truthful_Action",
            state_feature="phase=turn_start|table=A|risk=0-1/6|must_call_liar=False",
            death_prob_bucket="0-1/6",
            winner="p1",
            value_action=0.8,
            value_counterfactual=0.4,
            phi=0.4,
            rollout_samples=200,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_log = Path(temp_dir) / "g1.jsonl"
            baseline_log.write_text(
                json.dumps(
                    {
                        "turn": 1,
                        "player_id": "p1",
                        "skill_name": "Truthful_Action",
                        "observation": {
                            "player_id": "p1",
                            "phase": "turn_start",
                            "table_type": "A",
                            "must_call_liar": False,
                            "alive_players": ["p1", "p2"],
                            "private_hand": ["A", "Q"],
                            "player_states": {"p1": {"death_probability": 0.0}},
                            "pending_claim": None,
                        },
                        "action": {"type": "play_claim", "claim_rank": "A", "cards": ["A"]},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch(
                "liars_game_engine.analysis.task_k_gold_runner.load_settings",
                return_value=self._make_settings(),
            ), patch(
                "liars_game_engine.analysis.task_k_gold_runner.generate_baseline_logs",
                return_value=[baseline_log],
            ), patch(
                "liars_game_engine.analysis.task_k_gold_runner.asyncio.run",
                side_effect=lambda result: result,
            ), patch(
                "liars_game_engine.analysis.task_k_gold_runner.ShapleyAnalyzer.analyze_logs",
                return_value=([fake_attr], object()),
            ), patch(
                "liars_game_engine.analysis.task_k_gold_runner.ShapleyAnalyzer.export_credit_report",
                side_effect=lambda attributions, output_path: Path(output_path),
            ):
                summary = run_task_k_gold_pipeline(
                    output_dir=temp_dir,
                    game_count=1,
                    rollout_samples=200,
                    max_workers=24,
                )

            attributed_dir = Path(summary["attributed_dir"])
            attributed_log = attributed_dir / "g1.jsonl"
            self.assertTrue(attributed_log.exists())

            attributed_records = [json.loads(line) for line in attributed_log.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(attributed_records), 1)
            self.assertAlmostEqual(attributed_records[0]["shapley_value"], 0.4)
            self.assertAlmostEqual(attributed_records[0]["phi"], 0.4)

    def test_task_k_main_parses_cli_overrides(self) -> None:
        with patch.object(
            task_k_gold_runner,
            "run_task_k_gold_pipeline",
            return_value={"credit_report": "logs/task_k_gold_smoke/credit_report_final.csv"},
        ) as mocked_run, patch.object(
            sys,
            "argv",
            [
                "task_k_gold_runner.py",
                "--game-count",
                "10",
                "--rollout-samples",
                "2",
                "--output-dir",
                "logs/task_k_gold_smoke",
                "--max-workers",
                "4",
            ],
        ):
            task_k_gold_runner.main()

        mocked_run.assert_called_once_with(
            game_count=10,
            rollout_samples=2,
            output_dir="logs/task_k_gold_smoke",
            max_workers=4,
        )

    def test_task_l_pipeline_returns_comparison_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "liars_game_engine.analysis.task_l_proxy_refine_runner.load_settings",
            return_value=self._make_settings(),
        ), patch(
            "liars_game_engine.analysis.task_l_proxy_refine_runner.generate_negative_logs_until_records",
            return_value={
                "negative_log_dir": str(Path(temp_dir) / "negative_logs"),
                "record_count": 100,
                "game_count": 5,
                "log_paths": [Path(temp_dir) / "negative_logs" / "g1.jsonl"],
            },
        ), patch(
            "liars_game_engine.analysis.task_l_proxy_refine_runner.train_value_proxy",
            side_effect=[
                {
                    "val_mse": 0.20,
                    "best_val_mse": 0.20,
                    "model_path": str(Path(temp_dir) / "elite.pt"),
                },
                {
                    "val_mse": 0.18,
                    "best_val_mse": 0.18,
                    "model_path": str(Path(temp_dir) / "value_proxy_mlp_v2.pt"),
                },
            ],
        ), patch(
            "liars_game_engine.analysis.task_l_proxy_refine_runner.run_proxy_alignment_for_model",
            side_effect=[
                {"pearson_correlation": 0.33, "mae": 0.09, "speedup_ratio": 450.0},
                {"pearson_correlation": 0.41, "mae": 0.08, "speedup_ratio": 430.0},
            ],
        ):
            summary = run_task_l_proxy_refine_pipeline(output_dir=temp_dir, negative_record_target=100)

        self.assertEqual(summary["negative_record_count"], 100)
        self.assertTrue(str(summary["mixed_model_path"]).endswith("value_proxy_mlp_v2.pt"))
        self.assertIn("elite_alignment", summary)
        self.assertIn("mixed_alignment", summary)


if __name__ == "__main__":
    unittest.main()
