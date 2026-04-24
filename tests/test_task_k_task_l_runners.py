import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from liars_game_engine.analysis.shapley_analyzer import ShapleyAttribution
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

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "liars_game_engine.analysis.task_k_gold_runner.load_settings",
            return_value=self._make_settings(),
        ), patch(
            "liars_game_engine.analysis.task_k_gold_runner.generate_baseline_logs",
            return_value=[Path(temp_dir) / "g1.jsonl"],
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

    def test_task_k_pipeline_writes_progress_log_every_50_games_with_percent_and_eta(self) -> None:
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

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "liars_game_engine.analysis.task_k_gold_runner.load_settings",
            return_value=self._make_settings(),
        ), patch(
            "liars_game_engine.analysis.task_k_gold_runner.generate_baseline_logs",
            side_effect=[
                [Path(temp_dir) / "g1.jsonl" for _ in range(50)],
                [Path(temp_dir) / "g2.jsonl" for _ in range(50)],
            ],
        ), patch(
            "liars_game_engine.analysis.task_k_gold_runner.asyncio.run",
            side_effect=lambda result: result,
        ), patch(
            "liars_game_engine.analysis.task_k_gold_runner.ShapleyAnalyzer.analyze_logs",
            side_effect=[([fake_attr], object()), ([fake_attr], object())],
        ), patch(
            "liars_game_engine.analysis.task_k_gold_runner.ShapleyAnalyzer.export_credit_report",
            side_effect=lambda attributions, output_path: Path(output_path),
        ), patch(
            "liars_game_engine.analysis.task_k_gold_runner.time.perf_counter",
            side_effect=[10.0, 30.0, 40.0, 70.0],
        ):
            summary = run_task_k_gold_pipeline(
                output_dir=temp_dir,
                game_count=100,
                rollout_samples=200,
                max_workers=24,
                progress_interval_games=50,
            )

            progress_path = Path(summary["progress_log"])
            progress_lines = progress_path.read_text(encoding="utf-8").splitlines()
            self.assertTrue(progress_path.exists())

        self.assertEqual(len(progress_lines), 2)
        self.assertIn("completed_games=50/100", progress_lines[0])
        self.assertIn("percent=50.0%", progress_lines[0])
        self.assertIn("eta_seconds=20.000000", progress_lines[0])
        self.assertIn("progress_bar=[##########----------]", progress_lines[0])
        self.assertIn("completed_games=100/100", progress_lines[1])
        self.assertIn("percent=100.0%", progress_lines[1])
        self.assertIn("eta_seconds=0.000000", progress_lines[1])
        self.assertIn("progress_log", summary)

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
