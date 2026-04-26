import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

from liars_game_engine.analysis.shapley_analyzer import ShapleyAttribution
from liars_game_engine.analysis.task_k_gold_runner import main, run_task_k_gold_pipeline
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
            (Path(temp_dir) / "g1.jsonl").write_text(
                '{"turn": 1, "player_id": "p1", "skill_name": "Truthful_Action"}\n',
                encoding="utf-8",
            )
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
            (Path(temp_dir) / "g1.jsonl").write_text(
                '{"turn": 1, "player_id": "p1", "skill_name": "Truthful_Action"}\n',
                encoding="utf-8",
            )
            (Path(temp_dir) / "g2.jsonl").write_text(
                '{"turn": 1, "player_id": "p1", "skill_name": "Truthful_Action"}\n',
                encoding="utf-8",
            )
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

    def test_task_k_pipeline_writes_diagnostics_events(self) -> None:
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
            (Path(temp_dir) / "g1.jsonl").write_text(
                '{"turn": 1, "player_id": "p1", "skill_name": "Truthful_Action"}\n',
                encoding="utf-8",
            )
            summary = run_task_k_gold_pipeline(
                output_dir=Path(temp_dir) / "output",
                game_count=1,
                rollout_samples=200,
                max_workers=2,
                progress_interval_games=1,
            )

            diagnostics_path = Path(summary["diagnostics_log"])
            events = [json.loads(line) for line in diagnostics_path.read_text(encoding="utf-8").splitlines()]

            self.assertTrue(diagnostics_path.exists())
            self.assertEqual(events[0]["event"], "pipeline_start")
            self.assertEqual(events[1]["event"], "batch_completed")
            self.assertEqual(events[-1]["event"], "pipeline_finished")
            self.assertIn("resident_memory_mb", events[0])
            self.assertIn("child_cpu_seconds", events[1])

    def test_task_k_pipeline_writes_failure_log_on_exception(self) -> None:
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
            side_effect=RuntimeError("boom"),
        ):
            (Path(temp_dir) / "g1.jsonl").write_text(
                '{"turn": 1, "player_id": "p1", "skill_name": "Truthful_Action"}\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "boom"):
                run_task_k_gold_pipeline(
                    output_dir=Path(temp_dir) / "output",
                    game_count=1,
                    rollout_samples=200,
                    max_workers=2,
                    progress_interval_games=1,
                )

            output_dir = Path(temp_dir) / "output"
            diagnostics_path = output_dir / "diagnostics.jsonl"
            failure_log_path = output_dir / "failure.log"
            events = [json.loads(line) for line in diagnostics_path.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(failure_log_path.exists())
            self.assertIn("RuntimeError: boom", failure_log_path.read_text(encoding="utf-8"))
            self.assertEqual(events[-1]["event"], "pipeline_failed")
            self.assertEqual(events[-1]["error_type"], "RuntimeError")
            self.assertIn("traceback", events[-1])

    def test_task_k_pipeline_reuses_existing_baseline_logs_without_regenerating(self) -> None:
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
            side_effect=AssertionError("baseline generation should not run"),
        ), patch(
            "liars_game_engine.analysis.task_k_gold_runner.ShapleyAnalyzer.analyze_logs",
            return_value=([fake_attr], object()),
        ) as analyze_logs, patch(
            "liars_game_engine.analysis.task_k_gold_runner.ShapleyAnalyzer.export_credit_report",
            side_effect=lambda attributions, output_path: Path(output_path),
        ):
            baseline_dir = Path(temp_dir) / "existing_baseline"
            baseline_dir.mkdir(parents=True)
            existing_logs = [
                baseline_dir / "g1.jsonl",
                baseline_dir / "g2.jsonl",
            ]
            for log_path in existing_logs:
                log_path.write_text("{}", encoding="utf-8")

            summary = run_task_k_gold_pipeline(
                output_dir=Path(temp_dir) / "output",
                existing_baseline_dir=baseline_dir,
                game_count=999,
                rollout_samples=200,
                max_workers=4,
                progress_interval_games=10,
            )

        analyze_logs.assert_called_once_with(existing_logs)
        self.assertEqual(summary["baseline_game_count"], 2)
        self.assertEqual(summary["baseline_dir"], str(baseline_dir))

    def test_task_k_pipeline_exports_attributed_logs_with_phi_labels(self) -> None:
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
            baseline_path = Path(temp_dir) / "g1.jsonl"
            baseline_path.write_text(
                (
                    '{"turn": 1, "player_id": "p1", "skill_name": "Truthful_Action"}\n'
                    '{"turn": 2, "player_id": "p2", "skill_name": "Calculated_Bluff"}\n'
                ),
                encoding="utf-8",
            )

            summary = run_task_k_gold_pipeline(
                output_dir=Path(temp_dir) / "output",
                game_count=1,
                rollout_samples=200,
                max_workers=2,
                progress_interval_games=1,
            )

            attributed_dir = Path(summary["attributed_dir"])
            attributed_path = attributed_dir / "g1.jsonl"
            attributed_lines = attributed_path.read_text(encoding="utf-8").splitlines()
            self.assertTrue(attributed_dir.exists())
            self.assertTrue(attributed_path.exists())
            self.assertEqual(len(attributed_lines), 2)
            self.assertIn('"shapley_value": 0.4', attributed_lines[0])
            self.assertIn('"phi": 0.4', attributed_lines[0])
            self.assertNotIn('"shapley_value"', attributed_lines[1])

    def test_task_k_main_parses_cli_arguments(self) -> None:
        with patch(
            "liars_game_engine.analysis.task_k_gold_runner.run_task_k_gold_pipeline",
            return_value={"credit_report": "logs/task_k_gold/credit_report_final.csv"},
        ) as run_pipeline, patch(
            "sys.argv",
            [
                "task_k_gold_runner.py",
                "--game-count",
                "2",
                "--rollout-samples",
                "5",
                "--max-workers",
                "2",
                "--progress-interval-games",
                "1",
                "--output-dir",
                "/tmp/task_k_smoke",
                "--existing-baseline-dir",
                "/tmp/existing_baseline",
            ],
        ):
            main()

        run_pipeline.assert_called_once_with(
            config_file="config/experiment.yaml",
            game_count=2,
            rollout_samples=5,
            output_dir="/tmp/task_k_smoke",
            max_workers=2,
            progress_interval_games=1,
            existing_baseline_dir="/tmp/existing_baseline",
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
