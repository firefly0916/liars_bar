import importlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from liars_game_engine.analysis.shapley_analyzer import ShapleyAttribution
from liars_game_engine.config.schema import AppSettings


class TaskKBackfillLabelsRunnerTest(unittest.TestCase):
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

    def test_backfill_pipeline_exports_attributed_logs_from_existing_baseline_logs(self) -> None:
        spec = importlib.util.find_spec("liars_game_engine.analysis.task_k_backfill_labels")
        self.assertIsNotNone(spec)
        module = importlib.import_module("liars_game_engine.analysis.task_k_backfill_labels")
        run_fn = getattr(module, "run_task_k_backfill_pipeline", None)
        self.assertIsNotNone(run_fn)

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
            log_root = Path(temp_dir) / "baseline_logs"
            log_root.mkdir()
            baseline_log = log_root / "g1.jsonl"
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
            output_dir = Path(temp_dir) / "attributed_logs"

            with patch.object(
                module,
                "load_settings",
                return_value=self._make_settings(),
            ), patch.object(
                module.ShapleyAnalyzer,
                "analyze_logs",
                return_value=([fake_attr], object()),
            ):
                summary = run_fn(
                    log_root=log_root,
                    output_dir=output_dir,
                    rollout_samples=200,
                    max_workers=4,
                )

            attributed_log = output_dir / "g1.jsonl"
            self.assertTrue(attributed_log.exists())
            record = json.loads(attributed_log.read_text(encoding="utf-8").splitlines()[0])
            self.assertAlmostEqual(record["shapley_value"], 0.4)
            self.assertAlmostEqual(record["phi"], 0.4)

        self.assertEqual(summary["baseline_log_count"], 1)
        self.assertEqual(summary["attribution_count"], 1)
        self.assertEqual(summary["rollout_samples"], 200)
        self.assertEqual(summary["max_workers"], 4)
        self.assertTrue(str(summary["attributed_dir"]).endswith("attributed_logs"))
        self.assertIn("progress_log", summary)

    def test_backfill_pipeline_writes_progress_log_per_batch(self) -> None:
        spec = importlib.util.find_spec("liars_game_engine.analysis.task_k_backfill_labels")
        self.assertIsNotNone(spec)
        module = importlib.import_module("liars_game_engine.analysis.task_k_backfill_labels")
        run_fn = getattr(module, "run_task_k_backfill_pipeline", None)
        self.assertIsNotNone(run_fn)

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
            log_root = Path(temp_dir) / "baseline_logs"
            log_root.mkdir()
            for idx in range(2):
                (log_root / f"g{idx + 1}.jsonl").write_text(
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

            output_dir = Path(temp_dir) / "attributed_logs"
            with patch.object(
                module,
                "load_settings",
                return_value=self._make_settings(),
            ), patch.object(
                module.ShapleyAnalyzer,
                "analyze_logs",
                side_effect=[([fake_attr], object()), ([fake_attr], object())],
            ), patch.object(
                module.time,
                "perf_counter",
                side_effect=[10.0, 20.0, 30.0, 50.0],
            ):
                summary = run_fn(
                    log_root=log_root,
                    output_dir=output_dir,
                    rollout_samples=200,
                    max_workers=4,
                    progress_interval_logs=1,
                )

            progress_path = Path(summary["progress_log"])
            progress_lines = progress_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(progress_lines), 2)
        self.assertIn("completed_logs=1/2", progress_lines[0])
        self.assertIn("percent=50.0%", progress_lines[0])
        self.assertIn("eta_seconds=10.000000", progress_lines[0])
        self.assertIn("completed_logs=2/2", progress_lines[1])
        self.assertIn("percent=100.0%", progress_lines[1])


if __name__ == "__main__":
    unittest.main()
