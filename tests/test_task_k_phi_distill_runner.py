import importlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from liars_game_engine.analysis.train_value_proxy import VALUE_PROXY_TARGET_PHI


class TaskKPhiDistillRunnerTest(unittest.TestCase):
    def test_extract_task_k_phi_dataset_aligns_state_action_and_same_record_phi(self) -> None:
        spec = importlib.util.find_spec("liars_game_engine.analysis.task_k_phi_distill_runner")
        self.assertIsNotNone(spec)
        module = importlib.import_module("liars_game_engine.analysis.task_k_phi_distill_runner")
        extract_fn = getattr(module, "extract_task_k_phi_dataset", None)
        self.assertIsNotNone(extract_fn)

        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "task-k-001.jsonl"
            log_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "turn": 4,
                                "player_id": "p2",
                                "skill_name": "Logical_Skepticism",
                                "shapley_value": 0.42,
                                "observation": {
                                    "player_id": "p2",
                                    "phase": "response_window",
                                    "table_type": "A",
                                    "must_call_liar": False,
                                    "alive_players": ["p1", "p2", "p3"],
                                    "private_hand": ["A", "Q", "K"],
                                    "player_states": {"p2": {"death_probability": 1.0 / 3.0}},
                                    "pending_claim": {"actor_id": "p1", "claim_rank": "A", "declared_count": 2},
                                },
                                "action": {"type": "challenge", "cards": []},
                            }
                        ),
                        json.dumps(
                            {
                                "turn": 5,
                                "player_id": "p2",
                                "skill_name": "Calculated_Bluff",
                                "shapley_value": -0.35,
                                "observation": {
                                    "player_id": "p2",
                                    "phase": "turn_start",
                                    "table_type": "K",
                                    "must_call_liar": False,
                                    "alive_players": ["p2", "p3"],
                                    "private_hand": ["Q", "Q"],
                                    "player_states": {"p2": {"death_probability": 1.0 / 6.0}},
                                    "pending_claim": None,
                                },
                                "action": {"type": "play_claim", "claim_rank": "K", "cards": ["Q"]},
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            samples = extract_fn(Path(temp_dir))

        self.assertEqual(len(samples), 2)
        first = samples[0]
        self.assertEqual(first.game_id, "task-k-001")
        self.assertEqual(first.turn, 4)
        self.assertEqual(first.player_id, "p2")
        self.assertEqual(first.skill_name, "Logical_Skepticism")
        self.assertEqual(first.action["type"], "challenge")
        self.assertAlmostEqual(first.phi, 0.42)
        self.assertEqual(len(first.normalized_features), 8)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in first.normalized_features))

        second = samples[1]
        self.assertEqual(second.turn, 5)
        self.assertEqual(second.action["type"], "play_claim")
        self.assertEqual(second.action["cards"], ["Q"])
        self.assertAlmostEqual(second.phi, -0.35)

    def test_task_k_phi_training_pipeline_exports_dataset_and_uses_phi_mode(self) -> None:
        spec = importlib.util.find_spec("liars_game_engine.analysis.task_k_phi_distill_runner")
        self.assertIsNotNone(spec)
        module = importlib.import_module("liars_game_engine.analysis.task_k_phi_distill_runner")
        run_fn = getattr(module, "run_task_k_phi_training_pipeline", None)
        self.assertIsNotNone(run_fn)

        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "task_k_logs"
            log_dir.mkdir()
            (log_dir / "task-k-001.jsonl").write_text(
                json.dumps(
                    {
                        "turn": 1,
                        "player_id": "p1",
                        "skill_name": "Truthful_Action",
                        "shapley_value": 0.15,
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
            output_dir = Path(temp_dir) / "distill_output"

            with patch.object(
                module,
                "train_value_proxy",
                return_value={"model_path": str(output_dir / "value_proxy_mlp.pt"), "val_mse": 0.12},
            ) as mocked_train:
                summary = run_fn(log_root=log_dir, output_dir=output_dir, epochs=12)

        self.assertEqual(summary["sample_count"], 1)
        self.assertTrue(str(summary["dataset_path"]).endswith("task_k_phi_dataset.jsonl"))
        self.assertEqual(summary["target_mode"], VALUE_PROXY_TARGET_PHI)
        mocked_train.assert_called_once()
        self.assertEqual(mocked_train.call_args.kwargs["target_mode"], VALUE_PROXY_TARGET_PHI)
        self.assertEqual(mocked_train.call_args.kwargs["epochs"], 12)


if __name__ == "__main__":
    unittest.main()
