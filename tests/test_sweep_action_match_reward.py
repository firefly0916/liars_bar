import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def _load_sweep_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "sweep_action_match_reward.py"
    spec = importlib.util.spec_from_file_location("sweep_action_match_reward", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class SweepActionMatchRewardTest(unittest.TestCase):
    def test_parse_weight_values_supports_csv_and_range_syntax(self) -> None:
        module = _load_sweep_module()

        self.assertEqual(module.parse_weight_values("0.4,0.45,0.5"), [0.4, 0.45, 0.5])
        self.assertEqual(module.parse_weight_values("0.4:0.5:0.05"), [0.4, 0.45, 0.5])

    def test_build_run_slug_formats_compact_weight_tag(self) -> None:
        module = _load_sweep_module()

        self.assertEqual(module.build_run_slug(0.35), "amrw-035")
        self.assertEqual(module.build_run_slug(0.4), "amrw-040")
        self.assertEqual(module.build_run_slug(0.55), "amrw-055")

    def test_run_sweep_collects_train_and_eval_metrics(self) -> None:
        module = _load_sweep_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            feat_repo = root / "feat"
            task_m_repo = root / "task_m"
            feat_repo.mkdir()
            task_m_repo.mkdir()

            calls: list[list[str]] = []

            def _fake_run_command(command, *, cwd, env=None):
                calls.append([str(item) for item in command])
                if "train_savi_alignment.py" in " ".join(str(item) for item in command):
                    output_path = Path(command[command.index("--output-path") + 1])
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_text(
                        json.dumps(
                            {
                                "completed_steps": 10,
                                "action_match_reward_weight": 0.35,
                                "final_adapter_path": str(output_path.parent / "checkpoints" / "final"),
                                "step_metrics": {"effective_step_count": 10, "signal_density_rate": 1.0},
                            }
                        ),
                        encoding="utf-8",
                    )
                else:
                    log_dir = Path(command[command.index("--log-dir") + 1])
                    log_dir.mkdir(parents=True, exist_ok=True)
                    (log_dir / "summary.json").write_text(
                        json.dumps(
                            {
                                "total_games": 1,
                                "llm_turn_count": 14,
                                "parse_error_rate": 0.0,
                                "resolution_adjustment_rate": 0.0,
                                "parse_error_count": 0,
                                "resolution_adjustment_count": 0,
                            }
                        ),
                        encoding="utf-8",
                    )

            with patch.object(module, "run_command", side_effect=_fake_run_command):
                summary = module.run_sweep(
                    weights=[0.35],
                    feat_repo_root=feat_repo,
                    task_m_repo_root=task_m_repo,
                    dataset_path=feat_repo / "dataset.jsonl",
                    policy_model_path=Path("/models/base"),
                    proxy_model_path=Path("/models/proxy.pt"),
                    task_m_config_path=task_m_repo / "config/experiment.yaml",
                    output_root=feat_repo / "logs" / "sweep",
                    task_m_games=1,
                    smoke_steps=10,
                    local_llm_max_new_tokens=192,
                )

            self.assertEqual(len(summary["runs"]), 1)
            run = summary["runs"][0]
            self.assertEqual(run["action_match_reward_weight"], 0.35)
            self.assertEqual(run["train"]["completed_steps"], 10)
            self.assertEqual(run["task_m"]["resolution_adjustment_rate"], 0.0)
            self.assertEqual(len(calls), 2)
            self.assertTrue(any("train_savi_alignment.py" in " ".join(call) for call in calls))
            self.assertTrue(any("run_llm_drill.py" in " ".join(call) for call in calls))


if __name__ == "__main__":
    unittest.main()
