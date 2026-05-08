import importlib
import unittest


class TrainingSignalInspectorTest(unittest.TestCase):
    def test_build_training_signal_report_summarizes_mask_and_signal_steps(self) -> None:
        module = importlib.import_module("liars_game_engine.analysis.training_signal_inspector")

        report = module.build_training_signal_report(
            {
                "requested_steps": 3,
                "completed_steps": 3,
                "step_metrics": {
                    "effective_step_count": 2,
                    "idle_step_count": 1,
                    "signalless_step_count": 1,
                    "signal_density_rate": 2.0 / 3.0,
                    "average_hicra_mask_intensity": 0.1,
                },
                "step_summaries": [
                    {
                        "step": 1,
                        "loss": 0.4,
                        "ev_gap": 0.2,
                        "reward_span": 0.5,
                        "idle_step": False,
                        "signalless_step": False,
                        "skip_update": False,
                        "reasoning_action_mismatch": True,
                        "mask_metrics": {
                            "non_zero_mask_count": 2,
                            "mask_hit_count": 1,
                            "average_hicra_mask_intensity": 0.3,
                        },
                    },
                    {
                        "step": 2,
                        "loss": 0.7,
                        "ev_gap": 0.1,
                        "reward_span": 0.0,
                        "idle_step": True,
                        "signalless_step": True,
                        "skip_update": True,
                        "reasoning_action_mismatch": False,
                        "mask_metrics": {
                            "non_zero_mask_count": 0,
                            "mask_hit_count": 0,
                            "average_hicra_mask_intensity": 0.0,
                        },
                    },
                    {
                        "step": 3,
                        "loss": 0.5,
                        "ev_gap": 0.4,
                        "reward_span": 0.2,
                        "idle_step": False,
                        "signalless_step": False,
                        "skip_update": False,
                        "reasoning_action_mismatch": True,
                        "mask_metrics": {
                            "non_zero_mask_count": 1,
                            "mask_hit_count": 0,
                            "average_hicra_mask_intensity": 0.0,
                        },
                    },
                ],
            }
        )

        self.assertEqual(report["summary"]["mask_nonzero_step_count"], 2)
        self.assertEqual(report["summary"]["mask_hit_step_count"], 1)
        self.assertAlmostEqual(report["summary"]["reasoning_action_mismatch_rate"], 2.0 / 3.0, places=6)
        self.assertEqual(report["step_rows"][0]["step"], 1)
        self.assertEqual(report["step_rows"][1]["step"], 2)
        self.assertEqual(report["idle_steps"], [2])
        self.assertEqual(report["signalless_steps"], [2])
        self.assertEqual(report["mask_miss_steps"], [2, 3])
        self.assertEqual(report["top_ev_gap_steps"][0]["step"], 3)
        self.assertEqual(report["top_loss_steps"][0]["step"], 2)

    def test_render_training_signal_markdown_includes_anomaly_sections(self) -> None:
        module = importlib.import_module("liars_game_engine.analysis.training_signal_inspector")

        report = module.build_training_signal_report(
            {
                "requested_steps": 2,
                "completed_steps": 2,
                "step_metrics": {
                    "effective_step_count": 0,
                    "idle_step_count": 2,
                    "signalless_step_count": 2,
                    "signal_density_rate": 0.0,
                    "average_hicra_mask_intensity": 0.0,
                },
                "step_summaries": [
                    {
                        "step": 1,
                        "loss": 0.2,
                        "ev_gap": 0.1,
                        "reward_span": 0.0,
                        "idle_step": True,
                        "signalless_step": True,
                        "skip_update": True,
                        "reasoning_action_mismatch": True,
                        "mask_metrics": {
                            "non_zero_mask_count": 0,
                            "mask_hit_count": 0,
                            "average_hicra_mask_intensity": 0.0,
                        },
                    }
                ],
            }
        )

        rendered = module.render_training_signal_markdown(report)
        self.assertIn("# Training Signal Report", rendered)
        self.assertIn("mask_never_activated", rendered)
        self.assertIn("Idle Steps", rendered)
        self.assertIn("Mask Miss Steps", rendered)


if __name__ == "__main__":
    unittest.main()
