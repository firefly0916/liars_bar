import importlib
import unittest


class ProxyRolloutCalibrationTest(unittest.TestCase):
    def test_normalize_proxy_state_dict_accepts_legacy_backbone_head_layout(self) -> None:
        module = importlib.import_module("liars_game_engine.analysis.proxy_rollout_calibration")

        normalized = module._normalize_proxy_state_dict(
            {
                "backbone.0.weight": 1,
                "backbone.0.bias": 2,
                "backbone.2.weight": 3,
                "backbone.2.bias": 4,
                "head.weight": 5,
                "head.bias": 6,
            }
        )

        self.assertEqual(normalized["network.0.weight"], 1)
        self.assertEqual(normalized["network.0.bias"], 2)
        self.assertEqual(normalized["network.2.weight"], 3)
        self.assertEqual(normalized["network.2.bias"], 4)
        self.assertEqual(normalized["network.4.weight"], 5)
        self.assertEqual(normalized["network.4.bias"], 6)

    def test_build_turn_alignment_report_orders_actions_and_computes_rank_agreement(self) -> None:
        module = importlib.import_module("liars_game_engine.analysis.proxy_rollout_calibration")

        report = module.build_turn_alignment_report(
            game_id="g1",
            turn=12,
            action_rows=[
                {"action_label": "play-1", "proxy_score": 0.2, "rollout_score": 0.8, "feature_vector": [1.0, 2.0]},
                {"action_label": "challenge", "proxy_score": 0.9, "rollout_score": 0.1, "feature_vector": [3.0, 4.0]},
                {"action_label": "play-2", "proxy_score": 0.4, "rollout_score": 0.5, "feature_vector": [5.0, 6.0]},
            ],
        )

        self.assertEqual(report["game_id"], "g1")
        self.assertEqual(report["turn"], 12)
        self.assertEqual(report["action_count"], 3)
        self.assertEqual(report["proxy_top1_action"], "challenge")
        self.assertEqual(report["rollout_top1_action"], "play-1")
        self.assertFalse(report["top1_match"])
        self.assertAlmostEqual(report["spearman_rank_correlation"], -1.0, places=6)

        ranked_actions = report["actions"]
        self.assertEqual(ranked_actions[0]["proxy_rank"], 1)
        self.assertEqual(ranked_actions[0]["action_label"], "challenge")
        self.assertEqual(ranked_actions[0]["rollout_rank"], 3)
        self.assertEqual(ranked_actions[0]["feature_vector"], [3.0, 4.0])

    def test_summarize_alignment_reports_aggregates_top1_and_rank_metrics(self) -> None:
        module = importlib.import_module("liars_game_engine.analysis.proxy_rollout_calibration")

        first = module.build_turn_alignment_report(
            game_id="g1",
            turn=1,
            action_rows=[
                {"action_label": "a", "proxy_score": 0.9, "rollout_score": 0.9},
                {"action_label": "b", "proxy_score": 0.1, "rollout_score": 0.1},
            ],
        )
        second = module.build_turn_alignment_report(
            game_id="g2",
            turn=2,
            action_rows=[
                {"action_label": "a", "proxy_score": 0.2, "rollout_score": 0.8},
                {"action_label": "b", "proxy_score": 0.8, "rollout_score": 0.2},
            ],
        )

        summary = module.summarize_alignment_reports(
            checkpoint_label="step-000045",
            turn_reports=[first, second],
        )

        self.assertEqual(summary["checkpoint_label"], "step-000045")
        self.assertEqual(summary["sample_size"], 2)
        self.assertAlmostEqual(summary["top1_match_rate"], 0.5, places=6)
        self.assertAlmostEqual(summary["mean_spearman_rank_correlation"], 0.0, places=6)
        self.assertAlmostEqual(summary["mean_action_count"], 2.0, places=6)
        self.assertEqual(summary["proxy_top1_win_count"], 1)
        self.assertEqual(summary["rollout_top1_win_count"], 1)


if __name__ == "__main__":
    unittest.main()
