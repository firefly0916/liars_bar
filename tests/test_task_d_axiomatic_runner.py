import json
import tempfile
import unittest
from pathlib import Path

from liars_game_engine.analysis.shapley_analyzer import ShapleyAttribution
from liars_game_engine.analysis.task_d_axiomatic_runner import (
    _build_probe_experiment_settings,
    compute_efficiency_error,
    compute_symmetry_deviation,
)
from liars_game_engine.config.schema import AppSettings


class TaskDAxiomaticRunnerTest(unittest.TestCase):
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
                        "agent_type": "langchain",
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

    def test_build_probe_experiment_settings_enables_probe_and_forces_four_mock(self) -> None:
        """作用: 验证 Task D 实验配置会强制 4 Mock + Probe 开关开启。

        输入:
        - 无（测试内部构造基础 settings）。

        返回:
        - 无。
        """
        settings = _build_probe_experiment_settings(self._make_settings())

        self.assertTrue(settings.runtime.enable_null_player_probe)
        self.assertEqual(len(settings.players), 4)
        self.assertEqual({player.agent_type for player in settings.players}, {"mock"})

    def test_compute_efficiency_error_matches_outcome_minus_initial_value(self) -> None:
        """作用: 验证有效性偏差按 sum(phi)-(Outcome-V_initial) 计算。

        输入:
        - 无（测试内构造临时日志和 attribution 样本）。

        返回:
        - 无。
        """
        attributions = [
            ShapleyAttribution(
                game_id="g1",
                turn=1,
                player_id="p1",
                skill_name="Truthful_Action",
                state_feature="phase=turn_start|table=A|risk=0-1/6|must_call_liar=False",
                death_prob_bucket="0-1/6",
                winner="p1",
                value_action=0.8,
                value_counterfactual=0.06,
                phi=0.74,
                rollout_samples=100,
            ),
            ShapleyAttribution(
                game_id="g1",
                turn=2,
                player_id="p2",
                skill_name="Calculated_Bluff",
                state_feature="phase=response_window|table=A|risk=1/6-1/3|must_call_liar=False",
                death_prob_bucket="1/6-1/3",
                winner="p1",
                value_action=0.2,
                value_counterfactual=0.44,
                phi=-0.24,
                rollout_samples=100,
            ),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "g1.jsonl"
            log_file.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "turn": 1,
                                "player_id": "p1",
                                "observation": {"alive_players": ["p1", "p2"]},
                                "action": {"type": "play_claim", "claim_rank": "A", "cards": ["A"]},
                                "skill_name": "Truthful_Action",
                                "skill_parameters": {},
                                "checkpoint": {"format": "pickle_base64_v1", "payload": "dummy"},
                                "step_result": {
                                    "events": [
                                        "p1 played 1 face-down card(s).",
                                    ]
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "turn": 2,
                                "player_id": "p2",
                                "observation": {"alive_players": ["p1", "p2"]},
                                "action": {"type": "challenge", "claim_rank": None, "cards": []},
                                "skill_name": "Calculated_Bluff",
                                "skill_parameters": {},
                                "checkpoint": {"format": "pickle_base64_v1", "payload": "dummy"},
                                "step_result": {
                                    "events": [
                                        "Roulette revealed LETHAL on p2; player eliminated.",
                                    ]
                                },
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = compute_efficiency_error(attributions=attributions, log_paths=[log_file], initial_value=0.25)

        self.assertEqual(summary["sample_count"], 1)
        self.assertAlmostEqual(summary["mean_abs_error"], 0.25, places=6)

    def test_compute_symmetry_deviation_tracks_equivalent_action_gap(self) -> None:
        """作用: 验证对称性偏离按等效动作组内 phi 差值统计。

        输入:
        - 无（测试内构造两条等效动作样本）。

        返回:
        - 无。
        """
        attributions = [
            ShapleyAttribution(
                game_id="g1",
                turn=1,
                player_id="p1",
                skill_name="Truthful_Action",
                state_feature="phase=turn_start|table=A|risk=0-1/6|must_call_liar=False",
                death_prob_bucket="0-1/6",
                winner="p1",
                value_action=0.6,
                value_counterfactual=0.5,
                phi=0.10,
                rollout_samples=100,
            ),
            ShapleyAttribution(
                game_id="g2",
                turn=1,
                player_id="p1",
                skill_name="Truthful_Action",
                state_feature="phase=turn_start|table=A|risk=0-1/6|must_call_liar=False",
                death_prob_bucket="0-1/6",
                winner="p1",
                value_action=0.64,
                value_counterfactual=0.50,
                phi=0.14,
                rollout_samples=100,
            ),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            log_paths: list[Path] = []
            for game_id in ["g1", "g2"]:
                log_file = Path(temp_dir) / f"{game_id}.jsonl"
                log_file.write_text(
                    json.dumps(
                        {
                            "turn": 1,
                            "player_id": "p1",
                            "observation": {"alive_players": ["p1", "p2", "p3", "p4"]},
                            "action": {"type": "play_claim", "claim_rank": "A", "cards": ["A"]},
                            "skill_name": "Truthful_Action",
                            "skill_parameters": {},
                            "checkpoint": {"format": "pickle_base64_v1", "payload": "dummy"},
                            "step_result": {"events": ["p1 played 1 face-down card(s)."]},
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                log_paths.append(log_file)

            symmetry = compute_symmetry_deviation(attributions=attributions, log_paths=log_paths)

        self.assertEqual(symmetry["pair_count"], 1)
        self.assertAlmostEqual(symmetry["mean_abs_diff"], 0.04, places=6)


if __name__ == "__main__":
    unittest.main()
