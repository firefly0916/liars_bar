import unittest

from liars_game_engine.agents.liar_planner import LiarPlanner, ObservationParser, ParameterResolver


class LiarPlannerTest(unittest.TestCase):
    def test_observation_parser_includes_required_sections_and_high_risk(self) -> None:
        """作用: 验证观察解析结果包含三段关键标签与高风险描述。

        输入:
        - 无（测试内构造 observation）。

        返回:
        - 无。
        """
        parser = ObservationParser()
        observation = {
            "player_id": "p1",
            "table_type": "A",
            "private_hand": ["A", "K"],
            "pile_history": [{"actor_id": "p2", "claim_rank": "A", "declared_count": 6, "cards": ["Q"]}],
            "legal_actions": [{"type": "play_claim", "claim_rank": "A", "min_cards": 1, "max_cards": 2}],
            "player_states": {
                "p1": {
                    "is_alive": True,
                    "is_safe": False,
                    "death_probability": 0.5,
                    "hand_count": 2,
                }
            },
        }

        text = parser.parse(observation)

        self.assertIn("[生存警告]", text)
        self.assertIn("极高风险局势", text)
        self.assertIn("[牌局情报]", text)
        self.assertIn("逻辑冲突", text)
        self.assertIn("[规则约束]", text)

    def test_observation_parser_marks_death_probability_one_as_dead_end(self) -> None:
        """作用: 验证致死率 1/1 会输出“绝境”提示。

        输入:
        - 无（测试内构造 observation）。

        返回:
        - 无。
        """
        parser = ObservationParser()
        observation = {
            "player_id": "p1",
            "table_type": "Q",
            "private_hand": ["K"],
            "pile_history": [],
            "legal_actions": [{"type": "challenge"}],
            "player_states": {"p1": {"death_probability": 1.0}},
        }

        text = parser.parse(observation)

        self.assertIn("绝境", text)

    def test_parameter_resolver_uses_deterministic_rounding_mapping(self) -> None:
        """作用: 验证 bluff_ratio + intended_total_cards 的确定性映射。

        输入:
        - 无（测试内构造手牌与参数）。

        返回:
        - 无。
        """
        resolver = ParameterResolver()
        resolved = resolver.resolve_strategic_drain(
            hand=["A", "JOKER", "K", "Q"],
            table_rank="A",
            skill_parameters={"bluff_ratio": 0.4, "intended_total_cards": 3},
        )

        self.assertEqual(resolved["resolved_fake_count"], 1)
        self.assertEqual(resolved["resolved_true_count"], 2)
        self.assertEqual(len(resolved["cards"]), 3)

    def test_parameter_resolver_enforces_mixed_true_card_when_available(self) -> None:
        """作用: 验证 Strategic_Drain 在有真牌时至少包含 1 张真牌。

        输入:
        - 无（测试内构造手牌与参数）。

        返回:
        - 无。
        """
        resolver = ParameterResolver()
        resolved = resolver.resolve_strategic_drain(
            hand=["A", "K", "Q"],
            table_rank="A",
            skill_parameters={"bluff_ratio": 0.95, "intended_total_cards": 2},
        )

        self.assertGreaterEqual(resolved["resolved_true_count"], 1)
        self.assertEqual(len(resolved["cards"]), 2)

    def test_liar_planner_auto_corrects_illegal_logical_skepticism(self) -> None:
        """作用: 验证无法质疑时会自动纠偏到 Truthful_Action 并记录偏差。

        输入:
        - 无（测试内构造 observation）。

        返回:
        - 无。
        """
        planner = LiarPlanner()
        observation = {
            "table_type": "A",
            "private_hand": ["A", "K"],
            "legal_actions": [{"type": "play_claim", "claim_rank": "A", "min_cards": 1, "max_cards": 2}],
        }

        outcome = planner.resolve_outcome(
            thought="I should challenge",
            selected_skill="Logical_Skepticism",
            skill_parameters={},
            observation=observation,
        )

        self.assertEqual(outcome.selected_skill, "Truthful_Action")
        self.assertIsNotNone(outcome.decision_bias)
        self.assertEqual(outcome.action.type, "play_claim")

    def test_null_probe_skill_marks_probe_type_and_uses_random_baseline_policy(self) -> None:
        """作用: 验证 Null_Probe_Skill 会标记 Probe 并按随机基线策略输出合法动作。

        输入:
        - 无（测试内构造 observation）。

        返回:
        - 无。
        """
        planner = LiarPlanner(enable_null_player_probe=True)
        observation = {
            "table_type": "A",
            "private_hand": ["A", "K"],
            "legal_actions": [{"type": "play_claim", "claim_rank": "A", "min_cards": 1, "max_cards": 2}],
        }

        outcome = planner.resolve_outcome(
            thought="Probe this state",
            selected_skill="Null_Probe_Skill",
            skill_parameters={},
            observation=observation,
        )

        self.assertEqual(outcome.selected_skill, "Null_Probe_Skill")
        self.assertEqual(outcome.skill_parameters.get("probe_type"), "Probe")
        self.assertEqual(outcome.skill_parameters.get("probe_policy"), "random_baseline")
        self.assertEqual(outcome.action.type, "play_claim")
        self.assertGreaterEqual(len(outcome.action.cards), 1)
        self.assertLessEqual(len(outcome.action.cards), 2)


if __name__ == "__main__":
    unittest.main()
