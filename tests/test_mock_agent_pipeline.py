import json
import unittest

from liars_game_engine.agents.factory import build_agents
from liars_game_engine.agents.mock_agent import MockAgent
from liars_game_engine.config.schema import AppSettings


class MockAgentPipelineTest(unittest.IsolatedAsyncioTestCase):
    async def test_mock_agent_outputs_skill_schema_and_resolved_action(self) -> None:
        """作用: 验证 MockAgent 走 Skill 管线并返回可执行动作。

        输入:
        - 无（测试内构造 observation）。

        返回:
        - 无。
        """
        agent = MockAgent(player_id="p1", model="m", prompt_profile="baseline", temperature=0.2, seed=1)
        observation = {
            "player_id": "p1",
            "phase": "turn_start",
            "current_player_id": "p1",
            "table_type": "A",
            "private_hand": ["A", "K", "Q"],
            "pending_claim": None,
            "must_call_liar": False,
            "legal_actions": [{"type": "play_claim", "claim_rank": "A", "min_cards": 1, "max_cards": 3}],
        }

        decision = await agent.act(observation)
        payload = json.loads(decision.raw_output)

        self.assertIn("selected_skill", payload)
        self.assertIn("skill_parameters", payload)
        self.assertIn(decision.action.type, {"play_claim", "challenge", "pass"})
        self.assertIsNotNone(decision.selected_skill)

    async def test_mock_agent_uses_skill_pipeline_for_forced_challenge(self) -> None:
        """作用: 验证强制质疑场景下 MockAgent 仍通过 Skill 决策触发 challenge。

        输入:
        - 无（测试内构造 observation）。

        返回:
        - 无。
        """
        agent = MockAgent(player_id="p1", model="m", prompt_profile="baseline", temperature=0.2, seed=2)
        observation = {
            "player_id": "p1",
            "phase": "response_window",
            "current_player_id": "p1",
            "table_type": "Q",
            "private_hand": [],
            "pending_claim": {"actor_id": "p2", "claim_rank": "Q", "declared_count": 1},
            "must_call_liar": True,
            "legal_actions": [{"type": "challenge"}],
        }

        decision = await agent.act(observation)

        self.assertEqual(decision.selected_skill, "Logical_Skepticism")
        self.assertEqual(decision.action.type, "challenge")

    async def test_build_agents_mounts_null_probe_skill_when_enabled(self) -> None:
        """作用: 验证开启开关后会把 Null_Probe_Skill 挂载到 Planner。

        输入:
        - 无（测试内构造 AppSettings 并调用 build_agents）。

        返回:
        - 无。
        """
        settings = AppSettings.from_dict(
            {
                "runtime": {
                    "max_turns": 12,
                    "random_seed": 2,
                    "fallback_action": "challenge",
                    "enable_null_player_probe": True,
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

        agents = build_agents(settings)
        first_agent = agents["p1"]

        self.assertIn("Null_Probe_Skill", first_agent.planner.available_skills)


if __name__ == "__main__":
    unittest.main()
