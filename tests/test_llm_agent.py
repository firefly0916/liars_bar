import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from liars_game_engine.agents.factory import build_agents
from liars_game_engine.agents.prompts import build_openai_messages, load_prompt_profile
from liars_game_engine.config.schema import AppSettings


class FakeAsyncOpenAI:
    instances: list["FakeAsyncOpenAI"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.chat = SimpleNamespace(completions=FakeCompletions())
        type(self).instances.append(self)


class FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"Reasoning":"继续吹牛还能保留主动权。","Action":{"type":"play_claim","claim_rank":"A","cards":["K"]}}'
                    )
                )
            ]
        )


class LlmAgentTest(unittest.IsolatedAsyncioTestCase):
    async def test_build_agents_uses_llm_agent_and_executes_single_openai_call(self) -> None:
        """作用: 验证新 llm agent_type 走单次 OpenAI-compatible 调用并解析动作 JSON。

        输入:
        - 无（测试内构造 settings、observation 与假 OpenAI 客户端）。

        返回:
        - 无。
        """
        FakeAsyncOpenAI.instances.clear()
        settings = AppSettings.from_dict(
            {
                "api": {
                    "api_key": "test-key",
                    "base_url": "http://127.0.0.1:9999/v1",
                },
                "players": [
                    {
                        "player_id": "p1",
                        "name": "Alice",
                        "agent_type": "llm",
                        "model": "unit-test-model",
                        "prompt_profile": "baseline",
                        "temperature": 0.1,
                    }
                ],
            }
        )
        observation = {
            "player_id": "p1",
            "phase": "turn_start",
            "current_player_id": "p1",
            "table_type": "A",
            "private_hand": ["K", "Q"],
            "pending_claim": None,
            "must_call_liar": False,
            "legal_actions": [{"type": "play_claim", "claim_rank": "A", "min_cards": 1, "max_cards": 2}],
        }

        agent = build_agents(settings)["p1"]

        self.assertEqual(agent.__class__.__name__, "LlmAgent")

        llm_agent_module = importlib.import_module("liars_game_engine.agents.llm_agent")
        with patch.object(llm_agent_module, "AsyncOpenAI", FakeAsyncOpenAI):
            decision = await agent.act(observation)

        self.assertEqual(decision.thought, "继续吹牛还能保留主动权。")
        self.assertEqual(decision.action.type, "play_claim")
        self.assertEqual(decision.action.claim_rank, "A")
        self.assertEqual(decision.action.cards, ["K"])

        self.assertEqual(len(FakeAsyncOpenAI.instances), 1)
        client = FakeAsyncOpenAI.instances[0]
        self.assertEqual(client.kwargs["api_key"], "test-key")
        self.assertEqual(client.kwargs["base_url"], "http://127.0.0.1:9999/v1")

        self.assertEqual(len(client.chat.completions.calls), 1)
        request = client.chat.completions.calls[0]
        self.assertEqual(request["model"], "unit-test-model")
        self.assertEqual(request["temperature"], 0.1)
        self.assertEqual(request["messages"][0]["role"], "system")
        self.assertEqual(request["messages"][1]["role"], "user")
        self.assertIn("Status Report", request["messages"][1]["content"])
        self.assertIn("Qualitative Context", request["messages"][1]["content"])
        self.assertIn("Decision Request", request["messages"][1]["content"])

    def test_build_openai_messages_uses_three_section_prompt_and_reasoning_action_contract(self) -> None:
        profile = load_prompt_profile("baseline")
        observation = {
            "player_id": "p1",
            "phase": "response_window",
            "current_player_id": "p1",
            "table_type": "A",
            "private_hand": ["A", "Q", "Q", "K"],
            "pending_claim": {"actor_id": "p2", "claim_rank": "A", "declared_count": 2},
            "must_call_liar": False,
            "alive_players": ["p1", "p2", "p3"],
            "player_states": {"p1": {"death_probability": 1.0 / 3.0}},
            "legal_actions": [
                {"type": "challenge"},
                {"type": "play_claim", "claim_rank": "A", "min_cards": 1, "max_cards": 2},
            ],
        }

        messages = build_openai_messages(profile, observation)

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        user_prompt = messages[1]["content"]
        self.assertIn("Status Report", user_prompt)
        self.assertIn("Qualitative Context", user_prompt)
        self.assertIn("Decision Request", user_prompt)
        self.assertIn("诚实度参考", user_prompt)
        self.assertIn("人设稳定性", user_prompt)
        self.assertIn("轮盘死亡概率", user_prompt)
        self.assertIn("Reasoning", user_prompt)
        self.assertIn("Action", user_prompt)
        self.assertNotIn("phi", user_prompt.lower())


if __name__ == "__main__":
    unittest.main()
