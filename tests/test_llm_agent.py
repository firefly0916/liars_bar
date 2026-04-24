import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from liars_game_engine.agents.factory import build_agents
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
                        content='{"thought":"Play the safe off-rank card.","action":{"type":"play_claim","claim_rank":"A","cards":["K"]}}'
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

        self.assertEqual(decision.thought, "Play the safe off-rank card.")
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


if __name__ == "__main__":
    unittest.main()
