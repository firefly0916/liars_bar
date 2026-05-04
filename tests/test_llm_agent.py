import importlib
import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from liars_game_engine.agents.action_resolver import resolve_action_from_intent
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
                        content='{"Reasoning":"A small bluff keeps the initiative without overcommitting.","Action":{"type":"play_claim","play_count":1,"true_card_count":0}}'
                    )
                )
            ]
        )


class LlmAgentTest(unittest.IsolatedAsyncioTestCase):
    async def test_build_agents_uses_llm_agent_and_executes_single_openai_call(self) -> None:
        """Verify the LLM agent path uses one OpenAI-compatible call and parses JSON."""
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

        self.assertEqual(decision.thought, "A small bluff keeps the initiative without overcommitting.")
        self.assertEqual(decision.action.type, "play_claim")
        self.assertEqual(decision.action.claim_rank, "A")
        self.assertEqual(decision.action.cards, ["K"])
        self.assertEqual(decision.action_intent["play_count"], 1)
        self.assertEqual(decision.action_intent["true_card_count"], 0)
        self.assertIsNone(decision.resolution_reason)

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

    async def test_llm_agent_supports_local_hf_backend_without_http_client(self) -> None:
        settings = AppSettings.from_dict(
            {
                "api": {
                    "api_key": "LOCAL",
                    "base_url": "local://hf",
                },
                "players": [
                    {
                        "player_id": "p1",
                        "name": "Alice",
                        "agent_type": "llm",
                        "model": "Qwen/Qwen2.5-0.5B-Instruct",
                        "prompt_profile": "baseline",
                        "temperature": 0.0,
                    }
                ],
            }
        )
        observation = {
            "player_id": "p1",
            "phase": "response_window",
            "current_player_id": "p1",
            "table_type": "A",
            "private_hand": ["Q", "K"],
            "pending_claim": {"actor_id": "p2", "claim_rank": "A", "declared_count": 2},
            "must_call_liar": False,
            "alive_players": ["p1", "p2", "p3"],
            "player_states": {"p1": {"death_probability": 1.0 / 3.0}},
            "legal_actions": [
                {"type": "challenge"},
                {"type": "play_claim", "claim_rank": "A", "min_cards": 1, "max_cards": 2},
            ],
        }

        agent = build_agents(settings)["p1"]
        llm_agent_module = importlib.import_module("liars_game_engine.agents.llm_agent")

        async def fake_local_completion(**kwargs):
            self.assertEqual(kwargs["model"], "Qwen/Qwen2.5-0.5B-Instruct")
            self.assertEqual(kwargs["temperature"], 0.0)
            self.assertEqual(kwargs["messages"][0]["role"], "system")
            return '{"Reasoning":"The roulette risk is too high, so challenge immediately.","Action":{"type":"challenge"}}'

        with patch.object(llm_agent_module, "AsyncOpenAI", None), patch.object(
            llm_agent_module,
            "generate_local_chat_completion",
            fake_local_completion,
            create=True,
        ):
            decision = await agent.act(observation)

        self.assertEqual(decision.thought, "The roulette risk is too high, so challenge immediately.")
        self.assertEqual(decision.action.type, "challenge")

    async def test_llm_agent_falls_back_when_local_backend_raises(self) -> None:
        settings = AppSettings.from_dict(
            {
                "api": {
                    "api_key": "LOCAL",
                    "base_url": "local://hf",
                },
                "players": [
                    {
                        "player_id": "p1",
                        "name": "Alice",
                        "agent_type": "llm",
                        "model": "Qwen/Qwen2.5-0.5B-Instruct",
                        "prompt_profile": "baseline",
                        "temperature": 0.0,
                    }
                ],
            }
        )
        agent = build_agents(settings)["p1"]
        llm_agent_module = importlib.import_module("liars_game_engine.agents.llm_agent")

        async def fake_local_completion(**kwargs):
            raise RuntimeError("model load failed")

        with patch.object(llm_agent_module, "generate_local_chat_completion", fake_local_completion):
            decision = await agent.act(
                {
                    "player_id": "p1",
                    "phase": "response_window",
                    "current_player_id": "p1",
                    "table_type": "A",
                    "private_hand": ["Q", "K"],
                    "pending_claim": {"actor_id": "p2", "claim_rank": "A", "declared_count": 2},
                    "must_call_liar": False,
                    "alive_players": ["p1", "p2", "p3"],
                    "player_states": {"p1": {"death_probability": 1.0 / 3.0}},
                    "legal_actions": [{"type": "challenge"}],
                }
            )

        self.assertEqual(decision.action.type, "challenge")
        self.assertIsNotNone(decision.parse_error)
        self.assertEqual(decision.parse_error.code, "E_AGENT_PROVIDER_UNAVAILABLE")

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
        self.assertIn("honesty_reference", user_prompt)
        self.assertIn("persona_stability", user_prompt)
        self.assertIn("roulette_death_probability", user_prompt)
        self.assertIn("Reasoning", user_prompt)
        self.assertIn("Action", user_prompt)
        self.assertIn("play_count", user_prompt)
        self.assertIn("true_card_count", user_prompt)
        self.assertNotIn('"claim_rank":', user_prompt)
        self.assertNotIn("phi", user_prompt.lower())
        self.assertNotIn("selected_skill", user_prompt.lower())
        self.assertNotIn("skill_parameters", user_prompt.lower())

    def test_build_openai_messages_omits_pass_when_pass_is_not_legal(self) -> None:
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
        user_prompt = messages[1]["content"].lower()

        self.assertNotIn("pass", user_prompt)
        self.assertIn("play_claim|challenge", user_prompt)

    def test_resolver_redirects_illegal_pass_to_minimum_risk_legal_play(self) -> None:
        observation = {
            "player_id": "p1",
            "phase": "response_window",
            "current_player_id": "p1",
            "table_type": "A",
            "private_hand": ["A", "Q", "K"],
            "pending_claim": {"actor_id": "p2", "claim_rank": "A", "declared_count": 2},
            "must_call_liar": False,
            "legal_actions": [
                {"type": "challenge"},
                {"type": "play_claim", "claim_rank": "A", "min_cards": 1, "max_cards": 2},
            ],
        }

        decision = resolve_action_from_intent(observation=observation, action_type="pass")

        self.assertEqual(decision.action.type, "play_claim")
        self.assertEqual(decision.action.claim_rank, "A")
        self.assertEqual(decision.action.cards, ["A"])
        self.assertIn("illegal_pass_redirection", str(decision.resolution_reason))

    async def test_llm_agent_consistently_downgrades_true_card_count_when_hand_cannot_support_intent(self) -> None:
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
            "private_hand": ["A", "Q", "K"],
            "pending_claim": None,
            "must_call_liar": False,
            "legal_actions": [{"type": "play_claim", "claim_rank": "A", "min_cards": 1, "max_cards": 3}],
        }

        class _IntentAsyncOpenAI:
            def __init__(self, **kwargs) -> None:
                self.chat = SimpleNamespace(
                    completions=SimpleNamespace(
                        create=self._create,
                    )
                )

            async def _create(self, **kwargs):
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content='{"Reasoning":"I want to pressure the table with two truthful cards.","Action":{"type":"play_claim","play_count":2,"true_card_count":2}}'
                            )
                        )
                    ]
                )

        agent = build_agents(settings)["p1"]
        llm_agent_module = importlib.import_module("liars_game_engine.agents.llm_agent")
        with patch.object(llm_agent_module, "AsyncOpenAI", _IntentAsyncOpenAI):
            decision = await agent.act(observation)

        self.assertEqual(decision.action.type, "play_claim")
        self.assertEqual(decision.action.claim_rank, "A")
        self.assertEqual(decision.action.cards, ["A", "Q"])
        self.assertEqual(decision.action_intent["play_count"], 2)
        self.assertEqual(decision.action_intent["true_card_count"], 2)
        self.assertIn("downgraded", str(decision.resolution_reason))
        self.assertIn("requested_true=2", str(decision.resolution_reason))
        self.assertIn("resolved_true=1", str(decision.resolution_reason))

    @unittest.skipUnless(importlib.util.find_spec("transformers"), "transformers not installed")
    def test_build_openai_messages_stays_under_400_tokens_for_task_m(self) -> None:
        from transformers import AutoTokenizer

        profile = load_prompt_profile("baseline")
        observation = {
            "player_id": "p1",
            "phase": "response_window",
            "current_player_id": "p1",
            "table_type": "A",
            "private_hand": ["Q", "K", "JOKER", "Q", "K"],
            "pending_claim": {"actor_id": "p2", "claim_rank": "A", "declared_count": 2},
            "must_call_liar": False,
            "alive_players": ["p1", "p2", "p3", "p4"],
            "player_states": {"p1": {"death_probability": 1.0 / 3.0}},
            "legal_actions": [
                {"type": "challenge"},
                {"type": "play_claim", "claim_rank": "A", "min_cards": 1, "max_cards": 3},
            ],
        }

        messages = build_openai_messages(profile, observation)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-0.5B-Instruct",
                trust_remote_code=True,
                local_files_only=True,
            )
        except OSError as exc:
            self.skipTest(f"local tokenizer cache unavailable: {exc}")
        prompt_tokens = tokenizer(messages[1]["content"], return_tensors="pt")["input_ids"].shape[-1]

        self.assertLess(
            prompt_tokens,
            400,
            f"Task M prompt is too long: {prompt_tokens} tokens",
        )


if __name__ == "__main__":
    unittest.main()
