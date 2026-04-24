import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from liars_game_engine.config.schema import AppSettings
from liars_game_engine.experiment.llm_drill import run_llm_drill, select_representative_decisions


class LlmDrillTest(unittest.IsolatedAsyncioTestCase):
    def _make_settings(self) -> AppSettings:
        return AppSettings.from_dict(
            {
                "api": {
                    "api_key": "LOCAL",
                    "base_url": "local://hf",
                },
                "runtime": {"max_turns": 40, "random_seed": 11, "fallback_action": "challenge"},
                "logging": {"run_log_dir": "logs/llm_drill", "level": "INFO"},
                "players": [
                    {
                        "player_id": "p1",
                        "name": "Llm",
                        "agent_type": "llm",
                        "model": "Qwen/Qwen2.5-0.5B-Instruct",
                        "prompt_profile": "baseline",
                        "temperature": 0.0,
                    },
                    {
                        "player_id": "p2",
                        "name": "Mock-2",
                        "agent_type": "mock",
                        "model": "mock-a",
                        "prompt_profile": "baseline",
                    },
                    {
                        "player_id": "p3",
                        "name": "Mock-3",
                        "agent_type": "mock",
                        "model": "mock-b",
                        "prompt_profile": "baseline",
                    },
                    {
                        "player_id": "p4",
                        "name": "Mock-4",
                        "agent_type": "mock",
                        "model": "mock-c",
                        "prompt_profile": "baseline",
                    },
                ],
                "rules": {
                    "deck_ranks": ["A", "K", "Q", "JOKER"],
                    "cards_per_player": 2,
                    "roulette_slots": 2,
                    "enable_items": False,
                },
            }
        )

    async def test_run_llm_drill_writes_game_logs_and_summary(self) -> None:
        settings = self._make_settings()

        async def fake_local_completion(**kwargs):
            user_prompt = kwargs["messages"][1]["content"]
            if "轮盘死亡概率: 50%" in user_prompt:
                return '{"Reasoning":"死亡风险偏高，先质疑止损。","Action":{"type":"challenge"}}'
            return '{"Reasoning":"当前风险可控，继续维持叙事。","Action":{"type":"play_claim","claim_rank":"A","cards":["K"]}}'

        with tempfile.TemporaryDirectory() as temp_dir:
            llm_agent_module = __import__("liars_game_engine.agents.llm_agent", fromlist=["dummy"])
            with patch.object(llm_agent_module, "generate_local_chat_completion", fake_local_completion):
                summary = await run_llm_drill(settings=settings, games=2, log_dir=temp_dir)

            self.assertEqual(summary["total_games"], 2)
            self.assertGreater(summary["llm_turn_count"], 0)
            self.assertIn("parse_error_rate", summary)
            self.assertLessEqual(summary["parse_error_rate"], 1.0)
            self.assertEqual(len(summary["game_summaries"]), 2)
            self.assertLessEqual(len(summary["representative_decisions"]), 3)

            games_dir = Path(temp_dir) / "games"
            self.assertEqual(len(list(games_dir.glob("*.jsonl"))), 2)

            summary_path = Path(temp_dir) / "summary.json"
            self.assertTrue(summary_path.exists())
            persisted = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["total_games"], 2)

    def test_select_representative_decisions_prefers_high_risk_and_low_honesty(self) -> None:
        decisions = [
            {
                "turn": 1,
                "death_probability": 0.1,
                "honesty_ratio": 0.8,
                "thought": "safe",
                "action": {"type": "play_claim"},
                "observation": {"player_id": "p1"},
            },
            {
                "turn": 2,
                "death_probability": 0.5,
                "honesty_ratio": 0.0,
                "thought": "danger",
                "action": {"type": "challenge"},
                "observation": {"player_id": "p1"},
            },
            {
                "turn": 3,
                "death_probability": 0.33,
                "honesty_ratio": 0.25,
                "thought": "suspicious",
                "action": {"type": "play_claim"},
                "observation": {"player_id": "p1"},
            },
        ]

        selected = select_representative_decisions(decisions, limit=2)

        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0]["thought"], "danger")
        self.assertEqual(selected[1]["thought"], "suspicious")

    def test_run_llm_drill_script_is_invokable_from_repo_root(self) -> None:
        completed = subprocess.run(
            [sys.executable, "scripts/run_llm_drill.py", "--help"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Run a lightweight LLM drill", completed.stdout)


if __name__ == "__main__":
    unittest.main()
