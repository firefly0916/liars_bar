import json
import tempfile
import unittest

from liars_game_engine.agents.factory import build_agents
from liars_game_engine.config.schema import AppSettings
from liars_game_engine.engine.environment import GameEnvironment
from liars_game_engine.experiment.logger import ExperimentLogger
from liars_game_engine.experiment.orchestrator import GameOrchestrator


class OrchestratorLoggingTest(unittest.IsolatedAsyncioTestCase):
    def _make_settings(self) -> AppSettings:
        """作用: 构造编排器日志测试使用的最小可运行配置。

        输入:
        - 无。

        返回:
        - AppSettings: 可用于初始化环境和 Agent 的配置对象。
        """
        return AppSettings.from_dict(
            {
                "runtime": {"max_turns": 12, "random_seed": 3, "fallback_action": "challenge"},
                "parser": {"max_retries": 3, "allow_markdown_json": True, "allow_key_alias": True},
                "logging": {"run_log_dir": "logs/runs", "level": "INFO"},
                "players": [
                    {
                        "player_id": "p1",
                        "name": "Alice",
                        "agent_type": "mock",
                        "model": "openai/gpt-4o-mini",
                        "prompt_profile": "baseline",
                    },
                    {
                        "player_id": "p2",
                        "name": "Bob",
                        "agent_type": "mock",
                        "model": "anthropic/claude-3.5-sonnet",
                        "prompt_profile": "baseline",
                    },
                ],
                "rules": {
                    "deck_ranks": ["A", "K"],
                    "cards_per_player": 1,
                    "roulette_slots": 1,
                    "enable_items": False,
                },
            }
        )

    async def test_orchestrator_records_turn_logs(self) -> None:
        """作用: 验证 orchestrator 会写入包含关键字段的回合日志。

        输入:
        - 无（测试内创建临时目录并运行对局）。

        返回:
        - 无。
        """
        settings = self._make_settings()

        with tempfile.TemporaryDirectory() as temp_dir:
            settings.logging.run_log_dir = temp_dir
            env = GameEnvironment(settings)
            agents = build_agents(settings)
            logger = ExperimentLogger(base_dir=temp_dir, game_id="test-game")
            orchestrator = GameOrchestrator(
                env=env,
                agents=agents,
                logger=logger,
                fallback_action=settings.runtime.fallback_action,
                max_turns=settings.runtime.max_turns,
            )

            summary = await orchestrator.run_game_loop()

            self.assertGreaterEqual(summary["turns_played"], 1)
            self.assertTrue(logger.log_file.exists())

            lines = logger.log_file.read_text(encoding="utf-8").splitlines()
            self.assertGreaterEqual(len(lines), 1)

            first_record = json.loads(lines[0])
            self.assertIn("player_id", first_record)
            self.assertIn("observation", first_record)
            self.assertIn("action", first_record)
            self.assertIn("step_result", first_record)
            self.assertIn("trace_id", first_record)
            self.assertIn("skill_name", first_record)
            self.assertIn("skill_parameters", first_record)
            self.assertIn("checkpoint", first_record)
            self.assertIn("state_features", first_record)
            self.assertIn("log_version", first_record)
            self.assertIn("skill_category", first_record)
            self.assertEqual(first_record["checkpoint"]["format"], "pickle_base64_v1")
            self.assertTrue(first_record["checkpoint"]["payload"])
            self.assertEqual(first_record["log_version"], "v2_probe")

            state_features = first_record["state_features"]
            self.assertEqual(
                set(state_features.keys()),
                {
                    "phase",
                    "table_type",
                    "must_call_liar",
                    "alive_player_count",
                    "hand_count",
                    "death_probability",
                },
            )


if __name__ == "__main__":
    unittest.main()
