import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from liars_game_engine.config.loader import load_settings


class ConfigLoaderTest(unittest.TestCase):
    def test_load_settings_merges_env_and_yaml(self) -> None:
        """作用: 验证 .env 与 YAML 能按预期合并到设置对象。

        输入:
        - 无（测试内构造临时配置文件）。

        返回:
        - 无。
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            env_file = base / ".env"
            env_file.write_text(
                "OPENROUTER_API_KEY=test-key\nOPENROUTER_BASE_URL=https://openrouter.ai/api/v1\n",
                encoding="utf-8",
            )

            config_file = base / "experiment.yaml"
            config_file.write_text(
                """
api:
  timeout_seconds: 45
runtime:
  max_turns: 20
  random_seed: 7
  fallback_action: challenge
  enable_null_player_probe: true
parser:
  max_retries: 3
  allow_markdown_json: true
  allow_key_alias: true
logging:
  run_log_dir: logs/runs
  level: INFO
players:
  - player_id: p1
    name: Alice
    agent_type: mock
    model: openai/gpt-4o-mini
    prompt_profile: baseline
    temperature: 0.2
  - player_id: p2
    name: Bob
    agent_type: langchain
    model: anthropic/claude-3.5-sonnet
    prompt_profile: deceptive
    temperature: 0.5
rules:
  deck_ranks: [A, K, Q, J]
  cards_per_player: 5
  roulette_slots: 6
  enable_items: false
""",
                encoding="utf-8",
            )

            settings = load_settings(config_file=config_file, env_file=env_file)

            self.assertEqual(settings.api.openrouter_api_key, "test-key")
            self.assertEqual(settings.api.openrouter_base_url, "https://openrouter.ai/api/v1")
            self.assertEqual(settings.api.timeout_seconds, 45)
            self.assertEqual(settings.players[1].model, "anthropic/claude-3.5-sonnet")
            self.assertEqual(settings.runtime.fallback_action, "challenge")
            self.assertTrue(settings.runtime.enable_null_player_probe)

    def test_load_settings_supports_generic_api_keys_from_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            config_file = base / "experiment.yaml"
            config_file.write_text(
                """
api:
  api_key: generic-key
  base_url: http://127.0.0.1:8000/v1
  timeout_seconds: 30
players: []
""",
                encoding="utf-8",
            )

            settings = load_settings(config_file=config_file, env_file=base / ".env.missing")

            self.assertEqual(settings.api.openrouter_api_key, "generic-key")
            self.assertEqual(settings.api.openrouter_base_url, "http://127.0.0.1:8000/v1")
            self.assertEqual(settings.api.timeout_seconds, 30)

    def test_process_environment_overrides_dotenv_with_openai_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            env_file = base / ".env"
            env_file.write_text(
                "OPENROUTER_API_KEY=dotenv-key\nOPENROUTER_BASE_URL=https://openrouter.ai/api/v1\n",
                encoding="utf-8",
            )
            config_file = base / "experiment.yaml"
            config_file.write_text("players: []\n", encoding="utf-8")

            with patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "runtime-key",
                    "OPENAI_BASE_URL": "http://127.0.0.1:8000/v1",
                    "OPENAI_TIMEOUT_SECONDS": "15",
                },
                clear=False,
            ):
                settings = load_settings(config_file=config_file, env_file=env_file)

            self.assertEqual(settings.api.openrouter_api_key, "runtime-key")
            self.assertEqual(settings.api.openrouter_base_url, "http://127.0.0.1:8000/v1")
            self.assertEqual(settings.api.timeout_seconds, 15)

    def test_repo_experiment_config_is_drill_ready_for_task_m(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "",
                "OPENAI_BASE_URL": "",
                "OPENROUTER_API_KEY": "",
                "OPENROUTER_BASE_URL": "",
                "VLLM_API_KEY": "",
                "VLLM_BASE_URL": "",
            },
            clear=False,
        ):
            settings = load_settings(
                config_file=Path("config/experiment.yaml"),
                env_file=Path(".env.missing"),
            )

        self.assertEqual(settings.api.openrouter_api_key, "LOCAL")
        self.assertEqual(settings.api.openrouter_base_url, "local://hf")
        self.assertEqual(len(settings.players), 4)
        self.assertEqual(settings.players[0].agent_type, "llm")
        self.assertEqual(settings.players[0].model, "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertEqual(sum(1 for player in settings.players if player.agent_type == "mock"), 3)


if __name__ == "__main__":
    unittest.main()
