from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ApiSettings:
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    timeout_seconds: int = 60

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ApiSettings":
        """作用: 兼容通用 API 字段并构建 API 配置对象。

        输入:
        - raw: 原始 API 配置字典，允许使用 api_key/base_url 或历史字段名。

        返回:
        - ApiSettings: 标准化后的 API 配置实例。
        """
        normalized = dict(raw)
        if "api_key" in normalized and "openrouter_api_key" not in normalized:
            normalized["openrouter_api_key"] = normalized.pop("api_key")
        if "base_url" in normalized and "openrouter_base_url" not in normalized:
            normalized["openrouter_base_url"] = normalized.pop("base_url")
        return cls(**normalized)


@dataclass
class RuntimeSettings:
    max_turns: int = 120
    random_seed: int = 42
    fallback_action: str = "challenge"
    enable_null_player_probe: bool = False
    null_probe_action_probability: float = 0.12


@dataclass
class ParserSettings:
    max_retries: int = 3
    allow_markdown_json: bool = True
    allow_key_alias: bool = True


@dataclass
class LoggingSettings:
    run_log_dir: str = "logs/runs"
    level: str = "INFO"


@dataclass
class PlayerConfig:
    player_id: str
    name: str
    agent_type: str
    model: str
    prompt_profile: str
    temperature: float = 0.2


@dataclass
class RulesSettings:
    deck_ranks: list[str] = field(default_factory=lambda: ["A", "K", "Q", "JOKER"])
    cards_per_player: int = 5
    roulette_slots: int = 6
    enable_items: bool = False


@dataclass
class AppSettings:
    api: ApiSettings = field(default_factory=ApiSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    parser: ParserSettings = field(default_factory=ParserSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    players: list[PlayerConfig] = field(default_factory=list)
    rules: RulesSettings = field(default_factory=RulesSettings)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AppSettings":
        """作用: 从原始字典构建强类型配置对象。

        输入:
        - raw: 由 YAML/.env 聚合后的原始配置字典。

        返回:
        - AppSettings: 完整配置实例。
        """
        players = [PlayerConfig(**player) for player in raw.get("players", [])]
        return cls(
            api=ApiSettings.from_dict(raw.get("api", {})),
            runtime=RuntimeSettings(**raw.get("runtime", {})),
            parser=ParserSettings(**raw.get("parser", {})),
            logging=LoggingSettings(**raw.get("logging", {})),
            players=players,
            rules=RulesSettings(**raw.get("rules", {})),
        )
