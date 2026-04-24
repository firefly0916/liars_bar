from __future__ import annotations

from pathlib import Path
import os

import yaml

from .schema import AppSettings


def _parse_dotenv(env_file: Path) -> dict[str, str]:
    """作用: 读取 .env 文件并解析为键值字典。

    输入:
    - env_file: .env 文件路径。

    返回:
    - dict[str, str]: 解析后的环境变量映射。
    """
    if not env_file.exists():
        return {}

    parsed: dict[str, str] = {}
    for line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def load_settings(config_file: Path | str = "config/experiment.yaml", env_file: Path | str = ".env") -> AppSettings:
    """作用: 合并 YAML 与 .env 配置并构造 AppSettings。

    输入:
    - config_file: 实验 YAML 配置路径。
    - env_file: .env 路径。

    返回:
    - AppSettings: 完整可用的运行配置对象。
    """
    config_path = Path(config_file)
    env_path = Path(env_file)

    yaml_data = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    if yaml_data is None:
        yaml_data = {}

    settings = AppSettings.from_dict(yaml_data)

    merged_env = _parse_dotenv(env_path)
    merged_env.update({key: value for key, value in os.environ.items() if value})

    api_key_override = (
        merged_env.get("OPENAI_API_KEY")
        or merged_env.get("OPENROUTER_API_KEY")
        or merged_env.get("VLLM_API_KEY")
    )
    if api_key_override:
        settings.api.openrouter_api_key = api_key_override

    base_url_override = (
        merged_env.get("OPENAI_BASE_URL")
        or merged_env.get("OPENROUTER_BASE_URL")
        or merged_env.get("VLLM_BASE_URL")
    )
    if base_url_override:
        settings.api.openrouter_base_url = base_url_override

    timeout_override = (
        merged_env.get("OPENAI_TIMEOUT_SECONDS")
        or merged_env.get("OPENROUTER_TIMEOUT_SECONDS")
        or merged_env.get("VLLM_TIMEOUT_SECONDS")
    )
    if timeout_override:
        settings.api.timeout_seconds = int(timeout_override)

    return settings
