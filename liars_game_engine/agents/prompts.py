from __future__ import annotations

from pathlib import Path

import yaml


DEFAULT_PROFILE = {
    "system": "You are a strategic Liar's Bar player.",
    "instruction": (
        "Return JSON with keys thought and action. "
        "action.type must be one of play_claim, challenge, pass."
    ),
    "output_schema": {
        "thought": "string",
        "action": {
            "type": "play_claim|challenge|pass",
            "claim_rank": "string|null",
            "cards": ["string"],
        },
    },
}


def load_prompt_profile(profile_name: str, profiles_dir: Path | str = "prompts/profiles") -> dict[str, object]:
    """作用: 加载指定 profile，不存在时回落到默认模板。

    输入:
    - profile_name: profile 文件名（不含扩展名）。
    - profiles_dir: profile 目录路径。

    返回:
    - dict[str, object]: 合并默认值后的 profile 配置。
    """
    base_dir = Path(profiles_dir)
    profile_path = base_dir / f"{profile_name}.yaml"
    if not profile_path.exists():
        return DEFAULT_PROFILE

    with profile_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}

    merged = dict(DEFAULT_PROFILE)
    merged.update(loaded)
    return merged


def build_prompt(profile: dict[str, object], observation: dict[str, object]) -> str:
    """作用: 将 profile 与 observation 拼装为提示词文本。

    输入:
    - profile: 包含 system/instruction 等字段的模板字典。
    - observation: 当前局面观测。

    返回:
    - str: 发送给模型的完整 prompt。
    """
    return (
        f"SYSTEM:\n{profile['system']}\n\n"
        f"INSTRUCTION:\n{profile['instruction']}\n\n"
        f"OBSERVATION:\n{observation}\n\n"
        "Respond in JSON only."
    )
