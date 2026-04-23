from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class ExperimentLogger:
    def __init__(self, base_dir: str | Path, game_id: str) -> None:
        """作用: 初始化实验日志写入器并创建日志文件路径。

        输入:
        - base_dir: 日志目录。
        - game_id: 本局实验唯一标识。

        返回:
        - 无。
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.game_id = game_id
        self.log_file = self.base_dir / f"{game_id}.jsonl"

    def record_turn(self, payload: dict[str, object]) -> None:
        """作用: 追加写入单回合 JSONL 日志记录。

        输入:
        - payload: 回合记录数据字典。

        返回:
        - 无。
        """
        enriched = dict(payload)
        enriched.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        with self.log_file.open("a", encoding="utf-8") as file:
            file.write(json.dumps(enriched, ensure_ascii=False) + "\n")
