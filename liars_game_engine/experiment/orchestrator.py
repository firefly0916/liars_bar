from __future__ import annotations

from uuid import uuid4

from liars_game_engine.agents.base_agent import BaseAgent
from liars_game_engine.engine.environment import GameEnvironment
from liars_game_engine.engine.game_state import ActionModel
from liars_game_engine.experiment.logger import ExperimentLogger


class GameOrchestrator:
    LOG_VERSION = "v2_probe"

    def __init__(
        self,
        env: GameEnvironment,
        agents: dict[str, BaseAgent],
        logger: ExperimentLogger,
        fallback_action: str,
        max_turns: int,
    ) -> None:
        """作用: 初始化编排器与运行参数。

        输入:
        - env: 游戏环境实例。
        - agents: 玩家到 Agent 的映射。
        - logger: JSONL 日志写入器。
        - fallback_action: 非法动作后的降级动作类型。
        - max_turns: 最大回合数限制。

        返回:
        - 无。
        """
        self.env = env
        self.agents = agents
        self.logger = logger
        self.fallback_action = fallback_action
        self.max_turns = max_turns
        self.turn_checkpoints: dict[str, dict[str, object]] = {}

    @staticmethod
    def _build_state_features(observation: dict[str, object], player_id: str) -> dict[str, object]:
        """作用: 从 observation 提取任务 C 所需的 6 个核心状态特征。

        输入:
        - observation: 当前玩家观测。
        - player_id: 当前行动玩家 ID。

        返回:
        - dict[str, object]: phase/table_type/must_call_liar/alive_player_count/hand_count/death_probability。
        """
        player_states = observation.get("player_states", {})
        current_player_state = player_states.get(player_id, {}) if isinstance(player_states, dict) else {}
        if not isinstance(current_player_state, dict):
            current_player_state = {}

        alive_players = observation.get("alive_players", [])
        private_hand = observation.get("private_hand", [])

        death_probability_raw = current_player_state.get("death_probability", 0.0)
        try:
            death_probability = float(death_probability_raw)
        except (TypeError, ValueError):
            death_probability = 0.0

        return {
            "phase": str(observation.get("phase", "")),
            "table_type": str(observation.get("table_type", "A")),
            "must_call_liar": bool(observation.get("must_call_liar", False)),
            "alive_player_count": len(alive_players) if isinstance(alive_players, list) else 0,
            "hand_count": len(private_hand) if isinstance(private_hand, list) else 0,
            "death_probability": max(0.0, min(1.0, death_probability)),
        }

    async def run_game_loop(self) -> dict[str, object]:
        """作用: 执行完整对局主循环并记录每回合日志。

        输入:
        - 无。

        返回:
        - dict[str, object]: 对局摘要（回合数、是否结束、赢家、日志路径）。
        """
        turns_played = 0

        while turns_played < self.max_turns and not self.env.is_game_over():
            player_id = self.env.get_current_player()
            if player_id not in self.agents:
                break

            agent = self.agents[player_id]
            observation = self.env.get_observation_for(player_id)
            checkpoint = self.env.save_checkpoint()
            decision = await agent.act(observation)

            step_result = self.env.step(player_id, decision.action)
            fallback_used = False
            fallback_reason = None

            if not step_result.success:
                fallback_used = True
                fallback_reason = step_result.error_reason
                step_result = self.env.step(player_id, ActionModel(type=self.fallback_action))

            trace_id = f"turn-{turns_played + 1}-{uuid4().hex[:8]}"
            self.turn_checkpoints[trace_id] = checkpoint
            serialized_checkpoint = self.env.serialize_checkpoint(checkpoint)
            self.logger.record_turn(
                {
                    "log_version": self.LOG_VERSION,
                    "trace_id": trace_id,
                    "turn": turns_played + 1,
                    "player_id": player_id,
                    "observation": observation,
                    "thought": decision.thought,
                    "action": {
                        "type": decision.action.type,
                        "claim_rank": decision.action.claim_rank,
                        "cards": decision.action.cards,
                    },
                    "raw_output": decision.raw_output,
                    "skill_name": decision.selected_skill,
                    "skill_category": (
                        "Probe"
                        if str(decision.selected_skill or "") == "Null_Probe_Skill"
                        else "Standard"
                    ),
                    "skill_parameters": decision.skill_parameters,
                    "state_features": self._build_state_features(observation=observation, player_id=player_id),
                    "decision_bias": decision.decision_bias,
                    "checkpoint": {
                        "format": "pickle_base64_v1",
                        "payload": serialized_checkpoint,
                    },
                    "parser_error": (
                        {
                            "code": decision.parse_error.code,
                            "message": decision.parse_error.message,
                        }
                        if decision.parse_error
                        else None
                    ),
                    "fallback_used": fallback_used,
                    "fallback_reason": fallback_reason,
                    "step_result": {
                        "success": step_result.success,
                        "events": step_result.events,
                        "error_code": step_result.error_code,
                        "error_reason": step_result.error_reason,
                    },
                }
            )

            turns_played += 1

        alive_players = [
            player_id
            for player_id, runtime in self.env.state.players.items()
            if not runtime.eliminated
        ]
        winner = alive_players[0] if len(alive_players) == 1 else None
        return {
            "turns_played": turns_played,
            "game_over": self.env.is_game_over(),
            "winner": winner,
            "log_file": str(self.logger.log_file),
        }
