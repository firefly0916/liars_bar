from __future__ import annotations

import json
import random

from liars_game_engine.agents.base_agent import AgentDecision, BaseAgent
from liars_game_engine.agents.liar_planner import LiarPlanner


class MockAgent(BaseAgent):
    def __init__(
        self,
        player_id: str,
        model: str,
        prompt_profile: str,
        temperature: float,
        seed: int = 0,
        enable_null_player_probe: bool = False,
    ) -> None:
        """作用: 初始化可复现的离线 MockAgent。

        输入:
        - player_id: 玩家 ID。
        - model: 模型名占位字段。
        - prompt_profile: prompt profile 名称。
        - temperature: 温度占位字段。
        - seed: 随机种子，控制策略随机性。

        返回:
        - 无。
        """
        super().__init__(player_id=player_id, model=model, prompt_profile=prompt_profile, temperature=temperature)
        self.rng = random.Random(seed)
        self.enable_null_player_probe = enable_null_player_probe
        self.planner = LiarPlanner(enable_null_player_probe=enable_null_player_probe)

    async def act(self, observation: dict[str, object]) -> AgentDecision:
        """作用: 按内置启发式策略输出挑战、出牌或跳过动作。

        输入:
        - observation: 当前玩家可见状态，含 phase、private_hand、pending_claim 等。

        返回:
        - AgentDecision: 结构化动作与对应 thought/raw_output。
        """
        phase = str(observation.get("phase", ""))
        hand = [str(card) for card in observation.get("private_hand", [])]
        pending_claim = observation.get("pending_claim")
        table_type = str(observation.get("table_type", "A"))
        must_call_liar = bool(observation.get("must_call_liar", False))
        legal_actions = observation.get("legal_actions", [])
        legal_types = {
            str(item.get("type", ""))
            for item in legal_actions
            if isinstance(item, dict)
        }

        selected_skill = "Truthful_Action"
        skill_parameters: dict[str, object] = {}
        thought = "I will play safely with truthful action."

        if phase == "response_window" and isinstance(pending_claim, dict):
            actor_id = pending_claim.get("actor_id")
            if actor_id != self.player_id and (must_call_liar or self.rng.random() < 0.45):
                selected_skill = "Logical_Skepticism"
                thought = "I suspect bluff, challenge now."
                skill_parameters = {}

        if selected_skill != "Logical_Skepticism" and hand:
            true_cards = [card for card in hand if card in {table_type, "JOKER"}]
            fake_cards = [card for card in hand if card not in {table_type, "JOKER"}]

            if true_cards and fake_cards and self.rng.random() < 0.2:
                selected_skill = "Strategic_Drain"
                intended_total = self.rng.randint(1, min(3, len(hand)))
                skill_parameters = {
                    "bluff_ratio": round(self.rng.uniform(0.2, 0.8), 2),
                    "intended_total_cards": intended_total,
                }
                thought = "I will use mixed play to drain cards while masking intent."
            elif not true_cards and len(fake_cards) >= 2 and self.rng.random() < 0.35:
                selected_skill = "Aggressive_Deception"
                skill_parameters = {"intended_total_cards": min(3, len(fake_cards))}
                thought = "I need to dump fake cards quickly with aggressive deception."
            elif not true_cards:
                selected_skill = "Calculated_Bluff"
                skill_parameters = {"intended_total_cards": 1}
                thought = "I have no true cards, so I will bluff cautiously."
            else:
                selected_skill = "Truthful_Action"
                skill_parameters = {}
                thought = "I can play true cards and avoid roulette risk."

        if not hand and "challenge" in legal_types:
            selected_skill = "Logical_Skepticism"
            skill_parameters = {}
            thought = "No cards left; challenge is my only strong option."

        if not hand and "pass" in legal_types and "challenge" not in legal_types:
            selected_skill = "Truthful_Action"
            skill_parameters = {}
            thought = "No cards left; planner should route to pass."

        if self.enable_null_player_probe and hand and "play_claim" in legal_types and self.rng.random() < 0.12:
            selected_skill = "Null_Probe_Skill"
            skill_parameters = {"probe_type": "Probe"}
            thought = "Probe mode: execute random-baseline legal action for null-player analysis."

        outcome = self.planner.resolve_outcome(
            thought=thought,
            selected_skill=selected_skill,
            skill_parameters=skill_parameters,
            observation=observation,
        )

        raw_output = json.dumps(
            {
                "thought": thought,
                "selected_skill": selected_skill,
                "skill_parameters": skill_parameters,
            },
            ensure_ascii=False,
        )

        return AgentDecision(
            thought=outcome.thought,
            action=outcome.action,
            raw_output=raw_output,
            selected_skill=outcome.selected_skill,
            skill_parameters=outcome.skill_parameters,
            decision_bias=outcome.decision_bias,
        )
