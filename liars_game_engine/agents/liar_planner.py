from __future__ import annotations

from dataclasses import dataclass
import random

from liars_game_engine.engine.game_state import ActionModel, JOKER_RANK


SKILL_DEFINITIONS: dict[str, str] = {
    "Truthful_Action": "诚实出牌：优先打出 1-3 张真实桌面牌或 Joker，可绝对规避被质疑惩罚。",
    "Calculated_Bluff": "谨慎吹牛：仅用 1 张假牌伪装，风险较低，适合消耗无用牌。",
    "Aggressive_Deception": "激进强攻：一次打出 2-3 张假牌，快速清手但显著提高被质疑概率。",
    "Logical_Skepticism": "逻辑质疑：直接挑战上家声明，适合牌池和手牌推断为高概率必假时。",
    "Strategic_Drain": "策略性消牌：混合真假牌出牌，可通过 bluff_ratio 调节真假比例。",
}
NULL_PROBE_SKILL_NAME = "Null_Probe_Skill"
NULL_PROBE_SKILL_DESCRIPTION = "空玩家探测：按随机基线玩家策略采样合法动作用于公理验证对照。"


def build_skill_definitions(enable_null_player_probe: bool) -> dict[str, str]:
    definitions = dict(SKILL_DEFINITIONS)
    if enable_null_player_probe:
        definitions[NULL_PROBE_SKILL_NAME] = NULL_PROBE_SKILL_DESCRIPTION
    return definitions


@dataclass
class PlannerOutcome:
    thought: str
    selected_skill: str
    skill_parameters: dict[str, object]
    action: ActionModel
    decision_bias: str | None = None


class ObservationParser:
    TABLE_TRUTH_CAPACITY = 8

    @staticmethod
    def _is_true_card(card: str, table_rank: str) -> bool:
        return card == table_rank or card == JOKER_RANK

    def parse(self, observation: dict[str, object]) -> str:
        """作用: 将结构化 observation 转为 LLM 可读叙述。

        输入:
        - observation: 环境观测字典。

        返回:
        - str: 含关键标签的叙述文本。
        """
        player_id = str(observation.get("player_id", ""))
        table_rank = str(observation.get("table_type", "A"))
        private_hand = [str(card) for card in observation.get("private_hand", [])]
        pile_history = observation.get("pile_history", [])
        legal_actions = observation.get("legal_actions", [])
        player_states = observation.get("player_states", {})

        death_probability = 0.0
        if isinstance(player_states, dict):
            state = player_states.get(player_id, {})
            if isinstance(state, dict):
                death_probability = float(state.get("death_probability", 0.0))

        if death_probability >= 0.999999:
            survival_text = "绝境"
        elif death_probability > 0.3:
            survival_text = "极高风险局势"
        else:
            survival_text = "可控风险"

        declared_count = 0
        if isinstance(pile_history, list):
            for claim in pile_history:
                if not isinstance(claim, dict):
                    continue
                claim_rank = str(claim.get("claim_rank", table_rank))
                if claim_rank == table_rank:
                    declared_count += int(claim.get("declared_count", 0))

        hand_true_count = sum(1 for card in private_hand if self._is_true_card(card, table_rank))
        logical_conflict = declared_count + hand_true_count > self.TABLE_TRUTH_CAPACITY

        if logical_conflict:
            intel_text = (
                f"逻辑冲突：你手中有 {hand_true_count} 张 {table_rank}/Joker 真牌，"
                f"桌面已声明 {declared_count} 张 {table_rank}。"
                f"总量上限为 8（6{table_rank}+2Joker），当前牌池极大概率存在虚假。"
            )
        else:
            intel_text = (
                f"逻辑冲突评估：你手中真牌 {hand_true_count} 张，桌面声明 {declared_count} 张，"
                f"尚未超过总量上限 8，但牌池风险在累积。"
            )

        readable_actions: list[str] = []
        if isinstance(legal_actions, list):
            for item in legal_actions:
                if not isinstance(item, dict):
                    continue
                action_type = str(item.get("type", ""))
                if action_type == "play_claim":
                    min_cards = item.get("min_cards", 1)
                    max_cards = item.get("max_cards", 3)
                    claim_rank = item.get("claim_rank", table_rank)
                    readable_actions.append(f"play_claim({min_cards}-{max_cards}张, claim_rank={claim_rank})")
                elif action_type:
                    readable_actions.append(action_type)

        rule_text = "、".join(readable_actions) if readable_actions else "无合法动作"

        return (
            f"[生存警告] 当前致死率 {death_probability:.2f}，判定为{survival_text}。\n"
            f"[牌局情报] {intel_text}\n"
            f"[规则约束] 当前可选原始动作：{rule_text}。"
        )


class ParameterResolver:
    @staticmethod
    def _split_hand(hand: list[str], table_rank: str) -> tuple[list[str], list[str]]:
        true_cards = [card for card in hand if card == table_rank or card == JOKER_RANK]
        fake_cards = [card for card in hand if card not in {table_rank, JOKER_RANK}]
        return true_cards, fake_cards

    @staticmethod
    def _clamp_ratio(raw_ratio: object) -> float:
        try:
            ratio = float(raw_ratio)
        except (TypeError, ValueError):
            return 0.5
        return max(0.0, min(1.0, ratio))

    @staticmethod
    def _clamp_total(raw_total: object, hand_size: int) -> int:
        try:
            total = int(raw_total)
        except (TypeError, ValueError):
            total = 2
        total = max(1, min(3, total))
        return min(total, hand_size)

    def resolve_strategic_drain(
        self,
        hand: list[str],
        table_rank: str,
        skill_parameters: dict[str, object],
    ) -> dict[str, object]:
        """作用: 将 Strategic_Drain 的比例参数映射为具体出牌组合。

        输入:
        - hand: 当前手牌。
        - table_rank: 本轮桌面牌型。
        - skill_parameters: LLM 提供参数（含 bluff_ratio/intended_total_cards）。

        返回:
        - dict[str, object]: 解析后的出牌卡组与统计字段。
        """
        if not hand:
            return {
                "cards": [],
                "bluff_ratio": 0.0,
                "intended_total_cards": 0,
                "resolved_true_count": 0,
                "resolved_fake_count": 0,
            }

        bluff_ratio = self._clamp_ratio(skill_parameters.get("bluff_ratio"))
        intended_total = self._clamp_total(skill_parameters.get("intended_total_cards"), len(hand))

        true_cards, fake_cards = self._split_hand(hand, table_rank)

        fake_count = round(intended_total * bluff_ratio)
        true_count = intended_total - fake_count

        if true_cards and true_count == 0:
            true_count = 1
            fake_count = intended_total - true_count

        if true_count > len(true_cards):
            shortage = true_count - len(true_cards)
            true_count = len(true_cards)
            fake_count += shortage

        if fake_count > len(fake_cards):
            shortage = fake_count - len(fake_cards)
            fake_count = len(fake_cards)
            true_count += shortage

        if true_count + fake_count < intended_total:
            shortage = intended_total - (true_count + fake_count)
            add_true = min(shortage, len(true_cards) - true_count)
            true_count += max(0, add_true)
            shortage -= max(0, add_true)
            add_fake = min(shortage, len(fake_cards) - fake_count)
            fake_count += max(0, add_fake)

        selected_cards: list[str] = []
        remain_true = true_count
        remain_fake = fake_count
        for card in hand:
            is_true = card == table_rank or card == JOKER_RANK
            if is_true and remain_true > 0:
                selected_cards.append(card)
                remain_true -= 1
            elif (not is_true) and remain_fake > 0:
                selected_cards.append(card)
                remain_fake -= 1

            if remain_true == 0 and remain_fake == 0:
                break

        return {
            "cards": selected_cards,
            "bluff_ratio": bluff_ratio,
            "intended_total_cards": intended_total,
            "resolved_true_count": true_count,
            "resolved_fake_count": fake_count,
        }


class SkillExecutioner:
    def __init__(self, parameter_resolver: ParameterResolver | None = None) -> None:
        self.parameter_resolver = parameter_resolver or ParameterResolver()

    @staticmethod
    def _split_hand(hand: list[str], table_rank: str) -> tuple[list[str], list[str]]:
        true_cards = [card for card in hand if card == table_rank or card == JOKER_RANK]
        fake_cards = [card for card in hand if card not in {table_rank, JOKER_RANK}]
        return true_cards, fake_cards

    def _build_null_probe_action(self, observation: dict[str, object]) -> tuple[ActionModel, dict[str, object]]:
        table_rank = str(observation.get("table_type", "A"))
        hand = [str(card) for card in observation.get("private_hand", [])]

        legal_templates = [
            item
            for item in observation.get("legal_actions", [])
            if isinstance(item, dict)
        ]

        probe_meta: dict[str, object] = {
            "probe_type": "Probe",
            "probe_policy": "random_baseline",
        }

        if not legal_templates:
            if hand:
                return (
                    ActionModel(type="play_claim", claim_rank=table_rank, cards=hand[:1]),
                    {**probe_meta, "resolved_total_cards": 1, "sampled_action_type": "play_claim"},
                )
            return ActionModel(type="pass"), {**probe_meta, "sampled_action_type": "pass"}

        selected_template = random.choice(legal_templates)
        action_type = str(selected_template.get("type", ""))

        if action_type in {"challenge", "pass"}:
            return ActionModel(type=action_type), {**probe_meta, "sampled_action_type": action_type}

        if not hand:
            fallback_type = "challenge" if any(str(item.get("type", "")) == "challenge" for item in legal_templates) else "pass"
            return ActionModel(type=fallback_type), {**probe_meta, "sampled_action_type": fallback_type}

        min_cards = int(selected_template.get("min_cards", 1))
        max_cards = int(selected_template.get("max_cards", min(3, len(hand))))
        max_cards = max(1, min(max_cards, len(hand)))
        min_cards = max(1, min(min_cards, max_cards))

        draw_count = random.randint(min_cards, max_cards)
        shuffled_hand = list(hand)
        random.shuffle(shuffled_hand)
        cards = shuffled_hand[:draw_count]
        claim_rank = str(selected_template.get("claim_rank", table_rank))

        return (
            ActionModel(type="play_claim", claim_rank=claim_rank, cards=cards),
            {
                **probe_meta,
                "resolved_total_cards": len(cards),
                "sampled_action_type": "play_claim",
            },
        )

    def execute(
        self,
        selected_skill: str,
        skill_parameters: dict[str, object],
        observation: dict[str, object],
    ) -> tuple[ActionModel, dict[str, object]]:
        """作用: 将 Skill 决策落地为可执行动作。

        输入:
        - selected_skill: Planner 选择的技能名。
        - skill_parameters: 技能参数。
        - observation: 当前观察。

        返回:
        - tuple[ActionModel, dict[str, object]]: 结构化动作与解析后参数。
        """
        table_rank = str(observation.get("table_type", "A"))
        hand = [str(card) for card in observation.get("private_hand", [])]
        true_cards, fake_cards = self._split_hand(hand, table_rank)

        if selected_skill == "Logical_Skepticism":
            return ActionModel(type="challenge"), dict(skill_parameters)

        if selected_skill == "Truthful_Action":
            cards = true_cards[:3] if true_cards else hand[:1]
            return (
                ActionModel(type="play_claim", claim_rank=table_rank, cards=cards),
                {**skill_parameters, "resolved_total_cards": len(cards)},
            )

        if selected_skill == "Calculated_Bluff":
            cards = fake_cards[:1] if fake_cards else (true_cards[:1] or hand[:1])
            return (
                ActionModel(type="play_claim", claim_rank=table_rank, cards=cards),
                {**skill_parameters, "resolved_total_cards": len(cards)},
            )

        if selected_skill == "Aggressive_Deception":
            if len(fake_cards) >= 2:
                cards = fake_cards[: min(3, len(fake_cards))]
            else:
                cards = (fake_cards + true_cards)[: min(3, len(hand))]
                if not cards and hand:
                    cards = hand[:1]
            return (
                ActionModel(type="play_claim", claim_rank=table_rank, cards=cards),
                {**skill_parameters, "resolved_total_cards": len(cards)},
            )

        if selected_skill == "Strategic_Drain":
            resolved = self.parameter_resolver.resolve_strategic_drain(hand=hand, table_rank=table_rank, skill_parameters=skill_parameters)
            cards = [str(card) for card in resolved.get("cards", [])]
            if not cards and hand:
                cards = hand[:1]
            enriched_parameters = {**skill_parameters, **resolved}
            return ActionModel(type="play_claim", claim_rank=table_rank, cards=cards), enriched_parameters

        if selected_skill == NULL_PROBE_SKILL_NAME:
            base_action, base_parameters = self._build_null_probe_action(observation=observation)
            return base_action, {**skill_parameters, **base_parameters}

        fallback_cards = true_cards[:1] or hand[:1]
        return (
            ActionModel(type="play_claim", claim_rank=table_rank, cards=fallback_cards),
            {**skill_parameters, "resolved_total_cards": len(fallback_cards)},
        )


class LiarPlanner:
    def __init__(
        self,
        observation_parser: ObservationParser | None = None,
        executioner: SkillExecutioner | None = None,
        enable_null_player_probe: bool = False,
    ) -> None:
        self.observation_parser = observation_parser or ObservationParser()
        self.executioner = executioner or SkillExecutioner()
        self.enable_null_player_probe = enable_null_player_probe
        self.available_skills = build_skill_definitions(enable_null_player_probe=enable_null_player_probe)

    def build_skill_prompt_block(self) -> str:
        lines = [f"你必须从以下 {len(self.available_skills)} 个 Skill 中选择一个："]
        for name, desc in self.available_skills.items():
            lines.append(f"- {name}: {desc}")
        lines.append(
            "对于 Strategic_Drain 技能，你可以通过 bluff_ratio 自由调节真假牌比例。"
            "0.1 表示几乎全是真牌，0.9 表示几乎全是吹牛。"
        )
        lines.append(
            "输出 JSON Schema：{\"thought\": string, \"selected_skill\": string, \"skill_parameters\": object}。"
        )
        return "\n".join(lines)

    def resolve_outcome(
        self,
        thought: str,
        selected_skill: str,
        skill_parameters: dict[str, object],
        observation: dict[str, object],
    ) -> PlannerOutcome:
        """作用: 对选中技能执行合法性修正并生成动作。

        输入:
        - thought: LLM 思考文本。
        - selected_skill: LLM 选中的技能名。
        - skill_parameters: 技能参数。
        - observation: 当前观察。

        返回:
        - PlannerOutcome: 最终执行决策。
        """
        legal_actions = observation.get("legal_actions", [])
        legal_types = {
            str(item.get("type", ""))
            for item in legal_actions
            if isinstance(item, dict)
        }

        decision_bias = None
        resolved_skill = selected_skill
        resolved_parameters = dict(skill_parameters)

        if selected_skill not in self.available_skills:
            resolved_skill = "Truthful_Action"
            resolved_parameters = {}
            decision_bias = f"decision_bias: {selected_skill} is unavailable, auto-corrected to Truthful_Action"

        if selected_skill == "Logical_Skepticism" and "challenge" not in legal_types:
            resolved_skill = "Truthful_Action"
            resolved_parameters = {}
            decision_bias = "decision_bias: Logical_Skepticism is illegal here, auto-corrected to Truthful_Action"

        action, resolved_parameters = self.executioner.execute(
            selected_skill=resolved_skill,
            skill_parameters=resolved_parameters,
            observation=observation,
        )

        if action.type not in legal_types and legal_types:
            if "challenge" in legal_types:
                action = ActionModel(type="challenge")
                resolved_skill = "Logical_Skepticism"
                resolved_parameters = {}
            elif "play_claim" in legal_types:
                fallback_action, fallback_params = self.executioner.execute(
                    selected_skill="Truthful_Action",
                    skill_parameters={},
                    observation=observation,
                )
                action = fallback_action
                resolved_skill = "Truthful_Action"
                resolved_parameters = fallback_params
            elif "pass" in legal_types:
                action = ActionModel(type="pass")
                resolved_skill = "Truthful_Action"
                resolved_parameters = {}

        if resolved_skill == NULL_PROBE_SKILL_NAME:
            resolved_parameters = {**resolved_parameters, "probe_type": "Probe"}

        return PlannerOutcome(
            thought=thought,
            selected_skill=resolved_skill,
            skill_parameters=resolved_parameters,
            action=action,
            decision_bias=decision_bias,
        )
