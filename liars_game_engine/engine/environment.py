from __future__ import annotations

import base64
import copy
import pickle
import random
from collections import Counter

from liars_game_engine.config.schema import AppSettings
from liars_game_engine.engine.game_state import ActionModel, ClaimState, GamePhase, JOKER_RANK, PlayerRuntimeState, RuntimeGameState, StepResult
from liars_game_engine.engine.rules.challenge_rule import ChallengeRule
from liars_game_engine.engine.rules.declare_rule import DeclareRule
from liars_game_engine.engine.rules.roulette_rule import RouletteRule


class GameEnvironment:
    TABLE_DECK = ("A", "K", "Q")
    LIAR_RANK_COUNTS = {"A": 6, "K": 6, "Q": 6}
    JOKER_COUNT = 2

    def __init__(self, settings: AppSettings) -> None:
        """作用: 初始化环境状态、随机座位顺序、首轮信息与初始手牌。

        输入:
        - settings: 应用配置对象（玩家、规则、随机种子等）。

        返回:
        - 无。
        """
        self.settings = settings
        self.rng = random.Random(settings.runtime.random_seed)
        self.declare_rule = DeclareRule()
        self.challenge_rule = ChallengeRule()
        self.roulette_rule = RouletteRule(settings.rules.roulette_slots, self.rng)

        player_ids = [player.player_id for player in settings.players]
        if len(player_ids) < 2 or len(player_ids) > 4:
            raise ValueError("Liar's Bar basic rules require 2 to 4 players")
        self.rng.shuffle(player_ids)

        players = {
            player_id: PlayerRuntimeState(
                player_id=player_id,
                hand=[],
                revolver_deck=self.roulette_rule.build_revolver_deck(),
            )
            for player_id in player_ids
        }

        initial_player = self.rng.choice(player_ids)
        self.state = RuntimeGameState(
            players=players,
            turn_order=player_ids,
            current_player_id=initial_player,
            phase=GamePhase.TURN_START,
            round_index=1,
            table_type=self._draw_table_type(),
            first_turn_of_round=True,
        )
        self._deal_round_cards()

    def _build_liar_deck(self) -> list[str]:
        """作用: 构建标准 Liar 牌堆（A/K/Q 各 6 张 + 2 张 JOKER）。

        输入:
        - 无。

        返回:
        - list[str]: 未洗牌的牌面列表。
        """
        deck: list[str] = []
        for rank, count in self.LIAR_RANK_COUNTS.items():
            deck.extend([rank] * count)
        deck.extend([JOKER_RANK] * self.JOKER_COUNT)
        return deck

    def _draw_table_type(self) -> str:
        """作用: 随机抽取本轮桌牌类型（A/K/Q）。

        输入:
        - 无。

        返回:
        - str: 本轮 table_type。
        """
        return self.rng.choice(self.TABLE_DECK)

    def _deal_round_cards(self) -> None:
        """作用: 给所有存活玩家重发本轮手牌。

        输入:
        - 无。

        返回:
        - 无。
        """
        for player in self.state.players.values():
            player.hand.clear()

        alive_players = [pid for pid, runtime in self.state.players.items() if not runtime.eliminated]
        cards_per_player = self.settings.rules.cards_per_player
        required_cards = cards_per_player * len(alive_players)

        deck = self._build_liar_deck()
        if required_cards > len(deck):
            raise ValueError("configured cards_per_player exceeds basic deck capacity")

        self.rng.shuffle(deck)
        for player_id in alive_players:
            for _ in range(cards_per_player):
                self.state.players[player_id].hand.append(deck.pop())

    def _players_with_cards(self) -> list[str]:
        """作用: 获取当前仍有手牌且未淘汰的玩家顺序列表。

        输入:
        - 无。

        返回:
        - list[str]: 仍持牌玩家 ID 列表。
        """
        return [
            player_id
            for player_id in self.state.turn_order
            if not self.state.players[player_id].eliminated and self.state.players[player_id].hand
        ]

    def get_current_player(self) -> str:
        """作用: 返回当前轮到行动的玩家 ID。

        输入:
        - 无。

        返回:
        - str: 当前玩家 ID。
        """
        return self.state.current_player_id

    def save_checkpoint(self) -> dict[str, object]:
        """作用: 保存当前环境可回放快照（状态 + 随机数状态）。

        输入:
        - 无。

        返回:
        - dict[str, object]: 可用于 load_checkpoint 的快照。
        """
        return {
            "state": copy.deepcopy(self.state),
            "rng_state": copy.deepcopy(self.rng.getstate()),
        }

    def load_checkpoint(self, checkpoint: dict[str, object]) -> None:
        """作用: 将环境恢复到先前保存的快照。

        输入:
        - checkpoint: save_checkpoint 输出的快照字典。

        返回:
        - 无。
        """
        state = checkpoint.get("state")
        rng_state = checkpoint.get("rng_state")

        if not isinstance(state, RuntimeGameState):
            raise ValueError("checkpoint.state must be RuntimeGameState")
        if rng_state is None:
            raise ValueError("checkpoint.rng_state is required")

        self.state = copy.deepcopy(state)
        self.rng.setstate(rng_state)

    def serialize_checkpoint(self, checkpoint: dict[str, object]) -> str:
        """作用: 将 checkpoint 编码为可写入 JSONL 的 Base64 文本。

        输入:
        - checkpoint: save_checkpoint 返回的快照字典。

        返回:
        - str: Base64 编码后的字符串。
        """
        payload = pickle.dumps(checkpoint)
        return base64.b64encode(payload).decode("ascii")

    def deserialize_checkpoint(self, encoded_checkpoint: str) -> dict[str, object]:
        """作用: 将日志中的 Base64 checkpoint 还原为可 load 的字典。

        输入:
        - encoded_checkpoint: Base64 编码快照。

        返回:
        - dict[str, object]: 可直接传入 load_checkpoint 的快照。
        """
        try:
            payload = base64.b64decode(encoded_checkpoint.encode("ascii"))
            decoded = pickle.loads(payload)
        except Exception as error:
            raise ValueError("invalid encoded checkpoint payload") from error

        if not isinstance(decoded, dict):
            raise ValueError("decoded checkpoint must be dict")
        return decoded

    def get_legal_actions(self, player_id: str) -> list[dict[str, object]]:
        """作用: 枚举当前玩家在当前状态下的合法动作模板。

        输入:
        - player_id: 请求动作集合的玩家 ID。

        返回:
        - list[dict[str, object]]: 可执行动作描述列表。
        """
        if player_id not in self.state.players:
            return []

        if self.is_game_over():
            return []

        if self.state.current_player_id != player_id:
            return []

        player = self.state.players[player_id]
        if player.eliminated:
            return []

        if self._must_call_liar(player_id):
            return [{"type": "challenge"}]

        legal_actions: list[dict[str, object]] = []

        if player.hand and self.state.phase in {GamePhase.TURN_START, GamePhase.DECLARE, GamePhase.RESPONSE_WINDOW}:
            legal_actions.append(
                {
                    "type": "play_claim",
                    "claim_rank": self.state.table_type,
                    "min_cards": 1,
                    "max_cards": min(3, len(player.hand)),
                }
            )

        if (
            self.state.phase == GamePhase.RESPONSE_WINDOW
            and self.state.pending_claim is not None
            and not self.state.first_turn_of_round
        ):
            legal_actions.append({"type": "challenge"})

        if self.state.phase == GamePhase.RESPONSE_WINDOW and not player.hand:
            legal_actions.append({"type": "pass"})

        return legal_actions

    def get_observation_for(self, player_id: str) -> dict[str, object]:
        """作用: 生成指定玩家可见的 observation 视图。

        输入:
        - player_id: 请求观测的玩家 ID。

        返回:
        - dict[str, object]: 提供给 Agent 的局面信息。
        """
        player = self.state.players[player_id]
        alive_players = [pid for pid, runtime in self.state.players.items() if not runtime.eliminated]

        pending_claim = None
        if self.state.pending_claim:
            pending_claim = {
                "actor_id": self.state.pending_claim.actor_id,
                "claim_rank": self.state.pending_claim.claim_rank,
                "declared_count": self.state.pending_claim.declared_count,
            }

        public_counts = {pid: len(runtime.hand) for pid, runtime in self.state.players.items()}
        player_states = {
            pid: {
                "is_alive": not runtime.eliminated,
                "is_safe": runtime.is_safe,
                "death_probability": runtime.death_probability,
                "hand_count": len(runtime.hand),
            }
            for pid, runtime in self.state.players.items()
        }
        pile_history = [
            {
                "actor_id": claim.actor_id,
                "claim_rank": claim.claim_rank,
                "declared_count": claim.declared_count,
                "cards": list(claim.cards),
            }
            for claim in self.state.pile_history
        ]

        return {
            "player_id": player_id,
            "phase": self.state.phase.value,
            "current_player_id": self.state.current_player_id,
            "alive_players": alive_players,
            "private_hand": list(player.hand),
            "public_card_counts": public_counts,
            "pending_claim": pending_claim,
            "turn_index": self.state.turn_index,
            "round_index": self.state.round_index,
            "table_type": self.state.table_type,
            "first_turn_of_round": self.state.first_turn_of_round,
            "must_call_liar": self._must_call_liar(player_id),
            "player_states": player_states,
            "pile_history": pile_history,
            "legal_actions": self.get_legal_actions(player_id),
        }

    def _next_alive_player(self, from_player: str) -> str:
        """作用: 从给定玩家开始，寻找下一个未淘汰玩家。

        输入:
        - from_player: 起始玩家 ID。

        返回:
        - str: 下一个存活玩家 ID；若不存在则返回起始玩家。
        """
        if not self.state.turn_order:
            return ""

        start_idx = self.state.turn_order.index(from_player)
        for offset in range(1, len(self.state.turn_order) + 1):
            candidate = self.state.turn_order[(start_idx + offset) % len(self.state.turn_order)]
            if not self.state.players[candidate].eliminated:
                return candidate
        return from_player

    def _next_player_with_cards(self, from_player: str) -> str:
        """作用: 从给定玩家后方寻找下一个“未淘汰且仍有手牌”的玩家。

        输入:
        - from_player: 起始玩家 ID。

        返回:
        - str: 下一个可行动玩家 ID；无则返回空字符串。
        """
        if not self.state.turn_order:
            return ""

        if from_player in self.state.turn_order:
            start_idx = self.state.turn_order.index(from_player)
        else:
            start_idx = -1

        for offset in range(1, len(self.state.turn_order) + 1):
            candidate = self.state.turn_order[(start_idx + offset) % len(self.state.turn_order)]
            runtime = self.state.players[candidate]
            if runtime.eliminated:
                continue
            if runtime.hand:
                return candidate
        return ""

    def _alive_count(self) -> int:
        """作用: 统计当前存活玩家数量。

        输入:
        - 无。

        返回:
        - int: 存活玩家数。
        """
        return sum(0 if player.eliminated else 1 for player in self.state.players.values())

    def is_game_over(self) -> bool:
        """作用: 判断游戏是否结束（仅剩 1 名存活玩家或显式 GAME_OVER）。

        输入:
        - 无。

        返回:
        - bool: True 表示游戏结束。
        """
        if self.state.phase == GamePhase.GAME_OVER:
            return True
        return self._alive_count() <= 1

    def _error(self, code: str, reason: str) -> StepResult:
        """作用: 统一构造失败结果。

        输入:
        - code: 错误码。
        - reason: 错误原因。

        返回:
        - StepResult: success=False 的标准结果。
        """
        return StepResult(success=False, events=[], error_code=code, error_reason=reason)

    def _must_call_liar(self, player_id: str) -> bool:
        """作用: 判断当前玩家是否被规则强制必须 call LIAR。

        输入:
        - player_id: 当前行动玩家 ID。

        返回:
        - bool: True 表示当前只能挑战，不能继续出牌。
        """
        if self.state.phase != GamePhase.RESPONSE_WINDOW:
            return False
        if self.state.pending_claim is None:
            return False

        players_with_cards = self._players_with_cards()
        return len(players_with_cards) == 1 and players_with_cards[0] == player_id

    def _resolve_round_starter(self, preferred_player: str) -> str:
        """作用: 根据优先玩家与存活状态确定下一轮起手玩家。

        输入:
        - preferred_player: 规则指定的优先起手玩家 ID。

        返回:
        - str: 实际可用的起手玩家 ID；无可用玩家时为空字符串。
        """
        if not self.state.turn_order:
            return ""

        if preferred_player and preferred_player in self.state.turn_order and not self.state.players[preferred_player].eliminated:
            return preferred_player

        if preferred_player and preferred_player in self.state.turn_order:
            return self._next_alive_player(preferred_player)

        for player_id in self.state.turn_order:
            if not self.state.players[player_id].eliminated:
                return player_id
        return ""

    def _start_new_round(self, start_player_hint: str) -> None:
        """作用: 开启新一轮并重置轮状态。

        输入:
        - start_player_hint: 规则建议的起手玩家 ID。

        返回:
        - 无。
        """
        if self._alive_count() <= 1:
            self.state.phase = GamePhase.GAME_OVER
            return

        self.state.round_index += 1
        self.state.table_type = self._draw_table_type()
        self.state.pending_claim = None
        self.state.pile_history.clear()
        self.state.phase = GamePhase.TURN_START
        self.state.first_turn_of_round = True
        self._deal_round_cards()

        starter = self._resolve_round_starter(start_player_hint)
        if starter and self.state.players[starter].hand:
            self.state.current_player_id = starter
            return

        fallback = self._next_player_with_cards(starter)
        if fallback:
            self.state.current_player_id = fallback
            return

        self.state.phase = GamePhase.GAME_OVER

    def _handle_play_claim(self, player_id: str, action: ActionModel) -> StepResult:
        """作用: 处理出牌声明动作并推进到响应窗口。

        输入:
        - player_id: 执行动作的玩家 ID。
        - action: 结构化 play_claim 动作。

        返回:
        - StepResult: 本次出牌结算结果。
        """
        validation = self.declare_rule.validate(self.state, player_id, action)
        if not validation.ok:
            return self._error(validation.error_code or "E_ACTION_RULE_VIOLATION", validation.error_reason or "invalid declare")

        if self._must_call_liar(player_id):
            return self._error("E_ACTION_RULE_VIOLATION", "only one player has cards left; must call LIAR")

        player = self.state.players[player_id]
        counter = Counter(action.cards)
        for card, count in counter.items():
            for _ in range(count):
                player.hand.remove(card)

        claim = ClaimState(
            actor_id=player_id,
            claim_rank=self.state.table_type,
            cards=list(action.cards),
            declared_count=len(action.cards),
        )
        self.state.pending_claim = claim
        self.state.pile_history.append(copy.deepcopy(claim))
        self.state.phase = GamePhase.RESPONSE_WINDOW
        self.state.first_turn_of_round = False
        self.state.turn_index += 1

        next_player = self._next_player_with_cards(player_id)
        events = [
            f"{player_id} played {len(action.cards)} face-down card(s).",
            f"Claim recorded against table type {self.state.table_type}.",
        ]

        if not next_player:
            events.append("No player with cards can respond; starting a new round.")
            self._start_new_round(start_player_hint=player_id)
            return StepResult(success=True, events=events)

        self.state.current_player_id = next_player
        events.append(f"Response window opened for {self.state.current_player_id}.")
        return StepResult(success=True, events=events)

    def _handle_challenge(self, player_id: str, action: ActionModel) -> StepResult:
        """作用: 处理 call LIAR（challenge）并进行惩罚与轮切换。

        输入:
        - player_id: 发起挑战的玩家 ID。
        - action: 结构化 challenge 动作。

        返回:
        - StepResult: 挑战结算结果。
        """
        validation = self.challenge_rule.validate(self.state, player_id, action)
        if not validation.ok:
            return self._error(validation.error_code or "E_ACTION_RULE_VIOLATION", validation.error_reason or "invalid challenge")

        if self.state.first_turn_of_round:
            return self._error("E_ACTION_RULE_VIOLATION", "cannot call LIAR on the first turn of a round")

        claim = self.state.pending_claim
        if claim is None:
            return self._error("E_ACTION_RULE_VIOLATION", "no pending claim to challenge")

        has_liar = self.challenge_rule.has_liar_cards(self.state)
        loser_id = claim.actor_id if has_liar else player_id
        loser = self.state.players[loser_id]

        self.state.phase = GamePhase.PENALTY
        roulette_outcome = self.roulette_rule.apply_penalty(loser)

        events = [
            f"{player_id} called LIAR on {claim.actor_id}.",
            "At least one revealed card is a Liar." if has_liar else "All revealed cards are Innocent.",
            roulette_outcome.event,
        ]

        self.state.pending_claim = None
        self.state.turn_index += 1

        if self._alive_count() <= 1:
            self.state.phase = GamePhase.GAME_OVER
            return StepResult(success=True, events=events)

        self._start_new_round(start_player_hint=loser_id)
        events.append(
            f"Round {self.state.round_index} started with table type {self.state.table_type} and starter {self.state.current_player_id}."
        )
        return StepResult(success=True, events=events)

    def _handle_pass(self, player_id: str) -> StepResult:
        """作用: 处理无手牌玩家在响应窗口中的跳过。

        输入:
        - player_id: 执行 pass 的玩家 ID。

        返回:
        - StepResult: pass 行为结算结果。
        """
        if self.state.phase != GamePhase.RESPONSE_WINDOW:
            return self._error("E_ENV_PHASE_MISMATCH", "pass only allowed in response window")

        if self.state.players[player_id].hand:
            return self._error("E_ACTION_RULE_VIOLATION", "pass is only allowed when player has no cards")

        self.state.turn_index += 1
        next_player = self._next_player_with_cards(player_id)
        if not next_player:
            self._start_new_round(start_player_hint=player_id)
            return StepResult(success=True, events=[f"{player_id} was skipped; new round started."])

        self.state.current_player_id = next_player
        return StepResult(success=True, events=[f"{player_id} had no cards and was skipped."])

    def step(self, player_id: str, action: ActionModel) -> StepResult:
        """作用: 环境统一动作入口，分发到具体处理逻辑。

        输入:
        - player_id: 发起动作的玩家 ID。
        - action: 动作对象，支持 play_claim/challenge/pass。

        返回:
        - StepResult: 当前动作结算结果。
        """
        if self.is_game_over():
            self.state.phase = GamePhase.GAME_OVER
            return self._error("E_ENV_PHASE_MISMATCH", "game is already over")

        if player_id != self.state.current_player_id:
            return self._error("E_ENV_PHASE_MISMATCH", "not current player turn")

        if self.state.players[player_id].eliminated:
            return self._error("E_ENV_PHASE_MISMATCH", "eliminated player cannot act")

        if action.type == "play_claim":
            return self._handle_play_claim(player_id, action)

        if action.type == "challenge":
            return self._handle_challenge(player_id, action)

        if action.type == "pass":
            return self._handle_pass(player_id)

        return self._error("E_ACTION_SCHEMA_MISSING", f"unsupported action type: {action.type}")
