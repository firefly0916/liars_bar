from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


VALID_ACTION_TYPES = {"play_claim", "challenge", "pass"}
JOKER_RANK = "JOKER"
REVOLVER_LETHAL = "LETHAL"
REVOLVER_BLANK = "BLANK"


class GamePhase(str, Enum):
    TURN_START = "turn_start"
    DECLARE = "declare"
    RESPONSE_WINDOW = "response_window"
    RESOLUTION = "resolution"
    PENALTY = "penalty"
    TURN_END = "turn_end"
    GAME_OVER = "game_over"


@dataclass
class ActionModel:
    type: str
    claim_rank: str | None = None
    cards: list[str] = field(default_factory=list)


@dataclass
class ParseError:
    code: str
    message: str
    raw_output: str


@dataclass
class ParseResult:
    ok: bool
    action: ActionModel | None = None
    thought: str = ""
    error: ParseError | None = None


@dataclass
class ClaimState:
    actor_id: str
    claim_rank: str
    cards: list[str]
    declared_count: int


@dataclass
class PlayerRuntimeState:
    player_id: str
    hand: list[str] = field(default_factory=list)
    eliminated: bool = False
    revolver_deck: list[str] = field(default_factory=list)

    @property
    def is_safe(self) -> bool:
        """作用: 判断当前玩家是否已打空手牌。

        输入:
        - 无。

        返回:
        - bool: True 表示 hand 为空。
        """
        return not self.hand

    @property
    def death_probability(self) -> float:
        """作用: 计算玩家当前轮盘状态的即时致死概率。

        输入:
        - 无。

        返回:
        - float: 当前 lethal 比例，常见为 1/n。
        """
        if not self.revolver_deck:
            return 0.0
        return self.revolver_deck.count(REVOLVER_LETHAL) / len(self.revolver_deck)


@dataclass
class RuntimeGameState:
    players: dict[str, PlayerRuntimeState]
    turn_order: list[str]
    current_player_id: str
    phase: GamePhase = GamePhase.TURN_START
    pending_claim: ClaimState | None = None
    pile_history: list[ClaimState] = field(default_factory=list)
    turn_index: int = 0
    round_index: int = 1
    table_type: str = "A"
    first_turn_of_round: bool = True


@dataclass
class StepResult:
    success: bool
    events: list[str] = field(default_factory=list)
    error_code: str | None = None
    error_reason: str | None = None


@dataclass
class RouletteOutcome:
    fired: bool
    eliminated: bool
    event: str
