from __future__ import annotations

import asyncio
import csv
import json
import math
import random
import re
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from liars_game_engine.agents.factory import build_agents
from liars_game_engine.analysis.train_value_proxy import (
    VALUE_PROXY_INPUT_DIM,
    VALUE_PROXY_TARGET_PHI,
    VALUE_PROXY_TARGET_WINNER,
    ValueProxyMLP,
    build_value_proxy_feature_context,
    encode_value_proxy_features,
)
from liars_game_engine.config.schema import AppSettings
from liars_game_engine.engine.environment import GameEnvironment
from liars_game_engine.engine.game_state import ActionModel


LETHAL_EVENT_PATTERN = re.compile(r"Roulette revealed LETHAL on (?P<player_id>[^;]+); player eliminated\.")
DEFAULT_ROLLOUT_STEP_LIMIT = 100
CREDIT_REPORT_COLUMNS = ("skill_name", "death_prob_bucket", "avg_shapley_value", "sample_count")
DEATH_BUCKET_ORDER = {"0-1/6": 0, "1/6-1/3": 1, "1/3-1/2": 2, ">1/2": 3}
NULL_PROBE_SKILL_NAME = "Null_Probe_Skill"
BASELINE_MODE_RANDOM_NON_ORIGINAL = "random_non_original"
BASELINE_MODE_RANDOM_ALL_LEGAL = "expected_value_all_legal"
BASELINE_MODE_RANDOM_LEGAL_AGENT = "random_legal_agent"
BASELINE_MODE_FORCE_ORIGINAL = "force_original"
BASELINE_MODES = {
    BASELINE_MODE_RANDOM_NON_ORIGINAL,
    BASELINE_MODE_RANDOM_ALL_LEGAL,
    BASELINE_MODE_RANDOM_LEGAL_AGENT,
    BASELINE_MODE_FORCE_ORIGINAL,
}


@dataclass
class TurnTrajectory:
    game_id: str
    turn: int
    player_id: str
    observation: dict[str, object]
    action: ActionModel
    skill_name: str
    skill_parameters: dict[str, object]
    checkpoint_format: str
    checkpoint_payload: str
    decision_bias: str | None = None


@dataclass
class GameTrajectory:
    game_id: str
    turns: list[TurnTrajectory] = field(default_factory=list)
    winner: str | None = None


@dataclass
class ShapleyAttribution:
    game_id: str
    turn: int
    player_id: str
    skill_name: str
    state_feature: str
    death_prob_bucket: str
    winner: str | None
    value_action: float
    value_counterfactual: float
    phi: float
    rollout_samples: int


@dataclass
class CreditLedger:
    by_skill: dict[str, dict[str, float]] = field(default_factory=dict)
    by_state_feature: dict[str, dict[str, float]] = field(default_factory=dict)

    def add(self, attribution: ShapleyAttribution) -> None:
        """作用: 将单条边际贡献归并到技能和状态聚类表。

        输入:
        - attribution: 单回合归因结果。

        返回:
        - 无。
        """
        skill_bucket = self.by_skill.setdefault(
            attribution.skill_name,
            {"count": 0.0, "phi_sum": 0.0, "phi_avg": 0.0},
        )
        skill_bucket["count"] += 1.0
        skill_bucket["phi_sum"] += attribution.phi
        skill_bucket["phi_avg"] = skill_bucket["phi_sum"] / skill_bucket["count"]

        state_key = f"{attribution.state_feature}|{attribution.skill_name}"
        state_bucket = self.by_state_feature.setdefault(
            state_key,
            {"count": 0.0, "phi_sum": 0.0, "phi_avg": 0.0},
        )
        state_bucket["count"] += 1.0
        state_bucket["phi_sum"] += attribution.phi
        state_bucket["phi_avg"] = state_bucket["phi_sum"] / state_bucket["count"]


def _build_state_features_from_observation(observation: dict[str, object], player_id: str) -> dict[str, object]:
    return build_value_proxy_feature_context(observation=observation, player_id=player_id)


class ProxyValuePredictor:
    def __init__(
        self,
        model_path: Path | str,
        device: str | torch.device | None = None,
        output_mode: str = VALUE_PROXY_TARGET_WINNER,
    ) -> None:
        """作用: 加载训练好的价值代理模型，用于快速状态胜率估计。

        输入:
        - model_path: `.pt` 模型权重路径。
        - device: 推理设备；为空时自动选择 CUDA/CPU。

        返回:
        - 无。
        """
        resolved_device = device
        if resolved_device is None:
            resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = torch.device(resolved_device)
        self.model_path = Path(model_path)
        self.output_mode = output_mode
        self.model = ValueProxyMLP(
            input_dim=VALUE_PROXY_INPUT_DIM,
            hidden_dim=64,
            output_mode=output_mode,
        ).to(self.device)
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @staticmethod
    def encode_state_features(state_features: dict[str, object]) -> list[float]:
        return encode_value_proxy_features(state_features)

    def predict_state_features(self, state_features: dict[str, object]) -> float:
        encoded = self.encode_state_features(state_features)
        features = torch.tensor([encoded], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            prediction = self.model(features).item()
        if self.output_mode == VALUE_PROXY_TARGET_PHI:
            return max(-1.0, min(1.0, float(prediction)))
        return max(0.0, min(1.0, float(prediction)))


class LogIterator:
    def __init__(self, log_paths: list[Path]) -> None:
        self.log_paths = log_paths

    @staticmethod
    def _build_action(payload: dict[str, object]) -> ActionModel:
        return ActionModel(
            type=str(payload.get("type", "")),
            claim_rank=payload.get("claim_rank"),
            cards=[str(card) for card in payload.get("cards", [])],
        )

    @staticmethod
    def _infer_winner(records: list[dict[str, object]]) -> str | None:
        alive_players: set[str] = set()
        for record in records:
            observation = record.get("observation", {})
            if isinstance(observation, dict):
                alive = observation.get("alive_players", [])
                if isinstance(alive, list):
                    alive_players.update(str(player_id) for player_id in alive)

            step_result = record.get("step_result", {})
            events = step_result.get("events", []) if isinstance(step_result, dict) else []
            if not isinstance(events, list):
                continue

            for event in events:
                if not isinstance(event, str):
                    continue
                matched = LETHAL_EVENT_PATTERN.search(event)
                if matched:
                    alive_players.discard(matched.group("player_id"))

        if len(alive_players) == 1:
            return next(iter(alive_players))
        return None

    def iter_games(self) -> list[GameTrajectory]:
        """作用: 读取结构化日志并输出可用于归因分析的对局轨迹。

        输入:
        - 无。

        返回:
        - list[GameTrajectory]: 每个日志文件对应一局轨迹。
        """
        games: list[GameTrajectory] = []

        for log_path in self.log_paths:
            lines = Path(log_path).read_text(encoding="utf-8").splitlines()
            records = [json.loads(line) for line in lines if line.strip()]

            turns: list[TurnTrajectory] = []
            for record in records:
                action_payload = record.get("action", {})
                checkpoint = record.get("checkpoint", {})
                if not isinstance(action_payload, dict) or not isinstance(checkpoint, dict):
                    continue

                turns.append(
                    TurnTrajectory(
                        game_id=Path(log_path).stem,
                        turn=int(record.get("turn", 0)),
                        player_id=str(record.get("player_id", "")),
                        observation=record.get("observation", {}),
                        action=self._build_action(action_payload),
                        skill_name=str(record.get("skill_name", "Unknown")),
                        skill_parameters=(
                            record.get("skill_parameters", {})
                            if isinstance(record.get("skill_parameters", {}), dict)
                            else {}
                        ),
                        checkpoint_format=str(checkpoint.get("format", "")),
                        checkpoint_payload=str(checkpoint.get("payload", "")),
                        decision_bias=(
                            str(record.get("decision_bias"))
                            if record.get("decision_bias") is not None
                            else None
                        ),
                    )
                )

            games.append(
                GameTrajectory(
                    game_id=Path(log_path).stem,
                    turns=turns,
                    winner=self._infer_winner(records),
                )
            )

        return games


def _action_to_payload(action: ActionModel) -> dict[str, object]:
    return {
        "type": action.type,
        "claim_rank": action.claim_rank,
        "cards": list(action.cards),
    }


def _build_action_from_legal_template(
    env: GameEnvironment,
    player_id: str,
    legal_template: dict[str, object],
    rng: random.Random,
) -> ActionModel:
    action_type = str(legal_template.get("type", ""))
    if action_type == "challenge":
        return ActionModel(type="challenge")

    if action_type == "pass":
        return ActionModel(type="pass")

    hand = list(env.state.players[player_id].hand)
    if not hand:
        return ActionModel(type="pass")

    min_cards = int(legal_template.get("min_cards", 1))
    max_cards = int(legal_template.get("max_cards", min(3, len(hand))))
    max_cards = max(1, min(max_cards, len(hand)))
    min_cards = max(1, min(min_cards, max_cards))

    draw_count = rng.randint(min_cards, max_cards)
    selected_indices = list(range(len(hand)))
    rng.shuffle(selected_indices)
    cards = [hand[index] for index in selected_indices[:draw_count]]
    return ActionModel(type="play_claim", claim_rank=str(legal_template.get("claim_rank", env.state.table_type)), cards=cards)


def _build_random_action(env: GameEnvironment, player_id: str, rng: random.Random) -> ActionModel:
    legal = env.get_legal_actions(player_id)
    if not legal:
        return ActionModel(type="challenge")

    choice = rng.choice(legal)
    return _build_action_from_legal_template(env=env, player_id=player_id, legal_template=choice, rng=rng)


def _is_action_matching_legal_template(
    env: GameEnvironment,
    action: ActionModel,
    legal_template: dict[str, object],
) -> bool:
    action_type = str(legal_template.get("type", ""))
    if action_type != action.type:
        return False
    if action_type in {"challenge", "pass"}:
        return True
    if action_type != "play_claim":
        return False

    legal_claim_rank = str(legal_template.get("claim_rank", env.state.table_type))
    action_claim_rank = str(action.claim_rank or env.state.table_type)
    return legal_claim_rank == action_claim_rank


def _build_counterfactual_candidates(
    env: GameEnvironment,
    player_id: str,
    original_action: ActionModel,
    rng: random.Random,
    include_original: bool = False,
) -> list[ActionModel]:
    legal_actions = env.get_legal_actions(player_id)
    deduplicated: dict[str, ActionModel] = {}

    def _is_same_template(legal_template: dict[str, object]) -> bool:
        return _is_action_matching_legal_template(
            env=env,
            action=original_action,
            legal_template=legal_template,
        )

    for legal_template in legal_actions:
        if not isinstance(legal_template, dict):
            continue

        same_template = _is_same_template(legal_template)
        if not include_original and same_template:
            continue

        if same_template and include_original:
            candidate = original_action
        else:
            candidate = _build_action_from_legal_template(
                env=env,
                player_id=player_id,
                legal_template=legal_template,
                rng=rng,
            )

        payload_key = json.dumps(_action_to_payload(candidate), sort_keys=True)
        deduplicated[payload_key] = candidate

    if include_original and not deduplicated:
        payload_key = json.dumps(_action_to_payload(original_action), sort_keys=True)
        deduplicated[payload_key] = original_action

    return list(deduplicated.values())


def _build_counterfactual_action(
    env: GameEnvironment,
    player_id: str,
    original_action: ActionModel,
    sample_seed: int,
    baseline_mode: str = BASELINE_MODE_RANDOM_LEGAL_AGENT,
) -> ActionModel | None:
    """作用: 为反事实分支采样合法动作，支持多种基线模式。

    输入:
    - env: 已加载 checkpoint 的环境。
    - player_id: 当前决策玩家 ID。
    - original_action: 日志中的原动作。
    - sample_seed: 当前样本种子。
    - baseline_mode: 反事实基线策略。

    返回:
    - ActionModel | None: 反事实动作；若不存在可替代动作返回 None。
    """
    normalized_mode = baseline_mode if baseline_mode in BASELINE_MODES else BASELINE_MODE_RANDOM_LEGAL_AGENT
    if normalized_mode == BASELINE_MODE_FORCE_ORIGINAL:
        return original_action

    rng = random.Random(sample_seed)

    if normalized_mode == BASELINE_MODE_RANDOM_LEGAL_AGENT:
        candidates = _build_counterfactual_candidates(
            env=env,
            player_id=player_id,
            original_action=original_action,
            rng=rng,
            include_original=False,
        )
        if not candidates:
            return original_action
        return rng.choice(candidates)

    include_original = normalized_mode == BASELINE_MODE_RANDOM_ALL_LEGAL
    candidates = _build_counterfactual_candidates(
        env=env,
        player_id=player_id,
        original_action=original_action,
        rng=rng,
        include_original=include_original,
    )
    if not candidates:
        return original_action if include_original else None
    return rng.choice(candidates)


def _score_terminal_result(env: GameEnvironment, target_player_id: str) -> float:
    alive_players = [player_id for player_id, player_state in env.state.players.items() if not player_state.eliminated]
    if len(alive_players) != 1:
        return 0.5
    return 1.0 if alive_players[0] == target_player_id else 0.0


def _rollout_once(
    settings_raw: dict[str, object],
    encoded_checkpoint: str,
    initial_action_payload: dict[str, object],
    target_player_id: str,
    sample_seed: int,
    rollout_policy: str,
    counterfactual: bool,
    rollout_step_limit: int = DEFAULT_ROLLOUT_STEP_LIMIT,
    baseline_mode: str = BASELINE_MODE_RANDOM_LEGAL_AGENT,
) -> float:
    settings = AppSettings.from_dict(settings_raw)
    env = GameEnvironment(settings)
    checkpoint = env.deserialize_checkpoint(encoded_checkpoint)
    env.load_checkpoint(checkpoint)

    rng = random.Random(sample_seed)
    current_player = env.get_current_player()

    logged_action = ActionModel(
        type=str(initial_action_payload.get("type", "")),
        claim_rank=initial_action_payload.get("claim_rank"),
        cards=[str(card) for card in initial_action_payload.get("cards", [])],
    )

    if counterfactual:
        counter_action = _build_counterfactual_action(
            env=env,
            player_id=current_player,
            original_action=logged_action,
            sample_seed=sample_seed,
            baseline_mode=baseline_mode,
        )
        if counter_action is None:
            return 0.5
        first_action = counter_action
    else:
        first_action = logged_action

    first_result = env.step(current_player, first_action)
    if not first_result.success:
        fallback_action = _build_random_action(env, current_player, rng)
        fallback_result = env.step(current_player, fallback_action)
        if not fallback_result.success:
            return 0.5

    if env.is_game_over():
        return _score_terminal_result(env=env, target_player_id=target_player_id)

    mock_agents = build_agents(settings) if rollout_policy == "mock" else {}
    turns_guard = 0
    max_rollout_turns = max(0, int(rollout_step_limit))

    while not env.is_game_over() and turns_guard < max_rollout_turns:
        player_id = env.get_current_player()
        if rollout_policy == "mock" and player_id in mock_agents:
            observation = env.get_observation_for(player_id)
            decision = asyncio.run(mock_agents[player_id].act(observation))
            result = env.step(player_id, decision.action)
            if not result.success:
                result = env.step(player_id, _build_random_action(env, player_id, rng))
                if not result.success:
                    return 0.5
        else:
            action = _build_random_action(env, player_id, rng)
            result = env.step(player_id, action)
            if not result.success:
                result = env.step(player_id, _build_random_action(env, player_id, rng))
                if not result.success:
                    return 0.5

        turns_guard += 1

    if not env.is_game_over():
        return 0.5

    return _score_terminal_result(env=env, target_player_id=target_player_id)


class ShapleyAnalyzer:
    def __init__(
        self,
        settings: AppSettings,
        rollout_samples: int = 16,
        rollout_policy: str = "random",
        max_workers: int = 1,
        rollout_step_limit: int = DEFAULT_ROLLOUT_STEP_LIMIT,
        baseline_mode: str = BASELINE_MODE_RANDOM_LEGAL_AGENT,
    ) -> None:
        """作用: 初始化离线 Shapley 归因分析器。

        输入:
        - settings: 对局环境配置。
        - rollout_samples: 每个分支 rollout 次数。
        - rollout_policy: 未来模拟策略（random/mock）。
        - max_workers: 并行 worker 数。
        - rollout_step_limit: 每条回放路径的最大步数保护。
        - baseline_mode: 反事实基线策略。

        返回:
        - 无。
        """
        self.settings = settings
        self.rollout_samples = max(1, rollout_samples)
        self.rollout_policy = rollout_policy
        self.max_workers = max(1, max_workers)
        self.rollout_step_limit = max(0, rollout_step_limit)
        self.baseline_mode = baseline_mode if baseline_mode in BASELINE_MODES else BASELINE_MODE_RANDOM_LEGAL_AGENT
        self.settings_raw = asdict(settings)

    def _load_env_from_trajectory(self, trajectory: TurnTrajectory) -> GameEnvironment | None:
        if trajectory.checkpoint_format != "pickle_base64_v1" or not trajectory.checkpoint_payload:
            return None

        env = GameEnvironment(self.settings)
        checkpoint = env.deserialize_checkpoint(trajectory.checkpoint_payload)
        env.load_checkpoint(checkpoint)
        return env

    @staticmethod
    def _safe_float(raw_value: object, default: float = 0.0) -> float:
        try:
            return float(raw_value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _death_prob_bucket(cls, observation: dict[str, object]) -> str:
        player_id = str(observation.get("player_id", ""))
        player_states = observation.get("player_states", {})
        current_state = player_states.get(player_id, {}) if isinstance(player_states, dict) else {}
        if not isinstance(current_state, dict):
            current_state = {}

        death_probability = cls._safe_float(current_state.get("death_probability", 0.0), default=0.0)
        death_probability = max(0.0, min(1.0, death_probability))

        if death_probability <= 1.0 / 6.0:
            return "0-1/6"
        if death_probability <= 1.0 / 3.0:
            return "1/6-1/3"
        if death_probability <= 1.0 / 2.0:
            return "1/3-1/2"
        return ">1/2"

    @classmethod
    def _state_feature(cls, observation: dict[str, object]) -> str:
        must_call_liar = bool(observation.get("must_call_liar", False))
        phase = str(observation.get("phase", ""))
        table_type = str(observation.get("table_type", "A"))
        risk_bucket = cls._death_prob_bucket(observation)
        return f"phase={phase}|table={table_type}|risk={risk_bucket}|must_call_liar={must_call_liar}"

    def _run_rollout_batch(
        self,
        trajectory: TurnTrajectory,
        counterfactual: bool,
        baseline_mode: str | None = None,
    ) -> list[float]:
        payload = _action_to_payload(trajectory.action)
        counterfactual_offset = 0
        effective_baseline_mode = baseline_mode if baseline_mode is not None else self.baseline_mode

        args_list = [
            (
                self.settings_raw,
                trajectory.checkpoint_payload,
                payload,
                trajectory.player_id,
                self.settings.runtime.random_seed + counterfactual_offset + trajectory.turn * 10_000 + sample_idx,
                self.rollout_policy,
                counterfactual,
                self.rollout_step_limit,
                effective_baseline_mode,
            )
            for sample_idx in range(self.rollout_samples)
        ]

        if self.max_workers == 1:
            return [_rollout_once(*args) for args in args_list]

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(_rollout_once, *args) for args in args_list]
            return [future.result() for future in futures]

    def _build_attribution(
        self,
        trajectory: TurnTrajectory,
        winner: str | None,
        value_action: float,
        value_counter: float,
        rollout_samples: int,
    ) -> ShapleyAttribution:
        phi = value_action - value_counter
        return ShapleyAttribution(
            game_id=trajectory.game_id,
            turn=trajectory.turn,
            player_id=trajectory.player_id,
            skill_name=trajectory.skill_name,
            state_feature=self._state_feature(trajectory.observation),
            death_prob_bucket=self._death_prob_bucket(trajectory.observation),
            winner=winner,
            value_action=value_action,
            value_counterfactual=value_counter,
            phi=phi,
            rollout_samples=rollout_samples,
        )

    def attribute_step_rollout(
        self,
        trajectory: TurnTrajectory,
        winner: str | None = None,
    ) -> ShapleyAttribution | None:
        if trajectory.checkpoint_format != "pickle_base64_v1" or not trajectory.checkpoint_payload:
            return None

        action_scores = self._run_rollout_batch(trajectory, counterfactual=False)
        counter_scores = self._run_rollout_batch(trajectory, counterfactual=True)

        if not action_scores or not counter_scores:
            return None

        value_action = sum(action_scores) / len(action_scores)
        value_counter = sum(counter_scores) / len(counter_scores)
        return self._build_attribution(
            trajectory=trajectory,
            winner=winner,
            value_action=value_action,
            value_counter=value_counter,
            rollout_samples=self.rollout_samples,
        )

    def _build_proxy_feature_context(
        self,
        trajectory: TurnTrajectory,
        action: ActionModel,
    ) -> dict[str, object] | None:
        if not isinstance(trajectory.observation, dict):
            return None
        return build_value_proxy_feature_context(
            observation=trajectory.observation,
            player_id=trajectory.player_id,
            action=_action_to_payload(action),
        )

    def _build_proxy_legal_actions(self, trajectory: TurnTrajectory) -> list[ActionModel]:
        env = self._load_env_from_trajectory(trajectory)
        if env is None:
            return []

        rng = random.Random(self.settings.runtime.random_seed + trajectory.turn)
        candidates = _build_counterfactual_candidates(
            env=env,
            player_id=trajectory.player_id,
            original_action=trajectory.action,
            rng=rng,
            include_original=True,
        )
        if not candidates:
            return [trajectory.action]
        return candidates

    def attribute_step_proxy(
        self,
        trajectory: TurnTrajectory,
        winner: str | None,
        predictor: ProxyValuePredictor | object,
    ) -> ShapleyAttribution | None:
        if trajectory.checkpoint_format != "pickle_base64_v1" or not trajectory.checkpoint_payload:
            return None

        action_features = self._build_proxy_feature_context(trajectory=trajectory, action=trajectory.action)
        if action_features is None:
            return None

        legal_actions = self._build_proxy_legal_actions(trajectory)
        if not legal_actions:
            return None

        legal_scores: list[float] = []
        for legal_action in legal_actions:
            feature_context = self._build_proxy_feature_context(trajectory=trajectory, action=legal_action)
            if feature_context is None:
                continue
            legal_scores.append(float(predictor.predict_state_features(feature_context)))

        if not legal_scores:
            return None

        value_action = float(predictor.predict_state_features(action_features))
        value_counter = sum(legal_scores) / len(legal_scores)
        return self._build_attribution(
            trajectory=trajectory,
            winner=winner,
            value_action=value_action,
            value_counter=value_counter,
            rollout_samples=1,
        )

    def _sample_alignment_trajectories(
        self,
        log_paths: list[Path],
        sample_size: int,
        sample_seed: int,
    ) -> list[TurnTrajectory]:
        trajectories: list[TurnTrajectory] = []
        for game in LogIterator(log_paths).iter_games():
            for trajectory in game.turns:
                if trajectory.checkpoint_format == "pickle_base64_v1" and trajectory.checkpoint_payload:
                    trajectories.append(trajectory)

        if len(trajectories) <= sample_size:
            return trajectories

        rng = random.Random(sample_seed)
        return rng.sample(trajectories, sample_size)

    @staticmethod
    def _pearson_correlation(left: list[float], right: list[float]) -> float:
        if len(left) != len(right) or len(left) < 2:
            return 0.0

        mean_left = sum(left) / len(left)
        mean_right = sum(right) / len(right)
        numerator = sum((item - mean_left) * (other - mean_right) for item, other in zip(left, right))
        left_variance = sum((item - mean_left) ** 2 for item in left)
        right_variance = sum((item - mean_right) ** 2 for item in right)
        denominator = math.sqrt(left_variance * right_variance)
        if denominator <= 0.0:
            return 0.0
        return numerator / denominator

    def run_proxy_alignment(
        self,
        log_paths: list[Path],
        predictor: ProxyValuePredictor | object,
        sample_size: int = 20,
        sample_seed: int = 42,
    ) -> dict[str, float | int | bool]:
        sampled = self._sample_alignment_trajectories(
            log_paths=log_paths,
            sample_size=max(1, int(sample_size)),
            sample_seed=sample_seed,
        )

        winner_lookup = {game.game_id: game.winner for game in LogIterator(log_paths).iter_games()}

        rollout_results: list[ShapleyAttribution] = []
        start_rollout = time.perf_counter()
        for trajectory in sampled:
            attribution = self.attribute_step_rollout(trajectory=trajectory, winner=winner_lookup.get(trajectory.game_id))
            if attribution is not None:
                rollout_results.append(attribution)
        rollout_elapsed = max(0.0, time.perf_counter() - start_rollout)

        proxy_results: list[ShapleyAttribution] = []
        start_proxy = time.perf_counter()
        for trajectory in sampled:
            attribution = self.attribute_step_proxy(
                trajectory=trajectory,
                winner=winner_lookup.get(trajectory.game_id),
                predictor=predictor,
            )
            if attribution is not None:
                proxy_results.append(attribution)
        proxy_elapsed = max(0.0, time.perf_counter() - start_proxy)

        rollout_by_key = {(item.game_id, item.turn, item.player_id): item for item in rollout_results}
        proxy_by_key = {(item.game_id, item.turn, item.player_id): item for item in proxy_results}

        paired_keys = sorted(set(rollout_by_key) & set(proxy_by_key))
        rollout_phi = [rollout_by_key[key].phi for key in paired_keys]
        proxy_phi = [proxy_by_key[key].phi for key in paired_keys]

        mae = (
            sum(abs(left - right) for left, right in zip(rollout_phi, proxy_phi)) / len(paired_keys)
            if paired_keys
            else 0.0
        )
        correlation = self._pearson_correlation(rollout_phi, proxy_phi)
        speedup_ratio = rollout_elapsed / max(proxy_elapsed, 1e-9)

        return {
            "sample_size": len(paired_keys),
            "requested_sample_size": max(1, int(sample_size)),
            "pearson_correlation": correlation,
            "mae": mae,
            "speedup_ratio": speedup_ratio,
            "rollout_elapsed_seconds": rollout_elapsed,
            "proxy_elapsed_seconds": proxy_elapsed,
            "rollout_samples": self.rollout_samples,
            "alignment_passed": correlation > 0.75,
        }

    def run_direct_phi_alignment(
        self,
        log_paths: list[Path],
        predictor: ProxyValuePredictor | object,
        sample_size: int = 20,
        sample_seed: int = 42,
    ) -> dict[str, float | int | bool | str]:
        sampled = self._sample_alignment_trajectories(
            log_paths=log_paths,
            sample_size=max(1, int(sample_size)),
            sample_seed=sample_seed,
        )

        winner_lookup = {game.game_id: game.winner for game in LogIterator(log_paths).iter_games()}

        rollout_results: list[ShapleyAttribution] = []
        start_rollout = time.perf_counter()
        for trajectory in sampled:
            attribution = self.attribute_step_rollout(trajectory=trajectory, winner=winner_lookup.get(trajectory.game_id))
            if attribution is not None:
                rollout_results.append(attribution)
        rollout_elapsed = max(0.0, time.perf_counter() - start_rollout)

        predicted_by_key: dict[tuple[str, int, str], float] = {}
        start_proxy = time.perf_counter()
        for trajectory in sampled:
            feature_context = self._build_proxy_feature_context(trajectory=trajectory, action=trajectory.action)
            if feature_context is None:
                continue
            predicted_by_key[(trajectory.game_id, trajectory.turn, trajectory.player_id)] = float(
                predictor.predict_state_features(feature_context)
            )
        proxy_elapsed = max(0.0, time.perf_counter() - start_proxy)

        rollout_by_key = {(item.game_id, item.turn, item.player_id): item for item in rollout_results}
        paired_keys = sorted(set(rollout_by_key) & set(predicted_by_key))
        rollout_phi = [rollout_by_key[key].phi for key in paired_keys]
        proxy_phi = [predicted_by_key[key] for key in paired_keys]

        mae = (
            sum(abs(left - right) for left, right in zip(rollout_phi, proxy_phi)) / len(paired_keys)
            if paired_keys
            else 0.0
        )
        correlation = self._pearson_correlation(rollout_phi, proxy_phi)
        speedup_ratio = rollout_elapsed / max(proxy_elapsed, 1e-9)

        return {
            "sample_size": len(paired_keys),
            "requested_sample_size": max(1, int(sample_size)),
            "pearson_correlation": correlation,
            "mae": mae,
            "speedup_ratio": speedup_ratio,
            "rollout_elapsed_seconds": rollout_elapsed,
            "proxy_elapsed_seconds": proxy_elapsed,
            "rollout_samples": self.rollout_samples,
            "alignment_passed": correlation > 0.60,
            "prediction_mode": "direct_phi",
        }

    @staticmethod
    def export_alignment_report(report: dict[str, object], output_path: Path | str) -> Path:
        report_path = Path(output_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report_path

    def analyze_logs_proxy(
        self,
        log_paths: list[Path],
        predictor: ProxyValuePredictor | object,
    ) -> tuple[list[ShapleyAttribution], CreditLedger]:
        iterator = LogIterator(log_paths)
        games = iterator.iter_games()

        attributions: list[ShapleyAttribution] = []
        ledger = CreditLedger()

        for game in games:
            for trajectory in game.turns:
                attribution = self.attribute_step_proxy(
                    trajectory=trajectory,
                    winner=game.winner,
                    predictor=predictor,
                )
                if attribution is None:
                    continue
                attributions.append(attribution)
                ledger.add(attribution)

        return attributions, ledger

    def analyze_logs(self, log_paths: list[Path]) -> tuple[list[ShapleyAttribution], CreditLedger]:
        """作用: 对日志中每个决策回合计算边际贡献并汇总台账。

        输入:
        - log_paths: 一组 JSONL 轨迹日志路径。

        返回:
        - tuple[list[ShapleyAttribution], CreditLedger]: 逐回合归因与聚类结果。
        """
        iterator = LogIterator(log_paths)
        games = iterator.iter_games()

        attributions: list[ShapleyAttribution] = []
        ledger = CreditLedger()

        for game in games:
            for trajectory in game.turns:
                attribution = self.attribute_step_rollout(trajectory=trajectory, winner=game.winner)
                if attribution is None:
                    continue
                attributions.append(attribution)
                ledger.add(attribution)

        return attributions, ledger

    @staticmethod
    def export_credit_report(attributions: list[ShapleyAttribution], output_path: Path | str) -> Path:
        """作用: 聚合 Shapley 结果并导出任务 C 需要的 credit_report.csv。

        输入:
        - attributions: 逐决策点的归因结果。
        - output_path: CSV 输出路径。

        返回:
        - Path: 实际输出文件路径。
        """
        report_path = Path(output_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        aggregated: dict[tuple[str, str], dict[str, float]] = {}
        for attribution in attributions:
            key = (attribution.skill_name, attribution.death_prob_bucket)
            bucket = aggregated.setdefault(key, {"phi_sum": 0.0, "record_count": 0.0, "sample_count": 0.0})
            bucket["phi_sum"] += attribution.phi
            bucket["record_count"] += 1.0
            bucket["sample_count"] += float(attribution.rollout_samples)

        ordered_items = sorted(
            aggregated.items(),
            key=lambda item: (item[0][0], DEATH_BUCKET_ORDER.get(item[0][1], 99), item[0][1]),
        )

        with report_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(CREDIT_REPORT_COLUMNS))
            writer.writeheader()

            for (skill_name, death_prob_bucket), stats in ordered_items:
                avg_value = stats["phi_sum"] / stats["record_count"] if stats["record_count"] else 0.0
                writer.writerow(
                    {
                        "skill_name": skill_name,
                        "death_prob_bucket": death_prob_bucket,
                        "avg_shapley_value": f"{avg_value:.6f}",
                        "sample_count": str(int(stats["sample_count"])),
                    }
                )

        return report_path

    @staticmethod
    def summarize_probe_skill(attributions: list[ShapleyAttribution]) -> dict[str, float | int | str]:
        """作用: 专项汇总 Null_Probe_Skill 的 Shapley 统计。

        输入:
        - attributions: 全量归因列表。

        返回:
        - dict[str, float | int | str]: 仅针对 Null_Probe_Skill 的计数与均值。
        """
        probe_items = [item for item in attributions if item.skill_name == NULL_PROBE_SKILL_NAME]
        if not probe_items:
            return {
                "skill_name": NULL_PROBE_SKILL_NAME,
                "count": 0,
                "phi_sum": 0.0,
                "phi_avg": 0.0,
                "sample_count": 0,
            }

        phi_sum = sum(item.phi for item in probe_items)
        sample_count = sum(int(item.rollout_samples) for item in probe_items)
        return {
            "skill_name": NULL_PROBE_SKILL_NAME,
            "count": len(probe_items),
            "phi_sum": phi_sum,
            "phi_avg": phi_sum / len(probe_items),
            "sample_count": sample_count,
        }
