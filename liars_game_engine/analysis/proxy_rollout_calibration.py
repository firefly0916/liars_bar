from __future__ import annotations

import json
import math
import random
from concurrent.futures import ProcessPoolExecutor
from hashlib import md5
from itertools import combinations
from pathlib import Path
from statistics import mean

import torch
from torch import nn

from liars_game_engine.analysis.shapley_analyzer import (
    LogIterator,
    ShapleyAnalyzer,
    TurnTrajectory,
    _action_to_payload,
    _rollout_once,
)
from liars_game_engine.config.loader import load_settings
from liars_game_engine.engine.game_state import ActionModel
from liars_game_engine.analysis.train_value_proxy import VALUE_PROXY_INPUT_DIM, encode_value_proxy_features


VALUE_PROXY_TARGET_PHI = "phi"
VALUE_PROXY_TARGET_WINNER = "winner"


class _CompatValueProxyMLP(nn.Module):
    def __init__(self, input_dim: int = VALUE_PROXY_INPUT_DIM, hidden_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def _normalize_proxy_state_dict(state_dict: dict[str, object]) -> dict[str, object]:
    if any(str(key).startswith("network.") for key in state_dict):
        return dict(state_dict)

    legacy_to_current = {
        "backbone.0.weight": "network.0.weight",
        "backbone.0.bias": "network.0.bias",
        "backbone.2.weight": "network.2.weight",
        "backbone.2.bias": "network.2.bias",
        "head.weight": "network.4.weight",
        "head.bias": "network.4.bias",
    }
    normalized: dict[str, object] = {}
    for key, value in state_dict.items():
        normalized[str(legacy_to_current.get(str(key), str(key)))] = value
    return normalized


class CompatProxyValuePredictor:
    def __init__(
        self,
        model_path: Path | str,
        device: str | torch.device | None = None,
        output_mode: str = VALUE_PROXY_TARGET_PHI,
    ) -> None:
        resolved_device = device
        if resolved_device is None:
            resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = torch.device(resolved_device)
        self.model_path = Path(model_path)
        self.output_mode = str(output_mode)
        self.model = _CompatValueProxyMLP(input_dim=VALUE_PROXY_INPUT_DIM, hidden_dim=64).to(self.device)
        state_dict = torch.load(self.model_path, map_location=self.device)
        normalized_state_dict = _normalize_proxy_state_dict(state_dict)
        self.model.load_state_dict(normalized_state_dict)
        self.model.eval()

    @staticmethod
    def encode_state_features(state_features: dict[str, object]) -> list[float]:
        return encode_value_proxy_features(state_features)

    def predict_state_features(self, state_features: dict[str, object]) -> float:
        encoded = self.encode_state_features(state_features)
        features = torch.tensor([encoded], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            raw_prediction = self.model(features).item()
        if self.output_mode == VALUE_PROXY_TARGET_PHI:
            return max(-1.0, min(1.0, math.tanh(float(raw_prediction))))
        return max(0.0, min(1.0, 1.0 / (1.0 + math.exp(-float(raw_prediction)))))


def _action_label(payload: dict[str, object]) -> str:
    cards = payload.get("cards", [])
    rendered_cards = ",".join(str(card) for card in cards) if isinstance(cards, list) else ""
    claim_rank = str(payload.get("claim_rank", "") or "")
    return f"{str(payload.get('type', ''))}|{claim_rank}|{rendered_cards}"


def _rank_labels_by_score(action_rows: list[dict[str, object]], score_key: str) -> tuple[list[str], dict[str, int]]:
    ordered = sorted(
        action_rows,
        key=lambda item: (-float(item.get(score_key, 0.0) or 0.0), str(item.get("action_label", ""))),
    )
    labels = [str(item["action_label"]) for item in ordered]
    return labels, {label: index + 1 for index, label in enumerate(labels)}


def _spearman_rank_correlation(proxy_ranks: dict[str, int], rollout_ranks: dict[str, int]) -> float:
    labels = sorted(set(proxy_ranks) & set(rollout_ranks))
    if len(labels) <= 1:
        return 1.0

    count = len(labels)
    squared_diffs = sum((proxy_ranks[label] - rollout_ranks[label]) ** 2 for label in labels)
    return 1.0 - ((6.0 * squared_diffs) / (count * ((count**2) - 1)))


def build_turn_alignment_report(
    game_id: str,
    turn: int,
    action_rows: list[dict[str, object]],
    *,
    chosen_action_label: str | None = None,
) -> dict[str, object]:
    normalized_rows = []
    for row in action_rows:
        normalized = {
            "action_label": str(row["action_label"]),
            "proxy_score": float(row["proxy_score"]),
            "rollout_score": float(row["rollout_score"]),
        }
        if "feature_vector" in row:
            normalized["feature_vector"] = [float(item) for item in row.get("feature_vector", [])]
        normalized_rows.append(normalized)
    proxy_order, proxy_ranks = _rank_labels_by_score(normalized_rows, "proxy_score")
    rollout_order, rollout_ranks = _rank_labels_by_score(normalized_rows, "rollout_score")
    proxy_top1_action = proxy_order[0] if proxy_order else ""
    rollout_top1_action = rollout_order[0] if rollout_order else ""

    actions: list[dict[str, object]] = []
    action_lookup = {str(row["action_label"]): row for row in normalized_rows}
    for label in proxy_order:
        row = action_lookup[label]
        actions.append(
            {
                **row,
                "proxy_rank": proxy_ranks[label],
                "rollout_rank": rollout_ranks[label],
            }
        )

    proxy_top1_rollout_score = float(action_lookup.get(proxy_top1_action, {}).get("rollout_score", 0.0) or 0.0)
    rollout_top1_rollout_score = float(action_lookup.get(rollout_top1_action, {}).get("rollout_score", 0.0) or 0.0)
    chosen_proxy_rank = proxy_ranks.get(str(chosen_action_label)) if chosen_action_label else None
    chosen_rollout_rank = rollout_ranks.get(str(chosen_action_label)) if chosen_action_label else None
    chosen_rollout_score = (
        float(action_lookup.get(str(chosen_action_label), {}).get("rollout_score", 0.0) or 0.0)
        if chosen_action_label
        else None
    )

    return {
        "game_id": str(game_id),
        "turn": int(turn),
        "action_count": len(actions),
        "proxy_top1_action": proxy_top1_action,
        "rollout_top1_action": rollout_top1_action,
        "top1_match": proxy_top1_action == rollout_top1_action,
        "spearman_rank_correlation": _spearman_rank_correlation(proxy_ranks, rollout_ranks),
        "proxy_top1_rollout_score": proxy_top1_rollout_score,
        "rollout_top1_rollout_score": rollout_top1_rollout_score,
        "rollout_regret": max(0.0, rollout_top1_rollout_score - proxy_top1_rollout_score),
        "chosen_action_label": chosen_action_label,
        "chosen_action_proxy_rank": chosen_proxy_rank,
        "chosen_action_rollout_rank": chosen_rollout_rank,
        "chosen_action_rollout_score": chosen_rollout_score,
        "actions": actions,
    }


def summarize_alignment_reports(
    checkpoint_label: str,
    turn_reports: list[dict[str, object]],
) -> dict[str, object]:
    if not turn_reports:
        return {
            "checkpoint_label": str(checkpoint_label),
            "sample_size": 0,
            "top1_match_rate": 0.0,
            "mean_spearman_rank_correlation": 0.0,
            "mean_action_count": 0.0,
            "mean_rollout_regret": 0.0,
            "mean_chosen_action_rollout_rank": 0.0,
            "proxy_top1_win_count": 0,
            "rollout_top1_win_count": 0,
        }

    chosen_ranks = [
        float(report["chosen_action_rollout_rank"])
        for report in turn_reports
        if report.get("chosen_action_rollout_rank") is not None
    ]
    top1_match_count = sum(1 for report in turn_reports if bool(report.get("top1_match")))
    return {
        "checkpoint_label": str(checkpoint_label),
        "sample_size": len(turn_reports),
        "top1_match_rate": top1_match_count / len(turn_reports),
        "mean_spearman_rank_correlation": mean(float(report["spearman_rank_correlation"]) for report in turn_reports),
        "mean_action_count": mean(float(report["action_count"]) for report in turn_reports),
        "mean_rollout_regret": mean(float(report["rollout_regret"]) for report in turn_reports),
        "mean_chosen_action_rollout_rank": (mean(chosen_ranks) if chosen_ranks else 0.0),
        "proxy_top1_win_count": top1_match_count,
        "rollout_top1_win_count": top1_match_count,
    }


def _extract_death_probability(trajectory: TurnTrajectory) -> float:
    observation = trajectory.observation if isinstance(trajectory.observation, dict) else {}
    player_states = observation.get("player_states", {})
    current_state = player_states.get(trajectory.player_id, {}) if isinstance(player_states, dict) else {}
    if not isinstance(current_state, dict):
        return 0.0
    try:
        return max(0.0, min(1.0, float(current_state.get("death_probability", 0.0) or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _sample_llm_trajectories(
    log_paths: list[Path],
    llm_player_id: str,
    sample_size: int,
    sample_seed: int,
    risk_priority_fraction: float,
) -> list[TurnTrajectory]:
    trajectories: list[TurnTrajectory] = []
    for game in LogIterator(log_paths).iter_games():
        for trajectory in game.turns:
            if trajectory.player_id != llm_player_id:
                continue
            if trajectory.checkpoint_format != "pickle_base64_v1" or not trajectory.checkpoint_payload:
                continue
            trajectories.append(trajectory)

    if len(trajectories) <= sample_size:
        return sorted(trajectories, key=lambda item: (item.game_id, item.turn))

    priority_count = max(0, min(sample_size, int(round(sample_size * risk_priority_fraction))))
    ordered_by_risk = sorted(
        trajectories,
        key=lambda item: (-_extract_death_probability(item), item.game_id, item.turn),
    )
    selected_keys: set[tuple[str, int, str]] = set()
    selected: list[TurnTrajectory] = []
    for trajectory in ordered_by_risk[:priority_count]:
        key = (trajectory.game_id, trajectory.turn, trajectory.player_id)
        selected_keys.add(key)
        selected.append(trajectory)

    remaining = [
        trajectory
        for trajectory in trajectories
        if (trajectory.game_id, trajectory.turn, trajectory.player_id) not in selected_keys
    ]
    rng = random.Random(sample_seed)
    random_count = max(0, sample_size - len(selected))
    if random_count > 0 and remaining:
        selected.extend(rng.sample(remaining, min(random_count, len(remaining))))

    return sorted(selected, key=lambda item: (item.game_id, item.turn))


def _enumerate_legal_actions(analyzer: ShapleyAnalyzer, trajectory: TurnTrajectory) -> list[ActionModel]:
    env = analyzer._load_env_from_trajectory(trajectory)
    if env is None:
        return [trajectory.action]

    legal_templates = env.get_legal_actions(trajectory.player_id)
    hand = list(env.state.players[trajectory.player_id].hand)
    deduplicated: dict[str, ActionModel] = {}

    for template in legal_templates:
        if not isinstance(template, dict):
            continue
        action_type = str(template.get("type", ""))
        if action_type == "challenge":
            candidate = ActionModel(type="challenge")
            deduplicated[json.dumps(_action_to_payload(candidate), sort_keys=True)] = candidate
            continue
        if action_type == "pass":
            candidate = ActionModel(type="pass")
            deduplicated[json.dumps(_action_to_payload(candidate), sort_keys=True)] = candidate
            continue
        if action_type != "play_claim" or not hand:
            continue

        claim_rank = str(template.get("claim_rank", env.state.table_type))
        min_cards = max(1, int(template.get("min_cards", 1) or 1))
        max_cards = max(min_cards, int(template.get("max_cards", min_cards) or min_cards))
        max_cards = min(max_cards, len(hand))
        for play_count in range(min_cards, max_cards + 1):
            for card_indexes in combinations(range(len(hand)), play_count):
                cards = [str(hand[index]) for index in card_indexes]
                candidate = ActionModel(type="play_claim", claim_rank=claim_rank, cards=cards)
                deduplicated[json.dumps(_action_to_payload(candidate), sort_keys=True)] = candidate

    deduplicated[json.dumps(_action_to_payload(trajectory.action), sort_keys=True)] = trajectory.action
    return list(deduplicated.values()) or [trajectory.action]


def _trajectory_seed_offset(trajectory: TurnTrajectory, base_random_seed: int) -> int:
    digest = md5(f"{trajectory.game_id}:{trajectory.turn}:{trajectory.player_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) + int(base_random_seed)


def _evaluate_turn_alignment(
    analyzer: ShapleyAnalyzer,
    predictor: CompatProxyValuePredictor | object,
    trajectory: TurnTrajectory,
) -> dict[str, object]:
    legal_actions = _enumerate_legal_actions(analyzer=analyzer, trajectory=trajectory)
    rollout_inputs: list[tuple[str, tuple[object, ...]]] = []
    action_rows: list[dict[str, object]] = []
    turn_seed_offset = _trajectory_seed_offset(trajectory, analyzer.settings.runtime.random_seed)

    for action in legal_actions:
        label = _action_label(_action_to_payload(action))
        feature_context = analyzer._build_proxy_feature_context(trajectory=trajectory, action=action)
        feature_vector = (
            [float(item) for item in predictor.encode_state_features(feature_context)]
            if feature_context is not None
            else []
        )
        proxy_score = (
            float(predictor.predict_state_features(feature_context))
            if feature_context is not None
            else 0.0
        )
        action_rows.append(
            {
                "action_label": label,
                "proxy_score": proxy_score,
                "rollout_score": 0.0,
                "feature_vector": feature_vector,
            }
        )
        payload = _action_to_payload(action)
        for sample_idx in range(analyzer.rollout_samples):
            rollout_inputs.append(
                (
                    label,
                    (
                        analyzer.settings_raw,
                        trajectory.checkpoint_payload,
                        payload,
                        trajectory.player_id,
                        turn_seed_offset + sample_idx,
                        analyzer.rollout_policy,
                        False,
                        analyzer.rollout_step_limit,
                        analyzer.baseline_mode,
                    ),
                )
            )

    rollout_scores_by_label: dict[str, list[float]] = {str(row["action_label"]): [] for row in action_rows}
    if analyzer.max_workers == 1:
        for label, rollout_args in rollout_inputs:
            rollout_scores_by_label[label].append(float(_rollout_once(*rollout_args)))
    else:
        with ProcessPoolExecutor(max_workers=analyzer.max_workers) as executor:
            futures = [(label, executor.submit(_rollout_once, *rollout_args)) for label, rollout_args in rollout_inputs]
            for label, future in futures:
                rollout_scores_by_label[label].append(float(future.result()))

    for row in action_rows:
        label = str(row["action_label"])
        row["rollout_score"] = mean(rollout_scores_by_label[label]) if rollout_scores_by_label[label] else 0.0

    return build_turn_alignment_report(
        game_id=trajectory.game_id,
        turn=trajectory.turn,
        action_rows=action_rows,
        chosen_action_label=_action_label(_action_to_payload(trajectory.action)),
    )


def run_checkpoint_proxy_rollout_calibration(
    checkpoint_root: Path | str,
    *,
    model_path: Path | str,
    config_file: Path | str = "config/experiment.yaml",
    sample_size: int = 40,
    sample_seed: int = 4242,
    rollout_samples: int = 12,
    max_workers: int = 1,
    risk_priority_fraction: float = 0.5,
    llm_player_id: str | None = None,
    proxy_output_mode: str = VALUE_PROXY_TARGET_PHI,
) -> dict[str, object]:
    root = Path(checkpoint_root)
    task_m_summary = json.loads((root / "task_m" / "summary.json").read_text(encoding="utf-8"))
    resolved_llm_player_id = str(llm_player_id or task_m_summary.get("llm_player_id", "p1"))
    log_paths = sorted((root / "task_m" / "games").glob("*.jsonl"))
    settings = load_settings(config_file=config_file)
    analyzer = ShapleyAnalyzer(
        settings=settings,
        rollout_samples=rollout_samples,
        rollout_policy="random",
        max_workers=max(1, int(max_workers)),
    )
    predictor = CompatProxyValuePredictor(
        model_path=model_path,
        output_mode=proxy_output_mode,
    )
    sampled = _sample_llm_trajectories(
        log_paths=log_paths,
        llm_player_id=resolved_llm_player_id,
        sample_size=max(1, int(sample_size)),
        sample_seed=int(sample_seed),
        risk_priority_fraction=float(risk_priority_fraction),
    )
    turn_reports = [_evaluate_turn_alignment(analyzer=analyzer, predictor=predictor, trajectory=trajectory) for trajectory in sampled]
    summary = summarize_alignment_reports(checkpoint_label=root.name, turn_reports=turn_reports)
    summary["checkpoint_root"] = str(root)
    summary["llm_player_id"] = resolved_llm_player_id
    summary["log_count"] = len(log_paths)
    summary["requested_sample_size"] = int(sample_size)
    summary["sample_seed"] = int(sample_seed)
    summary["rollout_samples"] = int(rollout_samples)
    summary["risk_priority_fraction"] = float(risk_priority_fraction)
    summary["proxy_output_mode"] = str(proxy_output_mode)
    summary["mismatch_examples"] = sorted(
        [report for report in turn_reports if not bool(report.get("top1_match"))],
        key=lambda item: (-float(item.get("rollout_regret", 0.0) or 0.0), str(item.get("game_id", "")), int(item.get("turn", 0))),
    )[:5]
    return {
        "summary": summary,
        "turn_reports": turn_reports,
    }


def render_checkpoint_markdown(summary_rows: list[dict[str, object]]) -> str:
    headers = [
        "checkpoint_label",
        "sample_size",
        "top1_match_rate",
        "mean_spearman_rank_correlation",
        "mean_action_count",
        "mean_rollout_regret",
        "mean_chosen_action_rollout_rank",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["checkpoint_label"]),
                    str(row["sample_size"]),
                    f"{float(row['top1_match_rate']):.6f}",
                    f"{float(row['mean_spearman_rank_correlation']):.6f}",
                    f"{float(row['mean_action_count']):.6f}",
                    f"{float(row['mean_rollout_regret']):.6f}",
                    f"{float(row['mean_chosen_action_rollout_rank']):.6f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)
