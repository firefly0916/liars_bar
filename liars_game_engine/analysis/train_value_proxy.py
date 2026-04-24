from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from liars_game_engine.engine.game_state import JOKER_RANK


LETHAL_EVENT_PATTERN = re.compile(r"Roulette revealed LETHAL on (?P<player_id>[^;]+); player eliminated\.")
PHASE_INDEX = {
    "turn_start": 0.0,
    "declare": 1.0,
    "response_window": 2.0,
    "resolution": 3.0,
}
TABLE_INDEX = {
    "A": 0.0,
    "K": 1.0,
    "Q": 2.0,
}
VALUE_PROXY_INPUT_DIM = 8
DEFAULT_VALUE_PROXY_EPOCHS = 40
TRUTHFUL_DECK_CAP = 8.0


@dataclass
class ValueSample:
    game_id: str
    features: list[float]
    target: float


class ValueProxyMLP(nn.Module):
    def __init__(self, input_dim: int = VALUE_PROXY_INPUT_DIM, hidden_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def _extract_action_type(action: object) -> str:
    if isinstance(action, dict):
        return str(action.get("type", ""))
    return str(getattr(action, "type", ""))


def _extract_action_cards(action: object) -> list[str]:
    raw_cards: object
    if isinstance(action, dict):
        raw_cards = action.get("cards", [])
    else:
        raw_cards = getattr(action, "cards", [])

    if not isinstance(raw_cards, list):
        return []
    return [str(card) for card in raw_cards]


def _truthful_card_count(cards: list[str], table_type: str) -> int:
    truthful_cards = {table_type, JOKER_RANK}
    return sum(1 for card in cards if str(card) in truthful_cards)


def build_value_proxy_feature_context(
    state_features: dict[str, object] | None = None,
    observation: dict[str, object] | None = None,
    player_id: str | None = None,
    action: object | None = None,
) -> dict[str, object]:
    context = dict(state_features) if isinstance(state_features, dict) else {}
    normalized_observation = observation if isinstance(observation, dict) else {}
    resolved_player_id = player_id or str(normalized_observation.get("player_id", ""))

    if normalized_observation:
        player_states = normalized_observation.get("player_states", {})
        current_player_state = player_states.get(resolved_player_id, {}) if isinstance(player_states, dict) else {}
        if not isinstance(current_player_state, dict):
            current_player_state = {}

        alive_players = normalized_observation.get("alive_players", [])
        private_hand = normalized_observation.get("private_hand", [])
        pending_claim = normalized_observation.get("pending_claim", {})

        context.update(
            {
                "phase": str(normalized_observation.get("phase", context.get("phase", ""))),
                "table_type": str(normalized_observation.get("table_type", context.get("table_type", "A"))),
                "must_call_liar": bool(normalized_observation.get("must_call_liar", context.get("must_call_liar", False))),
                "alive_player_count": len(alive_players) if isinstance(alive_players, list) else 0,
                "hand_count": len(private_hand) if isinstance(private_hand, list) else 0,
                "death_probability": _clamp_unit(
                    _safe_float(current_player_state.get("death_probability", context.get("death_probability", 0.0)))
                ),
                "private_hand": [str(card) for card in private_hand] if isinstance(private_hand, list) else [],
                "pending_claim_declared_count": int(
                    _safe_float(
                        pending_claim.get("declared_count", context.get("pending_claim_declared_count", 0.0))
                        if isinstance(pending_claim, dict)
                        else context.get("pending_claim_declared_count", 0.0),
                        default=0.0,
                    )
                ),
            }
        )

    private_hand = context.get("private_hand", [])
    if not isinstance(private_hand, list):
        private_hand = []
    context["private_hand"] = [str(card) for card in private_hand]

    context["hand_count"] = int(
        _safe_float(context.get("hand_count", len(context["private_hand"])), default=float(len(context["private_hand"])))
    )
    context["alive_player_count"] = int(_safe_float(context.get("alive_player_count", 0.0), default=0.0))
    context["death_probability"] = _clamp_unit(_safe_float(context.get("death_probability", 0.0), default=0.0))
    context["pending_claim_declared_count"] = int(
        _safe_float(context.get("pending_claim_declared_count", 0.0), default=0.0)
    )
    context["action_type"] = _extract_action_type(action) or str(context.get("action_type", ""))
    context["action_cards"] = _extract_action_cards(action) or [
        str(card)
        for card in context.get("action_cards", [])
        if isinstance(context.get("action_cards", []), list)
    ]
    return context


def _infer_winner(records: list[dict[str, object]]) -> str | None:
    if not records:
        return None

    last_record = records[-1]
    observation = last_record.get("observation", {})
    alive_players = set()
    if isinstance(observation, dict):
        alive = observation.get("alive_players", [])
        if isinstance(alive, list):
            alive_players = {str(player_id) for player_id in alive}

    step_result = last_record.get("step_result", {})
    events = step_result.get("events", []) if isinstance(step_result, dict) else []
    if isinstance(events, list):
        for event in events:
            if not isinstance(event, str):
                continue
            matched = LETHAL_EVENT_PATTERN.search(event)
            if matched:
                alive_players.discard(matched.group("player_id"))

    if len(alive_players) == 1:
        return next(iter(alive_players))
    return None


def _build_feature_vector(state_features: dict[str, object]) -> list[float]:
    feature_context = build_value_proxy_feature_context(state_features=state_features)
    phase = str(feature_context.get("phase", ""))
    table_type = str(feature_context.get("table_type", "A"))
    must_call_liar = bool(feature_context.get("must_call_liar", False))
    alive_player_count = int(_safe_float(feature_context.get("alive_player_count", 0.0), default=0.0))
    hand_count = int(_safe_float(feature_context.get("hand_count", 0.0), default=0.0))
    death_probability = _clamp_unit(_safe_float(feature_context.get("death_probability", 0.0), default=0.0))
    private_hand = [str(card) for card in feature_context.get("private_hand", []) if isinstance(feature_context.get("private_hand", []), list)]
    action_type = str(feature_context.get("action_type", ""))
    action_cards = [
        str(card)
        for card in feature_context.get("action_cards", [])
        if isinstance(feature_context.get("action_cards", []), list)
    ]
    pending_claim_declared_count = int(
        _safe_float(feature_context.get("pending_claim_declared_count", 0.0), default=0.0)
    )

    phase_id = PHASE_INDEX.get(phase, 0.0)
    table_id = TABLE_INDEX.get(table_type, 0.0)
    truthful_cards_in_hand = _truthful_card_count(private_hand, table_type)
    hand_truth_ratio = truthful_cards_in_hand / len(private_hand) if private_hand else 0.0

    if action_type == "play_claim":
        action_consistency_score = _truthful_card_count(action_cards, table_type) / len(action_cards) if action_cards else 0.0
    elif action_type == "challenge":
        action_consistency_score = _clamp_unit((truthful_cards_in_hand + pending_claim_declared_count) / TRUTHFUL_DECK_CAP)
    else:
        action_consistency_score = 0.0

    return [
        phase_id / max(1.0, float(len(PHASE_INDEX) - 1)),
        table_id / max(1.0, float(len(TABLE_INDEX) - 1)),
        1.0 if must_call_liar else 0.0,
        _clamp_unit(alive_player_count / 4.0),
        _clamp_unit(hand_count / 8.0),
        death_probability,
        _clamp_unit(hand_truth_ratio),
        _clamp_unit(action_consistency_score),
    ]


def encode_value_proxy_features(state_features: dict[str, object]) -> list[float]:
    return _build_feature_vector(state_features)


def load_value_samples(log_root: Path) -> list[ValueSample]:
    samples: list[ValueSample] = []
    log_paths = sorted(log_root.rglob("*.jsonl"))
    if not log_paths:
        return samples

    for log_path in log_paths:
        records: list[dict[str, object]] = []
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                records.append(payload)

        if not records:
            continue

        game_id = log_path.stem
        winner = _infer_winner(records)

        for record in records:
            state_features = record.get("state_features", {})
            player_id = str(record.get("player_id", ""))
            if not player_id:
                continue

            feature_context = build_value_proxy_feature_context(
                state_features=state_features if isinstance(state_features, dict) else None,
                observation=record.get("observation") if isinstance(record.get("observation"), dict) else None,
                player_id=player_id,
                action=record.get("action"),
            )
            target = 1.0 if (winner is not None and player_id == winner) else 0.0
            samples.append(
                ValueSample(
                    game_id=game_id,
                    features=encode_value_proxy_features(feature_context),
                    target=target,
                )
            )

    return samples


def load_value_samples_from_roots(log_roots: Iterable[Path]) -> list[ValueSample]:
    samples: list[ValueSample] = []
    for log_root in log_roots:
        samples.extend(load_value_samples(Path(log_root)))
    return samples


def split_train_val(
    samples: list[ValueSample],
    val_ratio: float,
    seed: int,
) -> tuple[list[ValueSample], list[ValueSample]]:
    game_ids = sorted({sample.game_id for sample in samples})
    if not game_ids:
        return [], []

    rng = random.Random(seed)
    rng.shuffle(game_ids)

    val_count = max(1, int(len(game_ids) * val_ratio))
    if val_count >= len(game_ids) and len(game_ids) > 1:
        val_count = len(game_ids) - 1

    val_games = set(game_ids[:val_count])
    train_samples = [sample for sample in samples if sample.game_id not in val_games]
    val_samples = [sample for sample in samples if sample.game_id in val_games]
    return train_samples, val_samples


def _build_dataset(samples: list[ValueSample]) -> TensorDataset:
    feature_tensor = torch.tensor([sample.features for sample in samples], dtype=torch.float32)
    target_tensor = torch.tensor([[sample.target] for sample in samples], dtype=torch.float32)
    return TensorDataset(feature_tensor, target_tensor)


def _evaluate_mse(model: nn.Module, dataset: TensorDataset, device: torch.device) -> float:
    if len(dataset) == 0:
        return 0.0

    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    model.eval()
    total_loss = 0.0
    total_count = 0
    criterion = nn.MSELoss(reduction="sum")

    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            preds = model(features)
            loss = criterion(preds, targets)
            total_loss += float(loss.item())
            total_count += targets.shape[0]

    return total_loss / max(1, total_count)


def train_value_proxy(
    output_dir: Path,
    log_root: Path | None = None,
    log_roots: Iterable[Path] | None = None,
    val_ratio: float = 0.2,
    epochs: int = DEFAULT_VALUE_PROXY_EPOCHS,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    seed: int = 42,
    model_filename: str = "value_proxy_mlp.pt",
    metrics_filename: str = "value_proxy_metrics.json",
) -> dict[str, object]:
    random.seed(seed)
    torch.manual_seed(seed)

    resolved_log_roots = [Path(item) for item in log_roots] if log_roots is not None else []
    if not resolved_log_roots:
        if log_root is None:
            raise RuntimeError("Either log_root or log_roots must be provided")
        resolved_log_roots = [Path(log_root)]

    samples = load_value_samples_from_roots(resolved_log_roots)
    if not samples:
        raise RuntimeError(f"No samples found under log roots: {resolved_log_roots}")

    train_samples, val_samples = split_train_val(samples=samples, val_ratio=val_ratio, seed=seed)
    if not train_samples or not val_samples:
        raise RuntimeError("Train/validation split failed: insufficient game samples")

    train_dataset = _build_dataset(train_samples)
    val_dataset = _build_dataset(val_samples)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ValueProxyMLP(input_dim=VALUE_PROXY_INPUT_DIM, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_val_mse = float("inf")
    best_state_dict = None

    for _epoch in range(max(1, epochs)):
        model.train()
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)

            preds = model(features)
            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_mse = _evaluate_mse(model=model, dataset=val_dataset, device=device)
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state_dict = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    train_mse = _evaluate_mse(model=model, dataset=train_dataset, device=device)
    final_val_mse = _evaluate_mse(model=model, dataset=val_dataset, device=device)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / model_filename
    metrics_path = output_dir / metrics_filename

    torch.save(model.state_dict(), model_path)

    metrics = {
        "log_root": str(resolved_log_roots[0]) if len(resolved_log_roots) == 1 else "",
        "log_roots": [str(item) for item in resolved_log_roots],
        "train_sample_count": len(train_samples),
        "val_sample_count": len(val_samples),
        "total_sample_count": len(samples),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "seed": int(seed),
        "device": str(device),
        "train_mse": float(train_mse),
        "val_mse": float(final_val_mse),
        "best_val_mse": float(best_val_mse),
        "input_dim": VALUE_PROXY_INPUT_DIM,
        "model_path": str(model_path),
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train value proxy MLP on Task D trajectory logs")
    parser.add_argument("--log-root", type=Path, default=Path("logs/task_d_probe"))
    parser.add_argument("--output-dir", type=Path, default=Path("logs/task_d_probe"))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=DEFAULT_VALUE_PROXY_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metrics = train_value_proxy(
        log_root=args.log_root,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    print("Value proxy training finished:", json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
