from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from liars_game_engine.analysis.hicra_preprocessor import build_hicra_sample
from liars_game_engine.analysis.shapley_analyzer import ProxyValuePredictor
from liars_game_engine.analysis.token_alignment import recalibrate_alignment_sample
from liars_game_engine.analysis.train_value_proxy import (
    VALUE_PROXY_TARGET_PHI,
    build_value_proxy_feature_context,
)


DEFAULT_PROXY_MODEL_PATH = Path("models/proxy/value_proxy_mlp_distill.pt")
DEFAULT_SIGNAL_EPSILON = 1e-8
_CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def _ensure_english_only_text(value: object, *, field_name: str) -> str:
    text = str(value or "")
    if _CJK_PATTERN.search(text):
        raise ValueError(f"English-only validation failed for {field_name}: detected CJK content")
    return text


def _validate_alignment_record_language(record: dict[str, object]) -> dict[str, object]:
    _ensure_english_only_text(record.get("thought", ""), field_name="thought")
    _ensure_english_only_text(record.get("base_prompt", ""), field_name="base_prompt")
    _ensure_english_only_text(record.get("rendered_prompt", ""), field_name="rendered_prompt")

    messages = record.get("messages", [])
    if isinstance(messages, list):
        for index, message in enumerate(messages):
            if isinstance(message, dict):
                _ensure_english_only_text(message.get("content", ""), field_name=f"messages[{index}].content")

    strategic_tokens = record.get("strategic_tokens", [])
    if isinstance(strategic_tokens, list):
        for index, token in enumerate(strategic_tokens):
            if isinstance(token, dict):
                _ensure_english_only_text(token.get("token", ""), field_name=f"strategic_tokens[{index}].token")

    alignment_metadata = record.get("alignment_metadata", {})
    if isinstance(alignment_metadata, dict):
        for field_name in ("reasoning_text", "assistant_text", "rendered_text"):
            _ensure_english_only_text(
                alignment_metadata.get(field_name, ""),
                field_name=f"alignment_metadata.{field_name}",
            )
    return record


def load_alignment_records(dataset_path: Path | str) -> list[dict[str, object]]:
    path = Path(dataset_path)
    records: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError("Alignment dataset rows must be JSON objects")
        records.append(_validate_alignment_record_language(record))
    return records


def compute_mask_hit_metrics(record: dict[str, object]) -> dict[str, float | int]:
    alignment_metadata = record.get("alignment_metadata", {})
    token_weight_mask = record.get("token_weight_mask", [])
    alignments = alignment_metadata.get("strategic_token_alignments", []) if isinstance(alignment_metadata, dict) else []
    if not isinstance(alignments, list):
        alignments = []
    if not isinstance(token_weight_mask, list):
        token_weight_mask = []

    strategic_token_indices: set[int] = set()
    for alignment in alignments:
        if not isinstance(alignment, dict):
            continue
        token_indices = alignment.get("token_indices", [])
        if not isinstance(token_indices, list):
            continue
        strategic_token_indices.update(int(index) for index in token_indices)

    non_zero_mask_indices = {index for index, weight in enumerate(token_weight_mask) if float(weight) != 0.0}
    hit_count = len(strategic_token_indices & non_zero_mask_indices)
    strategic_count = len(strategic_token_indices)
    mask_hit_rate = (hit_count / strategic_count) if strategic_count else 0.0
    return {
        "strategic_alignment_count": len(alignments),
        "strategic_token_index_count": strategic_count,
        "non_zero_mask_count": len(non_zero_mask_indices),
        "mask_hit_count": hit_count,
        "mask_hit_rate": float(mask_hit_rate),
        "average_hicra_mask_intensity": (
            sum(abs(float(token_weight_mask[index])) for index in non_zero_mask_indices) / len(non_zero_mask_indices)
            if non_zero_mask_indices
            else 0.0
        ),
    }


def is_class_a_record(record: dict[str, object], *, ev_gap_threshold: float = 0.15) -> bool:
    return bool(record.get("reasoning_action_mismatch", False)) or float(record.get("ev_gap", 0.0) or 0.0) > float(ev_gap_threshold)


def compute_sampling_weight(
    record: dict[str, object],
    *,
    alpha: float = 1.0,
    ev_gap_threshold: float = 0.15,
) -> float:
    if not is_class_a_record(record, ev_gap_threshold=ev_gap_threshold):
        return 1.0
    return float(1.0 + (float(alpha) * float(record.get("ev_gap", 0.0) or 0.0)))


def should_skip_gradient_update(
    *,
    reward_span: float,
    mean_abs_advantage: float,
    non_zero_mask_count: int,
    epsilon: float = DEFAULT_SIGNAL_EPSILON,
) -> dict[str, object]:
    signalless_step = float(reward_span) <= float(epsilon) or float(mean_abs_advantage) <= float(epsilon)
    idle_step = bool(signalless_step) and int(non_zero_mask_count) == 0
    return {
        "skip_update": bool(signalless_step),
        "idle_step": bool(idle_step),
        "signalless_step": bool(signalless_step),
        "signal_epsilon": float(epsilon),
    }


def summarize_step_metrics(step_summaries: list[dict[str, object]]) -> dict[str, float | int]:
    total_steps = len(step_summaries)
    effective_step_count = sum(1 for step in step_summaries if not bool(step.get("skip_update", False)))
    idle_step_count = sum(1 for step in step_summaries if bool(step.get("idle_step", False)))
    signalless_step_count = sum(1 for step in step_summaries if bool(step.get("signalless_step", False)))
    non_zero_intensities = [
        float(step.get("mask_metrics", {}).get("average_hicra_mask_intensity", 0.0) or 0.0)
        for step in step_summaries
        if float(step.get("mask_metrics", {}).get("average_hicra_mask_intensity", 0.0) or 0.0) > 0.0
    ]
    return {
        "total_sampled_steps": total_steps,
        "effective_step_count": effective_step_count,
        "idle_step_count": idle_step_count,
        "signalless_step_count": signalless_step_count,
        "signal_density_rate": (effective_step_count / total_steps) if total_steps else 0.0,
        "average_hicra_mask_intensity": (
            sum(non_zero_intensities) / len(non_zero_intensities)
            if non_zero_intensities
            else 0.0
        ),
    }


def summarize_smoke_groups(groups: list[dict[str, object]]) -> dict[str, float | int]:
    reward_spans: list[float] = []
    all_rewards: list[float] = []
    high_ev_gap_mismatch_group_count = 0
    for group in groups:
        rewards = [
            float(candidate.get("reward_breakdown", {}).get("total_reward", 0.0) or 0.0)
            for candidate in group.get("candidates", [])
            if isinstance(candidate, dict)
        ]
        if rewards:
            reward_spans.append(max(rewards) - min(rewards))
            all_rewards.extend(rewards)
        ev_gap = float(group.get("ev_gap", 0.0) or 0.0)
        if bool(group.get("reasoning_action_mismatch", False)) and ev_gap > 0.15:
            high_ev_gap_mismatch_group_count += 1

    reward_mean = (sum(all_rewards) / len(all_rewards)) if all_rewards else 0.0
    reward_variance = (
        sum((reward - reward_mean) ** 2 for reward in all_rewards) / len(all_rewards)
        if all_rewards
        else 0.0
    )
    return {
        "group_count": len(groups),
        "high_ev_gap_mismatch_group_count": high_ev_gap_mismatch_group_count,
        "reward_variance": float(reward_variance),
        "max_group_reward_span": max(reward_spans) if reward_spans else 0.0,
        "mean_total_reward": float(reward_mean),
    }


def compute_group_relative_advantages(rewards: list[float]) -> list[float]:
    if not rewards:
        return []
    baseline = sum(rewards) / len(rewards)
    return [float(reward - baseline) for reward in rewards]


def _normalize_offset_mapping(offset_mapping: object) -> list[tuple[int, int]]:
    if not isinstance(offset_mapping, list):
        return []
    if offset_mapping and isinstance(offset_mapping[0], list):
        offset_mapping = offset_mapping[0]
    normalized: list[tuple[int, int]] = []
    for item in offset_mapping:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            normalized.append((int(item[0]), int(item[1])))
    return normalized


def _normalize_input_ids(value: object) -> list[int]:
    if isinstance(value, list):
        if value and isinstance(value[0], list):
            value = value[0]
        return [int(item) for item in value]
    return []


def _tokenize_rendered_text(tokenizer: object, rendered_text: str) -> tuple[list[int], list[int], list[tuple[int, int]]]:
    encoded = tokenizer(rendered_text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = _normalize_input_ids(encoded.get("input_ids", []))
    attention_mask = _normalize_input_ids(encoded.get("attention_mask", [1 for _ in input_ids]))
    if not attention_mask:
        attention_mask = [1 for _ in input_ids]
    offset_mapping = _normalize_offset_mapping(encoded.get("offset_mapping", []))
    return input_ids, attention_mask, offset_mapping


def _truncate_from_left(values: list, max_seq_len: int) -> list:
    if max_seq_len <= 0 or len(values) <= max_seq_len:
        return list(values)
    return list(values[-max_seq_len:])


def _find_overlapping_token_indices(
    offset_mapping: list[tuple[int, int]],
    span_start: int,
    span_end: int,
) -> list[int]:
    token_indices: list[int] = []
    for index, (token_start, token_end) in enumerate(offset_mapping):
        if token_end > span_start and token_start < span_end:
            token_indices.append(index)
    return token_indices


def prepare_candidate_training_example(
    candidate: dict[str, object],
    tokenizer: object,
    max_seq_len: int | None = None,
) -> dict[str, object]:
    import torch

    recalibrated = recalibrate_alignment_sample(sample=candidate, tokenizer=tokenizer)
    alignment_metadata = recalibrated.get("alignment_metadata", {})
    if not isinstance(alignment_metadata, dict):
        raise ValueError("alignment_metadata missing after recalibration")

    rendered_text = str(alignment_metadata.get("rendered_text", "") or "")
    assistant_span = alignment_metadata.get("assistant_rendered_span", {})
    if not isinstance(assistant_span, dict):
        assistant_span = {}
    assistant_start = int(assistant_span.get("start", 0) or 0)
    assistant_end = int(assistant_span.get("end", 0) or 0)

    input_ids, attention_mask, offset_mapping = _tokenize_rendered_text(tokenizer, rendered_text)
    assistant_token_indices = set(_find_overlapping_token_indices(offset_mapping, assistant_start, assistant_end))
    assistant_token_mask = [1.0 if index in assistant_token_indices else 0.0 for index in range(len(input_ids))]
    token_weight_mask = recalibrated.get("token_weight_mask", [])
    if not isinstance(token_weight_mask, list):
        token_weight_mask = []
    token_weight_mask = [float(weight) for weight in token_weight_mask]
    if len(token_weight_mask) < len(input_ids):
        token_weight_mask = token_weight_mask + [0.0 for _ in range(len(input_ids) - len(token_weight_mask))]
    else:
        token_weight_mask = token_weight_mask[: len(input_ids)]

    if max_seq_len is not None:
        capped_len = max(1, int(max_seq_len))
        input_ids = _truncate_from_left(input_ids, capped_len)
        attention_mask = _truncate_from_left(attention_mask, capped_len)
        assistant_token_mask = _truncate_from_left(assistant_token_mask, capped_len)
        token_weight_mask = _truncate_from_left(token_weight_mask, capped_len)

    label_token_mask = assistant_token_mask[1:]
    shifted_token_weights = token_weight_mask[1:]
    return {
        "rendered_text": rendered_text,
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
        "label_token_mask": torch.tensor(label_token_mask, dtype=torch.float32),
        "token_weight_mask": torch.tensor(shifted_token_weights, dtype=torch.float32),
        "assistant_token_count": int(sum(assistant_token_mask)),
        "active_label_count": int(sum(label_token_mask)),
        "hicra_non_zero_count": int(sum(1 for value in shifted_token_weights if float(value) != 0.0)),
        "alignment_metadata": alignment_metadata,
    }


def resolve_model_loading_options(
    *,
    device: str,
    torch_dtype: str,
    load_in_4bit: bool,
) -> dict[str, object]:
    resolved_dtype = _resolve_torch_dtype(torch_dtype)
    options: dict[str, object] = {
        "device_map": str(device),
        "use_gradient_checkpointing": True,
        "load_in_4bit": bool(load_in_4bit),
        "dtype": resolved_dtype,
    }
    if load_in_4bit:
        import torch
        from transformers import BitsAndBytesConfig

        options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=resolved_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        options["torch_dtype"] = resolved_dtype
    return options


def compute_candidate_loss_terms(
    *,
    logits,
    ref_logits,
    input_ids,
    label_token_mask,
    token_weight_mask,
    advantage: float,
    kl_beta: float,
    hicra_gamma: float,
) -> dict[str, object]:
    import torch
    import torch.nn.functional as F

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_ref_logits = ref_logits[:, :-1, :] if ref_logits is not None else None

    log_probs = F.log_softmax(shift_logits, dim=-1)
    gathered_log_probs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

    active_mask = label_token_mask.to(device=shift_logits.device, dtype=shift_logits.dtype)
    token_weights = token_weight_mask.to(device=shift_logits.device, dtype=shift_logits.dtype)
    hicra_coefficients = 1.0 + torch.abs(token_weights) * float(hicra_gamma)
    weighted_mask = active_mask * hicra_coefficients
    weight_denom = torch.clamp(weighted_mask.sum(), min=1.0)
    active_denom = torch.clamp(active_mask.sum(), min=1.0)

    mean_log_prob = (gathered_log_probs * active_mask).sum() / active_denom
    weighted_mean_log_prob = (gathered_log_probs * weighted_mask).sum() / weight_denom
    nll = -(gathered_log_probs * active_mask).sum() / active_denom
    weighted_nll = -(gathered_log_probs * weighted_mask).sum() / weight_denom

    if shift_ref_logits is not None:
        ref_log_probs = F.log_softmax(shift_ref_logits, dim=-1)
        policy_probs = torch.exp(log_probs)
        tokenwise_kl = torch.sum(policy_probs * (log_probs - ref_log_probs), dim=-1)
        kl_value = (tokenwise_kl * active_mask).sum() / active_denom
    else:
        kl_value = torch.zeros((), dtype=shift_logits.dtype, device=shift_logits.device)

    loss = -(float(advantage) * weighted_mean_log_prob) + (float(kl_beta) * kl_value)
    return {
        "loss": loss,
        "advantage": float(advantage),
        "mean_log_prob": float(mean_log_prob.detach().cpu().item()),
        "weighted_mean_log_prob": float(weighted_mean_log_prob.detach().cpu().item()),
        "nll": float(nll.detach().cpu().item()),
        "weighted_nll": float(weighted_nll.detach().cpu().item()),
        "kl_value": float(kl_value.detach().cpu().item()),
        "hicra_weight_mean": float((weighted_mask.sum() / active_denom).detach().cpu().item()),
        "active_label_count": int(active_mask.sum().detach().cpu().item()),
    }


def _score_candidate(
    predictor: ProxyValuePredictor | object,
    candidate: dict[str, object],
) -> float:
    state_features = candidate.get("state_features", {})
    if not isinstance(state_features, dict):
        state_features = {}
    action = candidate.get("action", {})
    if not isinstance(action, dict):
        action = {}
    feature_context = build_value_proxy_feature_context(
        state_features=state_features,
        observation=None,
        player_id=str(candidate.get("player_id", "")),
        action=action,
    )
    return float(predictor.predict_state_features(feature_context))


def _normalize_action(action: dict[str, object] | None) -> dict[str, object]:
    payload = dict(action or {})
    cards = payload.get("cards", [])
    if not isinstance(cards, list):
        cards = []
    return {
        "type": str(payload.get("type", "") or ""),
        "claim_rank": str(payload.get("claim_rank", "") or ""),
        "cards": [str(card) for card in cards],
    }


def _actions_equal(left: dict[str, object], right: dict[str, object]) -> bool:
    return _normalize_action(left) == _normalize_action(right)


def _extract_explicit_challenge_action(record: dict[str, object]) -> dict[str, object] | None:
    observation = record.get("observation", {})
    if not isinstance(observation, dict):
        return None
    legal_actions = observation.get("legal_actions", [])
    if not isinstance(legal_actions, list):
        return None
    for legal_action in legal_actions:
        normalized = _normalize_action(legal_action if isinstance(legal_action, dict) else {})
        if normalized["type"] == "challenge":
            return normalized
    return None


def _coerce_alignment_record(record: dict[str, object]) -> dict[str, object]:
    base = build_hicra_sample(record)
    merged = {**base, **dict(record)}
    merged["action"] = _normalize_action(merged.get("action") if isinstance(merged.get("action"), dict) else {})
    proxy_target_action = merged.get("proxy_target_action", {})
    if not isinstance(proxy_target_action, dict):
        proxy_target_action = {}
    merged["proxy_target_action"] = _normalize_action(proxy_target_action)
    merged["strategic_tokens"] = list(merged.get("strategic_tokens", [])) if isinstance(merged.get("strategic_tokens", []), list) else []
    merged["strategic_token_weight"] = float(merged.get("strategic_token_weight", 1.0) or 1.0)
    merged["token_penalty"] = float(merged.get("token_penalty", 0.0) or 0.0)
    merged["ev_gap"] = float(merged.get("ev_gap", 0.0) or 0.0)
    return merged


def _build_group_candidates(record: dict[str, object], group_size: int) -> list[dict[str, object]]:
    base_candidate = _coerce_alignment_record(record)
    chosen_action = _normalize_action(base_candidate.get("action", {}))
    proxy_target_action = _normalize_action(base_candidate.get("proxy_target_action", {}))
    explicit_challenge_action = _extract_explicit_challenge_action(base_candidate)

    candidate_specs: list[tuple[str, dict[str, object]]] = [("logged_action", chosen_action)]
    if proxy_target_action and not _actions_equal(proxy_target_action, chosen_action):
        candidate_specs.append(("proxy_target", proxy_target_action))
    if explicit_challenge_action and not any(
        _actions_equal(explicit_challenge_action, candidate_action) for _, candidate_action in candidate_specs
    ):
        candidate_specs.append(("legal_challenge", explicit_challenge_action))

    candidates: list[dict[str, object]] = []
    for index, (candidate_role, candidate_action) in enumerate(candidate_specs[: max(1, int(group_size))]):
        candidates.append(
            {
                **base_candidate,
                "action": dict(candidate_action),
                "candidate_index": index,
                "candidate_role": candidate_role,
            }
        )
    return candidates


def _compute_reward_breakdown(
    predictor: ProxyValuePredictor | object,
    candidate: dict[str, object],
) -> dict[str, float]:
    proxy_target_action = candidate.get("proxy_target_action", {})
    if not isinstance(proxy_target_action, dict):
        proxy_target_action = {}
    action_match_reward = 1.0 if _actions_equal(candidate.get("action", {}), proxy_target_action) else 0.0
    phi_dense_reward = _score_candidate(predictor, candidate)
    strategic_tokens = candidate.get("strategic_tokens", [])
    has_strategic_tokens = isinstance(strategic_tokens, list) and len(strategic_tokens) > 0
    strategic_token_weight = float(candidate.get("strategic_token_weight", 1.0) or 1.0)
    token_penalty = float(candidate.get("token_penalty", 0.0) or 0.0)
    hicra_penalty = token_penalty * strategic_token_weight if has_strategic_tokens and action_match_reward < 1.0 else 0.0
    total_reward = action_match_reward + phi_dense_reward + hicra_penalty
    return {
        "action_match_reward": float(action_match_reward),
        "phi_dense_reward": float(phi_dense_reward),
        "hicra_penalty": float(hicra_penalty),
        "total_reward": float(total_reward),
    }


def _build_scored_group(
    predictor: ProxyValuePredictor | object,
    record: dict[str, object],
    group_size: int,
) -> dict[str, object]:
    candidates = _build_group_candidates(record, group_size=group_size)
    rewards: list[float] = []
    for candidate in candidates:
        reward_breakdown = _compute_reward_breakdown(predictor, candidate)
        candidate["reward_breakdown"] = reward_breakdown
        rewards.append(float(reward_breakdown["total_reward"]))
    advantages = compute_group_relative_advantages(rewards)
    for candidate, advantage in zip(candidates, advantages, strict=False):
        candidate["advantage"] = float(advantage)
    return {
        "game_id": str(record.get("game_id", "")),
        "turn": int(record.get("turn", 0) or 0),
        "ev_gap": float(record.get("ev_gap", 0.0) or 0.0),
        "reasoning_action_mismatch": bool(record.get("reasoning_action_mismatch", False)),
        "candidate_count": len(candidates),
        "rewards": rewards,
        "advantages": advantages,
        "mask_metrics": compute_mask_hit_metrics(record),
        "candidates": candidates,
    }


def prioritize_smoke_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    def _sort_key(record: dict[str, object]) -> tuple[int, float, str, int]:
        is_anchor = bool(record.get("reasoning_action_mismatch", False)) and float(record.get("ev_gap", 0.0) or 0.0) > 0.15
        return (0 if is_anchor else 1, -float(record.get("ev_gap", 0.0) or 0.0), str(record.get("game_id", "")), int(record.get("turn", 0) or 0))

    return sorted(records, key=_sort_key)


def build_weighted_record_schedule(
    records: list[dict[str, object]],
    *,
    steps: int,
    anchor_ratio: float = 0.7,
    alpha: float = 1.0,
    ev_gap_threshold: float = 0.15,
) -> list[dict[str, object]]:
    if not records:
        return []
    anchor_records = [record for record in records if is_class_a_record(record, ev_gap_threshold=ev_gap_threshold)]
    standard_records = [record for record in records if not is_class_a_record(record, ev_gap_threshold=ev_gap_threshold)]
    prioritized_anchors = prioritize_smoke_records(anchor_records)
    prioritized_standard = prioritize_smoke_records(standard_records)
    anchor_index = 0
    standard_index = 0
    schedule: list[dict[str, object]] = []
    normalized_anchor_ratio = max(0.0, min(1.0, float(anchor_ratio)))
    for step_index in range(max(1, int(steps))):
        choose_anchor = ((step_index + 1) / max(1, int(steps))) <= normalized_anchor_ratio if step_index < int(steps * normalized_anchor_ratio) else False
        if prioritized_anchors and (choose_anchor or not prioritized_standard):
            weighted_anchors = sorted(
                prioritized_anchors,
                key=lambda record: (-compute_sampling_weight(record, alpha=alpha, ev_gap_threshold=ev_gap_threshold), -float(record.get("ev_gap", 0.0) or 0.0)),
            )
            record = weighted_anchors[anchor_index % len(weighted_anchors)]
            anchor_index += 1
        elif prioritized_standard:
            record = prioritized_standard[standard_index % len(prioritized_standard)]
            standard_index += 1
        else:
            record = prioritized_anchors[anchor_index % len(prioritized_anchors)]
            anchor_index += 1
        schedule.append(record)
    return schedule


def run_alignment_dry_run(
    dataset_path: Path | str,
    model_path: Path | str | None = None,
    group_size: int = 8,
) -> dict[str, object]:
    records = load_alignment_records(dataset_path)
    resolved_model_path = Path(model_path) if model_path is not None else DEFAULT_PROXY_MODEL_PATH
    predictor = ProxyValuePredictor(model_path=resolved_model_path, output_mode=VALUE_PROXY_TARGET_PHI)

    groups: list[dict[str, object]] = []
    for record in records:
        groups.append(_build_scored_group(predictor=predictor, record=record, group_size=group_size))

    summary = {
        "dataset_path": str(dataset_path),
        "model_path": str(resolved_model_path),
        "group_size": int(group_size),
        "record_count": len(records),
        "groups": groups,
    }
    summary["smoke_metrics"] = summarize_smoke_groups(groups)
    return summary


def _resolve_torch_dtype(dtype_name: str):
    import torch

    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = str(dtype_name or "auto").lower()
    if key == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if key not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return mapping[key]


def _build_training_components(
    model_name_or_path: str,
    learning_rate: float,
    *,
    device: str | None = None,
    torch_dtype: str = "auto",
    use_lora: bool = True,
    load_in_4bit: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    loading_options = resolve_model_loading_options(
        device=resolved_device,
        torch_dtype=torch_dtype,
        load_in_4bit=load_in_4bit,
    )
    model_kwargs = {
        "trust_remote_code": True,
        "use_cache": False,
        "low_cpu_mem_usage": True,
        "device_map": loading_options["device_map"],
    }
    if "quantization_config" in loading_options:
        model_kwargs["quantization_config"] = loading_options["quantization_config"]
    else:
        model_kwargs["torch_dtype"] = loading_options["torch_dtype"]
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_kwargs,
    )
    if not load_in_4bit:
        model.to(resolved_device)
    if hasattr(model, "config"):
        model.config.use_cache = False
    if loading_options["use_gradient_checkpointing"] and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    lora_enabled = False
    if use_lora:
        from peft import LoraConfig, get_peft_model

        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        lora_config = LoraConfig(
            r=int(lora_r),
            lora_alpha=int(lora_alpha),
            lora_dropout=float(lora_dropout),
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        lora_enabled = True

    model.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(learning_rate),
    )
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "model": model,
        "tokenizer": tokenizer,
        "optimizer": optimizer,
        "device": resolved_device,
        "torch_dtype": str(loading_options["dtype"]),
        "lora_enabled": lora_enabled,
        "trainable_parameter_count": int(trainable_params),
        "load_in_4bit": bool(load_in_4bit),
    }


def _compute_reference_logits(model, input_ids, attention_mask):
    if hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            return model(input_ids=input_ids, attention_mask=attention_mask).logits.detach()
    return None


def save_training_artifacts(
    *,
    model,
    tokenizer,
    optimizer,
    checkpoint_dir: Path | str,
    tag: str,
    lora_enabled: bool,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    import torch

    output_dir = Path(checkpoint_dir) / str(tag)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not hasattr(model, "save_pretrained"):
        raise ValueError("Training model does not support save_pretrained")
    model.save_pretrained(str(output_dir))

    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(str(output_dir))

    optimizer_state_path = output_dir / "optimizer.pt"
    if optimizer is not None and hasattr(optimizer, "state_dict"):
        torch.save(optimizer.state_dict(), optimizer_state_path)
        optimizer_state_path_value: str | None = str(optimizer_state_path)
    else:
        optimizer_state_path_value = None

    metadata_path_value: str | None = None
    if metadata is not None:
        metadata_path = output_dir / "training_snapshot.json"
        snapshot_payload = dict(metadata)
        snapshot_payload["tag"] = str(tag)
        snapshot_payload["path"] = str(output_dir)
        snapshot_payload["lora_enabled"] = bool(lora_enabled)
        metadata_path.write_text(json.dumps(snapshot_payload, ensure_ascii=True, indent=2), encoding="utf-8")
        metadata_path_value = str(metadata_path)

    return {
        "tag": str(tag),
        "path": str(output_dir),
        "lora_enabled": bool(lora_enabled),
        "metadata_path": metadata_path_value,
        "optimizer_state_path": optimizer_state_path_value,
    }


def run_smoke_training(
    dataset_path: Path | str,
    *,
    policy_model_path: str,
    model_path: Path | str | None = None,
    group_size: int = 8,
    steps: int = 10,
    learning_rate: float = 1e-4,
    kl_beta: float = 0.05,
    hicra_gamma: float = 1.0,
    max_grad_norm: float = 1.0,
    max_records: int | None = None,
    device: str | None = None,
    torch_dtype: str = "auto",
    use_lora: bool = True,
    load_in_4bit: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    max_seq_len: int | None = None,
    signal_epsilon: float = DEFAULT_SIGNAL_EPSILON,
    anchor_ratio: float = 0.7,
    anchor_alpha: float = 1.0,
    ev_gap_threshold: float = 0.15,
    checkpoint_dir: Path | str | None = None,
    save_every_steps: int | None = None,
    save_final_adapter: bool = False,
) -> dict[str, object]:
    import torch

    if (save_every_steps is not None or save_final_adapter) and checkpoint_dir is None:
        raise ValueError("checkpoint_dir is required when save_every_steps or save_final_adapter is enabled")

    records = prioritize_smoke_records(load_alignment_records(dataset_path))
    if max_records is not None:
        records = records[: max(1, int(max_records))]
    scheduled_records = build_weighted_record_schedule(
        records,
        steps=max(1, int(steps)),
        anchor_ratio=anchor_ratio,
        alpha=anchor_alpha,
        ev_gap_threshold=ev_gap_threshold,
    )
    resolved_model_path = Path(model_path) if model_path is not None else DEFAULT_PROXY_MODEL_PATH
    predictor = ProxyValuePredictor(model_path=resolved_model_path, output_mode=VALUE_PROXY_TARGET_PHI)
    components = _build_training_components(
        model_name_or_path=policy_model_path,
        learning_rate=learning_rate,
        device=device,
        torch_dtype=torch_dtype,
        use_lora=use_lora,
        load_in_4bit=load_in_4bit,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    model = components["model"]
    tokenizer = components["tokenizer"]
    optimizer = components["optimizer"]
    resolved_device = str(components["device"])

    steps = max(1, int(steps))
    normalized_save_every_steps = None
    if save_every_steps is not None and int(save_every_steps) > 0:
        normalized_save_every_steps = int(save_every_steps)
    step_summaries: list[dict[str, object]] = []
    checkpoint_events: list[dict[str, object]] = []
    for step_index in range(steps):
        record = scheduled_records[step_index % len(scheduled_records)]
        group = _build_scored_group(predictor=predictor, record=record, group_size=group_size)
        reward_span = max(group["rewards"]) - min(group["rewards"]) if group.get("rewards") else 0.0
        mean_advantage_abs = sum(abs(float(item)) for item in group.get("advantages", [])) / len(group.get("advantages", [])) if group.get("advantages") else 0.0
        mask_metrics = group["mask_metrics"]
        signal_flags = should_skip_gradient_update(
            reward_span=reward_span,
            mean_abs_advantage=mean_advantage_abs,
            non_zero_mask_count=int(mask_metrics.get("non_zero_mask_count", 0) or 0),
            epsilon=signal_epsilon,
        )
        optimizer.zero_grad(set_to_none=True)
        candidate_metrics: list[dict[str, object]] = []
        total_loss = None
        grad_norm = 0.0
        if not signal_flags["skip_update"]:
            for candidate in group["candidates"]:
                example = prepare_candidate_training_example(candidate, tokenizer, max_seq_len=max_seq_len)
                input_ids = example["input_ids"].to(resolved_device)
                attention_mask = example["attention_mask"].to(resolved_device)
                label_token_mask = example["label_token_mask"].to(resolved_device)
                token_weight_mask = example["token_weight_mask"].to(resolved_device)

                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                ref_logits = _compute_reference_logits(model, input_ids=input_ids, attention_mask=attention_mask)
                metrics = compute_candidate_loss_terms(
                    logits=logits,
                    ref_logits=ref_logits,
                    input_ids=input_ids,
                    label_token_mask=label_token_mask,
                    token_weight_mask=token_weight_mask,
                    advantage=float(candidate.get("advantage", 0.0) or 0.0),
                    kl_beta=float(kl_beta),
                    hicra_gamma=float(hicra_gamma),
                )
                candidate_metrics.append(
                    {
                        "candidate_index": int(candidate.get("candidate_index", 0) or 0),
                        "candidate_role": str(candidate.get("candidate_role", "")),
                        "advantage": float(candidate.get("advantage", 0.0) or 0.0),
                        "loss": float(metrics["loss"].detach().cpu().item()),
                        "nll": metrics["nll"],
                        "weighted_nll": metrics["weighted_nll"],
                        "kl_value": metrics["kl_value"],
                        "hicra_weight_mean": metrics["hicra_weight_mean"],
                        "active_label_count": metrics["active_label_count"],
                    }
                )
                total_loss = metrics["loss"] if total_loss is None else total_loss + metrics["loss"]

            total_loss = total_loss / max(1, len(group["candidates"]))
            total_loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(max_grad_norm)).detach().cpu().item())
            optimizer.step()
        else:
            total_loss = torch.zeros((), dtype=torch.float32, device=resolved_device)

        mean_candidate_loss = sum(item["loss"] for item in candidate_metrics) / len(candidate_metrics) if candidate_metrics else 0.0
        mean_kl = sum(item["kl_value"] for item in candidate_metrics) / len(candidate_metrics) if candidate_metrics else 0.0
        mean_weighted_nll = sum(item["weighted_nll"] for item in candidate_metrics) / len(candidate_metrics) if candidate_metrics else 0.0
        step_summaries.append(
            {
                "step": step_index + 1,
                "game_id": group["game_id"],
                "turn": group["turn"],
                "ev_gap": group["ev_gap"],
                "reasoning_action_mismatch": group["reasoning_action_mismatch"],
                "loss": float(total_loss.detach().cpu().item()),
                "mean_candidate_loss": float(mean_candidate_loss),
                "mean_weighted_nll": float(mean_weighted_nll),
                "mean_kl": float(mean_kl),
                "mean_abs_advantage": float(mean_advantage_abs),
                "reward_span": float(reward_span),
                "mask_metrics": mask_metrics,
                "grad_norm": grad_norm,
                "nonfinite_loss": bool(not math.isfinite(float(total_loss.detach().cpu().item()))),
                **signal_flags,
            }
        )
        if checkpoint_dir is not None and normalized_save_every_steps is not None and ((step_index + 1) % normalized_save_every_steps == 0):
            checkpoint_events.append(
                save_training_artifacts(
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    checkpoint_dir=checkpoint_dir,
                    tag=f"step-{step_index + 1:06d}",
                    lora_enabled=bool(components["lora_enabled"]),
                    metadata={
                        "step": step_index + 1,
                        "reason": "interval",
                        "dataset_path": str(dataset_path),
                        "policy_model_path": str(policy_model_path),
                        "step_summary": step_summaries[-1],
                    },
                )
            )

    if checkpoint_dir is not None and save_final_adapter:
        checkpoint_events.append(
            save_training_artifacts(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                checkpoint_dir=checkpoint_dir,
                tag="final",
                lora_enabled=bool(components["lora_enabled"]),
                metadata={
                    "step": len(step_summaries),
                    "reason": "final",
                    "dataset_path": str(dataset_path),
                    "policy_model_path": str(policy_model_path),
                    "step_metrics": summarize_step_metrics(step_summaries),
                },
            )
        )

    final_adapter_path = None
    for event in checkpoint_events:
        if str(event.get("tag", "")) == "final":
            final_adapter_path = str(event.get("path", ""))
            break
    return {
        "dataset_path": str(dataset_path),
        "policy_model_path": policy_model_path,
        "proxy_model_path": str(resolved_model_path),
        "group_size": int(group_size),
        "requested_steps": int(steps),
        "completed_steps": len(step_summaries),
        "records_used": len(records),
        "scheduled_anchor_ratio": float(anchor_ratio),
        "scheduled_anchor_alpha": float(anchor_alpha),
        "signal_epsilon": float(signal_epsilon),
        "trainable_parameter_count": int(components["trainable_parameter_count"]),
        "lora_enabled": bool(components["lora_enabled"]),
        "load_in_4bit": bool(components["load_in_4bit"]),
        "device": resolved_device,
        "torch_dtype": str(components["torch_dtype"]),
        "step_summaries": step_summaries,
        "step_metrics": summarize_step_metrics(step_summaries),
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else None,
        "save_every_steps": normalized_save_every_steps,
        "save_final_adapter": bool(save_final_adapter),
        "checkpoint_events": checkpoint_events,
        "final_adapter_path": final_adapter_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run GRPO data preparation for SAVI alignment.")
    parser.add_argument("dataset_path", help="Path to HICRA-labeled JSONL dataset.")
    parser.add_argument("--model-path", default=str(DEFAULT_PROXY_MODEL_PATH))
    parser.add_argument("--policy-model-path", default=None)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--kl-beta", type=float, default=0.05)
    parser.add_argument("--hicra-gamma", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--disable-lora", action="store_true", default=False)
    parser.add_argument("--load-in-4bit", action="store_true", default=False)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--signal-epsilon", type=float, default=DEFAULT_SIGNAL_EPSILON)
    parser.add_argument("--anchor-ratio", type=float, default=0.7)
    parser.add_argument("--anchor-alpha", type=float, default=1.0)
    parser.add_argument("--ev-gap-threshold", type=float, default=0.15)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--save-every-steps", type=int, default=None)
    parser.add_argument("--save-final-adapter", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--output-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        summary = run_alignment_dry_run(
            dataset_path=args.dataset_path,
            model_path=args.model_path,
            group_size=args.group_size,
        )
    else:
        if not args.policy_model_path:
            raise ValueError("--policy-model-path is required for smoke training mode")
        summary = run_smoke_training(
            dataset_path=args.dataset_path,
            policy_model_path=args.policy_model_path,
            model_path=args.model_path,
            group_size=args.group_size,
            steps=args.steps,
            learning_rate=args.learning_rate,
            kl_beta=args.kl_beta,
            hicra_gamma=args.hicra_gamma,
            max_grad_norm=args.max_grad_norm,
            max_records=args.max_records,
            device=args.device,
            torch_dtype=args.torch_dtype,
            use_lora=not bool(args.disable_lora),
            load_in_4bit=bool(args.load_in_4bit),
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            max_seq_len=args.max_seq_len,
            signal_epsilon=args.signal_epsilon,
            anchor_ratio=args.anchor_ratio,
            anchor_alpha=args.anchor_alpha,
            ev_gap_threshold=args.ev_gap_threshold,
            checkpoint_dir=args.checkpoint_dir,
            save_every_steps=args.save_every_steps,
            save_final_adapter=bool(args.save_final_adapter),
        )
    if args.output_path:
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
