from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from liars_game_engine.analysis.hicra_preprocessor import (
    build_hicra_sample,
    resolve_alignment_token_penalty,
)
from liars_game_engine.analysis.candidate_expansion import build_expanded_scored_candidates
from liars_game_engine.analysis.shapley_analyzer import ProxyValuePredictor
from liars_game_engine.analysis.token_alignment import recalibrate_alignment_sample
from liars_game_engine.analysis.token_alignment import build_action_only_assistant_response_text
from liars_game_engine.analysis.train_value_proxy import (
    VALUE_PROXY_TARGET_PHI,
    build_value_proxy_feature_context,
)


DEFAULT_PROXY_MODEL_PATH = Path("models/proxy/value_proxy_mlp_distill.pt")
DEFAULT_SIGNAL_EPSILON = 1e-8
DEFAULT_ACTION_MATCH_REWARD_WEIGHT = 0.25
DEFAULT_ACTION_MATCH_EV_GAP_THRESHOLD = 0.15
DEFAULT_ACTION_MATCH_PHI_THRESHOLD = -0.1
DEFAULT_PARSE_ERROR_PENALTY = -0.5
DEFAULT_FALLBACK_USED_PENALTY = -0.35
DEFAULT_RESOLUTION_REPAIR_PENALTY = -0.2
DEFAULT_PROTOCOL_PENALTY_WARMUP_START = 0
DEFAULT_PROTOCOL_PENALTY_WARMUP_END = 0
ABLATION_VARIANTS = {
    "full",
    "action_only_proxy",
    "action_only_proxy_action_loss_only",
    "action_only_proxy_format_plus_action_loss",
    "action_only_proxy_no_match_bonus",
    "expanded_action_proxy",
    "conservative_expanded_action_proxy",
    "conservative_expanded_action_scope",
    "conservative_expanded_action_json",
    "no_token_localization",
    "proxy_target_only",
    "logged_only",
    "random_target",
    "heuristic_target",
    "scoped_hicra",
    "scoped_hicra_soft_reward",
    "scoped_hicra_token_only",
    "scoped_hicra_mismatch_only",
    "scoped_hicra_reward_only",
    "scoped_hicra_weighted_sum",
    "scoped_hicra_adv_clip",
    "scoped_hicra_high_kl",
    "scoped_hicra_adv_reshape",
    "scoped_hicra_adv_reshape_clean",
    "hicra_clean_filter",
    "hicra_sequence_dpo",
    "standard_dpo_baseline",
}
TARGET_ONLY_VARIANTS = {"proxy_target_only", "random_target", "heuristic_target"}
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


def summarize_dataset_signal_metrics(records: list[dict[str, object]]) -> dict[str, float | int]:
    total_record_count = len(records)
    nonempty_strategic_token_count = 0
    effective_nonzero_weight_count = 0
    zero_penalty_nonempty_count = 0
    reasoning_action_mismatch_count = 0
    ev_gap_positive_count = 0

    for record in records:
        if bool(record.get("reasoning_action_mismatch", False)):
            reasoning_action_mismatch_count += 1
        if float(record.get("ev_gap", 0.0) or 0.0) > 0.0:
            ev_gap_positive_count += 1

        strategic_tokens = record.get("strategic_tokens", [])
        if not isinstance(strategic_tokens, list) or not strategic_tokens:
            continue
        nonempty_strategic_token_count += 1

        total_abs_weight = 0.0
        for token in strategic_tokens:
            if not isinstance(token, dict):
                continue
            penalty_signal = float(token.get("penalty_signal", 0.0) or 0.0)
            label_weight = float(token.get("weight", 1.0) or 1.0)
            total_abs_weight += abs(penalty_signal * label_weight)

        if total_abs_weight > 0.0:
            effective_nonzero_weight_count += 1
        else:
            zero_penalty_nonempty_count += 1

    return {
        "total_record_count": total_record_count,
        "nonempty_strategic_token_count": nonempty_strategic_token_count,
        "nonempty_strategic_token_rate": (
            nonempty_strategic_token_count / total_record_count if total_record_count else 0.0
        ),
        "effective_nonzero_weight_count": effective_nonzero_weight_count,
        "effective_nonzero_weight_rate": (
            effective_nonzero_weight_count / total_record_count if total_record_count else 0.0
        ),
        "effective_nonzero_within_nonempty_rate": (
            effective_nonzero_weight_count / nonempty_strategic_token_count
            if nonempty_strategic_token_count
            else 0.0
        ),
        "zero_penalty_nonempty_count": zero_penalty_nonempty_count,
        "reasoning_action_mismatch_count": reasoning_action_mismatch_count,
        "ev_gap_positive_count": ev_gap_positive_count,
    }


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


def has_effective_training_signal(record: dict[str, object], *, epsilon: float = DEFAULT_SIGNAL_EPSILON) -> bool:
    if bool(record.get("reasoning_action_mismatch", False)):
        return True
    if abs(float(record.get("ev_gap", 0.0) or 0.0)) > float(epsilon):
        return True
    if abs(float(record.get("token_penalty", 0.0) or 0.0)) > float(epsilon):
        return True
    strategic_tokens = record.get("strategic_tokens", [])
    if isinstance(strategic_tokens, list):
        for token in strategic_tokens:
            if not isinstance(token, dict):
                continue
            penalty_signal = abs(float(token.get("penalty_signal", 0.0) or 0.0))
            label_weight = abs(float(token.get("weight", 1.0) or 1.0))
            if penalty_signal * label_weight > float(epsilon):
                return True
    return False


def supports_trainable_candidate_group(
    record: dict[str, object],
    *,
    min_candidate_count: int = 2,
    candidate_mode: str = "full",
) -> bool:
    target_count = max(1, int(min_candidate_count))
    return len(_build_group_candidates(record, group_size=target_count, candidate_mode=candidate_mode)) >= target_count


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


def _find_action_field_assistant_span(assistant_text: str) -> tuple[int, int] | None:
    marker = '"Action"'
    marker_start = assistant_text.find(marker)
    if marker_start < 0:
        return None
    return marker_start, len(assistant_text)


def _build_label_token_mask(
    *,
    alignment_metadata: dict[str, object],
    offset_mapping: list[tuple[int, int]],
    token_count: int,
    label_scope: str,
) -> list[float]:
    assistant_span = alignment_metadata.get("assistant_rendered_span", {})
    if not isinstance(assistant_span, dict):
        assistant_span = {}
    assistant_start = int(assistant_span.get("start", 0) or 0)
    assistant_end = int(assistant_span.get("end", 0) or 0)
    normalized_scope = str(label_scope or "assistant")
    span_start = assistant_start
    span_end = assistant_end
    if normalized_scope == "action":
        assistant_text = str(alignment_metadata.get("assistant_text", "") or "")
        action_span = _find_action_field_assistant_span(assistant_text)
        if action_span is not None:
            span_start = assistant_start + int(action_span[0])
            span_end = assistant_start + int(action_span[1])
    elif normalized_scope not in {"assistant", "format_action"}:
        raise ValueError(f"Unsupported label scope: {label_scope}")
    token_indices = set(_find_overlapping_token_indices(offset_mapping, span_start, span_end))
    if normalized_scope == "format_action":
        reasoning_span = alignment_metadata.get("reasoning_rendered_span", {})
        if isinstance(reasoning_span, dict):
            reasoning_start = int(reasoning_span.get("start", 0) or 0)
            reasoning_end = int(reasoning_span.get("end", 0) or 0)
            token_indices.difference_update(
                _find_overlapping_token_indices(offset_mapping, reasoning_start, reasoning_end)
            )
    return [1.0 if index in token_indices else 0.0 for index in range(token_count)]


def prepare_candidate_training_example(
    candidate: dict[str, object],
    tokenizer: object,
    max_seq_len: int | None = None,
    label_scope: str = "assistant",
) -> dict[str, object]:
    import torch

    recalibrated = recalibrate_alignment_sample(sample=candidate, tokenizer=tokenizer)
    alignment_metadata = recalibrated.get("alignment_metadata", {})
    if not isinstance(alignment_metadata, dict):
        raise ValueError("alignment_metadata missing after recalibration")
    recalibrated_mask_metrics = compute_mask_hit_metrics(recalibrated)

    rendered_text = str(alignment_metadata.get("rendered_text", "") or "")
    input_ids, attention_mask, offset_mapping = _tokenize_rendered_text(tokenizer, rendered_text)
    label_token_source_mask = _build_label_token_mask(
        alignment_metadata=alignment_metadata,
        offset_mapping=offset_mapping,
        token_count=len(input_ids),
        label_scope=label_scope,
    )
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
        label_token_source_mask = _truncate_from_left(label_token_source_mask, capped_len)
        token_weight_mask = _truncate_from_left(token_weight_mask, capped_len)

    label_token_mask = label_token_source_mask[1:]
    shifted_token_weights = token_weight_mask[1:]
    return {
        "rendered_text": rendered_text,
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
        "label_token_mask": torch.tensor(label_token_mask, dtype=torch.float32),
        "token_weight_mask": torch.tensor(shifted_token_weights, dtype=torch.float32),
        "assistant_token_count": int(sum(label_token_source_mask)),
        "active_label_count": int(sum(label_token_mask)),
        "hicra_non_zero_count": int(sum(1 for value in shifted_token_weights if float(value) != 0.0)),
        "label_scope": str(label_scope or "assistant"),
        "alignment_metadata": alignment_metadata,
        "recalibrated_mask_metrics": recalibrated_mask_metrics,
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
    token_advantage_alpha: float = 0.0,
    token_advantage_mode: str = "none",
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
    strategic_token_mask = active_mask * (torch.abs(token_weights) > 0.0).to(dtype=shift_logits.dtype)

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

    base_advantage = float(advantage)
    strategic_advantage = base_advantage
    token_advantages = torch.full_like(gathered_log_probs, fill_value=base_advantage)
    if str(token_advantage_mode) == "hicra" and float(token_advantage_alpha) != 0.0:
        strategic_advantage = base_advantage + (float(token_advantage_alpha) * abs(base_advantage))
        token_advantages = torch.where(
            strategic_token_mask > 0.0,
            torch.full_like(gathered_log_probs, fill_value=strategic_advantage),
            token_advantages,
        )
        policy_loss = -((gathered_log_probs * active_mask * token_advantages).sum() / active_denom)
    else:
        policy_loss = -(base_advantage * weighted_mean_log_prob)
    loss = policy_loss + (float(kl_beta) * kl_value)
    token_advantage_mean = (token_advantages * active_mask).sum() / active_denom
    strategic_token_count = strategic_token_mask.sum()
    strategic_token_advantage_mean = (
        (token_advantages * strategic_token_mask).sum() / torch.clamp(strategic_token_count, min=1.0)
    )
    return {
        "loss": loss,
        "advantage": float(advantage),
        "token_advantage_mode": str(token_advantage_mode),
        "token_advantage_alpha": float(token_advantage_alpha),
        "token_advantage_mean": float(token_advantage_mean.detach().cpu().item()),
        "strategic_token_advantage_mean": float(strategic_token_advantage_mean.detach().cpu().item()),
        "strategic_token_count": int(strategic_token_count.detach().cpu().item()),
        "mean_log_prob": float(mean_log_prob.detach().cpu().item()),
        "weighted_mean_log_prob": float(weighted_mean_log_prob.detach().cpu().item()),
        "nll": float(nll.detach().cpu().item()),
        "weighted_nll": float(weighted_nll.detach().cpu().item()),
        "kl_value": float(kl_value.detach().cpu().item()),
        "hicra_weight_mean": float((weighted_mask.sum() / active_denom).detach().cpu().item()),
        "active_label_count": int(active_mask.sum().detach().cpu().item()),
    }


def compute_sequence_log_prob(
    *,
    logits,
    input_ids,
    label_token_mask,
) -> dict[str, object]:
    import torch
    import torch.nn.functional as F

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    gathered_log_probs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    active_mask = label_token_mask.to(device=shift_logits.device, dtype=shift_logits.dtype)
    active_denom = torch.clamp(active_mask.sum(), min=1.0)
    sequence_logp = (gathered_log_probs * active_mask).sum()
    mean_logp = sequence_logp / active_denom
    return {
        "sequence_logp": sequence_logp,
        "mean_logp": float(mean_logp.detach().cpu().item()),
        "active_label_count": int(active_mask.sum().detach().cpu().item()),
    }


def compute_dpo_pair_loss_from_logps(
    *,
    chosen_policy_logp,
    rejected_policy_logp,
    chosen_ref_logp,
    rejected_ref_logp,
    beta: float = 0.1,
) -> dict[str, object]:
    import torch
    import torch.nn.functional as F

    device = chosen_policy_logp.device if hasattr(chosen_policy_logp, "device") else None
    chosen_ref = (
        chosen_ref_logp
        if chosen_ref_logp is not None
        else torch.zeros((), dtype=chosen_policy_logp.dtype, device=device)
    )
    rejected_ref = (
        rejected_ref_logp
        if rejected_ref_logp is not None
        else torch.zeros((), dtype=rejected_policy_logp.dtype, device=device)
    )
    policy_logratio = chosen_policy_logp - rejected_policy_logp
    reference_logratio = chosen_ref - rejected_ref
    preference_logit = policy_logratio - reference_logratio
    loss = -F.logsigmoid(float(beta) * preference_logit)
    return {
        "loss": loss,
        "beta": float(beta),
        "policy_logratio": float(policy_logratio.detach().cpu().item()),
        "reference_logratio": float(reference_logratio.detach().cpu().item()),
        "preference_logit": float(preference_logit.detach().cpu().item()),
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


def _iter_distinct_legal_actions(record: dict[str, object]) -> list[dict[str, object]]:
    observation = record.get("observation", {})
    if not isinstance(observation, dict):
        return []
    legal_actions = observation.get("legal_actions", [])
    if not isinstance(legal_actions, list):
        return []
    distinct: list[dict[str, object]] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    for legal_action in legal_actions:
        normalized = _normalize_action(legal_action if isinstance(legal_action, dict) else {})
        key = (
            normalized.get("type", ""),
            normalized.get("claim_rank", ""),
            tuple(normalized.get("cards", [])),
        )
        if not key[0] or key in seen:
            continue
        seen.add(key)
        distinct.append(normalized)
    return distinct


def _record_stable_index(record: dict[str, object], modulo: int) -> int:
    if int(modulo) <= 0:
        return 0
    key = f"{record.get('game_id', '')}:{record.get('turn', '')}:{record.get('player_id', '')}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % int(modulo)


def _select_random_legal_target(record: dict[str, object]) -> dict[str, object]:
    chosen_action = _normalize_action(record.get("action") if isinstance(record.get("action"), dict) else {})
    legal_actions = [
        action for action in _iter_distinct_legal_actions(record) if not _actions_equal(action, chosen_action)
    ]
    if not legal_actions:
        legal_actions = _iter_distinct_legal_actions(record)
    if not legal_actions:
        return chosen_action
    return legal_actions[_record_stable_index(record, len(legal_actions))]


def _select_heuristic_target(record: dict[str, object]) -> dict[str, object]:
    chosen_action = _normalize_action(record.get("action") if isinstance(record.get("action"), dict) else {})
    legal_actions = _iter_distinct_legal_actions(record)
    if not legal_actions:
        return chosen_action

    challenge_action = next((action for action in legal_actions if action.get("type") == "challenge"), None)
    play_action = next((action for action in legal_actions if action.get("type") == "play_claim"), None)
    observation = record.get("observation", {})
    pending_claim = observation.get("pending_claim", {}) if isinstance(observation, dict) else {}
    declared_count = (
        int(pending_claim.get("declared_count", 0) or 0)
        if isinstance(pending_claim, dict)
        else 0
    )
    death_probability = float(record.get("death_probability", 0.0) or 0.0)
    ev_gap = float(record.get("ev_gap", 0.0) or 0.0)
    mismatch = bool(record.get("reasoning_action_mismatch", False))

    if challenge_action and (mismatch or ev_gap >= 0.15 or declared_count >= 2 or death_probability >= 0.25):
        return challenge_action
    return play_action or challenge_action or legal_actions[0]


def _clone_record(record: dict[str, object]) -> dict[str, object]:
    return json.loads(json.dumps(record, ensure_ascii=True))


def apply_ablation_variant(record: dict[str, object], variant: str) -> dict[str, object]:
    normalized_variant = str(variant or "full")
    if normalized_variant not in ABLATION_VARIANTS:
        raise ValueError(f"Unsupported ablation variant: {variant}")

    mutated = _clone_record(record)
    if normalized_variant in {
        "action_only_proxy",
        "logged_only",
        "random_target",
        "heuristic_target",
        "proxy_target_only",
        "standard_dpo_baseline",
        "expanded_action_proxy",
        "conservative_expanded_action_proxy",
        "conservative_expanded_action_scope",
        "action_only_proxy_no_match_bonus",
        "action_only_proxy_action_loss_only",
        "action_only_proxy_format_plus_action_loss",
        "conservative_expanded_action_json",
    }:
        mutated["strategic_tokens"] = []
        mutated["token_weight_mask"] = []
        mutated["token_penalty"] = 0.0
        mutated["strategic_token_weight"] = 0.0
        mutated["reasoning_action_mismatch"] = False

    if normalized_variant == "random_target":
        mutated["proxy_target_action"] = _select_random_legal_target(mutated)
    elif normalized_variant == "heuristic_target":
        mutated["proxy_target_action"] = _select_heuristic_target(mutated)

    return mutated


def _coerce_alignment_record(record: dict[str, object]) -> dict[str, object]:
    base = build_hicra_sample(record)
    merged = {**base, **dict(record)}
    merged["action"] = _normalize_action(merged.get("action") if isinstance(merged.get("action"), dict) else {})
    proxy_target_action = merged.get("proxy_target_action", {})
    if not isinstance(proxy_target_action, dict):
        proxy_target_action = {}
    merged["proxy_target_action"] = _normalize_action(proxy_target_action)
    strategic_tokens = list(merged.get("strategic_tokens", [])) if isinstance(merged.get("strategic_tokens", []), list) else []
    best_action_delta = float(merged.get("best_action_delta", 0.0) or 0.0)
    effective_token_penalty = resolve_alignment_token_penalty(
        ev_gap=float(merged.get("ev_gap", 0.0) or 0.0),
        best_action_delta=best_action_delta,
        reasoning_action_mismatch=bool(merged.get("reasoning_action_mismatch", False)),
        strategic_tokens=strategic_tokens,
    )
    normalized_tokens: list[dict[str, object]] = []
    for token in strategic_tokens:
        if not isinstance(token, dict):
            continue
        normalized = dict(token)
        existing_penalty_signal = float(normalized.get("penalty_signal", 0.0) or 0.0)
        if existing_penalty_signal == 0.0 and effective_token_penalty != 0.0:
            normalized["penalty_signal"] = effective_token_penalty
        elif "penalty_signal" not in normalized:
            normalized["penalty_signal"] = effective_token_penalty
        normalized_tokens.append(normalized)
    merged["strategic_tokens"] = normalized_tokens
    merged["strategic_token_weight"] = float(merged.get("strategic_token_weight", 1.0) or 1.0)
    merged["token_penalty"] = (
        effective_token_penalty
        if float(merged.get("token_penalty", 0.0) or 0.0) == 0.0 and effective_token_penalty != 0.0
        else float(merged.get("token_penalty", 0.0) or 0.0)
    )
    merged["ev_gap"] = float(merged.get("ev_gap", 0.0) or 0.0)
    merged["phi_chosen"] = float(merged.get("phi_chosen", 0.0) or 0.0)
    merged["fallback_used"] = bool(merged.get("fallback_used", False))
    resolution_reason = merged.get("resolution_reason")
    merged["resolution_reason"] = str(resolution_reason) if resolution_reason is not None else ""
    parse_error = merged.get("parse_error")
    merged["parse_error"] = parse_error if isinstance(parse_error, dict) else None
    return merged


def _without_hicra_token_signal(candidate: dict[str, object]) -> dict[str, object]:
    stripped = dict(candidate)
    stripped["strategic_tokens"] = []
    stripped["token_weight_mask"] = []
    stripped["token_penalty"] = 0.0
    stripped["strategic_token_weight"] = 0.0
    stripped["reasoning_action_mismatch"] = False
    return stripped


def _compute_protocol_penalty(candidate: dict[str, object]) -> tuple[float, list[str]]:
    return _compute_protocol_penalty_with_weights(
        candidate,
        parse_error_penalty=DEFAULT_PARSE_ERROR_PENALTY,
        fallback_used_penalty=DEFAULT_FALLBACK_USED_PENALTY,
        resolution_repair_penalty=DEFAULT_RESOLUTION_REPAIR_PENALTY,
    )


def _compute_protocol_penalty_with_weights(
    candidate: dict[str, object],
    *,
    parse_error_penalty: float,
    fallback_used_penalty: float,
    resolution_repair_penalty: float,
) -> tuple[float, list[str]]:
    penalty = 0.0
    reasons: list[str] = []

    parse_error = candidate.get("parse_error")
    if isinstance(parse_error, dict) and str(parse_error.get("code", "")).strip():
        penalty += float(parse_error_penalty)
        reasons.append("parse_error")

    if bool(candidate.get("fallback_used", False)):
        penalty += float(fallback_used_penalty)
        reasons.append("fallback_used")

    resolution_reason = str(candidate.get("resolution_reason", "") or "")
    if resolution_reason:
        resolution_penalty_tokens = (
            "illegal_pass_redirection",
            "claim_rank_forced_to_table_type",
            "redirected_to=",
            "repair",
        )
        for token in resolution_penalty_tokens:
            if token in resolution_reason:
                penalty += float(resolution_repair_penalty)
                reasons.append(token)
                break

    return float(penalty), reasons


def _compute_protocol_penalty_scale(
    *,
    step_index: int,
    warmup_start_step: int = DEFAULT_PROTOCOL_PENALTY_WARMUP_START,
    warmup_end_step: int = DEFAULT_PROTOCOL_PENALTY_WARMUP_END,
) -> float:
    normalized_start = max(0, int(warmup_start_step))
    normalized_end = max(0, int(warmup_end_step))
    if normalized_end <= normalized_start:
        return 1.0

    current_step = max(1, int(step_index))
    if current_step <= normalized_start:
        return 0.0
    if current_step >= normalized_end:
        return 1.0
    return float((current_step - normalized_start) / float(normalized_end - normalized_start))


def _resolve_action_match_trigger(
    candidate: dict[str, object],
    *,
    ev_gap_threshold: float = DEFAULT_ACTION_MATCH_EV_GAP_THRESHOLD,
    phi_threshold: float = DEFAULT_ACTION_MATCH_PHI_THRESHOLD,
) -> str:
    if not _actions_equal(candidate.get("action", {}), candidate.get("proxy_target_action", {})):
        return "not_matched"
    if bool(candidate.get("reasoning_action_mismatch", False)):
        return "reasoning_action_mismatch"
    if float(candidate.get("ev_gap", 0.0) or 0.0) > float(ev_gap_threshold):
        return "high_ev_gap"
    if float(candidate.get("phi_chosen", 0.0) or 0.0) < float(phi_threshold):
        return "negative_phi"
    return "none"


def _build_group_candidates(
    record: dict[str, object],
    group_size: int,
    *,
    candidate_mode: str = "full",
    scope_hicra_to_logged_action: bool = False,
) -> list[dict[str, object]]:
    base_candidate = _coerce_alignment_record(record)
    chosen_action = _normalize_action(base_candidate.get("action", {}))
    proxy_target_action = _normalize_action(base_candidate.get("proxy_target_action", {}))
    explicit_challenge_action = _extract_explicit_challenge_action(base_candidate)

    candidate_specs: list[tuple[str, dict[str, object]]] = [("logged_action", chosen_action)]
    if str(candidate_mode) == "logged_only":
        pass
    elif proxy_target_action and not _actions_equal(proxy_target_action, chosen_action):
        candidate_specs.append(("proxy_target", proxy_target_action))
    if str(candidate_mode) not in {"logged_only", "chosen_proxy"} and explicit_challenge_action and not any(
        _actions_equal(explicit_challenge_action, candidate_action) for _, candidate_action in candidate_specs
    ):
        candidate_specs.append(("legal_challenge", explicit_challenge_action))
    if str(candidate_mode) not in {"logged_only", "chosen_proxy"} and len(candidate_specs) < min(2, max(1, int(group_size))):
        for legal_action in _iter_distinct_legal_actions(base_candidate):
            if any(_actions_equal(legal_action, candidate_action) for _, candidate_action in candidate_specs):
                continue
            candidate_specs.append(("legal_alternative", legal_action))
            if len(candidate_specs) >= min(2, max(1, int(group_size))):
                break

    candidates: list[dict[str, object]] = []
    for index, (candidate_role, candidate_action) in enumerate(candidate_specs[: max(1, int(group_size))]):
        candidate = {
            **base_candidate,
            "action": dict(candidate_action),
            "candidate_index": index,
            "candidate_role": candidate_role,
        }
        if bool(scope_hicra_to_logged_action) and candidate_role != "logged_action":
            candidate = _without_hicra_token_signal(candidate)
        candidates.append(candidate)
    return candidates


def _compute_reward_breakdown(
    predictor: ProxyValuePredictor | object,
    candidate: dict[str, object],
    action_match_reward_weight: float = DEFAULT_ACTION_MATCH_REWARD_WEIGHT,
    parse_error_penalty: float = DEFAULT_PARSE_ERROR_PENALTY,
    fallback_used_penalty: float = DEFAULT_FALLBACK_USED_PENALTY,
    resolution_repair_penalty: float = DEFAULT_RESOLUTION_REPAIR_PENALTY,
    protocol_penalty_scale: float = 1.0,
    use_phi_dense_reward: bool = True,
    use_hicra_reward: bool = True,
    scope_hicra_to_logged_action: bool = False,
    hicra_reward_scale: float = 1.0,
    hicra_mismatch_only: bool = False,
    hicra_ev_gap_threshold: float = DEFAULT_ACTION_MATCH_EV_GAP_THRESHOLD,
    reward_component_weights: dict[str, float] | None = None,
) -> dict[str, float | bool | str | list[str]]:
    proxy_target_action = candidate.get("proxy_target_action", {})
    if not isinstance(proxy_target_action, dict):
        proxy_target_action = {}
    action_match_trigger = _resolve_action_match_trigger(candidate)
    action_match_reward = float(action_match_reward_weight) if action_match_trigger not in {"none", "not_matched"} else 0.0
    raw_phi_dense_reward = _score_candidate(predictor, candidate)
    phi_dense_reward = raw_phi_dense_reward if bool(use_phi_dense_reward) else 0.0
    strategic_tokens = candidate.get("strategic_tokens", [])
    has_strategic_tokens = isinstance(strategic_tokens, list) and len(strategic_tokens) > 0
    strategic_token_weight = float(candidate.get("strategic_token_weight", 1.0) or 1.0)
    token_penalty = float(candidate.get("token_penalty", 0.0) or 0.0)
    hicra_candidate_scope_allowed = (
        not bool(scope_hicra_to_logged_action)
        or str(candidate.get("candidate_role", "")) == "logged_action"
    )
    hicra_sample_allowed = (
        (
            bool(candidate.get("reasoning_action_mismatch", False))
            and float(candidate.get("ev_gap", 0.0) or 0.0) > float(hicra_ev_gap_threshold)
        )
        if bool(hicra_mismatch_only)
        else True
    )
    hicra_penalty = (
        token_penalty * strategic_token_weight * max(0.0, float(hicra_reward_scale))
        if (
            bool(use_hicra_reward)
            and hicra_candidate_scope_allowed
            and hicra_sample_allowed
            and has_strategic_tokens
            and action_match_reward < 1.0
        )
        else 0.0
    )
    protocol_penalty, protocol_penalty_reasons = _compute_protocol_penalty_with_weights(
        candidate,
        parse_error_penalty=parse_error_penalty,
        fallback_used_penalty=fallback_used_penalty,
        resolution_repair_penalty=resolution_repair_penalty,
    )
    protocol_penalty = float(protocol_penalty) * max(0.0, float(protocol_penalty_scale))
    component_weights = {
        "action_match_reward": 1.0,
        "phi_dense_reward": 1.0,
        "hicra_penalty": 1.0,
        "protocol_penalty": 1.0,
    }
    if isinstance(reward_component_weights, dict):
        for key in component_weights:
            if key in reward_component_weights:
                component_weights[key] = float(reward_component_weights[key])
    weighted_action_match_reward = action_match_reward * component_weights["action_match_reward"]
    weighted_phi_dense_reward = phi_dense_reward * component_weights["phi_dense_reward"]
    weighted_hicra_penalty = hicra_penalty * component_weights["hicra_penalty"]
    weighted_protocol_penalty = protocol_penalty * component_weights["protocol_penalty"]
    total_reward = (
        weighted_action_match_reward
        + weighted_phi_dense_reward
        + weighted_hicra_penalty
        + weighted_protocol_penalty
    )
    return {
        "action_match_reward": float(action_match_reward),
        "weighted_action_match_reward": float(weighted_action_match_reward),
        "action_match_triggered": bool(action_match_trigger not in {"none", "not_matched"}),
        "action_match_trigger": str(action_match_trigger),
        "phi_dense_reward": float(phi_dense_reward),
        "weighted_phi_dense_reward": float(weighted_phi_dense_reward),
        "raw_phi_dense_reward": float(raw_phi_dense_reward),
        "hicra_penalty": float(hicra_penalty),
        "weighted_hicra_penalty": float(weighted_hicra_penalty),
        "hicra_reward_scale": max(0.0, float(hicra_reward_scale)),
        "hicra_mismatch_only": bool(hicra_mismatch_only),
        "hicra_sample_allowed": bool(hicra_sample_allowed),
        "protocol_penalty": float(protocol_penalty),
        "weighted_protocol_penalty": float(weighted_protocol_penalty),
        "reward_component_weights": dict(component_weights),
        "protocol_penalty_scale": max(0.0, float(protocol_penalty_scale)),
        "protocol_violation": bool(protocol_penalty_reasons),
        "protocol_penalty_reasons": list(protocol_penalty_reasons),
        "total_reward": float(total_reward),
    }


def _build_scored_group(
    predictor: ProxyValuePredictor | object,
    record: dict[str, object],
    group_size: int,
    action_match_reward_weight: float = DEFAULT_ACTION_MATCH_REWARD_WEIGHT,
    parse_error_penalty: float = DEFAULT_PARSE_ERROR_PENALTY,
    fallback_used_penalty: float = DEFAULT_FALLBACK_USED_PENALTY,
    resolution_repair_penalty: float = DEFAULT_RESOLUTION_REPAIR_PENALTY,
    protocol_penalty_scale: float = 1.0,
    candidate_mode: str = "full",
    use_phi_dense_reward: bool = True,
    use_hicra_reward: bool = True,
    single_candidate_advantage: str = "centered",
    scope_hicra_to_logged_action: bool = False,
    hicra_reward_scale: float = 1.0,
    hicra_mismatch_only: bool = False,
    hicra_ev_gap_threshold: float = DEFAULT_ACTION_MATCH_EV_GAP_THRESHOLD,
    reward_component_weights: dict[str, float] | None = None,
    advantage_clip: float | None = None,
) -> dict[str, object]:
    expanded_candidate_pool: list[dict[str, object]] | None = None
    if str(candidate_mode) in {"expanded_proxy", "conservative_expanded_proxy"}:
        expanded = build_expanded_scored_candidates(
            score_candidate=lambda candidate: _score_candidate(predictor, candidate),
            record=record,
            group_size=group_size,
            selection_mode=(
                "conservative" if str(candidate_mode) == "conservative_expanded_proxy" else "legacy"
            ),
        )
        candidates = list(expanded["selected_candidates"])
        expanded_candidate_pool = list(expanded["expanded_candidate_pool"])
        if bool(scope_hicra_to_logged_action):
            candidates = [
                candidate
                if str(candidate.get("candidate_role", "")) == "logged_action"
                else _without_hicra_token_signal(candidate)
                for candidate in candidates
            ]
    else:
        candidates = _build_group_candidates(
            record,
            group_size=group_size,
            candidate_mode=candidate_mode,
            scope_hicra_to_logged_action=scope_hicra_to_logged_action,
        )
    rewards: list[float] = []
    for candidate in candidates:
        reward_breakdown = _compute_reward_breakdown(
            predictor,
            candidate,
            action_match_reward_weight=action_match_reward_weight,
            parse_error_penalty=parse_error_penalty,
            fallback_used_penalty=fallback_used_penalty,
            resolution_repair_penalty=resolution_repair_penalty,
            protocol_penalty_scale=protocol_penalty_scale,
            use_phi_dense_reward=use_phi_dense_reward,
            use_hicra_reward=use_hicra_reward,
            scope_hicra_to_logged_action=scope_hicra_to_logged_action,
            hicra_reward_scale=hicra_reward_scale,
            hicra_mismatch_only=hicra_mismatch_only,
            hicra_ev_gap_threshold=hicra_ev_gap_threshold,
            reward_component_weights=reward_component_weights,
        )
        candidate["reward_breakdown"] = reward_breakdown
        rewards.append(float(reward_breakdown["total_reward"]))
    if len(rewards) == 1 and str(single_candidate_advantage) == "raw":
        raw_advantages = [float(rewards[0])]
    else:
        raw_advantages = compute_group_relative_advantages(rewards)
    advantages = list(raw_advantages)
    if advantage_clip is not None and float(advantage_clip) > 0.0:
        clip_value = float(advantage_clip)
        advantages = [max(-clip_value, min(clip_value, float(advantage))) for advantage in raw_advantages]
    for candidate, advantage in zip(candidates, advantages, strict=False):
        candidate["advantage"] = float(advantage)
    raw_advantage_span = (max(raw_advantages) - min(raw_advantages)) if raw_advantages else 0.0
    advantage_span = (max(advantages) - min(advantages)) if advantages else 0.0
    return {
        "game_id": str(record.get("game_id", "")),
        "turn": int(record.get("turn", 0) or 0),
        "ev_gap": float(record.get("ev_gap", 0.0) or 0.0),
        "reasoning_action_mismatch": bool(record.get("reasoning_action_mismatch", False)),
        "candidate_mode": str(candidate_mode),
        "use_phi_dense_reward": bool(use_phi_dense_reward),
        "use_hicra_reward": bool(use_hicra_reward),
        "scope_hicra_to_logged_action": bool(scope_hicra_to_logged_action),
        "hicra_reward_scale": max(0.0, float(hicra_reward_scale)),
        "hicra_mismatch_only": bool(hicra_mismatch_only),
        "hicra_ev_gap_threshold": float(hicra_ev_gap_threshold),
        "reward_component_weights": (
            dict(reward_component_weights)
            if isinstance(reward_component_weights, dict)
            else {
                "action_match_reward": 1.0,
                "phi_dense_reward": 1.0,
                "hicra_penalty": 1.0,
                "protocol_penalty": 1.0,
            }
        ),
        "advantage_clip": float(advantage_clip) if advantage_clip is not None else None,
        "candidate_count": len(candidates),
        "rewards": rewards,
        "raw_advantages": raw_advantages,
        "advantages": advantages,
        "raw_advantage_span": float(raw_advantage_span),
        "advantage_span": float(advantage_span),
        "mask_metrics": compute_mask_hit_metrics(record),
        "candidates": candidates,
        "expanded_candidate_pool": expanded_candidate_pool or [],
    }


def _candidate_total_reward(candidate: dict[str, object]) -> float:
    reward_breakdown = candidate.get("reward_breakdown", {})
    if not isinstance(reward_breakdown, dict):
        return 0.0
    return float(reward_breakdown.get("total_reward", 0.0) or 0.0)


def build_dpo_preference_pair(
    group: dict[str, object],
    *,
    preference_pair_mode: str = "best_vs_logged",
) -> dict[str, object]:
    candidates = [
        candidate
        for candidate in group.get("candidates", [])
        if isinstance(candidate, dict)
    ]
    if len(candidates) < 2:
        raise ValueError("DPO preference pairs require at least two candidates")

    mode = str(preference_pair_mode or "best_vs_logged")
    ranked_candidates = sorted(candidates, key=_candidate_total_reward, reverse=True)
    chosen = ranked_candidates[0]
    if mode == "best_vs_logged":
        rejected = next(
            (
                candidate
                for candidate in candidates
                if str(candidate.get("candidate_role", "")) == "logged_action"
                and candidate is not chosen
            ),
            None,
        )
        if rejected is None:
            rejected = ranked_candidates[-1]
    elif mode == "best_vs_worst":
        rejected = ranked_candidates[-1]
    else:
        raise ValueError(f"Unsupported DPO preference pair mode: {preference_pair_mode}")

    if rejected is chosen:
        raise ValueError("DPO chosen and rejected candidates must be distinct")
    chosen_reward = _candidate_total_reward(chosen)
    rejected_reward = _candidate_total_reward(rejected)
    return {
        "chosen": chosen,
        "rejected": rejected,
        "chosen_reward": float(chosen_reward),
        "rejected_reward": float(rejected_reward),
        "preference_margin": float(chosen_reward - rejected_reward),
        "preference_pair_mode": mode,
    }


def _aggregate_recalibrated_mask_metrics(examples: list[dict[str, object]]) -> dict[str, float | int]:
    if not examples:
        return {
            "strategic_alignment_count": 0,
            "strategic_token_index_count": 0,
            "non_zero_mask_count": 0,
            "mask_hit_count": 0,
            "mask_hit_rate": 0.0,
            "average_hicra_mask_intensity": 0.0,
        }

    metrics_list = [
        example.get("recalibrated_mask_metrics", {})
        for example in examples
        if isinstance(example.get("recalibrated_mask_metrics", {}), dict)
    ]
    if not metrics_list:
        return {
            "strategic_alignment_count": 0,
            "strategic_token_index_count": 0,
            "non_zero_mask_count": 0,
            "mask_hit_count": 0,
            "mask_hit_rate": 0.0,
            "average_hicra_mask_intensity": 0.0,
        }

    positive_intensities = [
        float(item.get("average_hicra_mask_intensity", 0.0) or 0.0)
        for item in metrics_list
        if float(item.get("average_hicra_mask_intensity", 0.0) or 0.0) > 0.0
    ]
    return {
        "strategic_alignment_count": max(int(item.get("strategic_alignment_count", 0) or 0) for item in metrics_list),
        "strategic_token_index_count": max(
            int(item.get("strategic_token_index_count", 0) or 0) for item in metrics_list
        ),
        "non_zero_mask_count": max(int(item.get("non_zero_mask_count", 0) or 0) for item in metrics_list),
        "mask_hit_count": max(int(item.get("mask_hit_count", 0) or 0) for item in metrics_list),
        "mask_hit_rate": max(float(item.get("mask_hit_rate", 0.0) or 0.0) for item in metrics_list),
        "average_hicra_mask_intensity": (
            sum(positive_intensities) / len(positive_intensities) if positive_intensities else 0.0
        ),
    }


def resolve_ablation_settings(variant: str, *, hicra_gamma: float) -> dict[str, object]:
    normalized_variant = str(variant or "full")
    if normalized_variant not in ABLATION_VARIANTS:
        raise ValueError(f"Unsupported ablation variant: {variant}")

    settings: dict[str, object] = {
        "ablation_variant": normalized_variant,
        "candidate_mode": "full",
        "use_phi_dense_reward": True,
        "use_hicra_reward": True,
        "hicra_gamma": float(hicra_gamma),
        "single_candidate_advantage": "centered",
        "target_source": "proxy_target_action",
        "scope_hicra_to_logged_action": False,
        "hicra_reward_scale": 1.0,
        "hicra_mismatch_only": False,
        "hicra_ev_gap_threshold": DEFAULT_ACTION_MATCH_EV_GAP_THRESHOLD,
        "reward_component_weights": {
            "action_match_reward": 1.0,
            "phi_dense_reward": 1.0,
            "hicra_penalty": 1.0,
            "protocol_penalty": 1.0,
        },
        "advantage_clip": None,
        "kl_beta_multiplier": 1.0,
        "token_advantage_mode": "none",
        "token_advantage_alpha": 0.0,
        "label_scope": "assistant",
        "synthetic_candidate_label_scope": "assistant",
        "assistant_target_mode": "legacy",
        "record_filter_mode": "none",
        "training_objective": "group_relative",
        "dpo_beta": 0.1,
        "preference_pair_mode": "best_vs_logged",
    }
    if normalized_variant == "action_only_proxy":
        settings.update({"use_hicra_reward": False, "hicra_gamma": 0.0})
    elif normalized_variant == "action_only_proxy_action_loss_only":
        settings.update(
            {
                "use_hicra_reward": False,
                "hicra_gamma": 0.0,
                "target_source": "proxy_target_action_action_loss_only",
                "label_scope": "action",
            }
        )
    elif normalized_variant == "action_only_proxy_format_plus_action_loss":
        settings.update(
            {
                "use_hicra_reward": False,
                "hicra_gamma": 0.0,
                "target_source": "proxy_target_action_format_plus_action_loss",
                "label_scope": "format_action",
            }
        )
    elif normalized_variant == "action_only_proxy_no_match_bonus":
        settings.update(
            {
                "use_hicra_reward": False,
                "hicra_gamma": 0.0,
                "target_source": "proxy_target_action_without_action_match_bonus",
                "reward_component_weights": {
                    "action_match_reward": 0.0,
                    "phi_dense_reward": 1.0,
                    "hicra_penalty": 0.0,
                    "protocol_penalty": 1.0,
                },
            }
        )
    elif normalized_variant == "expanded_action_proxy":
        settings.update(
            {
                "candidate_mode": "expanded_proxy",
                "use_hicra_reward": False,
                "hicra_gamma": 0.0,
                "target_source": "expanded_proxy_ranked_legal_candidates",
            }
        )
    elif normalized_variant == "conservative_expanded_action_proxy":
        settings.update(
            {
                "candidate_mode": "conservative_expanded_proxy",
                "use_hicra_reward": False,
                "hicra_gamma": 0.0,
                "target_source": "conservative_expanded_proxy_ranked_legal_candidates",
            }
        )
    elif normalized_variant == "conservative_expanded_action_scope":
        settings.update(
            {
                "candidate_mode": "conservative_expanded_proxy",
                "use_hicra_reward": False,
                "hicra_gamma": 0.0,
                "target_source": "conservative_expanded_proxy_action_scope_synthetic_candidates",
                "label_scope": "assistant",
                "synthetic_candidate_label_scope": "action",
            }
        )
    elif normalized_variant == "conservative_expanded_action_json":
        settings.update(
            {
                "candidate_mode": "conservative_expanded_proxy",
                "use_hicra_reward": False,
                "hicra_gamma": 0.0,
                "target_source": "conservative_expanded_proxy_action_json_candidates",
                "assistant_target_mode": "action_json",
            }
        )
    elif normalized_variant == "scoped_hicra":
        settings.update({"scope_hicra_to_logged_action": True})
    elif normalized_variant == "scoped_hicra_soft_reward":
        settings.update(
            {
                "scope_hicra_to_logged_action": True,
                "hicra_reward_scale": 0.25,
                "hicra_gamma": min(float(hicra_gamma), 0.5),
            }
        )
    elif normalized_variant == "scoped_hicra_token_only":
        settings.update(
            {
                "scope_hicra_to_logged_action": True,
                "use_hicra_reward": False,
                "hicra_reward_scale": 0.0,
            }
        )
    elif normalized_variant == "scoped_hicra_mismatch_only":
        settings.update(
            {
                "scope_hicra_to_logged_action": True,
                "hicra_mismatch_only": True,
                "hicra_ev_gap_threshold": DEFAULT_ACTION_MATCH_EV_GAP_THRESHOLD,
            }
        )
    elif normalized_variant == "scoped_hicra_reward_only":
        settings.update(
            {
                "scope_hicra_to_logged_action": True,
                "hicra_gamma": 0.0,
            }
        )
    elif normalized_variant == "scoped_hicra_weighted_sum":
        settings.update(
            {
                "scope_hicra_to_logged_action": True,
                "reward_component_weights": {
                    "action_match_reward": 0.75,
                    "phi_dense_reward": 1.0,
                    "hicra_penalty": 0.5,
                    "protocol_penalty": 2.0,
                },
            }
        )
    elif normalized_variant == "scoped_hicra_adv_clip":
        settings.update(
            {
                "scope_hicra_to_logged_action": True,
                "advantage_clip": 0.3,
            }
        )
    elif normalized_variant == "scoped_hicra_high_kl":
        settings.update(
            {
                "scope_hicra_to_logged_action": True,
                "kl_beta_multiplier": 4.0,
            }
        )
    elif normalized_variant == "scoped_hicra_adv_reshape":
        settings.update(
            {
                "scope_hicra_to_logged_action": True,
                "use_hicra_reward": False,
                "hicra_reward_scale": 0.0,
                "hicra_gamma": 0.0,
                "token_advantage_mode": "hicra",
                "token_advantage_alpha": 0.2,
            }
        )
    elif normalized_variant == "scoped_hicra_adv_reshape_clean":
        settings.update(
            {
                "scope_hicra_to_logged_action": True,
                "use_hicra_reward": False,
                "hicra_reward_scale": 0.0,
                "hicra_gamma": 0.0,
                "token_advantage_mode": "hicra",
                "token_advantage_alpha": 0.2,
                "record_filter_mode": "hicra_clean",
            }
        )
    elif normalized_variant == "hicra_clean_filter":
        settings.update(
            {
                "use_hicra_reward": False,
                "hicra_reward_scale": 0.0,
                "hicra_gamma": 0.0,
                "record_filter_mode": "hicra_clean",
            }
        )
    elif normalized_variant == "hicra_sequence_dpo":
        settings.update(
            {
                "use_hicra_reward": False,
                "hicra_reward_scale": 0.0,
                "hicra_gamma": 0.0,
                "token_advantage_mode": "none",
                "token_advantage_alpha": 0.0,
                "training_objective": "dpo",
                "dpo_beta": 0.1,
                "preference_pair_mode": "best_vs_logged",
            }
        )
    elif normalized_variant == "standard_dpo_baseline":
        settings.update(
            {
                "use_hicra_reward": False,
                "hicra_reward_scale": 0.0,
                "hicra_gamma": 0.0,
                "token_advantage_mode": "none",
                "token_advantage_alpha": 0.0,
                "training_objective": "dpo",
                "dpo_beta": 0.02,
                "preference_pair_mode": "best_vs_worst",
                "target_source": "proxy_credit_preference_pair",
                "reward_component_weights": {
                    "action_match_reward": 0.0,
                    "phi_dense_reward": 1.0,
                    "hicra_penalty": 0.0,
                    "protocol_penalty": 0.0,
                },
            }
        )
    elif normalized_variant == "no_token_localization":
        settings.update({"hicra_gamma": 0.0})
    elif normalized_variant == "proxy_target_only":
        settings.update(
            {
                "candidate_mode": "chosen_proxy",
                "use_phi_dense_reward": False,
                "use_hicra_reward": False,
                "hicra_gamma": 0.0,
            }
        )
    elif normalized_variant == "logged_only":
        settings.update(
            {
                "candidate_mode": "logged_only",
                "use_hicra_reward": False,
                "hicra_gamma": 0.0,
                "single_candidate_advantage": "raw",
                "target_source": "logged_action",
            }
        )
    elif normalized_variant == "random_target":
        settings.update(
            {
                "candidate_mode": "chosen_proxy",
                "use_phi_dense_reward": False,
                "use_hicra_reward": False,
                "hicra_gamma": 0.0,
                "target_source": "deterministic_random_legal_action",
            }
        )
    elif normalized_variant == "heuristic_target":
        settings.update(
            {
                "candidate_mode": "chosen_proxy",
                "use_phi_dense_reward": False,
                "use_hicra_reward": False,
                "hicra_gamma": 0.0,
                "target_source": "handcrafted_heuristic_legal_action",
            }
        )
    return settings


SYNTHETIC_CANDIDATE_ROLES = {
    "truthful_play",
    "bluff_play",
    "legal_challenge",
    "legal_alternative",
}


def resolve_candidate_label_scope(settings: dict[str, object], candidate: dict[str, object]) -> str:
    role = str(candidate.get("candidate_role", "") or "")
    if role in SYNTHETIC_CANDIDATE_ROLES:
        return str(settings.get("synthetic_candidate_label_scope", settings.get("label_scope", "assistant")) or "assistant")
    return str(settings.get("label_scope", "assistant") or "assistant")


def apply_candidate_assistant_target_mode(
    candidate: dict[str, object],
    *,
    assistant_target_mode: str = "legacy",
) -> dict[str, object]:
    normalized_mode = str(assistant_target_mode or "legacy")
    if normalized_mode == "legacy":
        return candidate
    if normalized_mode != "action_json":
        raise ValueError(f"Unsupported assistant target mode: {assistant_target_mode}")
    mutated = dict(candidate)
    mutated["assistant_target"] = build_action_only_assistant_response_text(mutated)
    return mutated


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
    candidate_mode: str = "full",
) -> list[dict[str, object]]:
    if not records:
        return []
    min_candidate_count = 1 if str(candidate_mode) == "logged_only" else 2
    trainable_records = [
        record
        for record in records
        if supports_trainable_candidate_group(
            record,
            min_candidate_count=min_candidate_count,
            candidate_mode=candidate_mode,
        )
    ]
    fallback_records = [
        record
        for record in records
        if not supports_trainable_candidate_group(
            record,
            min_candidate_count=min_candidate_count,
            candidate_mode=candidate_mode,
        )
    ]
    anchor_records = [
        record for record in trainable_records if is_class_a_record(record, ev_gap_threshold=ev_gap_threshold)
    ]
    standard_records = [
        record for record in trainable_records if not is_class_a_record(record, ev_gap_threshold=ev_gap_threshold)
    ]
    signal_standard_records = [record for record in standard_records if has_effective_training_signal(record)]
    background_standard_records = [
        record for record in standard_records if not has_effective_training_signal(record)
    ]
    prioritized_anchors = prioritize_smoke_records(anchor_records)
    prioritized_signal_standard = prioritize_smoke_records(signal_standard_records)
    prioritized_background_standard = prioritize_smoke_records(background_standard_records)
    anchor_index = 0
    signal_standard_index = 0
    background_standard_index = 0
    schedule: list[dict[str, object]] = []
    normalized_anchor_ratio = max(0.0, min(1.0, float(anchor_ratio)))
    for step_index in range(max(1, int(steps))):
        choose_anchor = (
            ((step_index + 1) / max(1, int(steps))) <= normalized_anchor_ratio
            if step_index < int(steps * normalized_anchor_ratio)
            else False
        )
        if prioritized_anchors and (
            choose_anchor
            or (not prioritized_signal_standard and not prioritized_background_standard)
        ):
            weighted_anchors = sorted(
                prioritized_anchors,
                key=lambda record: (-compute_sampling_weight(record, alpha=alpha, ev_gap_threshold=ev_gap_threshold), -float(record.get("ev_gap", 0.0) or 0.0)),
            )
            record = weighted_anchors[anchor_index % len(weighted_anchors)]
            anchor_index += 1
        elif prioritized_signal_standard:
            record = prioritized_signal_standard[signal_standard_index % len(prioritized_signal_standard)]
            signal_standard_index += 1
        elif prioritized_background_standard:
            record = prioritized_background_standard[background_standard_index % len(prioritized_background_standard)]
            background_standard_index += 1
        elif fallback_records:
            record = fallback_records[step_index % len(fallback_records)]
        else:
            record = prioritized_anchors[anchor_index % len(prioritized_anchors)]
            anchor_index += 1
        schedule.append(record)
    return schedule


def filter_records_for_ablation(
    records: list[dict[str, object]],
    *,
    record_filter_mode: str | None = None,
    ev_gap_threshold: float = DEFAULT_ACTION_MATCH_EV_GAP_THRESHOLD,
) -> list[dict[str, object]]:
    mode = str(record_filter_mode or "none")
    if mode == "none":
        return list(records)
    if mode != "hicra_clean":
        raise ValueError(f"Unsupported record filter mode: {record_filter_mode}")
    filtered = [
        record
        for record in records
        if not (
            bool(record.get("reasoning_action_mismatch", False))
            and float(record.get("ev_gap", 0.0) or 0.0) > float(ev_gap_threshold)
        )
    ]
    return filtered or list(records)


def run_alignment_dry_run(
    dataset_path: Path | str,
    model_path: Path | str | None = None,
    group_size: int = 8,
    action_match_reward_weight: float = DEFAULT_ACTION_MATCH_REWARD_WEIGHT,
    ablation_variant: str = "full",
    hicra_gamma: float = 1.0,
) -> dict[str, object]:
    variant_settings = resolve_ablation_settings(ablation_variant, hicra_gamma=hicra_gamma)
    raw_records = load_alignment_records(dataset_path)
    records = [
        _coerce_alignment_record(apply_ablation_variant(record, str(variant_settings["ablation_variant"])))
        for record in raw_records
    ]
    records = filter_records_for_ablation(
        records,
        record_filter_mode=str(variant_settings["record_filter_mode"]),
        ev_gap_threshold=float(variant_settings["hicra_ev_gap_threshold"]),
    )
    resolved_model_path = Path(model_path) if model_path is not None else DEFAULT_PROXY_MODEL_PATH
    predictor = ProxyValuePredictor(model_path=resolved_model_path, output_mode=VALUE_PROXY_TARGET_PHI)

    groups: list[dict[str, object]] = []
    for record in records:
        groups.append(
            _build_scored_group(
                predictor=predictor,
                record=record,
                group_size=group_size,
                action_match_reward_weight=action_match_reward_weight,
                candidate_mode=str(variant_settings["candidate_mode"]),
                use_phi_dense_reward=bool(variant_settings["use_phi_dense_reward"]),
                use_hicra_reward=bool(variant_settings["use_hicra_reward"]),
                single_candidate_advantage=str(variant_settings["single_candidate_advantage"]),
                scope_hicra_to_logged_action=bool(variant_settings["scope_hicra_to_logged_action"]),
                hicra_reward_scale=float(variant_settings["hicra_reward_scale"]),
                hicra_mismatch_only=bool(variant_settings["hicra_mismatch_only"]),
                hicra_ev_gap_threshold=float(variant_settings["hicra_ev_gap_threshold"]),
                reward_component_weights=dict(variant_settings["reward_component_weights"]),
                advantage_clip=(
                    float(variant_settings["advantage_clip"])
                    if variant_settings["advantage_clip"] is not None
                    else None
                ),
            )
        )

    summary = {
        "dataset_path": str(dataset_path),
        "model_path": str(resolved_model_path),
        **variant_settings,
        "group_size": int(group_size),
        "action_match_reward_weight": float(action_match_reward_weight),
        "record_count": len(records),
        "dataset_signal_metrics": summarize_dataset_signal_metrics(records),
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
    save_optimizer_state: bool = True,
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
    if save_optimizer_state and optimizer is not None and hasattr(optimizer, "state_dict"):
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
    ablation_variant: str = "full",
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
    action_match_reward_weight: float = DEFAULT_ACTION_MATCH_REWARD_WEIGHT,
    parse_error_penalty: float = DEFAULT_PARSE_ERROR_PENALTY,
    fallback_used_penalty: float = DEFAULT_FALLBACK_USED_PENALTY,
    resolution_repair_penalty: float = DEFAULT_RESOLUTION_REPAIR_PENALTY,
    protocol_penalty_warmup_start: int = DEFAULT_PROTOCOL_PENALTY_WARMUP_START,
    protocol_penalty_warmup_end: int = DEFAULT_PROTOCOL_PENALTY_WARMUP_END,
    checkpoint_dir: Path | str | None = None,
    save_every_steps: int | None = None,
    save_final_adapter: bool = False,
    save_optimizer_state: bool = True,
) -> dict[str, object]:
    import torch

    if (save_every_steps is not None or save_final_adapter) and checkpoint_dir is None:
        raise ValueError("checkpoint_dir is required when save_every_steps or save_final_adapter is enabled")

    variant_settings = resolve_ablation_settings(ablation_variant, hicra_gamma=hicra_gamma)
    raw_records = load_alignment_records(dataset_path)
    normalized_records = [
        _coerce_alignment_record(apply_ablation_variant(record, str(variant_settings["ablation_variant"])))
        for record in raw_records
    ]
    normalized_records = filter_records_for_ablation(
        normalized_records,
        record_filter_mode=str(variant_settings["record_filter_mode"]),
        ev_gap_threshold=float(variant_settings["hicra_ev_gap_threshold"]),
    )
    records = prioritize_smoke_records(normalized_records)
    if max_records is not None:
        records = records[: max(1, int(max_records))]
    scheduled_records = build_weighted_record_schedule(
        records,
        steps=max(1, int(steps)),
        anchor_ratio=anchor_ratio,
        alpha=anchor_alpha,
        ev_gap_threshold=ev_gap_threshold,
        candidate_mode=str(variant_settings["candidate_mode"]),
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
        protocol_penalty_scale = _compute_protocol_penalty_scale(
            step_index=step_index + 1,
            warmup_start_step=protocol_penalty_warmup_start,
            warmup_end_step=protocol_penalty_warmup_end,
        )
        group = _build_scored_group(
            predictor=predictor,
            record=record,
            group_size=group_size,
            action_match_reward_weight=action_match_reward_weight,
            parse_error_penalty=parse_error_penalty,
            fallback_used_penalty=fallback_used_penalty,
            resolution_repair_penalty=resolution_repair_penalty,
            protocol_penalty_scale=protocol_penalty_scale,
            candidate_mode=str(variant_settings["candidate_mode"]),
            use_phi_dense_reward=bool(variant_settings["use_phi_dense_reward"]),
            use_hicra_reward=bool(variant_settings["use_hicra_reward"]),
            single_candidate_advantage=str(variant_settings["single_candidate_advantage"]),
            scope_hicra_to_logged_action=bool(variant_settings["scope_hicra_to_logged_action"]),
            hicra_reward_scale=float(variant_settings["hicra_reward_scale"]),
            hicra_mismatch_only=bool(variant_settings["hicra_mismatch_only"]),
            hicra_ev_gap_threshold=float(variant_settings["hicra_ev_gap_threshold"]),
            reward_component_weights=dict(variant_settings["reward_component_weights"]),
            advantage_clip=(
                float(variant_settings["advantage_clip"])
                if variant_settings["advantage_clip"] is not None
                else None
            ),
        )
        reward_span = max(group["rewards"]) - min(group["rewards"]) if group.get("rewards") else 0.0
        mean_advantage_abs = sum(abs(float(item)) for item in group.get("advantages", [])) / len(group.get("advantages", [])) if group.get("advantages") else 0.0
        mask_metrics = group["mask_metrics"]
        preference_pair: dict[str, object] | None = None
        if str(variant_settings["training_objective"]) == "dpo":
            try:
                preference_pair = build_dpo_preference_pair(
                    group,
                    preference_pair_mode=str(variant_settings["preference_pair_mode"]),
                )
                mean_advantage_abs = abs(float(preference_pair.get("preference_margin", 0.0) or 0.0))
            except ValueError:
                preference_pair = None
                mean_advantage_abs = 0.0
        signal_flags = should_skip_gradient_update(
            reward_span=reward_span,
            mean_abs_advantage=mean_advantage_abs,
            non_zero_mask_count=int(mask_metrics.get("non_zero_mask_count", 0) or 0),
            epsilon=signal_epsilon,
        )
        optimizer.zero_grad(set_to_none=True)
        candidate_metrics: list[dict[str, object]] = []
        candidate_examples: list[dict[str, object]] = []
        total_loss = None
        grad_norm = 0.0
        if not signal_flags["skip_update"]:
            if str(variant_settings["training_objective"]) == "dpo":
                if preference_pair is None:
                    total_loss = torch.zeros((), dtype=torch.float32, device=resolved_device)
                else:
                    pair_examples: dict[str, dict[str, object]] = {}
                    pair_logps: dict[str, object] = {}
                    for pair_role in ("chosen", "rejected"):
                        candidate = preference_pair[pair_role]
                        candidate_label_scope = resolve_candidate_label_scope(variant_settings, candidate)
                        training_candidate = apply_candidate_assistant_target_mode(
                            candidate,
                            assistant_target_mode=str(variant_settings["assistant_target_mode"]),
                        )
                        example = prepare_candidate_training_example(
                            training_candidate,
                            tokenizer,
                            max_seq_len=max_seq_len,
                            label_scope=candidate_label_scope,
                        )
                        candidate_examples.append(example)
                        pair_examples[pair_role] = example
                        input_ids = example["input_ids"].to(resolved_device)
                        attention_mask = example["attention_mask"].to(resolved_device)
                        label_token_mask = example["label_token_mask"].to(resolved_device)

                        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                        ref_logits = _compute_reference_logits(model, input_ids=input_ids, attention_mask=attention_mask)
                        policy_logp_metrics = compute_sequence_log_prob(
                            logits=logits,
                            input_ids=input_ids,
                            label_token_mask=label_token_mask,
                        )
                        if ref_logits is not None:
                            ref_logp_metrics = compute_sequence_log_prob(
                                logits=ref_logits,
                                input_ids=input_ids,
                                label_token_mask=label_token_mask,
                            )
                            pair_logps[f"{pair_role}_ref_logp"] = ref_logp_metrics["sequence_logp"]
                            reference_mean_logp = ref_logp_metrics["mean_logp"]
                        else:
                            pair_logps[f"{pair_role}_ref_logp"] = None
                            reference_mean_logp = 0.0
                        pair_logps[f"{pair_role}_policy_logp"] = policy_logp_metrics["sequence_logp"]
                        candidate_metrics.append(
                            {
                                "candidate_index": int(candidate.get("candidate_index", 0) or 0),
                                "candidate_role": str(candidate.get("candidate_role", "")),
                                "dpo_pair_role": pair_role,
                                "reward": _candidate_total_reward(candidate),
                                "policy_mean_logp": policy_logp_metrics["mean_logp"],
                                "reference_mean_logp": reference_mean_logp,
                                "active_label_count": policy_logp_metrics["active_label_count"],
                                "label_scope": candidate_label_scope,
                            }
                        )
                    dpo_metrics = compute_dpo_pair_loss_from_logps(
                        chosen_policy_logp=pair_logps["chosen_policy_logp"],
                        rejected_policy_logp=pair_logps["rejected_policy_logp"],
                        chosen_ref_logp=pair_logps["chosen_ref_logp"],
                        rejected_ref_logp=pair_logps["rejected_ref_logp"],
                        beta=float(variant_settings["dpo_beta"]),
                    )
                    total_loss = dpo_metrics["loss"]
                    candidate_metrics.append(
                        {
                            "candidate_role": "dpo_pair",
                            "loss": float(total_loss.detach().cpu().item()),
                            "dpo_beta": dpo_metrics["beta"],
                            "policy_logratio": dpo_metrics["policy_logratio"],
                            "reference_logratio": dpo_metrics["reference_logratio"],
                            "preference_logit": dpo_metrics["preference_logit"],
                            "preference_margin": float(preference_pair.get("preference_margin", 0.0) or 0.0),
                        }
                    )
            else:
                for candidate in group["candidates"]:
                    candidate_label_scope = resolve_candidate_label_scope(variant_settings, candidate)
                    training_candidate = apply_candidate_assistant_target_mode(
                        candidate,
                        assistant_target_mode=str(variant_settings["assistant_target_mode"]),
                    )
                    example = prepare_candidate_training_example(
                        training_candidate,
                        tokenizer,
                        max_seq_len=max_seq_len,
                        label_scope=candidate_label_scope,
                    )
                    candidate_examples.append(example)
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
                        kl_beta=float(kl_beta) * float(variant_settings["kl_beta_multiplier"]),
                        hicra_gamma=float(variant_settings["hicra_gamma"]),
                        token_advantage_mode=str(variant_settings["token_advantage_mode"]),
                        token_advantage_alpha=float(variant_settings["token_advantage_alpha"]),
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
                            "token_advantage_mode": metrics["token_advantage_mode"],
                            "token_advantage_mean": metrics["token_advantage_mean"],
                            "strategic_token_advantage_mean": metrics["strategic_token_advantage_mean"],
                            "strategic_token_count": metrics["strategic_token_count"],
                            "active_label_count": metrics["active_label_count"],
                            "label_scope": candidate_label_scope,
                        }
                    )
                    total_loss = metrics["loss"] if total_loss is None else total_loss + metrics["loss"]

                total_loss = total_loss / max(1, len(group["candidates"]))
            total_loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(max_grad_norm)).detach().cpu().item())
            optimizer.step()
        else:
            total_loss = torch.zeros((), dtype=torch.float32, device=resolved_device)

        effective_mask_metrics = (
            _aggregate_recalibrated_mask_metrics(candidate_examples)
            if candidate_examples
            else group["mask_metrics"]
        )

        loss_metrics = [float(item["loss"]) for item in candidate_metrics if "loss" in item]
        kl_metrics = [float(item["kl_value"]) for item in candidate_metrics if "kl_value" in item]
        weighted_nll_metrics = [float(item["weighted_nll"]) for item in candidate_metrics if "weighted_nll" in item]
        mean_candidate_loss = sum(loss_metrics) / len(loss_metrics) if loss_metrics else 0.0
        mean_kl = sum(kl_metrics) / len(kl_metrics) if kl_metrics else 0.0
        mean_weighted_nll = (
            sum(weighted_nll_metrics) / len(weighted_nll_metrics)
            if weighted_nll_metrics
            else 0.0
        )
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
                "mask_metrics": effective_mask_metrics,
                "raw_mask_metrics": mask_metrics,
                "grad_norm": grad_norm,
                "protocol_penalty_scale": float(protocol_penalty_scale),
                "training_objective": str(variant_settings["training_objective"]),
                "preference_pair_mode": str(variant_settings["preference_pair_mode"]),
                "preference_margin": (
                    float(preference_pair.get("preference_margin", 0.0) or 0.0)
                    if isinstance(preference_pair, dict)
                    else 0.0
                ),
                "candidate_metrics": candidate_metrics,
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
                    save_optimizer_state=bool(save_optimizer_state),
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
                save_optimizer_state=bool(save_optimizer_state),
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
        **variant_settings,
        "group_size": int(group_size),
        "action_match_reward_weight": float(action_match_reward_weight),
        "parse_error_penalty": float(parse_error_penalty),
        "fallback_used_penalty": float(fallback_used_penalty),
        "resolution_repair_penalty": float(resolution_repair_penalty),
        "effective_kl_beta": float(kl_beta) * float(variant_settings["kl_beta_multiplier"]),
        "protocol_penalty_warmup_start": int(protocol_penalty_warmup_start),
        "protocol_penalty_warmup_end": int(protocol_penalty_warmup_end),
        "requested_steps": int(steps),
        "completed_steps": len(step_summaries),
        "records_used": len(records),
        "scheduled_anchor_ratio": float(anchor_ratio),
        "scheduled_anchor_alpha": float(anchor_alpha),
        "signal_epsilon": float(signal_epsilon),
        "dataset_signal_metrics": summarize_dataset_signal_metrics(records),
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
        "save_optimizer_state": bool(save_optimizer_state),
        "checkpoint_events": checkpoint_events,
        "final_adapter_path": final_adapter_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train isolated CRC-GRA ablation variants for SAVI alignment.")
    parser.add_argument("dataset_path", help="Path to HICRA-labeled JSONL dataset.")
    parser.add_argument("--model-path", default=str(DEFAULT_PROXY_MODEL_PATH))
    parser.add_argument("--policy-model-path", default=None)
    parser.add_argument("--ablation-variant", choices=sorted(ABLATION_VARIANTS), default="full")
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
    parser.add_argument("--action-match-reward-weight", type=float, default=DEFAULT_ACTION_MATCH_REWARD_WEIGHT)
    parser.add_argument("--parse-error-penalty", type=float, default=DEFAULT_PARSE_ERROR_PENALTY)
    parser.add_argument("--fallback-used-penalty", type=float, default=DEFAULT_FALLBACK_USED_PENALTY)
    parser.add_argument("--resolution-repair-penalty", type=float, default=DEFAULT_RESOLUTION_REPAIR_PENALTY)
    parser.add_argument("--protocol-penalty-warmup-start", type=int, default=DEFAULT_PROTOCOL_PENALTY_WARMUP_START)
    parser.add_argument("--protocol-penalty-warmup-end", type=int, default=DEFAULT_PROTOCOL_PENALTY_WARMUP_END)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--save-every-steps", type=int, default=None)
    parser.add_argument("--save-final-adapter", action="store_true", default=False)
    parser.add_argument("--skip-optimizer-state", action="store_true", default=False)
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
            action_match_reward_weight=args.action_match_reward_weight,
            ablation_variant=args.ablation_variant,
            hicra_gamma=args.hicra_gamma,
        )
    else:
        if not args.policy_model_path:
            raise ValueError("--policy-model-path is required for smoke training mode")
        summary = run_smoke_training(
            dataset_path=args.dataset_path,
            policy_model_path=args.policy_model_path,
            model_path=args.model_path,
            ablation_variant=args.ablation_variant,
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
            action_match_reward_weight=args.action_match_reward_weight,
            parse_error_penalty=args.parse_error_penalty,
            fallback_used_penalty=args.fallback_used_penalty,
            resolution_repair_penalty=args.resolution_repair_penalty,
            protocol_penalty_warmup_start=args.protocol_penalty_warmup_start,
            protocol_penalty_warmup_end=args.protocol_penalty_warmup_end,
            checkpoint_dir=args.checkpoint_dir,
            save_every_steps=args.save_every_steps,
            save_final_adapter=bool(args.save_final_adapter),
            save_optimizer_state=not bool(args.skip_optimizer_state),
        )
    if args.output_path:
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
