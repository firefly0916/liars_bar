from __future__ import annotations

import json
import re

from liars_game_engine.engine.game_state import JOKER_RANK


SPECIAL_TOKEN_PATTERN = re.compile(r"<\|[^>]+?\|>")
SPAN_OVERLAP_RULE = "token_end > span_start and token_start < span_end"


def _derive_action_target(sample: dict[str, object]) -> dict[str, object]:
    action = sample.get("action", {})
    if not isinstance(action, dict):
        action = {}
    action_type = str(action.get("type", "") or "")
    if action_type in {"challenge", "pass"}:
        return {
            "type": action_type,
            "play_count": None,
            "true_card_count": None,
        }

    cards = [str(card) for card in action.get("cards", [])] if isinstance(action.get("cards", []), list) else []
    play_count = len(cards)
    observation = sample.get("observation", {})
    table_type = ""
    if isinstance(observation, dict):
        table_type = str(observation.get("table_type", "") or "")
    if not table_type:
        table_type = str(action.get("claim_rank", "") or "")
    truthful_cards = {table_type, JOKER_RANK} if table_type else {JOKER_RANK}
    true_card_count = sum(1 for card in cards if str(card) in truthful_cards)
    return {
        "type": action_type,
        "play_count": play_count if play_count > 0 else None,
        "true_card_count": true_card_count if play_count > 0 else 0,
    }


def build_assistant_response_text(sample: dict[str, object]) -> str:
    payload = {
        "Reasoning": str(sample.get("thought", "") or ""),
        "Action": _derive_action_target(sample),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _render_messages(messages: list[dict[str, str]], tokenizer: object) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
        except TypeError:
            return str(tokenizer.apply_chat_template(messages, tokenize=False))
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages)


def _tokenize_with_offsets(rendered_text: str, tokenizer: object) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer(rendered_text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = encoded.get("input_ids", [])
    offset_mapping = encoded.get("offset_mapping", [])
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    if offset_mapping and isinstance(offset_mapping[0], list):
        offset_mapping = offset_mapping[0]
    normalized_offsets = [(int(start), int(end)) for start, end in offset_mapping]
    return [int(token_id) for token_id in input_ids], normalized_offsets


def _find_assistant_text(rendered_text: str, assistant_text: str) -> tuple[int, int]:
    start = rendered_text.rfind(assistant_text)
    if start < 0:
        raise ValueError("Assistant text not found inside rendered text; cannot align token spans.")
    return start, start + len(assistant_text)


def _find_reasoning_span(assistant_text: str, reasoning_text: str) -> tuple[int, int]:
    start = assistant_text.find(reasoning_text)
    if start < 0:
        raise ValueError("Reasoning text not found inside assistant text; cannot align strategic spans.")
    return start, start + len(reasoning_text)


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


def _distribute_weight(total_weight: float, token_count: int, strategy: str) -> list[float]:
    if token_count <= 0:
        return []
    if strategy == "equal":
        share = total_weight / token_count
        return [round(float(share), 12) for _ in range(token_count)]
    if strategy == "first_token":
        return [round(float(total_weight), 12)] + [0.0 for _ in range(token_count - 1)]
    raise ValueError(f"Unsupported weight distribution strategy: {strategy}")


def _ensure_assistant_message(messages: list[dict[str, str]], assistant_text: str) -> list[dict[str, str]]:
    normalized = [{"role": str(message["role"]), "content": str(message["content"])} for message in messages]
    assistant_indices = [index for index, message in enumerate(normalized) if message["role"] == "assistant"]
    if assistant_indices:
        normalized[assistant_indices[-1]] = {"role": "assistant", "content": assistant_text}
        return normalized
    return [*normalized, {"role": "assistant", "content": assistant_text}]


def build_alignment_metadata(
    sample: dict[str, object],
    tokenizer: object,
    messages: list[dict[str, str]] | None = None,
    assistant_text: str | None = None,
    weight_distribution_strategy: str = "equal",
) -> dict[str, object]:
    reasoning_text = str(sample.get("thought", "") or "")
    resolved_assistant_text = assistant_text or build_assistant_response_text(sample)
    resolved_messages = _ensure_assistant_message(messages or [], assistant_text=resolved_assistant_text)
    rendered_text = _render_messages(resolved_messages, tokenizer=tokenizer)
    input_ids, offset_mapping = _tokenize_with_offsets(rendered_text, tokenizer=tokenizer)

    assistant_start, assistant_end = _find_assistant_text(rendered_text, resolved_assistant_text)
    reasoning_assistant_start, reasoning_assistant_end = _find_reasoning_span(
        resolved_assistant_text,
        reasoning_text,
    )
    reasoning_rendered_start = assistant_start + reasoning_assistant_start
    reasoning_rendered_end = assistant_start + reasoning_assistant_end

    strategic_token_alignments: list[dict[str, object]] = []
    strategic_tokens = sample.get("strategic_tokens", [])
    if not isinstance(strategic_tokens, list):
        strategic_tokens = []
    for token in strategic_tokens:
        if not isinstance(token, dict):
            continue
        raw_start = int(token.get("start", 0) or 0)
        raw_end = int(token.get("end", 0) or 0)
        assistant_span_start = reasoning_assistant_start + raw_start
        assistant_span_end = reasoning_assistant_start + raw_end
        rendered_span_start = reasoning_rendered_start + raw_start
        rendered_span_end = reasoning_rendered_start + raw_end
        token_indices = _find_overlapping_token_indices(offset_mapping, rendered_span_start, rendered_span_end)
        token_spans = [
            {"start": offset_mapping[index][0], "end": offset_mapping[index][1]}
            for index in token_indices
        ]
        penalty_signal = float(token.get("penalty_signal", 0.0) or 0.0)
        label_weight = float(token.get("weight", 1.0) or 1.0)
        total_weight = penalty_signal * label_weight
        distributed_weights = _distribute_weight(total_weight, len(token_indices), weight_distribution_strategy)
        strategic_token_alignments.append(
            {
                "label": str(token.get("label", "") or ""),
                "token": str(token.get("token", "") or ""),
                "raw_char_span": {"start": raw_start, "end": raw_end},
                "assistant_char_span": {"start": assistant_span_start, "end": assistant_span_end},
                "rendered_char_span": {"start": rendered_span_start, "end": rendered_span_end},
                "token_indices": token_indices,
                "token_spans": token_spans,
                "allocation_strategy": weight_distribution_strategy,
                "penalty_signal": penalty_signal,
                "label_weight": label_weight,
                "distributed_weights": distributed_weights,
            }
        )

    special_token_spans = [
        {"token": match.group(0), "start": match.start(), "end": match.end()}
        for match in SPECIAL_TOKEN_PATTERN.finditer(rendered_text)
    ]
    return {
        "reasoning_text": reasoning_text,
        "assistant_text": resolved_assistant_text,
        "rendered_text": rendered_text,
        "assistant_rendered_span": {"start": assistant_start, "end": assistant_end},
        "reasoning_assistant_span": {"start": reasoning_assistant_start, "end": reasoning_assistant_end},
        "reasoning_rendered_span": {"start": reasoning_rendered_start, "end": reasoning_rendered_end},
        "template_prefix_char_count": assistant_start,
        "special_token_spans": special_token_spans,
        "token_count": len(input_ids),
        "mapping_rules": {
            "render_protocol": "chat_template" if hasattr(tokenizer, "apply_chat_template") else "plain_text",
            "chat_template_offset_applied": bool(assistant_start > 0),
            "span_overlap_rule": SPAN_OVERLAP_RULE,
            "weight_distribution_strategy": weight_distribution_strategy,
            "assistant_span_source": "assistant_text_substring_search",
            "reasoning_span_source": "reasoning_text_substring_search_inside_assistant",
        },
        "strategic_token_alignments": strategic_token_alignments,
    }


def build_token_weight_mask(alignment_metadata: dict[str, object]) -> list[float]:
    token_count = int(alignment_metadata.get("token_count", 0) or 0)
    mask = [0.0 for _ in range(max(0, token_count))]
    alignments = alignment_metadata.get("strategic_token_alignments", [])
    if not isinstance(alignments, list):
        return mask
    for alignment in alignments:
        if not isinstance(alignment, dict):
            continue
        token_indices = alignment.get("token_indices", [])
        distributed_weights = alignment.get("distributed_weights", [])
        if not isinstance(token_indices, list) or not isinstance(distributed_weights, list):
            continue
        for index, weight in zip(token_indices, distributed_weights, strict=False):
            if 0 <= int(index) < len(mask):
                mask[int(index)] += float(weight)
    return mask


def align_sample_to_tokens(
    sample: dict[str, object],
    tokenizer: object,
    messages: list[dict[str, str]] | None = None,
    assistant_text: str | None = None,
    weight_distribution_strategy: str = "equal",
) -> dict[str, object]:
    alignment_metadata = build_alignment_metadata(
        sample=sample,
        tokenizer=tokenizer,
        messages=messages,
        assistant_text=assistant_text,
        weight_distribution_strategy=weight_distribution_strategy,
    )
    return {
        "alignment_metadata": alignment_metadata,
        "token_weight_mask": build_token_weight_mask(alignment_metadata),
    }
