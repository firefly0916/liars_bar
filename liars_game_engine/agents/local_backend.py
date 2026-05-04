from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import lru_cache
import os
from typing import Any


class LocalBackendUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class LocalModelBundle:
    tokenizer: Any
    model: Any


def _resolve_local_files_only() -> bool:
    raw = str(os.environ.get("LOCAL_LLM_LOCAL_FILES_ONLY", "1")).strip().lower()
    return raw not in {"0", "false", "no"}


def _resolve_local_adapter_path(adapter_path: str | None = None) -> str | None:
    resolved = adapter_path or os.environ.get("LOCAL_LLM_ADAPTER_PATH")
    if resolved is None:
        return None
    text = str(resolved).strip()
    return text or None


@lru_cache(maxsize=4)
def _load_model_bundle(
    model_name: str,
    device: str | None = None,
    device_map: str | None = None,
    adapter_path: str | None = None,
) -> LocalModelBundle:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as error:  # pragma: no cover - depends on optional local runtime deps.
        raise LocalBackendUnavailableError("transformers is required for local://hf backend") from error

    try:
        import torch
    except Exception:  # pragma: no cover - torch is a required runtime dependency in this repo.
        torch = None

    resolved_adapter_path = _resolve_local_adapter_path(adapter_path)
    local_files_only = _resolve_local_files_only()
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_adapter_path or model_name,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": local_files_only,
    }
    if device_map:
        model_kwargs["device_map"] = device_map
    elif torch is not None and device and device.startswith("cuda") and torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if resolved_adapter_path:
        try:
            from peft import PeftModel
        except Exception as error:  # pragma: no cover - depends on optional local runtime deps.
            raise LocalBackendUnavailableError("peft is required for local://hf LoRA adapter loading") from error
        model = PeftModel.from_pretrained(model, resolved_adapter_path, local_files_only=local_files_only)
    if (
        torch is not None
        and device
        and device.startswith("cuda")
        and torch.cuda.is_available()
        and device_map is None
        and hasattr(model, "to")
    ):
        model = model.to(device)
    return LocalModelBundle(tokenizer=tokenizer, model=model)


def _build_prompt(messages: list[dict[str, str]], tokenizer: Any) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n\n".join(f"{item.get('role', 'user').upper()}:\n{item.get('content', '')}" for item in messages) + "\n\nASSISTANT:\n"


def _generate_local_chat_completion_sync(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_new_tokens: int,
    device: str | None = None,
    device_map: str | None = None,
    adapter_path: str | None = None,
) -> str:
    try:
        import torch
    except Exception:  # pragma: no cover - torch is a required runtime dependency in this repo.
        torch = None

    if torch is not None:
        torch.set_num_threads(1)

    bundle = _load_model_bundle(
        model,
        device=device,
        device_map=device_map,
        adapter_path=adapter_path,
    )
    if hasattr(bundle.model, "eval"):
        bundle.model.eval()
    prompt = _build_prompt(messages, bundle.tokenizer)
    inputs = bundle.tokenizer(prompt, return_tensors="pt")
    can_move_to_device = bool(device and device_map is None)
    if device and device.startswith("cuda"):
        can_move_to_device = bool(torch is not None and torch.cuda.is_available())
    if can_move_to_device:
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0.0,
        "use_cache": True,
    }
    if temperature > 0.0:
        generation_kwargs["temperature"] = temperature

    output_ids = bundle.model.generate(**inputs, **generation_kwargs)
    prompt_token_count = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0][prompt_token_count:]
    return bundle.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


async def generate_local_chat_completion(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_new_tokens: int = 96,
    device: str | None = None,
    device_map: str | None = None,
) -> str:
    resolved_device = device or os.environ.get("LOCAL_LLM_DEVICE")
    resolved_device_map = device_map or os.environ.get("LOCAL_LLM_DEVICE_MAP")
    resolved_adapter_path = _resolve_local_adapter_path()
    resolved_max_new_tokens = int(os.environ.get("LOCAL_LLM_MAX_NEW_TOKENS", str(max_new_tokens)))
    return await asyncio.to_thread(
        _generate_local_chat_completion_sync,
        model,
        messages,
        temperature,
        resolved_max_new_tokens,
        resolved_device,
        resolved_device_map,
        resolved_adapter_path,
    )
