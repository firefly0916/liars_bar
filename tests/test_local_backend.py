import sys
import types
import unittest
from unittest.mock import patch

from liars_game_engine.agents import local_backend


class LocalBackendTest(unittest.TestCase):
    def test_load_model_bundle_uses_local_files_only_for_local_hf_backend(self) -> None:
        calls: list[tuple[str, dict[str, object]]] = []

        class FakeTokenizer:
            @staticmethod
            def from_pretrained(model_name: str, **kwargs):
                calls.append(("tokenizer", {"model_name": model_name, **kwargs}))
                return object()

        class FakeModel:
            @staticmethod
            def from_pretrained(model_name: str, **kwargs):
                calls.append(("model", {"model_name": model_name, **kwargs}))
                return object()

        fake_transformers = types.SimpleNamespace(
            AutoTokenizer=FakeTokenizer,
            AutoModelForCausalLM=FakeModel,
        )

        local_backend._load_model_bundle.cache_clear()
        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            local_backend._load_model_bundle("Qwen/Qwen2.5-0.5B-Instruct")

        self.assertEqual(len(calls), 2)
        self.assertTrue(all(call[1]["local_files_only"] is True for call in calls))
        self.assertTrue(all(call[1]["trust_remote_code"] is True for call in calls))

    def test_load_model_bundle_accepts_device_map_auto(self) -> None:
        calls: list[tuple[str, dict[str, object]]] = []

        class FakeTokenizer:
            @staticmethod
            def from_pretrained(model_name: str, **kwargs):
                calls.append(("tokenizer", {"model_name": model_name, **kwargs}))
                return object()

        class FakeModel:
            @staticmethod
            def from_pretrained(model_name: str, **kwargs):
                calls.append(("model", {"model_name": model_name, **kwargs}))
                return object()

        fake_transformers = types.SimpleNamespace(
            AutoTokenizer=FakeTokenizer,
            AutoModelForCausalLM=FakeModel,
        )

        local_backend._load_model_bundle.cache_clear()
        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            local_backend._load_model_bundle("Qwen/Qwen2.5-0.5B-Instruct", device_map="auto")

        model_call = next(payload for kind, payload in calls if kind == "model")
        self.assertEqual(model_call["device_map"], "auto")

    def test_generate_local_chat_completion_sync_uses_eval_and_kv_cache(self) -> None:
        generate_calls: list[dict[str, object]] = []

        class FakeTokenizer:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                return "PROMPT"

            def __call__(self, prompt: str, return_tensors: str):
                return {"input_ids": type("Ids", (), {"shape": (1, 3)})()}

            def decode(self, generated_ids, skip_special_tokens=True):
                return '{"Reasoning":"short","Action":{"type":"challenge"}}'

        class FakeModel:
            def __init__(self) -> None:
                self.eval_called = False

            def eval(self):
                self.eval_called = True
                return self

            def generate(self, **kwargs):
                generate_calls.append(kwargs)
                return [[11, 12, 13, 14]]

        model = FakeModel()
        bundle = local_backend.LocalModelBundle(tokenizer=FakeTokenizer(), model=model)

        with patch.object(local_backend, "_load_model_bundle", return_value=bundle):
            text = local_backend._generate_local_chat_completion_sync(
                model="Qwen/Qwen2.5-0.5B-Instruct",
                messages=[{"role": "user", "content": "hi"}],
                temperature=0.0,
                max_new_tokens=24,
            )

        self.assertTrue(model.eval_called)
        self.assertEqual(text, '{"Reasoning":"short","Action":{"type":"challenge"}}')
        self.assertEqual(len(generate_calls), 1)
        self.assertFalse(generate_calls[0]["do_sample"])
        self.assertTrue(generate_calls[0]["use_cache"])

    def test_generate_local_chat_completion_sync_moves_inputs_to_cuda_when_requested(self) -> None:
        class FakeTensor:
            shape = (1, 3)

            def __init__(self) -> None:
                self.moved_to: str | None = None

            def to(self, device: str):
                self.moved_to = device
                return self

        class FakeTokenizer:
            def __init__(self) -> None:
                self.last_inputs: dict[str, FakeTensor] | None = None

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                return "PROMPT"

            def __call__(self, prompt: str, return_tensors: str):
                self.last_inputs = {"input_ids": FakeTensor(), "attention_mask": FakeTensor()}
                return self.last_inputs

            def decode(self, generated_ids, skip_special_tokens=True):
                return '{"Reasoning":"short","Action":{"type":"challenge"}}'

        class FakeModel:
            def eval(self):
                return self

            def generate(self, **kwargs):
                return [[11, 12, 13, 14]]

        tokenizer = FakeTokenizer()
        bundle = local_backend.LocalModelBundle(tokenizer=tokenizer, model=FakeModel())
        with patch("torch.cuda.is_available", return_value=True), patch.object(
            local_backend,
            "_load_model_bundle",
            return_value=bundle,
        ):
            local_backend._generate_local_chat_completion_sync(
                model="Qwen/Qwen2.5-0.5B-Instruct",
                messages=[{"role": "user", "content": "hi"}],
                temperature=0.0,
                max_new_tokens=24,
                device="cuda",
            )

        self.assertIsNotNone(tokenizer.last_inputs)
        self.assertEqual(tokenizer.last_inputs["input_ids"].moved_to, "cuda")
        self.assertEqual(tokenizer.last_inputs["attention_mask"].moved_to, "cuda")


if __name__ == "__main__":
    unittest.main()
