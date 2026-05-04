import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


def _load_train_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "train_savi_alignment.py"
    spec = importlib.util.spec_from_file_location("train_savi_alignment", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class TrainSaviAlignmentTest(unittest.TestCase):
    @staticmethod
    def _cjk_text() -> str:
        return "".join(chr(codepoint) for codepoint in (0x6211, 0x6000, 0x7591, 0x5BF9, 0x65B9))

    class _FakeQwenTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            rendered = ""
            for message in messages:
                rendered += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            if add_generation_prompt:
                rendered += "<|im_start|>assistant\n"
            return rendered

        def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, return_tensors=None):
            input_ids = list(range(1, len(text) + 1))
            payload = {
                "input_ids": input_ids,
                "attention_mask": [1 for _ in input_ids],
            }
            if return_offsets_mapping:
                payload["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
            if return_tensors == "pt":
                payload = {key: torch.tensor([value], dtype=torch.long) for key, value in payload.items() if key != "offset_mapping"}
            return payload

    def test_compute_group_relative_advantages_centers_rewards(self) -> None:
        module = _load_train_module()
        advantages = module.compute_group_relative_advantages([0.1, 0.4, -0.2])
        self.assertEqual(len(advantages), 3)
        self.assertAlmostEqual(sum(advantages), 0.0, places=6)
        self.assertGreater(advantages[1], advantages[0])
        self.assertLess(advantages[2], 0.0)

    def test_run_alignment_dry_run_applies_group_rewards_and_penalties(self) -> None:
        module = _load_train_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_path = root / "savi_alignment_train.jsonl"
            dataset_path.write_text(
                json.dumps(
                    {
                        "game_id": "g1",
                        "turn": 2,
                        "player_id": "p1",
                        "thought": "High risk bluff in a long game.",
                        "action": {"type": "challenge", "claim_rank": "", "cards": []},
                        "proxy_target_action": {"type": "play_claim", "claim_rank": "A", "cards": ["A"]},
                        "token_penalty": -0.25,
                        "ev_gap": 0.25,
                        "sample_type": "Reasoning-Action Mismatch",
                        "reasoning_action_mismatch": True,
                        "strategic_tokens": [
                            {"label": "risk", "token": "risk", "weight": 1.5},
                            {"label": "skepticism", "token": "doubt", "weight": 1.4},
                        ],
                        "strategic_token_weight": 1.5,
                        "state_features": {
                            "table_type": "A",
                            "private_hand": ["A", "Q"],
                            "action_type": "challenge",
                            "action_cards": [],
                            "death_probability": 0.2,
                            "pending_claim_declared_count": 0,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            model_path = root / "value_proxy_mlp_distill.pt"
            model_path.write_text("stub", encoding="utf-8")

            class _FakePredictor:
                def __init__(self, model_path: str | Path, output_mode: str) -> None:
                    self.model_path = Path(model_path)
                    self.output_mode = output_mode

                def predict_state_features(self, state_features: dict[str, object]) -> float:
                    action_type = str(state_features.get("action_type", ""))
                    cards = list(state_features.get("action_cards", []))
                    if action_type == "play_claim" and cards == ["A"]:
                        return 0.35
                    if action_type == "challenge":
                        return -0.05
                    return 0.1

            with patch.object(module, "ProxyValuePredictor", _FakePredictor):
                summary = module.run_alignment_dry_run(
                    dataset_path=dataset_path,
                    model_path=model_path,
                    group_size=3,
                )

            self.assertEqual(summary["record_count"], 1)
            self.assertEqual(summary["group_size"], 3)
            self.assertEqual(summary["groups"][0]["candidate_count"], 3)
            self.assertEqual(len(summary["groups"][0]["rewards"]), 3)
            self.assertEqual(len(summary["groups"][0]["advantages"]), 3)
            candidates = summary["groups"][0]["candidates"]
            self.assertEqual(candidates[0]["action"]["type"], "challenge")
            self.assertEqual(candidates[1]["action"]["type"], "play_claim")
            self.assertEqual(candidates[1]["action"]["cards"], ["A"])
            self.assertIn("reward_breakdown", candidates[0])
            self.assertIn("reward_breakdown", candidates[1])
            self.assertLess(candidates[0]["reward_breakdown"]["hicra_penalty"], 0.0)
            self.assertEqual(candidates[1]["reward_breakdown"]["action_match_reward"], 1.0)
            self.assertGreater(
                candidates[1]["reward_breakdown"]["total_reward"],
                candidates[0]["reward_breakdown"]["total_reward"],
            )
            self.assertAlmostEqual(sum(summary["groups"][0]["advantages"]), 0.0, places=6)
            self.assertEqual(summary["smoke_metrics"]["high_ev_gap_mismatch_group_count"], 1)

    def test_compute_mask_hit_metrics_reports_non_zero_mask_coverage(self) -> None:
        module = _load_train_module()

        metrics = module.compute_mask_hit_metrics(
            {
                "alignment_metadata": {
                    "strategic_token_alignments": [
                        {"token_indices": [4, 5]},
                        {"token_indices": [8]},
                    ]
                },
                "token_weight_mask": [0.0, 0.0, 0.0, 0.0, -0.2, -0.2, 0.0, 0.0, -0.1],
            }
        )

        self.assertEqual(metrics["strategic_alignment_count"], 2)
        self.assertEqual(metrics["strategic_token_index_count"], 3)
        self.assertEqual(metrics["non_zero_mask_count"], 3)
        self.assertAlmostEqual(metrics["mask_hit_rate"], 1.0, places=6)

    def test_summarize_smoke_groups_reports_reward_variance_and_high_gap_anchor_count(self) -> None:
        module = _load_train_module()
        summary = module.summarize_smoke_groups(
            [
                {
                    "ev_gap": 0.21,
                    "reasoning_action_mismatch": True,
                    "candidates": [
                        {"reward_breakdown": {"total_reward": -0.1}},
                        {"reward_breakdown": {"total_reward": 0.4}},
                    ],
                },
                {
                    "ev_gap": 0.05,
                    "reasoning_action_mismatch": False,
                    "candidates": [
                        {"reward_breakdown": {"total_reward": 0.1}},
                        {"reward_breakdown": {"total_reward": 0.2}},
                    ],
                },
            ]
        )

        self.assertEqual(summary["group_count"], 2)
        self.assertEqual(summary["high_ev_gap_mismatch_group_count"], 1)
        self.assertGreater(summary["reward_variance"], 0.0)
        self.assertGreater(summary["max_group_reward_span"], 0.0)

    def test_prepare_candidate_training_example_marks_assistant_tokens_and_hicra_weights(self) -> None:
        module = _load_train_module()
        tokenizer = self._FakeQwenTokenizer()
        candidate = {
            "thought": "I suspect the claim is false, so challenge is the safer action.",
            "action": {"type": "challenge", "claim_rank": "", "cards": []},
            "observation": {"table_type": "A"},
            "messages": [
                {"role": "system", "content": "You are a careful Liar's Bar player."},
                {"role": "user", "content": "Return JSON only."},
            ],
            "strategic_tokens": [
                {
                    "label": "skepticism",
                    "token": "suspect",
                    "start": 2,
                    "end": 9,
                    "weight": 1.4,
                    "penalty_signal": -0.3,
                }
            ],
        }

        example = module.prepare_candidate_training_example(candidate, tokenizer)

        self.assertGreater(example["assistant_token_count"], 0)
        self.assertGreater(example["active_label_count"], 0)
        self.assertGreater(example["hicra_non_zero_count"], 0)
        self.assertEqual(example["input_ids"].shape, example["attention_mask"].shape)
        self.assertEqual(example["label_token_mask"].shape[0], example["input_ids"].shape[1] - 1)

    def test_compute_candidate_loss_terms_combines_advantage_kl_and_hicra(self) -> None:
        module = _load_train_module()
        logits = torch.tensor(
            [[[0.0, 0.0, 0.0], [0.2, 0.1, -0.3], [0.5, -0.2, 0.0], [0.1, 0.3, -0.1]]],
            dtype=torch.float32,
        )
        ref_logits = torch.tensor(
            [[[0.0, 0.0, 0.0], [0.0, 0.2, -0.1], [0.3, 0.0, 0.1], [0.2, 0.0, 0.0]]],
            dtype=torch.float32,
        )
        input_ids = torch.tensor([[0, 1, 2, 1]], dtype=torch.long)
        label_token_mask = torch.tensor([0.0, 1.0, 1.0], dtype=torch.float32)
        token_weight_mask = torch.tensor([0.0, -0.2, -0.4], dtype=torch.float32)

        metrics = module.compute_candidate_loss_terms(
            logits=logits,
            ref_logits=ref_logits,
            input_ids=input_ids,
            label_token_mask=label_token_mask,
            token_weight_mask=token_weight_mask,
            advantage=0.35,
            kl_beta=0.1,
            hicra_gamma=2.0,
        )

        self.assertGreater(metrics["active_label_count"], 0)
        self.assertGreater(metrics["hicra_weight_mean"], 1.0)
        self.assertGreaterEqual(metrics["kl_value"], 0.0)
        self.assertTrue(torch.isfinite(metrics["loss"]))

    def test_prepare_candidate_training_example_truncates_from_left_and_keeps_assistant_tail(self) -> None:
        module = _load_train_module()
        tokenizer = self._FakeQwenTokenizer()
        candidate = {
            "thought": "I suspect the claim is false, so challenge is the safer action.",
            "action": {"type": "challenge", "claim_rank": "", "cards": []},
            "observation": {"table_type": "A"},
            "messages": [
                {"role": "system", "content": "S" * 80},
                {"role": "user", "content": "U" * 120},
            ],
            "strategic_tokens": [
                {
                    "label": "skepticism",
                    "token": "suspect",
                    "start": 2,
                    "end": 9,
                    "weight": 1.4,
                    "penalty_signal": -0.3,
                }
            ],
        }

        example = module.prepare_candidate_training_example(candidate, tokenizer, max_seq_len=256)

        self.assertEqual(example["input_ids"].shape[1], 256)
        self.assertGreater(example["active_label_count"], 0)
        self.assertGreater(example["hicra_non_zero_count"], 0)

    def test_load_alignment_records_rejects_cjk_content(self) -> None:
        module = _load_train_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "alignment.jsonl"
            dataset_path.write_text(
                json.dumps(
                    {
                        "game_id": "g-cjk",
                        "turn": 1,
                        "player_id": "p1",
                        "thought": self._cjk_text(),
                        "messages": [
                            {"role": "system", "content": "You are a careful Liar's Bar player."},
                            {"role": "user", "content": "Return JSON only."},
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "English-only"):
                module.load_alignment_records(dataset_path)

    def test_resolve_model_loading_options_prefers_4bit_on_cuda(self) -> None:
        module = _load_train_module()

        options = module.resolve_model_loading_options(
            device="cuda",
            torch_dtype="bf16",
            load_in_4bit=True,
        )

        self.assertTrue(options["use_gradient_checkpointing"])
        self.assertTrue(options["load_in_4bit"])
        self.assertEqual(options["device_map"], "cuda")

    def test_compute_sampling_weight_uses_ev_gap_bias_for_class_a(self) -> None:
        module = _load_train_module()

        class_a = module.compute_sampling_weight(
            {
                "ev_gap": 0.4,
                "reasoning_action_mismatch": True,
            },
            alpha=2.0,
        )
        class_b = module.compute_sampling_weight(
            {
                "ev_gap": 0.05,
                "reasoning_action_mismatch": False,
            },
            alpha=2.0,
        )

        self.assertAlmostEqual(class_a, 1.8, places=6)
        self.assertAlmostEqual(class_b, 1.0, places=6)

    def test_should_skip_gradient_update_uses_robust_thresholds(self) -> None:
        module = _load_train_module()

        skip = module.should_skip_gradient_update(
            reward_span=1e-10,
            mean_abs_advantage=1e-12,
            non_zero_mask_count=0,
            epsilon=1e-8,
        )
        keep = module.should_skip_gradient_update(
            reward_span=0.2,
            mean_abs_advantage=0.05,
            non_zero_mask_count=2,
            epsilon=1e-8,
        )

        self.assertTrue(skip["skip_update"])
        self.assertTrue(skip["idle_step"])
        self.assertTrue(skip["signalless_step"])
        self.assertFalse(keep["skip_update"])
        self.assertFalse(keep["idle_step"])
        self.assertFalse(keep["signalless_step"])

    def test_summarize_step_metrics_reports_signal_density_and_mask_intensity(self) -> None:
        module = _load_train_module()

        summary = module.summarize_step_metrics(
            [
                {
                    "skip_update": False,
                    "idle_step": False,
                    "signalless_step": False,
                    "mask_metrics": {
                        "non_zero_mask_count": 2,
                        "average_hicra_mask_intensity": 0.35,
                    },
                },
                {
                    "skip_update": True,
                    "idle_step": True,
                    "signalless_step": True,
                    "mask_metrics": {
                        "non_zero_mask_count": 0,
                        "average_hicra_mask_intensity": 0.0,
                    },
                },
            ]
        )

        self.assertEqual(summary["effective_step_count"], 1)
        self.assertEqual(summary["idle_step_count"], 1)
        self.assertEqual(summary["signalless_step_count"], 1)
        self.assertAlmostEqual(summary["signal_density_rate"], 0.5, places=6)
        self.assertAlmostEqual(summary["average_hicra_mask_intensity"], 0.35, places=6)


if __name__ == "__main__":
    unittest.main()
