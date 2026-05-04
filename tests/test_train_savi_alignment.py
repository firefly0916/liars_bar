import importlib.util
import itertools
import json
import tempfile
import unittest
from contextlib import nullcontext
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

        def save_pretrained(self, output_dir: str) -> None:
            path = Path(output_dir)
            path.mkdir(parents=True, exist_ok=True)
            (path / "tokenizer.json").write_text("{}", encoding="utf-8")

    class _FakeTrainModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
            self.saved_paths: list[str] = []

        def forward(self, input_ids=None, attention_mask=None):
            seq_len = int(input_ids.shape[1])
            logits = self.scale.view(1, 1, 1).expand(1, seq_len, 3)
            return type("FakeOutput", (), {"logits": logits})()

        def save_pretrained(self, output_dir: str) -> None:
            path = Path(output_dir)
            path.mkdir(parents=True, exist_ok=True)
            (path / "adapter_model.bin").write_text("stub", encoding="utf-8")
            self.saved_paths.append(str(path))

        def disable_adapter(self):
            return nullcontext()

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
                        "observation": {
                            "legal_actions": [
                                {"type": "challenge"},
                                {"type": "play_claim", "claim_rank": "A", "min_cards": 1, "max_cards": 1},
                            ]
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
                    action_match_reward_weight=0.25,
                )

            self.assertEqual(summary["record_count"], 1)
            self.assertEqual(summary["group_size"], 3)
            self.assertEqual(summary["action_match_reward_weight"], 0.25)
            self.assertEqual(summary["groups"][0]["candidate_count"], 2)
            self.assertEqual(len(summary["groups"][0]["rewards"]), 2)
            self.assertEqual(len(summary["groups"][0]["advantages"]), 2)
            candidates = summary["groups"][0]["candidates"]
            self.assertEqual(candidates[0]["action"]["type"], "challenge")
            self.assertEqual(candidates[1]["action"]["type"], "play_claim")
            self.assertEqual(candidates[1]["action"]["cards"], ["A"])
            self.assertIn("reward_breakdown", candidates[0])
            self.assertIn("reward_breakdown", candidates[1])
            self.assertLess(candidates[0]["reward_breakdown"]["hicra_penalty"], 0.0)
            self.assertEqual(candidates[1]["reward_breakdown"]["action_match_reward"], 0.25)
            self.assertGreater(
                candidates[1]["reward_breakdown"]["total_reward"],
                candidates[0]["reward_breakdown"]["total_reward"],
            )
            self.assertAlmostEqual(sum(summary["groups"][0]["advantages"]), 0.0, places=6)
            self.assertEqual(summary["smoke_metrics"]["high_ev_gap_mismatch_group_count"], 1)

    def test_compute_reward_breakdown_respects_action_match_reward_weight(self) -> None:
        module = _load_train_module()

        class _FakePredictor:
            def predict_state_features(self, state_features: dict[str, object]) -> float:
                return 0.4

        breakdown = module._compute_reward_breakdown(
            predictor=_FakePredictor(),
            candidate={
                "action": {"type": "play_claim", "claim_rank": "A", "cards": ["A"]},
                "proxy_target_action": {"type": "play_claim", "claim_rank": "A", "cards": ["A"]},
                "state_features": {"action_type": "play_claim", "action_cards": ["A"]},
                "strategic_tokens": [],
                "strategic_token_weight": 1.0,
                "token_penalty": 0.0,
            },
            action_match_reward_weight=0.3,
        )

        self.assertAlmostEqual(breakdown["action_match_reward"], 0.3, places=6)
        self.assertAlmostEqual(breakdown["phi_dense_reward"], 0.4, places=6)
        self.assertAlmostEqual(breakdown["total_reward"], 0.7, places=6)

    def test_build_group_candidates_deduplicates_proxy_target_repeats_and_keeps_challenge_path(self) -> None:
        module = _load_train_module()

        candidates = module._build_group_candidates(
            {
                "game_id": "g2",
                "turn": 9,
                "player_id": "p1",
                "thought": "Challenge is dangerous but available.",
                "action": {"type": "play_claim", "claim_rank": "Q", "cards": ["Q"]},
                "proxy_target_action": {"type": "play_claim", "claim_rank": "Q", "cards": ["Q"]},
                "observation": {
                    "legal_actions": [
                        {"type": "challenge"},
                        {"type": "play_claim", "claim_rank": "Q", "min_cards": 1, "max_cards": 1},
                    ]
                },
            },
            group_size=8,
        )

        self.assertEqual(len(candidates), 2)
        self.assertEqual([candidate["candidate_role"] for candidate in candidates], ["logged_action", "legal_challenge"])
        self.assertEqual(candidates[1]["action"]["type"], "challenge")

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

    def test_save_training_artifacts_writes_adapter_tokenizer_metadata_and_optimizer_state(self) -> None:
        module = _load_train_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "checkpoints"
            model = self._FakeTrainModel()
            tokenizer = self._FakeQwenTokenizer()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

            artifact = module.save_training_artifacts(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                checkpoint_dir=checkpoint_dir,
                tag="step-000002",
                lora_enabled=True,
                metadata={"step": 2, "reason": "interval"},
            )

            saved_path = checkpoint_dir / "step-000002"
            self.assertEqual(artifact["path"], str(saved_path))
            self.assertEqual(artifact["tag"], "step-000002")
            self.assertTrue((saved_path / "adapter_model.bin").exists())
            self.assertTrue((saved_path / "tokenizer.json").exists())
            self.assertTrue((saved_path / "optimizer.pt").exists())
            metadata = json.loads((saved_path / "training_snapshot.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["step"], 2)
            self.assertEqual(metadata["reason"], "interval")

    def test_run_smoke_training_records_interval_and_final_checkpoint_events(self) -> None:
        module = _load_train_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "checkpoints"
            model = self._FakeTrainModel()
            tokenizer = self._FakeQwenTokenizer()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

            components = {
                "model": model,
                "tokenizer": tokenizer,
                "optimizer": optimizer,
                "device": "cpu",
                "torch_dtype": "torch.float32",
                "lora_enabled": True,
                "trainable_parameter_count": 1,
                "load_in_4bit": False,
            }

            record = {
                "game_id": "g1",
                "turn": 1,
                "player_id": "p1",
                "thought": "Play carefully and stay consistent.",
                "action": {"type": "challenge", "claim_rank": "", "cards": []},
                "proxy_target_action": {"type": "play_claim", "claim_rank": "A", "cards": ["A"]},
                "messages": [{"role": "user", "content": "Return JSON only."}],
                "ev_gap": 0.2,
                "reasoning_action_mismatch": True,
            }
            group = {
                "game_id": "g1",
                "turn": 1,
                "ev_gap": 0.2,
                "reasoning_action_mismatch": True,
                "rewards": [0.1, 0.4],
                "advantages": [-0.15, 0.15],
                "mask_metrics": {
                    "strategic_alignment_count": 1,
                    "strategic_token_index_count": 1,
                    "non_zero_mask_count": 1,
                    "mask_hit_count": 1,
                    "mask_hit_rate": 1.0,
                    "average_hicra_mask_intensity": 0.2,
                },
                "candidates": [
                    {"candidate_index": 0, "candidate_role": "logged_action", "advantage": -0.15},
                    {"candidate_index": 1, "candidate_role": "proxy_target", "advantage": 0.15},
                ],
            }

            example = {
                "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
                "label_token_mask": torch.tensor([1.0, 1.0], dtype=torch.float32),
                "token_weight_mask": torch.tensor([0.0, -0.2], dtype=torch.float32),
            }
            losses = itertools.cycle([0.6, 0.3, 0.5, 0.2])
            saved_tags: list[str] = []

            def _fake_compute_candidate_loss_terms(**kwargs):
                loss_value = next(losses)
                loss = model.scale * 0 + torch.tensor(loss_value, dtype=torch.float32)
                return {
                    "loss": loss,
                    "nll": loss_value,
                    "weighted_nll": loss_value,
                    "kl_value": 0.01,
                    "hicra_weight_mean": 1.2,
                    "active_label_count": 2,
                }

            def _fake_save_training_artifacts(**kwargs):
                saved_tags.append(str(kwargs["tag"]))
                return {
                    "tag": str(kwargs["tag"]),
                    "path": str(Path(kwargs["checkpoint_dir"]) / str(kwargs["tag"])),
                    "step": kwargs["metadata"]["step"],
                    "reason": kwargs["metadata"]["reason"],
                }

            with patch.object(module, "load_alignment_records", return_value=[record]), patch.object(
                module,
                "_build_training_components",
                return_value=components,
            ), patch.object(
                module,
                "ProxyValuePredictor",
            ) as predictor_cls, patch.object(
                module,
                "_build_scored_group",
                return_value=group,
            ), patch.object(
                module,
                "prepare_candidate_training_example",
                return_value=example,
            ), patch.object(
                module,
                "compute_candidate_loss_terms",
                side_effect=_fake_compute_candidate_loss_terms,
            ), patch.object(
                module,
                "save_training_artifacts",
                side_effect=_fake_save_training_artifacts,
            ):
                predictor_cls.return_value = object()
                summary = module.run_smoke_training(
                    dataset_path="dataset.jsonl",
                    policy_model_path="policy",
                    model_path="proxy.pt",
                    group_size=2,
                    steps=2,
                    checkpoint_dir=checkpoint_dir,
                    save_every_steps=1,
                    save_final_adapter=True,
                )

            self.assertEqual(saved_tags, ["step-000001", "step-000002", "final"])
            self.assertEqual(len(summary["checkpoint_events"]), 3)
            self.assertEqual(summary["checkpoint_events"][-1]["tag"], "final")
            self.assertEqual(summary["final_adapter_path"], str(checkpoint_dir / "final"))


if __name__ == "__main__":
    unittest.main()
