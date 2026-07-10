import importlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class _CharTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "".join(f"<|{message['role']}|>{message['content']}" for message in messages)

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        result = {"input_ids": list(range(len(text)))}
        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return result


class CRCGRAAblationTest(unittest.TestCase):
    def _record(self) -> dict[str, object]:
        return {
            "game_id": "g1",
            "turn": 2,
            "player_id": "p1",
            "thought": "High risk bluff in a long game.",
            "action": {"type": "challenge", "claim_rank": "", "cards": []},
            "proxy_target_action": {"type": "play_claim", "claim_rank": "A", "cards": ["A"]},
            "token_penalty": -0.25,
            "ev_gap": 0.25,
            "reasoning_action_mismatch": True,
            "strategic_tokens": [{"label": "risk", "token": "risk", "weight": 1.5}],
            "strategic_token_weight": 1.5,
            "state_features": {
                "table_type": "A",
                "private_hand": ["A", "Q"],
                "action_type": "challenge",
                "action_cards": [],
                "death_probability": 0.2,
            },
            "observation": {
                "pending_claim": {"declared_count": 2},
                "legal_actions": [
                    {"type": "challenge"},
                    {"type": "play_claim", "claim_rank": "A", "min_cards": 1, "max_cards": 1},
                ],
            },
        }

    def test_action_only_proxy_removes_reasoning_token_signal(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is not installed in this local environment")
        module = importlib.import_module("scripts.train_crc_gra_ablation")

        mutated = module.apply_ablation_variant(self._record(), "action_only_proxy")

        self.assertEqual(mutated["strategic_tokens"], [])
        self.assertEqual(mutated["token_penalty"], 0.0)
        self.assertFalse(mutated["reasoning_action_mismatch"])

    def test_logged_only_uses_raw_single_candidate_advantage(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is not installed in this local environment")
        module = importlib.import_module("scripts.train_crc_gra_ablation")

        class _FakePredictor:
            def predict_state_features(self, state_features: dict[str, object]) -> float:
                return 0.2

        record = module.apply_ablation_variant(self._record(), "logged_only")
        group = module._build_scored_group(
            predictor=_FakePredictor(),
            record=record,
            group_size=8,
            candidate_mode="logged_only",
            use_hicra_reward=False,
            single_candidate_advantage="raw",
        )

        self.assertEqual(group["candidate_count"], 1)
        self.assertEqual(group["candidates"][0]["candidate_role"], "logged_action")
        self.assertAlmostEqual(group["advantages"][0], group["rewards"][0])

    def test_target_only_variants_disable_proxy_dense_reward(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is not installed in this local environment")
        module = importlib.import_module("scripts.train_crc_gra_ablation")

        settings = module.resolve_ablation_settings("proxy_target_only", hicra_gamma=1.0)

        self.assertEqual(settings["candidate_mode"], "chosen_proxy")
        self.assertFalse(settings["use_phi_dense_reward"])
        self.assertFalse(settings["use_hicra_reward"])
        self.assertEqual(settings["hicra_gamma"], 0.0)

    def test_scoped_hicra_variant_keeps_proxy_credit_and_scopes_token_signal(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is not installed in this local environment")
        module = importlib.import_module("scripts.train_crc_gra_ablation")

        settings = module.resolve_ablation_settings("scoped_hicra", hicra_gamma=1.0)

        self.assertEqual(settings["candidate_mode"], "full")
        self.assertTrue(settings["use_phi_dense_reward"])
        self.assertTrue(settings["use_hicra_reward"])
        self.assertEqual(settings["hicra_gamma"], 1.0)
        self.assertTrue(settings["scope_hicra_to_logged_action"])

    def test_default_candidates_preserve_legacy_hicra_token_signal(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is not installed in this local environment")
        module = importlib.import_module("scripts.train_crc_gra_ablation")

        candidates = module._build_group_candidates(self._record(), group_size=8, candidate_mode="full")
        by_role = {candidate["candidate_role"]: candidate for candidate in candidates}

        self.assertIn("logged_action", by_role)
        self.assertIn("proxy_target", by_role)
        self.assertTrue(by_role["logged_action"]["strategic_tokens"])
        self.assertTrue(by_role["proxy_target"]["strategic_tokens"])
        self.assertEqual(by_role["proxy_target"]["token_penalty"], -0.25)

    def test_scoped_hicra_token_signal_is_scoped_to_logged_candidate(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is not installed in this local environment")
        module = importlib.import_module("scripts.train_crc_gra_ablation")

        candidates = module._build_group_candidates(
            self._record(),
            group_size=8,
            candidate_mode="full",
            scope_hicra_to_logged_action=True,
        )
        by_role = {candidate["candidate_role"]: candidate for candidate in candidates}

        self.assertIn("logged_action", by_role)
        self.assertIn("proxy_target", by_role)
        self.assertTrue(by_role["logged_action"]["strategic_tokens"])
        self.assertEqual(by_role["logged_action"]["token_penalty"], -0.25)
        self.assertEqual(by_role["proxy_target"]["strategic_tokens"], [])
        self.assertEqual(by_role["proxy_target"]["token_penalty"], 0.0)
        self.assertFalse(by_role["proxy_target"]["reasoning_action_mismatch"])

    def test_scoped_hicra_reward_penalty_no_longer_cancels_across_synthetic_candidates(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is not installed in this local environment")
        module = importlib.import_module("scripts.train_crc_gra_ablation")

        class _FlatPredictor:
            def predict_state_features(self, state_features: dict[str, object]) -> float:
                return 0.0

        group = module._build_scored_group(
            predictor=_FlatPredictor(),
            record=self._record(),
            group_size=8,
            candidate_mode="full",
            use_phi_dense_reward=True,
            use_hicra_reward=True,
            action_match_reward_weight=0.0,
            scope_hicra_to_logged_action=True,
        )
        by_role = {candidate["candidate_role"]: candidate for candidate in group["candidates"]}

        logged_reward = by_role["logged_action"]["reward_breakdown"]
        proxy_reward = by_role["proxy_target"]["reward_breakdown"]
        self.assertLess(logged_reward["hicra_penalty"], 0.0)
        self.assertEqual(proxy_reward["hicra_penalty"], 0.0)
        self.assertLess(logged_reward["total_reward"], proxy_reward["total_reward"])

    def test_scoped_hicra_soft_reward_scales_hicra_penalty(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is not installed in this local environment")
        module = importlib.import_module("scripts.train_crc_gra_ablation")

        class _FlatPredictor:
            def predict_state_features(self, state_features: dict[str, object]) -> float:
                return 0.0

        settings = module.resolve_ablation_settings("scoped_hicra_soft_reward", hicra_gamma=1.0)
        group = module._build_scored_group(
            predictor=_FlatPredictor(),
            record=self._record(),
            group_size=8,
            candidate_mode=str(settings["candidate_mode"]),
            use_phi_dense_reward=True,
            use_hicra_reward=True,
            action_match_reward_weight=0.0,
            scope_hicra_to_logged_action=bool(settings["scope_hicra_to_logged_action"]),
            hicra_reward_scale=float(settings["hicra_reward_scale"]),
        )
        logged_reward = {candidate["candidate_role"]: candidate for candidate in group["candidates"]}[
            "logged_action"
        ]["reward_breakdown"]

        self.assertEqual(settings["hicra_gamma"], 0.5)
        self.assertAlmostEqual(logged_reward["hicra_penalty"], -0.25 * 1.5 * 0.25)

    def test_scoped_hicra_token_only_keeps_tokens_but_removes_reward_penalty(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is not installed in this local environment")
        module = importlib.import_module("scripts.train_crc_gra_ablation")

        class _FlatPredictor:
            def predict_state_features(self, state_features: dict[str, object]) -> float:
                return 0.0

        settings = module.resolve_ablation_settings("scoped_hicra_token_only", hicra_gamma=1.0)
        group = module._build_scored_group(
            predictor=_FlatPredictor(),
            record=self._record(),
            group_size=8,
            candidate_mode=str(settings["candidate_mode"]),
            use_phi_dense_reward=True,
            use_hicra_reward=bool(settings["use_hicra_reward"]),
            action_match_reward_weight=0.0,
            scope_hicra_to_logged_action=bool(settings["scope_hicra_to_logged_action"]),
            hicra_reward_scale=float(settings["hicra_reward_scale"]),
        )
        by_role = {candidate["candidate_role"]: candidate for candidate in group["candidates"]}

        self.assertFalse(settings["use_hicra_reward"])
        self.assertEqual(settings["hicra_gamma"], 1.0)
        self.assertTrue(by_role["logged_action"]["strategic_tokens"])
        self.assertEqual(by_role["proxy_target"]["strategic_tokens"], [])
        self.assertEqual(by_role["logged_action"]["reward_breakdown"]["hicra_penalty"], 0.0)

    def test_scoped_hicra_mismatch_only_gates_reward_to_high_gap_mismatch(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is not installed in this local environment")
        module = importlib.import_module("scripts.train_crc_gra_ablation")

        class _FlatPredictor:
            def predict_state_features(self, state_features: dict[str, object]) -> float:
                return 0.0

        settings = module.resolve_ablation_settings("scoped_hicra_mismatch_only", hicra_gamma=1.0)
        non_mismatch_record = self._record()
        non_mismatch_record["reasoning_action_mismatch"] = False
        non_mismatch_record["ev_gap"] = 0.25
        non_mismatch_group = module._build_scored_group(
            predictor=_FlatPredictor(),
            record=non_mismatch_record,
            group_size=8,
            candidate_mode=str(settings["candidate_mode"]),
            use_phi_dense_reward=True,
            use_hicra_reward=True,
            action_match_reward_weight=0.0,
            scope_hicra_to_logged_action=bool(settings["scope_hicra_to_logged_action"]),
            hicra_reward_scale=float(settings["hicra_reward_scale"]),
            hicra_mismatch_only=bool(settings["hicra_mismatch_only"]),
            hicra_ev_gap_threshold=float(settings["hicra_ev_gap_threshold"]),
        )
        mismatch_group = module._build_scored_group(
            predictor=_FlatPredictor(),
            record=self._record(),
            group_size=8,
            candidate_mode=str(settings["candidate_mode"]),
            use_phi_dense_reward=True,
            use_hicra_reward=True,
            action_match_reward_weight=0.0,
            scope_hicra_to_logged_action=bool(settings["scope_hicra_to_logged_action"]),
            hicra_reward_scale=float(settings["hicra_reward_scale"]),
            hicra_mismatch_only=bool(settings["hicra_mismatch_only"]),
            hicra_ev_gap_threshold=float(settings["hicra_ev_gap_threshold"]),
        )

        non_mismatch_logged = {candidate["candidate_role"]: candidate for candidate in non_mismatch_group["candidates"]}[
            "logged_action"
        ]
        mismatch_logged = {candidate["candidate_role"]: candidate for candidate in mismatch_group["candidates"]}[
            "logged_action"
        ]
        self.assertTrue(settings["hicra_mismatch_only"])
        self.assertEqual(non_mismatch_logged["reward_breakdown"]["hicra_penalty"], 0.0)
        self.assertLess(mismatch_logged["reward_breakdown"]["hicra_penalty"], 0.0)

    def test_scoped_hicra_reward_only_disables_token_loss_weighting(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is not installed in this local environment")
        module = importlib.import_module("scripts.train_crc_gra_ablation")

        settings = module.resolve_ablation_settings("scoped_hicra_reward_only", hicra_gamma=1.0)

        self.assertTrue(settings["scope_hicra_to_logged_action"])
        self.assertTrue(settings["use_hicra_reward"])
        self.assertEqual(settings["hicra_gamma"], 0.0)
        self.assertEqual(settings["hicra_reward_scale"], 1.0)

    def test_scoped_hicra_weighted_sum_uses_explicit_reward_weights(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is not installed in this local environment")
        module = importlib.import_module("scripts.train_crc_gra_ablation")

        class _FlatPredictor:
            def predict_state_features(self, state_features: dict[str, object]) -> float:
                return 0.0

        settings = module.resolve_ablation_settings("scoped_hicra_weighted_sum", hicra_gamma=1.0)
        group = module._build_scored_group(
            predictor=_FlatPredictor(),
            record=self._record(),
            group_size=8,
            candidate_mode=str(settings["candidate_mode"]),
            use_phi_dense_reward=True,
            use_hicra_reward=True,
            action_match_reward_weight=0.0,
            scope_hicra_to_logged_action=bool(settings["scope_hicra_to_logged_action"]),
            reward_component_weights=dict(settings["reward_component_weights"]),
        )
        logged_reward = {candidate["candidate_role"]: candidate for candidate in group["candidates"]}[
            "logged_action"
        ]["reward_breakdown"]

        self.assertEqual(settings["reward_component_weights"]["hicra_penalty"], 0.5)
        self.assertAlmostEqual(logged_reward["weighted_hicra_penalty"], -0.25 * 1.5 * 0.5)
        self.assertAlmostEqual(logged_reward["total_reward"], logged_reward["weighted_hicra_penalty"])

    def test_scoped_hicra_adv_clip_clips_group_relative_advantages(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is not installed in this local environment")
        module = importlib.import_module("scripts.train_crc_gra_ablation")

        class _RolePredictor:
            def predict_state_features(self, state_features: dict[str, object]) -> float:
                return 10.0 if state_features.get("action_type") == "play_claim" else -10.0

        settings = module.resolve_ablation_settings("scoped_hicra_adv_clip", hicra_gamma=1.0)
        group = module._build_scored_group(
            predictor=_RolePredictor(),
            record=self._record(),
            group_size=8,
            candidate_mode=str(settings["candidate_mode"]),
            use_phi_dense_reward=True,
            use_hicra_reward=True,
            scope_hicra_to_logged_action=bool(settings["scope_hicra_to_logged_action"]),
            advantage_clip=float(settings["advantage_clip"]),
        )

        self.assertEqual(settings["advantage_clip"], 0.3)
        self.assertTrue(all(abs(float(item)) <= 0.3 for item in group["advantages"]))
        self.assertGreater(group["raw_advantage_span"], group["advantage_span"])

    def test_scoped_hicra_high_kl_applies_variant_kl_multiplier(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is not installed in this local environment")
        module = importlib.import_module("scripts.train_crc_gra_ablation")

        settings = module.resolve_ablation_settings("scoped_hicra_high_kl", hicra_gamma=1.0)

        self.assertTrue(settings["scope_hicra_to_logged_action"])
        self.assertEqual(settings["kl_beta_multiplier"], 4.0)

    def test_hicra_advantage_reshape_dampens_negative_strategic_token_penalty(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is not installed in this local environment")
        import torch

        module = importlib.import_module("scripts.train_crc_gra_ablation")
        logits = torch.zeros((1, 3, 5), dtype=torch.float32)
        input_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
        label_token_mask = torch.tensor([1.0, 1.0], dtype=torch.float32)
        token_weight_mask = torch.tensor([1.0, 0.0], dtype=torch.float32)

        metrics = module.compute_candidate_loss_terms(
            logits=logits,
            ref_logits=None,
            input_ids=input_ids,
            label_token_mask=label_token_mask,
            token_weight_mask=token_weight_mask,
            advantage=-1.0,
            kl_beta=0.0,
            hicra_gamma=0.0,
            token_advantage_alpha=0.2,
            token_advantage_mode="hicra",
        )

        self.assertAlmostEqual(metrics["strategic_token_advantage_mean"], -0.8, places=6)
        self.assertAlmostEqual(metrics["token_advantage_mean"], -0.9, places=6)

    def test_hicra_advantage_rescue_variant_settings_disable_old_penalty_paths(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is not installed in this local environment")
        module = importlib.import_module("scripts.train_crc_gra_ablation")

        settings = module.resolve_ablation_settings("scoped_hicra_adv_reshape", hicra_gamma=1.0)

        self.assertTrue(settings["scope_hicra_to_logged_action"])
        self.assertFalse(settings["use_hicra_reward"])
        self.assertEqual(settings["hicra_reward_scale"], 0.0)
        self.assertEqual(settings["hicra_gamma"], 0.0)
        self.assertEqual(settings["token_advantage_mode"], "hicra")
        self.assertEqual(settings["token_advantage_alpha"], 0.2)

    def test_hicra_clean_filter_removes_high_gap_mismatch_records(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is not installed in this local environment")
        module = importlib.import_module("scripts.train_crc_gra_ablation")

        clean = self._record()
        clean["game_id"] = "clean"
        clean["reasoning_action_mismatch"] = False
        dirty = self._record()
        dirty["game_id"] = "dirty"
        dirty["reasoning_action_mismatch"] = True
        dirty["ev_gap"] = 0.25

        filtered = module.filter_records_for_ablation(
            [clean, dirty],
            record_filter_mode="hicra_clean",
            ev_gap_threshold=0.15,
        )

        self.assertEqual([record["game_id"] for record in filtered], ["clean"])

    def test_hicra_sequence_dpo_variant_uses_sequence_preference_objective(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is not installed in this local environment")
        module = importlib.import_module("scripts.train_crc_gra_ablation")

        settings = module.resolve_ablation_settings("hicra_sequence_dpo", hicra_gamma=1.0)

        self.assertEqual(settings["training_objective"], "dpo")
        self.assertFalse(settings["use_hicra_reward"])
        self.assertEqual(settings["hicra_gamma"], 0.0)
        self.assertEqual(settings["token_advantage_mode"], "none")
        self.assertEqual(settings["preference_pair_mode"], "best_vs_logged")

    def test_standard_dpo_baseline_uses_proxy_only_preference_without_hicra(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is not installed in this local environment")
        module = importlib.import_module("scripts.train_crc_gra_ablation")

        settings = module.resolve_ablation_settings("standard_dpo_baseline", hicra_gamma=1.0)
        mutated = module.apply_ablation_variant(self._record(), "standard_dpo_baseline")

        self.assertEqual(settings["training_objective"], "dpo")
        self.assertEqual(settings["preference_pair_mode"], "best_vs_worst")
        self.assertEqual(settings["target_source"], "proxy_credit_preference_pair")
        self.assertEqual(settings["dpo_beta"], 0.02)
        self.assertFalse(settings["use_hicra_reward"])
        self.assertEqual(settings["hicra_gamma"], 0.0)
        self.assertEqual(settings["token_advantage_mode"], "none")
        self.assertEqual(settings["reward_component_weights"]["action_match_reward"], 0.0)
        self.assertEqual(settings["reward_component_weights"]["phi_dense_reward"], 1.0)
        self.assertEqual(mutated["strategic_tokens"], [])
        self.assertEqual(mutated["token_penalty"], 0.0)
        self.assertFalse(mutated["reasoning_action_mismatch"])

    def test_expanded_action_proxy_variant_keeps_action_only_training_signal(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is not installed in this local environment")
        module = importlib.import_module("scripts.train_crc_gra_ablation")

        settings = module.resolve_ablation_settings("expanded_action_proxy", hicra_gamma=1.0)
        mutated = module.apply_ablation_variant(self._record(), "expanded_action_proxy")

        self.assertEqual(settings["candidate_mode"], "expanded_proxy")
        self.assertFalse(settings["use_hicra_reward"])
        self.assertEqual(settings["hicra_gamma"], 0.0)
        self.assertEqual(mutated["strategic_tokens"], [])
        self.assertEqual(mutated["token_penalty"], 0.0)
        self.assertFalse(mutated["reasoning_action_mismatch"])

    def test_conservative_expanded_action_proxy_uses_conservative_candidate_mode(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is not installed in this local environment")
        module = importlib.import_module("scripts.train_crc_gra_ablation")

        settings = module.resolve_ablation_settings("conservative_expanded_action_proxy", hicra_gamma=1.0)
        mutated = module.apply_ablation_variant(self._record(), "conservative_expanded_action_proxy")

        self.assertEqual(settings["candidate_mode"], "conservative_expanded_proxy")
        self.assertFalse(settings["use_hicra_reward"])
        self.assertEqual(settings["hicra_gamma"], 0.0)
        self.assertEqual(settings["target_source"], "conservative_expanded_proxy_ranked_legal_candidates")
        self.assertEqual(mutated["strategic_tokens"], [])
        self.assertEqual(mutated["token_penalty"], 0.0)
        self.assertFalse(mutated["reasoning_action_mismatch"])

    def test_conservative_expanded_action_scope_uses_action_labels_for_synthetic_candidates(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is not installed in this local environment")
        module = importlib.import_module("scripts.train_crc_gra_ablation")

        settings = module.resolve_ablation_settings("conservative_expanded_action_scope", hicra_gamma=1.0)

        self.assertEqual(settings["candidate_mode"], "conservative_expanded_proxy")
        self.assertEqual(settings["label_scope"], "assistant")
        self.assertEqual(settings["synthetic_candidate_label_scope"], "action")
        self.assertFalse(settings["use_hicra_reward"])
        self.assertEqual(settings["hicra_gamma"], 0.0)
        self.assertEqual(
            module.resolve_candidate_label_scope(settings, {"candidate_role": "logged_action"}),
            "assistant",
        )
        self.assertEqual(
            module.resolve_candidate_label_scope(settings, {"candidate_role": "proxy_target"}),
            "assistant",
        )
        self.assertEqual(
            module.resolve_candidate_label_scope(settings, {"candidate_role": "truthful_play"}),
            "action",
        )
        self.assertEqual(
            module.resolve_candidate_label_scope(settings, {"candidate_role": "bluff_play"}),
            "action",
        )
        self.assertEqual(
            module.resolve_candidate_label_scope(settings, {"candidate_role": "legal_challenge"}),
            "action",
        )

    def test_action_only_assistant_target_renders_without_reasoning(self) -> None:
        token_alignment = importlib.import_module("liars_game_engine.analysis.token_alignment")

        rendered = token_alignment.build_action_only_assistant_response_text(self._record())
        decoded = json.loads(rendered)

        self.assertEqual(set(decoded.keys()), {"Action"})
        self.assertEqual(decoded["Action"]["type"], "challenge")

    def test_conservative_expanded_action_json_settings_use_action_json_targets(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is not installed in this local environment")
        module = importlib.import_module("scripts.train_crc_gra_ablation")

        settings = module.resolve_ablation_settings("conservative_expanded_action_json", hicra_gamma=1.0)

        self.assertEqual(settings["candidate_mode"], "conservative_expanded_proxy")
        self.assertFalse(settings["use_hicra_reward"])
        self.assertEqual(settings["hicra_gamma"], 0.0)
        self.assertEqual(settings["assistant_target_mode"], "action_json")

    def test_action_json_candidate_training_example_omits_reasoning(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is not installed in this local environment")
        module = importlib.import_module("scripts.train_crc_gra_ablation")
        settings = module.resolve_ablation_settings("conservative_expanded_action_json", hicra_gamma=1.0)

        candidate = module.apply_candidate_assistant_target_mode(
            self._record(),
            assistant_target_mode=str(settings["assistant_target_mode"]),
        )
        example = module.prepare_candidate_training_example(candidate, _CharTokenizer())
        assistant_text = str(example["alignment_metadata"]["assistant_text"])

        self.assertNotIn("Reasoning", assistant_text)
        self.assertIn("Action", assistant_text)
        self.assertGreater(example["active_label_count"], 0)

    def test_action_only_no_match_bonus_removes_only_action_match_reward(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is not installed in this local environment")
        module = importlib.import_module("scripts.train_crc_gra_ablation")

        settings = module.resolve_ablation_settings("action_only_proxy_no_match_bonus", hicra_gamma=1.0)
        mutated = module.apply_ablation_variant(self._record(), "action_only_proxy_no_match_bonus")

        self.assertEqual(settings["candidate_mode"], "full")
        self.assertFalse(settings["use_hicra_reward"])
        self.assertEqual(settings["hicra_gamma"], 0.0)
        self.assertEqual(settings["reward_component_weights"]["action_match_reward"], 0.0)
        self.assertEqual(settings["reward_component_weights"]["phi_dense_reward"], 1.0)
        self.assertEqual(mutated["strategic_tokens"], [])
        self.assertEqual(mutated["token_penalty"], 0.0)
        self.assertFalse(mutated["reasoning_action_mismatch"])

    def test_action_loss_only_variant_trains_only_action_span(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is not installed in this local environment")
        module = importlib.import_module("scripts.train_crc_gra_ablation")

        class _CharTokenizer:
            pad_token = "<pad>"
            eos_token = "<eos>"

            def __call__(
                self,
                text: str,
                add_special_tokens: bool = False,
                return_offsets_mapping: bool = False,
            ) -> dict[str, object]:
                return {
                    "input_ids": [min(255, ord(char)) for char in text],
                    "attention_mask": [1 for _ in text],
                    "offset_mapping": [(index, index + 1) for index, _ in enumerate(text)],
                }

        settings = module.resolve_ablation_settings("action_only_proxy_action_loss_only", hicra_gamma=1.0)
        mutated = module.apply_ablation_variant(self._record(), "action_only_proxy_action_loss_only")
        assistant_example = module.prepare_candidate_training_example(
            mutated,
            _CharTokenizer(),
            label_scope="assistant",
        )
        action_example = module.prepare_candidate_training_example(
            mutated,
            _CharTokenizer(),
            label_scope=str(settings["label_scope"]),
        )

        self.assertEqual(settings["candidate_mode"], "full")
        self.assertFalse(settings["use_hicra_reward"])
        self.assertEqual(settings["hicra_gamma"], 0.0)
        self.assertEqual(settings["label_scope"], "action")
        self.assertLess(action_example["active_label_count"], assistant_example["active_label_count"])
        self.assertGreater(action_example["active_label_count"], 0)
        self.assertEqual(mutated["strategic_tokens"], [])
        self.assertEqual(mutated["token_penalty"], 0.0)
        self.assertFalse(mutated["reasoning_action_mismatch"])

    def test_format_plus_action_variant_masks_reasoning_content_only(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is not installed in this local environment")
        module = importlib.import_module("scripts.train_crc_gra_ablation")

        class _CharTokenizer:
            pad_token = "<pad>"
            eos_token = "<eos>"

            def __call__(
                self,
                text: str,
                add_special_tokens: bool = False,
                return_offsets_mapping: bool = False,
            ) -> dict[str, object]:
                return {
                    "input_ids": [min(255, ord(char)) for char in text],
                    "attention_mask": [1 for _ in text],
                    "offset_mapping": [(index, index + 1) for index, _ in enumerate(text)],
                }

        settings = module.resolve_ablation_settings("action_only_proxy_format_plus_action_loss", hicra_gamma=1.0)
        mutated = module.apply_ablation_variant(self._record(), "action_only_proxy_format_plus_action_loss")
        assistant_example = module.prepare_candidate_training_example(
            mutated,
            _CharTokenizer(),
            label_scope="assistant",
        )
        action_example = module.prepare_candidate_training_example(
            mutated,
            _CharTokenizer(),
            label_scope="action",
        )
        format_action_example = module.prepare_candidate_training_example(
            mutated,
            _CharTokenizer(),
            label_scope=str(settings["label_scope"]),
        )

        self.assertEqual(settings["candidate_mode"], "full")
        self.assertFalse(settings["use_hicra_reward"])
        self.assertEqual(settings["hicra_gamma"], 0.0)
        self.assertEqual(settings["label_scope"], "format_action")
        self.assertGreater(format_action_example["active_label_count"], action_example["active_label_count"])
        self.assertLess(format_action_example["active_label_count"], assistant_example["active_label_count"])
        self.assertEqual(mutated["strategic_tokens"], [])
        self.assertEqual(mutated["token_penalty"], 0.0)
        self.assertFalse(mutated["reasoning_action_mismatch"])

    def test_expanded_action_proxy_adds_concrete_truthful_and_bluff_candidates(self) -> None:
        module = importlib.import_module("liars_game_engine.analysis.candidate_expansion")

        class _CardAwarePredictor:
            def predict_state_features(self, state_features: dict[str, object]) -> float:
                cards = state_features.get("action_cards", [])
                if cards == ["A"]:
                    return 0.5
                if cards == ["K"]:
                    return -0.4
                return 0.0

        record = self._record()
        record["action"] = {"type": "challenge", "claim_rank": "", "cards": []}
        record["proxy_target_action"] = {"type": "play_claim", "claim_rank": "A", "cards": ["A"]}
        record["state_features"]["private_hand"] = ["A", "K", "Q"]
        record["observation"]["table_type"] = "A"
        record["observation"]["legal_actions"] = [
            {"type": "challenge"},
            {"type": "play_claim", "claim_rank": "A", "min_cards": 1, "max_cards": 2},
        ]

        group = module.build_expanded_scored_candidates(
            score_candidate=lambda candidate: _CardAwarePredictor().predict_state_features(candidate["state_features"]),
            record=record,
            group_size=4,
        )
        roles = {candidate["candidate_role"] for candidate in group["selected_candidates"]}

        self.assertLessEqual(len(group["selected_candidates"]), 4)
        self.assertIn("logged_action", roles)
        self.assertIn("proxy_target", roles)
        self.assertIn("bluff_play", roles)
        self.assertIn("truthful_play", {candidate["candidate_role"] for candidate in group["expanded_candidate_pool"]})

    def test_dpo_preference_pair_prefers_higher_reward_candidate_against_logged(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is not installed in this local environment")
        module = importlib.import_module("scripts.train_crc_gra_ablation")

        group = {
            "candidates": [
                {
                    "candidate_role": "logged_action",
                    "reward_breakdown": {"total_reward": -0.1},
                },
                {
                    "candidate_role": "proxy_target",
                    "reward_breakdown": {"total_reward": 0.4},
                },
            ]
        }

        pair = module.build_dpo_preference_pair(group, preference_pair_mode="best_vs_logged")

        self.assertEqual(pair["chosen"]["candidate_role"], "proxy_target")
        self.assertEqual(pair["rejected"]["candidate_role"], "logged_action")
        self.assertAlmostEqual(pair["preference_margin"], 0.5)

    def test_compute_dpo_pair_loss_uses_policy_and_reference_logprob_gap(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is not installed in this local environment")
        import torch

        module = importlib.import_module("scripts.train_crc_gra_ablation")

        chosen_policy = torch.tensor(-1.0)
        rejected_policy = torch.tensor(-3.0)
        chosen_ref = torch.tensor(-2.0)
        rejected_ref = torch.tensor(-2.5)

        metrics = module.compute_dpo_pair_loss_from_logps(
            chosen_policy_logp=chosen_policy,
            rejected_policy_logp=rejected_policy,
            chosen_ref_logp=chosen_ref,
            rejected_ref_logp=rejected_ref,
            beta=0.1,
        )

        self.assertGreater(metrics["preference_logit"], 0.0)
        self.assertLess(float(metrics["loss"].detach().cpu().item()), 0.7)

    def test_dry_run_records_ablation_metadata(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is not installed in this local environment")
        module = importlib.import_module("scripts.train_crc_gra_ablation")

        class _FakePredictor:
            def __init__(self, model_path: str | Path, output_mode: str) -> None:
                self.model_path = Path(model_path)
                self.output_mode = output_mode

            def predict_state_features(self, state_features: dict[str, object]) -> float:
                action_type = str(state_features.get("action_type", ""))
                return 0.3 if action_type == "play_claim" else -0.1

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_path = root / "savi.jsonl"
            dataset_path.write_text(json.dumps(self._record()) + "\n", encoding="utf-8")
            proxy_path = root / "proxy.pt"
            proxy_path.write_text("stub", encoding="utf-8")

            with patch.object(module, "ProxyValuePredictor", _FakePredictor):
                summary = module.run_alignment_dry_run(
                    dataset_path=dataset_path,
                    model_path=proxy_path,
                    ablation_variant="no_token_localization",
                    hicra_gamma=1.0,
                )

        self.assertEqual(summary["ablation_variant"], "no_token_localization")
        self.assertEqual(summary["hicra_gamma"], 0.0)
        self.assertTrue(summary["use_hicra_reward"])
        self.assertEqual(summary["groups"][0]["candidate_mode"], "full")


if __name__ == "__main__":
    unittest.main()
