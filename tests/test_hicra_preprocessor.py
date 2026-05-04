import importlib
import json
import tempfile
import unittest
from pathlib import Path


class HicraPreprocessorTest(unittest.TestCase):
    def test_extract_strategic_tokens_finds_risk_bluff_game_and_skepticism_language(self) -> None:
        module = importlib.import_module("liars_game_engine.analysis.hicra_preprocessor")
        tokens = module.extract_strategic_tokens(
            "The risk is high, but this bluff can win the mind game if I challenge that suspicious claim."
        )

        labels = [item["label"] for item in tokens]
        self.assertIn("risk", labels)
        self.assertIn("bluff", labels)
        self.assertIn("game", labels)
        self.assertIn("skepticism", labels)
        self.assertTrue(all(float(item["weight"]) >= 1.0 for item in tokens))

    def test_build_hicra_sample_adds_skill_and_token_weights(self) -> None:
        module = importlib.import_module("liars_game_engine.analysis.hicra_preprocessor")
        sample = module.build_hicra_sample(
            record={
                "game_id": "g1",
                "turn": 3,
                "player_id": "p1",
                "thought": "Need to manage risk and sell this bluff in a longer game.",
                "action": {"type": "play_claim", "claim_rank": "A", "cards": ["Q"]},
                "state_features": {
                    "table_type": "A",
                    "private_hand": ["Q", "K"],
                    "action_type": "play_claim",
                    "action_cards": ["Q"],
                    "death_probability": 0.2,
                    "pending_claim_declared_count": 0,
                },
            },
            implicit_skill_label="Calculated_Bluff",
        )

        self.assertEqual(sample["implicit_skill_label"], "Calculated_Bluff")
        self.assertGreaterEqual(sample["strategic_token_weight"], 1.5)
        self.assertGreaterEqual(len(sample["strategic_tokens"]), 2)
        self.assertEqual(sample["action"]["type"], "play_claim")

    def test_build_alignment_sample_marks_reasoning_action_mismatch(self) -> None:
        module = importlib.import_module("liars_game_engine.analysis.hicra_preprocessor")
        sample = module.build_alignment_sample(
            record={
                "game_id": "g-high-gap",
                "turn": 17,
                "player_id": "p1",
                "thought": "The risk is high, but this mind game makes me doubt the previous claim and want to challenge.",
                "action": {"type": "challenge", "cards": []},
                "observation": {
                    "player_id": "p1",
                    "phase": "response_window",
                    "table_type": "K",
                    "must_call_liar": False,
                    "alive_players": ["p1", "p2"],
                    "private_hand": ["Q", "K"],
                    "player_states": {"p1": {"death_probability": 0.2}},
                    "pending_claim": {"actor_id": "p2", "claim_rank": "K", "declared_count": 1},
                },
            },
            audit_row={
                "game_id": "g-high-gap",
                "turn": "17",
                "player_id": "p1",
                "action_type": "challenge",
                "action_claim_rank": "",
                "action_cards": "",
                "phi_chosen": "-0.06",
                "best_action_type": "play_claim",
                "best_action_claim_rank": "K",
                "best_action_cards": "K",
                "phi_best": "0.34",
                "ev_gap": "0.40",
                "is_potential_point": "1",
            },
            potential_point_threshold=0.15,
        )

        self.assertEqual(sample["sample_type"], "Reasoning-Action Mismatch")
        self.assertTrue(sample["reasoning_action_mismatch"])
        self.assertAlmostEqual(sample["token_penalty"], -0.4, places=6)
        self.assertAlmostEqual(sample["ev_gap"], 0.4, places=6)
        self.assertEqual(sample["proxy_target_action"]["type"], "play_claim")
        self.assertEqual(sample["proxy_target_action"]["claim_rank"], "K")
        self.assertEqual(sample["proxy_target_action"]["cards"], ["K"])
        self.assertEqual(sample["chosen_implicit_skill_label"], "Logical_Skepticism")
        self.assertEqual(sample["proxy_target_implicit_skill_label"], "Truthful_Action")
        self.assertGreaterEqual(len(sample["strategic_tokens"]), 2)
        self.assertTrue(all("penalty_signal" in token for token in sample["strategic_tokens"]))

    def test_export_savi_alignment_dataset_writes_expected_jsonl(self) -> None:
        module = importlib.import_module("liars_game_engine.analysis.hicra_preprocessor")
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "savi_alignment_train.jsonl"
            path = module.export_savi_alignment_dataset(
                samples=[
                    {
                        "game_id": "g1",
                        "turn": 1,
                        "player_id": "p1",
                        "sample_type": "Standard",
                        "ev_gap": 0.0,
                    },
                    {
                        "game_id": "g2",
                        "turn": 2,
                        "player_id": "p1",
                        "sample_type": "Reasoning-Action Mismatch",
                        "ev_gap": 0.2,
                    },
                ],
                output_path=output_path,
            )

            self.assertEqual(path, output_path)
            self.assertTrue(path.exists())
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            payload = json.loads(lines[1])
            self.assertEqual(payload["sample_type"], "Reasoning-Action Mismatch")
            self.assertAlmostEqual(payload["ev_gap"], 0.2, places=6)


if __name__ == "__main__":
    unittest.main()
