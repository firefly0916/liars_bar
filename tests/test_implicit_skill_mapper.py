import importlib
import unittest

from liars_game_engine.analysis.train_value_proxy import build_value_proxy_feature_context


class ImplicitSkillMapperTest(unittest.TestCase):
    def test_map_implicit_skill_detects_truthful_action(self) -> None:
        module = importlib.import_module("liars_game_engine.analysis.implicit_skill_mapper")
        state_features = build_value_proxy_feature_context(
            observation={
                "player_id": "p1",
                "phase": "turn_start",
                "table_type": "A",
                "must_call_liar": False,
                "alive_players": ["p1", "p2"],
                "private_hand": ["A", "JOKER", "Q"],
                "player_states": {"p1": {"death_probability": 0.1}},
                "pending_claim": None,
            },
            player_id="p1",
            action={"type": "play_claim", "claim_rank": "A", "cards": ["A"]},
        )

        label = module.map_implicit_skill(
            state_features=state_features,
            action={"type": "play_claim", "claim_rank": "A", "cards": ["A"]},
        )
        self.assertEqual(label, "Truthful_Action")

    def test_map_implicit_skill_detects_calculated_and_aggressive_bluff(self) -> None:
        module = importlib.import_module("liars_game_engine.analysis.implicit_skill_mapper")
        state_features = build_value_proxy_feature_context(
            observation={
                "player_id": "p1",
                "phase": "turn_start",
                "table_type": "A",
                "must_call_liar": False,
                "alive_players": ["p1", "p2", "p3"],
                "private_hand": ["Q", "K", "Q"],
                "player_states": {"p1": {"death_probability": 0.2}},
                "pending_claim": None,
            },
            player_id="p1",
            action={"type": "play_claim", "claim_rank": "A", "cards": ["Q"]},
        )

        cautious = module.map_implicit_skill(
            state_features=state_features,
            action={"type": "play_claim", "claim_rank": "A", "cards": ["Q"]},
        )
        aggressive = module.map_implicit_skill(
            state_features=state_features,
            action={"type": "play_claim", "claim_rank": "A", "cards": ["Q", "K"]},
        )
        self.assertEqual(cautious, "Calculated_Bluff")
        self.assertEqual(aggressive, "Aggressive_Deception")

    def test_map_implicit_skill_detects_skepticism_and_drain(self) -> None:
        module = importlib.import_module("liars_game_engine.analysis.implicit_skill_mapper")
        skeptical_state = build_value_proxy_feature_context(
            observation={
                "player_id": "p1",
                "phase": "response_window",
                "table_type": "A",
                "must_call_liar": False,
                "alive_players": ["p1", "p2"],
                "private_hand": ["A", "Q"],
                "player_states": {"p1": {"death_probability": 0.55}},
                "pending_claim": {"actor_id": "p2", "claim_rank": "A", "declared_count": 3},
            },
            player_id="p1",
            action={"type": "challenge", "cards": []},
        )
        drain_state = build_value_proxy_feature_context(
            observation={
                "player_id": "p1",
                "phase": "turn_start",
                "table_type": "A",
                "must_call_liar": False,
                "alive_players": ["p1", "p2", "p3"],
                "private_hand": ["A", "Q", "K", "JOKER"],
                "player_states": {"p1": {"death_probability": 0.15}},
                "pending_claim": None,
            },
            player_id="p1",
            action={"type": "play_claim", "claim_rank": "A", "cards": ["A", "Q"]},
        )

        skepticism = module.map_implicit_skill(
            state_features=skeptical_state,
            action={"type": "challenge", "cards": []},
        )
        drain = module.map_implicit_skill(
            state_features=drain_state,
            action={"type": "play_claim", "claim_rank": "A", "cards": ["A", "Q"]},
        )
        self.assertEqual(skepticism, "Logical_Skepticism")
        self.assertEqual(drain, "Strategic_Drain")


if __name__ == "__main__":
    unittest.main()
