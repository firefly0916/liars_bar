import unittest

from liars_game_engine.analysis.candidate_expansion import build_expanded_scored_candidates


class ConservativeCandidateExpansionTest(unittest.TestCase):
    def _record(self) -> dict[str, object]:
        return {
            "game_id": "g1",
            "turn": 7,
            "player_id": "p1",
            "thought": "I should keep pressure without taking a blind challenge.",
            "action": {"type": "play_claim", "claim_rank": "A", "cards": ["A"]},
            "proxy_target_action": {"type": "play_claim", "claim_rank": "A", "cards": ["Q"]},
            "state_features": {"table_type": "A"},
            "observation": {
                "table_type": "A",
                "private_hand": ["A", "Q", "K"],
                "legal_actions": [
                    {"type": "play_claim", "claim_rank": "A", "min_cards": 1, "max_cards": 3},
                    {"type": "challenge"},
                ],
            },
        }

    def test_conservative_selection_excludes_low_proxy_challenge(self) -> None:
        scores = {
            "logged_action": 0.30,
            "proxy_target": 0.55,
            "truthful_play": 0.30,
            "bluff_play": 0.50,
            "legal_challenge": 0.05,
        }

        result = build_expanded_scored_candidates(
            score_candidate=lambda candidate: scores[str(candidate["candidate_role"])],
            record=self._record(),
            group_size=4,
            selection_mode="conservative",
        )

        roles = [str(candidate["candidate_role"]) for candidate in result["selected_candidates"]]
        self.assertNotIn("legal_challenge", roles)
        self.assertIn("logged_action", roles)
        self.assertIn("proxy_target", roles)
        self.assertLessEqual(len(roles), 4)

    def test_conservative_selection_keeps_high_proxy_challenge(self) -> None:
        scores = {
            "logged_action": 0.30,
            "proxy_target": 0.45,
            "truthful_play": 0.30,
            "bluff_play": 0.10,
            "legal_challenge": 0.70,
        }

        result = build_expanded_scored_candidates(
            score_candidate=lambda candidate: scores[str(candidate["candidate_role"])],
            record=self._record(),
            group_size=4,
            selection_mode="conservative",
        )

        roles = [str(candidate["candidate_role"]) for candidate in result["selected_candidates"]]
        self.assertIn("legal_challenge", roles)
        self.assertLessEqual(len(roles), 4)


if __name__ == "__main__":
    unittest.main()
