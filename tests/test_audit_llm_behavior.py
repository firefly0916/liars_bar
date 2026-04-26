import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class AuditLlmBehaviorTest(unittest.TestCase):
    def test_classify_reasoning_confidence_distinguishes_strong_from_hedged(self) -> None:
        module = importlib.import_module("liars_game_engine.analysis.audit_llm_behavior")

        strong = module.classify_reasoning_confidence("必须立刻质疑，对方明显在诈唬，我肯定这步最优。")
        hedged = module.classify_reasoning_confidence("也许可以质疑，但我不太确定，先保守一点。")

        self.assertEqual(strong["label"], "strong")
        self.assertGreater(strong["score"], 0)
        self.assertIn("必须", strong["strong_signals"])
        self.assertEqual(hedged["label"], "hedged")
        self.assertGreater(hedged["hedge_count"], 0)

    def test_run_llm_behavior_audit_writes_conflict_report(self) -> None:
        module = importlib.import_module("liars_game_engine.analysis.audit_llm_behavior")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            log_dir = root / "logs" / "llm_drill"
            games_dir = log_dir / "games"
            games_dir.mkdir(parents=True)
            (log_dir / "summary.json").write_text(
                json.dumps({"llm_player_id": "p1", "total_games": 1}, ensure_ascii=False),
                encoding="utf-8",
            )
            (games_dir / "llm-drill-01.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "turn": 1,
                                "player_id": "p1",
                                "thought": "必须立刻质疑，对方明显在诈唬，我肯定这步最优。",
                                "raw_output": '{"Reasoning":"必须立刻质疑，对方明显在诈唬，我肯定这步最优。","Action":{"type":"challenge"}}',
                                "action": {"type": "challenge", "cards": []},
                                "observation": {
                                    "player_id": "p1",
                                    "phase": "response_window",
                                    "table_type": "A",
                                    "must_call_liar": False,
                                    "alive_players": ["p1", "p2", "p3"],
                                    "private_hand": ["Q", "K"],
                                    "player_states": {"p1": {"death_probability": 0.5}},
                                    "pending_claim": {"actor_id": "p2", "claim_rank": "A", "declared_count": 2},
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "turn": 2,
                                "player_id": "p1",
                                "thought": "也许可以继续跟注，但我不太确定。",
                                "raw_output": '{"Reasoning":"也许可以继续跟注，但我不太确定。","Action":{"type":"play_claim","claim_rank":"A","cards":["K"]}}',
                                "action": {"type": "play_claim", "claim_rank": "A", "cards": ["K"]},
                                "observation": {
                                    "player_id": "p1",
                                    "phase": "turn_start",
                                    "table_type": "A",
                                    "must_call_liar": False,
                                    "alive_players": ["p1", "p2"],
                                    "private_hand": ["A", "K"],
                                    "player_states": {"p1": {"death_probability": 0.1}},
                                    "pending_claim": None,
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "turn": 3,
                                "player_id": "p2",
                                "thought": "mock turn",
                                "raw_output": "",
                                "action": {"type": "challenge", "cards": []},
                                "observation": {"player_id": "p2"},
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            output_dir = root / "audit_output"

            class _FakePredictor:
                def __init__(self, model_path: str | Path, output_mode: str) -> None:
                    self.model_path = Path(model_path)
                    self.output_mode = output_mode

                def predict_state_features(self, state_features: dict[str, object]) -> float:
                    if state_features.get("action_type") == "challenge":
                        return -0.25
                    return -0.2

            with patch.object(module, "ProxyValuePredictor", _FakePredictor):
                summary = module.run_llm_behavior_audit(
                    log_root=log_dir,
                    model_path=root / "value_proxy_mlp_distill.pt",
                    output_dir=output_dir,
                    phi_threshold=-0.1,
                )

            self.assertEqual(summary["llm_player_id"], "p1")
            self.assertEqual(summary["audited_turn_count"], 2)
            self.assertEqual(summary["conflict_count"], 1)
            self.assertTrue(Path(summary["conflict_cases_path"]).exists())
            self.assertTrue(Path(summary["summary_path"]).exists())

            conflict_lines = Path(summary["conflict_cases_path"]).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(conflict_lines), 1)
            conflict = json.loads(conflict_lines[0])
            self.assertEqual(conflict["turn"], 1)
            self.assertEqual(conflict["reasoning_confidence"]["label"], "strong")
            self.assertLess(conflict["phi_pred"], -0.1)


if __name__ == "__main__":
    unittest.main()
