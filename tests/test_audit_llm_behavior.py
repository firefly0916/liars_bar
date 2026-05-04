import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from liars_game_engine.analysis.shapley_analyzer import GameTrajectory, TurnTrajectory
from liars_game_engine.engine.game_state import ActionModel


class AuditLlmBehaviorTest(unittest.TestCase):
    def test_classify_reasoning_confidence_distinguishes_strong_from_hedged(self) -> None:
        module = importlib.import_module("liars_game_engine.analysis.audit_llm_behavior")

        strong = module.classify_reasoning_confidence("We must challenge immediately. This is clearly the optimal line.")
        hedged = module.classify_reasoning_confidence("Maybe challenge, but I am uncertain and prefer a conservative line.")

        self.assertEqual(strong["label"], "strong")
        self.assertGreater(strong["score"], 0)
        self.assertIn("must", strong["strong_signals"])
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
                                "thought": "We must challenge immediately. This is clearly the optimal line.",
                                "raw_output": '{"Reasoning":"We must challenge immediately. This is clearly the optimal line.","Action":{"type":"challenge"}}',
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
                                "thought": "Maybe continue the line, but I am not certain.",
                                "raw_output": '{"Reasoning":"Maybe continue the line, but I am not certain.","Action":{"type":"play_claim","claim_rank":"A","cards":["K"]}}',
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
            model_path = root / "value_proxy_mlp_distill.pt"
            model_path.write_text("stub", encoding="utf-8")

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
                    model_path=model_path,
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

    def test_run_llm_behavior_audit_writes_ev_gap_distribution(self) -> None:
        module = importlib.import_module("liars_game_engine.analysis.audit_llm_behavior")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            log_dir = root / "logs" / "task_m_llm_drill" / "sample"
            games_dir = log_dir / "games"
            games_dir.mkdir(parents=True)
            log_path = games_dir / "llm-drill-01.jsonl"
            log_path.write_text(
                json.dumps(
                    {
                        "turn": 7,
                        "player_id": "p1",
                        "thought": "This looks like a controlled probing line.",
                        "raw_output": '{"Reasoning":"This looks like a controlled probing line.","Action":{"type":"play_claim","claim_rank":"K","cards":["K"]}}',
                        "action": {"type": "play_claim", "claim_rank": "K", "cards": ["K"]},
                        "observation": {"player_id": "p1", "table_type": "A"},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            (log_dir / "summary.json").write_text(
                json.dumps({"llm_player_id": "p1", "total_games": 1}, ensure_ascii=False),
                encoding="utf-8",
            )
            output_dir = root / "task_n_ev_gap_audit"
            model_dir = root / "models" / "proxy"
            model_dir.mkdir(parents=True)
            model_path = model_dir / "value_proxy_mlp_distill.pt"
            model_path.write_text("stub", encoding="utf-8")

            trajectory = TurnTrajectory(
                game_id="llm-drill-01",
                turn=7,
                player_id="p1",
                observation={"player_id": "p1", "table_type": "A"},
                action=ActionModel(type="play_claim", claim_rank="K", cards=["K"]),
                skill_name="Unknown",
                skill_parameters={},
                checkpoint_format="pickle_base64_v1",
                checkpoint_payload="placeholder",
            )
            game = GameTrajectory(game_id="llm-drill-01", turns=[trajectory], winner=None)

            class _FakeLogIterator:
                def __init__(self, log_paths: list[Path]) -> None:
                    self.log_paths = log_paths

                def iter_games(self) -> list[GameTrajectory]:
                    return [game]

            class _FakeAnalyzer:
                def __init__(self, *args, **kwargs) -> None:
                    pass

                def _build_proxy_legal_actions(self, trajectory: TurnTrajectory) -> list[ActionModel]:
                    return [
                        ActionModel(type="play_claim", claim_rank="A", cards=["Q"]),
                        ActionModel(type="challenge"),
                        trajectory.action,
                    ]

            class _FakePredictor:
                def __init__(self, model_path: str | Path, output_mode: str) -> None:
                    self.model_path = Path(model_path)
                    self.output_mode = output_mode

                def predict_state_features(self, state_features: dict[str, object]) -> float:
                    action_type = str(state_features.get("action_type", ""))
                    action_cards = list(state_features.get("action_cards", []))
                    if action_type == "challenge":
                        return -0.15
                    if action_cards == ["Q"]:
                        return 0.35
                    return -0.05

            def _fake_feature_builder(*, observation: dict[str, object], player_id: str, action: dict[str, object]) -> dict[str, object]:
                return {
                    "player_id": player_id,
                    "table_type": observation.get("table_type", "A"),
                    "action_type": action.get("type", ""),
                    "action_cards": list(action.get("cards", [])),
                }

            with (
                patch.object(module, "LogIterator", _FakeLogIterator),
                patch.object(module, "ShapleyAnalyzer", _FakeAnalyzer),
                patch.object(module, "ProxyValuePredictor", _FakePredictor),
                patch.object(module, "build_value_proxy_feature_context", _fake_feature_builder),
            ):
                summary = module.run_llm_behavior_audit(
                    log_root=log_dir,
                    model_path=model_path,
                    output_dir=output_dir,
                    phi_threshold=-0.1,
                )

            self.assertEqual(summary["audited_turn_count"], 1)
            self.assertIn("avg_ev_gap", summary)
            self.assertAlmostEqual(summary["avg_ev_gap"], 0.4, places=6)
            self.assertEqual(summary["high_ev_gap_turn_count"], 1)
            self.assertTrue(Path(summary["ev_gap_distribution_path"]).exists())

            rows = Path(summary["ev_gap_distribution_path"]).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 2)
            header = rows[0].split(",")
            self.assertIn("phi_chosen", header)
            self.assertIn("phi_best", header)
            self.assertIn("ev_gap", header)
            self.assertIn("best_action_type", header)

            data = rows[1].split(",")
            ev_gap_idx = header.index("ev_gap")
            self.assertEqual(data[header.index("turn")], "7")
            self.assertEqual(data[header.index("best_action_type")], "play_claim")
            self.assertAlmostEqual(float(data[ev_gap_idx]), 0.4, places=6)

    def test_run_llm_behavior_audit_writes_formal_task_1_1_outputs(self) -> None:
        module = importlib.import_module("liars_game_engine.analysis.audit_llm_behavior")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            log_dir = root / "logs" / "task_m_llm_drill" / "sample"
            games_dir = log_dir / "games"
            games_dir.mkdir(parents=True)
            log_path = games_dir / "llm-drill-02.jsonl"
            log_path.write_text(
                json.dumps(
                    {
                        "turn": 9,
                        "player_id": "p1",
                        "thought": "Probe first and avoid overcommitting.",
                        "raw_output": '{"Reasoning":"Probe first and avoid overcommitting.","Action":{"type":"play_claim","claim_rank":"A","cards":["A"]}}',
                        "action": {"type": "play_claim", "claim_rank": "A", "cards": ["A"]},
                        "observation": {"player_id": "p1", "table_type": "A"},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            (log_dir / "summary.json").write_text(
                json.dumps({"llm_player_id": "p1", "total_games": 1}, ensure_ascii=False),
                encoding="utf-8",
            )
            output_dir = root / "task_n_ev_gap_audit"
            model_dir = root / "models" / "proxy"
            model_dir.mkdir(parents=True)
            model_path = model_dir / "value_proxy_mlp_distill.pt"
            model_path.write_text("stub", encoding="utf-8")

            trajectory = TurnTrajectory(
                game_id="llm-drill-02",
                turn=9,
                player_id="p1",
                observation={"player_id": "p1", "table_type": "A"},
                action=ActionModel(type="play_claim", claim_rank="A", cards=["A"]),
                skill_name="Unknown",
                skill_parameters={},
                checkpoint_format="pickle_base64_v1",
                checkpoint_payload="placeholder",
            )
            game = GameTrajectory(game_id="llm-drill-02", turns=[trajectory], winner=None)

            class _FakeLogIterator:
                def __init__(self, log_paths: list[Path]) -> None:
                    self.log_paths = log_paths

                def iter_games(self) -> list[GameTrajectory]:
                    return [game]

            class _FakeAnalyzer:
                def __init__(self, *args, **kwargs) -> None:
                    pass

                def _build_proxy_legal_actions(self, trajectory: TurnTrajectory) -> list[ActionModel]:
                    return [
                        trajectory.action,
                        ActionModel(type="play_claim", claim_rank="A", cards=["K"]),
                    ]

            class _FakePredictor:
                def __init__(self, model_path: str | Path, output_mode: str) -> None:
                    self.model_path = Path(model_path)
                    self.output_mode = output_mode

                def predict_state_features(self, state_features: dict[str, object]) -> float:
                    action_cards = list(state_features.get("action_cards", []))
                    if action_cards == ["K"]:
                        return 0.17
                    return 0.05

            def _fake_feature_builder(*, observation: dict[str, object], player_id: str, action: dict[str, object]) -> dict[str, object]:
                return {
                    "player_id": player_id,
                    "table_type": observation.get("table_type", "A"),
                    "action_type": action.get("type", ""),
                    "action_cards": list(action.get("cards", [])),
                }

            with (
                patch.object(module, "LogIterator", _FakeLogIterator),
                patch.object(module, "ShapleyAnalyzer", _FakeAnalyzer),
                patch.object(module, "ProxyValuePredictor", _FakePredictor),
                patch.object(module, "build_value_proxy_feature_context", _fake_feature_builder),
            ):
                summary = module.run_llm_behavior_audit(
                    log_root=log_dir,
                    model_path=model_path,
                    output_dir=output_dir,
                    phi_threshold=-0.1,
                )

            self.assertEqual(summary["audited_turn_count"], 1)
            self.assertAlmostEqual(summary["avg_ev_gap"], 0.12, places=6)
            self.assertEqual(summary["high_ev_gap_turn_count"], 0)
            self.assertEqual(summary["potential_point_ev_gap_threshold"], 0.15)
            self.assertTrue(Path(summary["task_1_1_ev_gap_report_path"]).exists())
            self.assertTrue(Path(summary["ev_gap_heatmap_path"]).exists())

            report = json.loads(Path(summary["task_1_1_ev_gap_report_path"]).read_text(encoding="utf-8"))
            self.assertEqual(report["high_ev_gap_turn_count"], 0)
            self.assertEqual(report["potential_point_ev_gap_threshold"], 0.15)
            self.assertEqual(report["ev_gap_heatmap_path"], summary["ev_gap_heatmap_path"])

            rows = Path(summary["ev_gap_distribution_path"]).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 2)
            header = rows[0].split(",")
            data = rows[1].split(",")
            self.assertEqual(data[header.index("is_potential_point")], "0")


if __name__ == "__main__":
    unittest.main()
