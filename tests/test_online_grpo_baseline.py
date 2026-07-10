import importlib
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


class OnlineGRPOBaselineTest(unittest.TestCase):
    def test_group_advantages_are_reward_centered_without_crc_fields(self) -> None:
        module = importlib.import_module("scripts.train_online_grpo_baseline")

        records = [
            {"reward": 1.0, "group_id": "g1", "phi_chosen": 99.0, "ev_gap": 99.0},
            {"reward": 0.0, "group_id": "g1", "phi_chosen": -99.0, "ev_gap": -99.0},
            {"reward": -1.0, "group_id": "g1"},
            {"reward": 0.5, "group_id": "g2"},
            {"reward": 0.5, "group_id": "g2"},
        ]

        module.assign_group_advantages(records)

        self.assertAlmostEqual(sum(record["advantage"] for record in records[:3]), 0.0)
        self.assertAlmostEqual(records[0]["advantage"], 1.0)
        self.assertAlmostEqual(records[1]["advantage"], 0.0)
        self.assertAlmostEqual(records[2]["advantage"], -1.0)
        self.assertAlmostEqual(records[3]["advantage"], 0.0)
        self.assertAlmostEqual(records[4]["advantage"], 0.0)
        self.assertNotIn("crc_reward", records[0])

    def test_protocol_reward_ignores_crc_proxy_and_ev_gap_fields(self) -> None:
        module = importlib.import_module("scripts.train_online_grpo_baseline")

        clean = module.compute_decision_reward(
            terminal_reward=1.0,
            parser_error=None,
            fallback_used=False,
            step_success=True,
            phi_chosen=-100.0,
            ev_gap=100.0,
            proxy_best_action={"type": "challenge"},
        )
        penalized = module.compute_decision_reward(
            terminal_reward=1.0,
            parser_error={"code": "E_PARSE"},
            fallback_used=True,
            step_success=False,
            phi_chosen=100.0,
            ev_gap=0.0,
            proxy_best_action={"type": "play_claim"},
        )

        self.assertAlmostEqual(clean, 1.0)
        self.assertLess(penalized, clean)

    def test_trainer_cli_exposes_online_grpo_contract(self) -> None:
        script_path = REPO_ROOT / "scripts" / "train_online_grpo_baseline.py"
        self.assertTrue(script_path.exists())

        completed = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )

        self.assertIn("online GRPO-style LoRA baseline", completed.stdout)
        self.assertIn("--group-size", completed.stdout)
        self.assertIn("--rollout-games-per-update", completed.stdout)
        self.assertIn("--save-final-adapter", completed.stdout)

    def test_runner_is_isolated_and_defaults_to_200_formal_games(self) -> None:
        script_path = REPO_ROOT / "scripts" / "run_online_grpo_baseline.sh"
        self.assertTrue(script_path.exists())
        content = script_path.read_text(encoding="utf-8")

        self.assertIn('FORMAL_GAMES="${FORMAL_GAMES:-200}"', content)
        self.assertIn('GRPO_GROUP_SIZE="${GRPO_GROUP_SIZE:-4}"', content)
        self.assertIn("train_online_grpo_baseline.py", content)
        self.assertIn('--group-size "${GRPO_GROUP_SIZE}"', content)
        self.assertIn("python scripts/run_llm_drill.py", content)
        self.assertIn('LOCAL_LLM_ADAPTER_PATH="${ONLINE_GRPO_ADAPTER_PATH}"', content)
        self.assertIn("-u LOCAL_LLM_ADAPTER_PATH", content)
        self.assertNotIn("run_protocol_anchor_pipeline.sh", content)
        self.assertNotIn("run_protocol_anchor_diagnostic_round1.sh", content)

    def test_openspec_documents_online_grpo_requirements(self) -> None:
        spec_path = (
            REPO_ROOT
            / "openspec"
            / "changes"
            / "add-online-grpo-baseline"
            / "specs"
            / "online-grpo-baseline-workflow"
            / "spec.md"
        )
        self.assertTrue(spec_path.exists())
        spec = spec_path.read_text(encoding="utf-8")

        self.assertIn("### Requirement: Online GRPO group rollout collection", spec)
        self.assertIn("### Requirement: Group-normalized objective", spec)
        self.assertIn("### Requirement: Isolated 200-game formal evaluation", spec)


if __name__ == "__main__":
    unittest.main()
