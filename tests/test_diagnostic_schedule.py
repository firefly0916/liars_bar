import importlib
import unittest


class DiagnosticScheduleTest(unittest.TestCase):
    def test_build_checkpoint_schedule_uses_dense_then_sparse_intervals(self) -> None:
        module = importlib.import_module("liars_game_engine.analysis.diagnostic_schedule")

        tags = module.build_checkpoint_schedule(
            max_step=200,
            dense_until_step=100,
            dense_interval=5,
            sparse_interval=10,
            include_final=True,
        )

        self.assertEqual(tags[:5], ["step-000005", "step-000010", "step-000015", "step-000020", "step-000025"])
        self.assertIn("step-000100", tags)
        self.assertIn("step-000110", tags)
        self.assertIn("step-000200", tags)
        self.assertEqual(tags[-1], "final")
        self.assertEqual(tags.count("step-000100"), 1)
        self.assertEqual(len(tags), 31)


if __name__ == "__main__":
    unittest.main()
