import unittest
from pathlib import Path


class TaskMServerScriptTest(unittest.TestCase):
    def test_task_m_server_script_exists_and_documents_tail_targets(self) -> None:
        script_path = Path(__file__).resolve().parent.parent / "scripts" / "run_task_m_drill_server.sh"
        self.assertTrue(script_path.exists())

        content = script_path.read_text(encoding="utf-8")
        self.assertIn('ENV_NAME="${CONDA_ENV_NAME:-liar_bar}"', content)
        self.assertIn("nohup conda run -n", content)
        self.assertIn("python scripts/run_llm_drill.py", content)
        self.assertIn('stdout.log', content)
        self.assertIn('progress.log', content)
        self.assertIn('tail -f "$OUT_DIR/progress.log"', content)
        self.assertIn('tail -f "$OUT_DIR/stdout.log"', content)


if __name__ == "__main__":
    unittest.main()
