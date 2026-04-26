import unittest
from pathlib import Path


class ServerScriptsTest(unittest.TestCase):
    def test_remote_run_task_k_uses_timestamped_run_directory(self) -> None:
        script_path = Path(__file__).resolve().parent.parent / "scripts" / "remote_run.sh"
        content = script_path.read_text(encoding="utf-8")

        self.assertIn('RUN_DIR="logs/task_k_gold/${TIMESTAMP}"', content)
        self.assertIn('RUN_LOG="${RUN_DIR}/run.log"', content)
        self.assertIn('--output-dir "${RUN_DIR}"', content)
        self.assertIn('echo "Output dir: ${RUN_DIR}"', content)
        self.assertIn('echo "Progress log: ${RUN_DIR}/progress.log"', content)

    def test_run_task_k_gold_server_uses_timestamped_run_directory(self) -> None:
        script_path = Path(__file__).resolve().parent.parent / "scripts" / "run_task_k_gold_server.sh"
        content = script_path.read_text(encoding="utf-8")

        self.assertIn('run_dir="logs/task_k_gold/${timestamp}"', content)
        self.assertIn('run_log="${run_dir}/run.log"', content)
        self.assertIn('--output-dir "${run_dir}"', content)
        self.assertIn('echo "Output dir -> ${run_dir}"', content)
        self.assertIn('echo "Progress checkpoints -> ${run_dir}/progress.log"', content)


if __name__ == "__main__":
    unittest.main()
