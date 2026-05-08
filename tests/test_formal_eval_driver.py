import importlib
import importlib.util
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path


class FormalEvalDriverTest(unittest.TestCase):
    def test_load_checkpoint_tags_prefers_explicit_tags_and_deduplicates(self) -> None:
        module = importlib.import_module("liars_game_engine.analysis.formal_eval_driver")

        tags = module.load_checkpoint_tags(
            selection_payload={"selected": [{"tag": "step-000010"}, {"tag": "final"}]},
            explicit_tags=["step-000140", "step-000045", "step-000140"],
        )

        self.assertEqual(tags, ["step-000140", "step-000045"])

    def test_build_formal_eval_plan_uses_train_checkpoints_and_output_roots(self) -> None:
        module = importlib.import_module("liars_game_engine.analysis.formal_eval_driver")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "train" / "checkpoints" / "step-000140").mkdir(parents=True, exist_ok=True)
            (root / "train" / "checkpoints" / "step-000045").mkdir(parents=True, exist_ok=True)

            plan = module.build_formal_eval_plan(
                run_root=root,
                checkpoint_tags=["step-000140", "step-000045"],
                output_dir_name="formal_eval_manual",
                games=100,
            )

        self.assertEqual(plan["checkpoint_tags"], ["step-000140", "step-000045"])
        self.assertTrue(str(root / "formal_eval_manual").endswith(plan["formal_root"]))
        first_entry = plan["entries"][0]
        self.assertEqual(first_entry["tag"], "step-000140")
        self.assertTrue(first_entry["checkpoint_path"].endswith("train/checkpoints/step-000140"))
        self.assertTrue(first_entry["task_m_log_path"].endswith("formal_eval_manual/step-000140/task_m_stdout.log"))
        self.assertEqual(first_entry["games"], 100)

    def test_load_checkpoint_tags_from_selection_file_preserves_selected_order(self) -> None:
        module = importlib.import_module("liars_game_engine.analysis.formal_eval_driver")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            selection_path = root / "selection.json"
            selection_path.write_text(
                json.dumps({"selected": [{"tag": "step-000150"}, {"tag": "step-000200"}]}),
                encoding="utf-8",
            )

            payload = module.load_selection_payload(selection_path)
            tags = module.load_checkpoint_tags(selection_payload=payload, explicit_tags=[])

        self.assertEqual(tags, ["step-000150", "step-000200"])

    def test_run_selected_formal_eval_streams_child_output_to_log_while_running(self) -> None:
        script_path = Path(__file__).resolve().parent.parent / "scripts" / "run_selected_formal_eval.py"
        spec = importlib.util.spec_from_file_location("run_selected_formal_eval_script", script_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            log_path = root / "stream.log"
            command = [
                sys.executable,
                "-c",
                "import time; print('start', flush=True); time.sleep(0.4); print('end', flush=True)",
            ]
            failure: list[BaseException] = []

            def _runner() -> None:
                try:
                    module._run_command(command, cwd=root, env=dict(os.environ), log_path=log_path)
                except BaseException as exc:  # pragma: no cover
                    failure.append(exc)

            thread = threading.Thread(target=_runner)
            thread.start()
            time.sleep(0.15)

            self.assertTrue(log_path.exists())
            partial = log_path.read_text(encoding="utf-8")
            self.assertIn("start", partial)
            self.assertNotIn("end", partial)

            thread.join(timeout=2.0)
            self.assertFalse(thread.is_alive())
            self.assertEqual(failure, [])
            final = log_path.read_text(encoding="utf-8")
            self.assertIn("end", final)


if __name__ == "__main__":
    unittest.main()
