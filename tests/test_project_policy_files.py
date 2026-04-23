import unittest
from pathlib import Path


class ProjectPolicyFilesTest(unittest.TestCase):
    def test_required_policy_files_exist(self) -> None:
        """作用: 验证项目强制策略文件都已存在。

        输入:
        - 无。

        返回:
        - 无。
        """
        project_root = Path(__file__).resolve().parent.parent

        required_files = [
            project_root / "PROJECT_MEMORY.md",
            project_root / "logs" / "CHANGELOG_DAILY.md",
            project_root / "logs" / "CHANGELOG_DETAILED.md",
        ]

        for path in required_files:
            self.assertTrue(path.exists(), f"Missing required policy file: {path}")


if __name__ == "__main__":
    unittest.main()
