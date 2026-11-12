import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run


class LauncherTests(unittest.TestCase):
    def test_first_run_installs_and_second_run_reuses_environment(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "requirements.txt").write_text("customtkinter>=5.2.2\n", encoding="utf-8")
            def create_environment(command, **kwargs):
                if "venv" in command:
                    binary = root / ".venv-run" / ("Scripts/python.exe" if run.sys.platform == "win32" else "bin/python")
                    binary.parent.mkdir(parents=True)
                    binary.touch()
            with patch.object(run, "ROOT", root), patch.object(run.subprocess, "run", side_effect=create_environment) as setup, patch.object(run.subprocess, "call", return_value=0) as launch:
                self.assertEqual(run.main(), 0)
                self.assertEqual(setup.call_count, 2)
                setup.reset_mock()
                self.assertEqual(run.main(), 0)
                setup.assert_not_called()
                self.assertEqual(launch.call_count, 2)
                self.assertEqual(launch.call_args.kwargs["cwd"], root)


if __name__ == "__main__":
    unittest.main()
