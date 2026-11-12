"""Verify a built ZIP from a different folder using the current Python environment."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

ARCHIVE = Path(__file__).resolve().parents[1] / "dist/EnglishVietnamese-biLSTM.zip"


class DistributionTests(unittest.TestCase):
    @unittest.skipUnless(ARCHIVE.is_file(), "Build the distribution first")
    def test_relocated_gui_autoloads_checkpoint_and_translates(self):
        with tempfile.TemporaryDirectory(prefix="bilstm release ") as folder:
            with zipfile.ZipFile(ARCHIVE) as archive:
                self.assertIsNone(archive.testzip())
                self.assertFalse(any("agents.md" in name.lower() or ".venv" in name or "__pycache__" in name for name in archive.namelist()))
                archive.extractall(folder)
            project = Path(folder) / "EnglishVietnamese-biLSTM"
            code = '''
import time
from pathlib import Path
from unittest.mock import patch
import torch
import app_gui
torch.set_num_threads(2)
assert app_gui.ROOT == Path.cwd()
with patch("app_gui.messagebox.showerror") as errors:
    app = app_gui.TranslationApp()
    app.withdraw()
    def wait():
        deadline = time.monotonic() + 30
        while app.busy and time.monotonic() < deadline:
            app.update()
            time.sleep(0.01)
        assert not app.busy, "Worker timeout"
        errors.assert_not_called()
    wait()
    assert app.trainer is not None
    assert "output_dir" not in app.trainer.config
    app._start("translate")
    wait()
    assert "Tensor shapes:" in app.details.get("1.0", "end")
    app._close()
print("Relocated GUI, checkpoint autoload and translation passed")
'''
            result = subprocess.run([sys.executable, "-c", code], cwd=project,
                                    env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                                    capture_output=True, text=True, encoding="utf-8", timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
