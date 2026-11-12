import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from app_gui import TranslationApp


class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.app = TranslationApp(autoload=False)
        cls.app.withdraw()

    @classmethod
    def tearDownClass(cls):
        cls.app._close()

    def setUp(self):
        self.errors = []
        self.app.report_callback_exception = lambda *args: self.errors.append(args)
        self.dialog = patch("app_gui.messagebox.showerror")
        self.showerror = self.dialog.start()
        self.directory = tempfile.TemporaryDirectory()

    def tearDown(self):
        if self.app.busy:
            self.app._cancel()
            deadline = time.monotonic() + 30
            while self.app.busy and time.monotonic() < deadline:
                self.app.update()
                time.sleep(0.01)
        self.dialog.stop()
        self.directory.cleanup()

    def wait_for_worker(self):
        deadline = time.monotonic() + 30
        while self.app.busy and time.monotonic() < deadline:
            self.app.update()
            time.sleep(0.01)
        self.assertFalse(self.app.busy, "GUI worker did not finish")
        self.assertFalse(self.errors)
        self.showerror.assert_not_called()

    def train(self):
        config = dict(embedding_size=8, hidden_size=12, epochs=1,
                      batch_size=16, max_length=12, seed=42,
                      learning_rate=0.001, output_dir=self.directory.name)
        with patch.object(self.app, "_config", return_value=config):
            self.app._start("train")
        self.wait_for_worker()

    def test_train_translate_save_load(self):
        self.train()
        self.assertIsNotNone(self.app.trainer)
        self.app._start("translate")
        self.wait_for_worker()
        self.assertIn("Tensor shapes:", self.app.details.get("1.0", "end"))
        path = str(Path(self.directory.name) / "roundtrip.pt")
        self.app._start("save", path)
        self.wait_for_worker()
        self.app._start("load", path)
        self.wait_for_worker()
        self.assertEqual(self.app.entries["hidden_size"].get(), "12")

    def test_load_ignores_unrelated_training_fields(self):
        self.train()
        self.app.entries["epochs"].delete(0, "end")
        self.app.entries["epochs"].insert(0, "invalid")
        self.app._start("load", str(Path(self.directory.name) / "bilstm.pt"))
        self.wait_for_worker()
        self.assertEqual(self.app.entries["epochs"].get(), "1")


if __name__ == "__main__":
    unittest.main()
