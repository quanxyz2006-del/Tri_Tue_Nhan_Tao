import tempfile
import threading
import unittest
from pathlib import Path

import torch

from src.data_processor import DataProcessor, tokenize
from src.encoder import Encoder
from src.seq2seq_model import Seq2Seq
from src.trainer import Trainer, TrainingCancelled
from utils.constants import EOS, PAD, SOS, UNK

ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_unicode_and_shift(self):
        self.assertEqual(tokenize("TO\u0302I!"), ["t\u00f4i", "!"])
        processor = DataProcessor(4)
        processor.vocab_en.fit(["hello"])
        processor.vocab_vi.fit(["xin chao ban"])
        source, inputs, targets = processor.batch([("hello", "xin chao ban"), ("unknown", "xin")])
        self.assertEqual(source[1, 0].item(), UNK)
        self.assertEqual(inputs[0, 0].item(), SOS)
        self.assertEqual(targets[0, -1].item(), EOS)
        self.assertTrue(torch.equal(inputs[0, 1:], targets[0, :-1]))
        self.assertEqual(targets[1, -1].item(), PAD)

    def test_split_has_no_source_leakage(self):
        processor = DataProcessor()
        splits = processor.prepare(ROOT / "data/eng-vie.txt")
        sets = {name: {en for en, _ in rows} for name, rows in splits.items()}
        self.assertFalse(sets["train"] & sets["test"])
        self.assertFalse(sets["train"] & sets["val"])
        train_words = {word for en, _ in splits["train"] for word in tokenize(en)}
        self.assertEqual(set(processor.vocab_en.idx2word[4:]), train_words)

    def test_padding_does_not_change_encoder_state(self):
        for bidirectional in (False, True):
            encoder = Encoder(12, 5, 7, bidirectional).eval()
            _, plain = encoder(torch.tensor([[4, 5, 6]]))
            outputs, padded = encoder(torch.tensor([[4, 5, 6, 0, 0]]))
            self.assertEqual(outputs.shape[-1], 14 if bidirectional else 7)
            for a, b in zip(plain, padded):
                self.assertEqual(tuple(b.shape), (1, 1, 7))
                torch.testing.assert_close(a, b)

    def test_eos_stops_generation(self):
        model = Seq2Seq(8, 8, 4, 5)
        with torch.no_grad():
            model.decoder.output.weight.zero_()
            model.decoder.output.bias.zero_()
            model.decoder.output.bias[EOS] = 100
        ids, _, steps = model.generate(torch.tensor([[4, 5]]), 10)
        self.assertEqual(ids, [])
        self.assertEqual(len(steps), 1)

    def test_training_ablation_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(dict(embedding_size=8, hidden_size=12, epochs=2, max_seconds=0,
                                   batch_size=16, max_length=12, output_dir=directory))
            trainer.prepare(ROOT / "data/eng-vie.txt")
            results = trainer.ablation()
            self.assertEqual(set(results), {"LSTM", "biLSTM"})
            for result in results.values():
                self.assertEqual(len(result["history"]), 1)
                self.assertGreater(result["history"][0]["train_loss"], 0)
                self.assertTrue(all(0 <= value <= 100 for value in result["bleu"].values()))
            before = trainer.translate("hello")
            restored = Trainer().load(Path(directory) / "bilstm.pt")
            self.assertEqual(before, restored.translate("hello"))
            event = threading.Event()
            event.set()
            with self.assertRaises(TrainingCancelled):
                trainer.fit(stop_event=event)


if __name__ == "__main__":
    unittest.main()
