"""Training, held-out evaluation, reproducible ablation and checkpoints."""
import copy
import json
import math
import pickle
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data_processor import DataProcessor, tokenize
from src.seq2seq_model import Seq2Seq
from utils.constants import PAD
from utils.helpers import bleu_scores

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = dict(embedding_size=128, hidden_size=256, learning_rate=0.001,
                epochs=10, batch_size=32, max_length=40, seed=42)


class TrainingCancelled(RuntimeError):
    pass


class Trainer:
    def __init__(self, config=None):
        self.config = {**DEFAULTS, **(config or {})}
        for key in ("embedding_size", "hidden_size", "epochs", "batch_size", "max_length"):
            value = self.config[key]
            if not isinstance(value, int) or value < (2 if key == "max_length" else 1):
                raise ValueError(f"Invalid {key}: {value}")
        lr = self.config["learning_rate"]
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError("learning_rate must be positive and finite")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = None
        self.model = None
        self.splits = None
        self.result = {}

    def prepare(self, path):
        self.processor = DataProcessor(self.config["max_length"])
        self.splits = self.processor.prepare(path, self.config["seed"])
        return {name: len(pairs) for name, pairs in self.splits.items()}

    @staticmethod
    def _check_stop(stop_event):
        if stop_event is not None and stop_event.is_set():
            raise TrainingCancelled("Training cancelled")

    def _epoch(self, loader, optimizer=None, stop_event=None):
        self.model.train(optimizer is not None)
        total_loss = correct = count = 0
        criterion = nn.CrossEntropyLoss(ignore_index=PAD, reduction="sum")
        for batch in loader:
            self._check_stop(stop_event)
            source, inputs, targets = (tensor.to(self.device) for tensor in batch)
            with torch.set_grad_enabled(optimizer is not None):
                logits = self.model(source, inputs)
                mask = targets.ne(PAD)
                tokens = int(mask.sum())
                loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                if optimizer is not None:
                    optimizer.zero_grad()
                    (loss / tokens).backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
            total_loss += loss.item()
            correct += int(((logits.argmax(-1) == targets) & mask).sum())
            count += tokens
        return total_loss / count, correct / count

    def fit(self, bidirectional=True, callback=None, stop_event=None):
        if self.splits is None:
            raise ValueError("Prepare a dataset before training")
        cfg = self.config
        random.seed(cfg["seed"])
        torch.manual_seed(cfg["seed"])
        self.bidirectional = bidirectional
        self.model = Seq2Seq(len(self.processor.vocab_en), len(self.processor.vocab_vi),
                            cfg["embedding_size"], cfg["hidden_size"], bidirectional).to(self.device)
        generator = torch.Generator().manual_seed(cfg["seed"])
        train_loader = DataLoader(self.splits["train"], batch_size=cfg["batch_size"],
                                  shuffle=True, generator=generator, collate_fn=self.processor.batch)
        val_loader = DataLoader(self.splits["val"], batch_size=cfg["batch_size"],
                                collate_fn=self.processor.batch)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg["learning_rate"])
        history, best_state, best_loss = [], None, float("inf")
        started = time.perf_counter()
        for epoch in range(1, cfg["epochs"] + 1):
            train_loss, train_acc = self._epoch(train_loader, optimizer, stop_event)
            val_loss, val_acc = self._epoch(val_loader, stop_event=stop_event)
            row = dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                       train_accuracy=train_acc, val_accuracy=val_acc)
            history.append(row)
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
            if callback:
                callback({**row, "model": "biLSTM" if bidirectional else "LSTM"})
            # Finish the current epoch before restoring the best validation state.
            if time.perf_counter() - started >= cfg.get("max_seconds", float("inf")):
                break
        seconds = time.perf_counter() - started
        self.model.load_state_dict(best_state)
        references, hypotheses = [], []
        grouped = defaultdict(list)
        for en, vi in self.splits["test"]:
            grouped[en].append(tokenize(vi))
        for en, refs in grouped.items():
            self._check_stop(stop_event)
            references.append(refs)
            hypotheses.append(self.translate(en)["output_tokens"])
        self.result = dict(history=history, bleu=bleu_scores(references, hypotheses),
                           seconds=seconds, best_val_loss=best_loss,
                           parameters=sum(p.numel() for p in self.model.parameters()),
                           split_sizes={k: len(v) for k, v in self.splits.items()},
                           weight_shapes={k: list(v.shape) for k, v in self.model.state_dict().items()})
        name = "bilstm" if bidirectional else "lstm"
        output = Path(cfg.get("output_dir", ROOT / "artifacts"))
        self.save(output / f"{name}.pt")
        (output / f"{name}_metrics.json").write_text(
            json.dumps(self.result, indent=2), encoding="utf-8")
        with (output / "processed_vocab.pkl").open("wb") as stream:
            pickle.dump(self.processor.state(), stream)
        return self.result

    def ablation(self, callback=None, stop_event=None):
        results = {}
        for name, bidirectional in (("LSTM", False), ("biLSTM", True)):
            self._check_stop(stop_event)
            results[name] = self.fit(bidirectional, callback, stop_event)
        return results

    def translate(self, text):
        if self.model is None or self.processor is None:
            raise ValueError("Train or load a model before translating")
        ids = self.processor.source(text)
        source = torch.tensor([ids], dtype=torch.long, device=self.device)
        output, shapes, steps = self.model.generate(source, self.config["max_length"])
        words = self.processor.vocab_vi.idx2word
        output_tokens = [words[index] for index in output]
        for step in steps:
            step["token"] = words[step["id"]]
        return dict(text=" ".join(output_tokens), output_tokens=output_tokens,
                    tokens=tokenize(text)[:self.config["max_length"]], ids=ids,
                    truncated=len(tokenize(text)) > self.config["max_length"],
                    shapes=shapes, steps=steps)

    def save(self, path):
        if self.model is None:
            raise ValueError("No model to save")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(version=1, config=self.config, processor=self.processor.state(),
                        bidirectional=self.bidirectional, weights=self.model.state_dict(),
                        result=self.result), path)

    def load(self, path):
        state = torch.load(path, map_location=self.device, weights_only=True)
        if state.get("version") != 1:
            raise ValueError("Unsupported checkpoint format")
        validated = Trainer(state["config"])
        self.config = validated.config
        self.processor = DataProcessor.from_state(state["processor"])
        self.bidirectional = state["bidirectional"]
        self.model = Seq2Seq(len(self.processor.vocab_en), len(self.processor.vocab_vi),
                            self.config["embedding_size"], self.config["hidden_size"],
                            self.bidirectional).to(self.device)
        self.model.load_state_dict(state["weights"])
        self.model.eval()
        self.result = state.get("result", {})
        self.splits = None
        return self
