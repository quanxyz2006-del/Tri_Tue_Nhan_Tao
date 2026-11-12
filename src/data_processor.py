"""Unicode normalization, training-only vocabularies and padded batches."""
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

from utils.constants import EOS, PAD, SOS, SPECIAL_TOKENS, UNK


def tokenize(text):
    text = unicodedata.normalize("NFC", text).lower()
    return re.findall(r"[^\W_]+|[.,!?;:'-]", text, flags=re.UNICODE)


class Vocabulary:
    def __init__(self, words=None):
        self.idx2word = list(SPECIAL_TOKENS if words is None else words)
        self.word2idx = {word: i for i, word in enumerate(self.idx2word)}

    def fit(self, sentences):
        for sentence in sentences:
            for token in tokenize(sentence):
                if token not in self.word2idx:
                    self.word2idx[token] = len(self.idx2word)
                    self.idx2word.append(token)

    def encode(self, tokens):
        return [self.word2idx.get(token, UNK) for token in tokens]

    def __len__(self):
        return len(self.idx2word)


class DataProcessor:
    def __init__(self, max_length=40):
        if max_length < 2:
            raise ValueError("max_length must be at least 2")
        self.max_length = max_length
        self.vocab_en, self.vocab_vi = Vocabulary(), Vocabulary()

    def prepare(self, path, seed=42):
        groups = defaultdict(list)
        seen = set()
        with Path(path).open(encoding="utf-8-sig") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip() or line.startswith("#"):
                    continue
                columns = line.rstrip("\r\n").split("\t")
                if len(columns) < 2:
                    raise ValueError(f"Line {number}: expected English<TAB>Vietnamese")
                en, vi = (" ".join(tokenize(value)) for value in columns[:2])
                if not en or not vi:
                    raise ValueError(f"Line {number}: empty normalized sentence")
                if (en, vi) not in seen:
                    groups[en].append((en, vi))
                    seen.add((en, vi))
        keys = list(groups)
        if len(keys) < 10:
            raise ValueError("At least 10 distinct English sentences are required")
        random.Random(seed).shuffle(keys)
        n = max(1, int(len(keys) * 0.1))
        # Keep all translations of the same source in one partition.
        splits = {name: [pair for key in part for pair in groups[key]]
                  for name, part in (("test", keys[:n]), ("val", keys[n:2*n]),
                                     ("train", keys[2*n:]))}
        self.vocab_en.fit(en for en, _ in splits["train"])
        self.vocab_vi.fit(vi for _, vi in splits["train"])
        return splits

    def source(self, text):
        tokens = tokenize(text)
        if not tokens:
            raise ValueError("Enter a nonempty English sentence")
        return self.vocab_en.encode(tokens[:self.max_length])

    def batch(self, pairs):
        sources, inputs, targets = [], [], []
        for en, vi in pairs:
            sources.append(torch.tensor(self.source(en), dtype=torch.long))
            ids = self.vocab_vi.encode(tokenize(vi)[:self.max_length - 1])
            inputs.append(torch.tensor([SOS] + ids, dtype=torch.long))
            targets.append(torch.tensor(ids + [EOS], dtype=torch.long))
        return tuple(pad_sequence(items, batch_first=True, padding_value=PAD)
                     for items in (sources, inputs, targets))

    def state(self):
        return {"max_length": self.max_length, "en": self.vocab_en.idx2word,
                "vi": self.vocab_vi.idx2word}

    @classmethod
    def from_state(cls, state):
        processor = cls(state["max_length"])
        processor.vocab_en = Vocabulary(state["en"])
        processor.vocab_vi = Vocabulary(state["vi"])
        return processor
