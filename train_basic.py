"""Train a short-sentence model with a bounded training budget."""
import json
import random
from pathlib import Path

import torch

from src.data_processor import tokenize
from src.trainer import Trainer

ROOT = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(4)
    corpus = ROOT / "data/manythings-vie/vie.txt"
    selected = ROOT / "data/eng-vie-basic.txt"
    if corpus.is_file():
        rows = []
        for line in corpus.read_text(encoding="utf-8-sig").splitlines():
            columns = line.split("\t")
            if len(columns) >= 2:
                en, vi = map(tokenize, columns[:2])
                if 2 <= len(en) <= 10 and 2 <= len(vi) <= 16:
                    rows.append(line)
        random.Random(42).shuffle(rows)
        selected.write_text("\n".join(rows[:8000]) + "\n", encoding="utf-8")
    output = ROOT / "artifacts/basic_15min"
    trainer = Trainer(dict(embedding_size=96, hidden_size=192, epochs=100,
                          batch_size=64, max_length=18, learning_rate=0.003,
                          seed=42, max_seconds=720, output_dir=str(output)))
    sizes = trainer.prepare(selected)
    print(json.dumps(dict(device=str(trainer.device), splits=sizes,
                          vocabulary_en=len(trainer.processor.vocab_en),
                          vocabulary_vi=len(trainer.processor.vocab_vi))), flush=True)
    result = trainer.fit(callback=lambda row: print(json.dumps(row), flush=True))
    samples = []
    for en, vi in trainer.splits["test"][:12]:
        samples.append(dict(english=en, reference=vi, prediction=trainer.translate(en)["text"]))
    restored = Trainer().load(output / "bilstm.pt")
    assert restored.translate("Hello.") == trainer.translate("Hello.")
    report = dict(bleu=result["bleu"], seconds=result["seconds"],
                  epochs=len(result["history"]), samples=samples,
                  checkpoint_verified=True)
    (output / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
