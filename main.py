import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="English-Vietnamese biLSTM translator")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--data", default=str(Path(__file__).parent / "data" / "eng-vie.txt"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--checkpoint")
    parser.add_argument("--text")
    args = parser.parse_args()
    if args.train or args.ablation or args.checkpoint:
        from src.trainer import Trainer
        trainer = Trainer({"epochs": args.epochs})
        if args.train or args.ablation:
            trainer.prepare(args.data)
            result = (trainer.ablation if args.ablation else trainer.fit)(callback=print)
            print(json.dumps(result, indent=2))
        else:
            trainer.load(args.checkpoint)
        if args.text:
            print(trainer.translate(args.text)["text"])
    else:
        from app_gui import main as gui_main
        gui_main()


if __name__ == "__main__":
    main()
