"""Build a portable source distribution with the trained checkpoint."""
from pathlib import Path
import io
import zipfile

import torch

ROOT = Path(__file__).resolve().parent


def main():
    files = [ROOT / name for name in (
        "README.md", "requirements.txt", "app_gui.py", "main.py", "run.py",
        "START_WINDOWS.bat", "start_macos_linux.sh", "train_basic.py", "package_project.py",
        "data/eng-vie.txt", "data/eng-vie-basic.txt", "data/manythings-vie/_about.txt",
        "artifacts/basic_15min/bilstm.pt", "artifacts/basic_15min/bilstm_metrics.json",
        "artifacts/basic_15min/evaluation.json")]
    for folder in ("src", "utils", "tests"):
        files.extend(sorted((ROOT / folder).glob("*.py")))
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"Required release file missing: {path}")
    output = ROOT / "dist/EnglishVietnamese-biLSTM.zip"
    output.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            name = "EnglishVietnamese-biLSTM/" + path.relative_to(ROOT).as_posix()
            if path.suffix == ".pt":
                state = torch.load(path, map_location="cpu", weights_only=True)
                state["config"].pop("output_dir", None)
                buffer = io.BytesIO()
                torch.save(state, buffer)
                archive.writestr(name, buffer.getvalue())
            else:
                archive.write(path, name)
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
    print(f"Created {output} ({output.stat().st_size / 1024 / 1024:.1f} MiB)")


if __name__ == "__main__":
    main()
