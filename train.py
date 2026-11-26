from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    runpy.run_path(str(ROOT / "training" / "train.py"), run_name="__main__")
