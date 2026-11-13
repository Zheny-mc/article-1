from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acfsa.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline RAG with ACFSA context filtering")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"), help="YAML configuration file")
    args = parser.parse_args()
    output = run_experiment(args.config)
    print(f"Эксперимент завершён. Результаты: {output}")


if __name__ == "__main__":
    main()
