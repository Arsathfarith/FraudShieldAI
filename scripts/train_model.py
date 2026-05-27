from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fraudshield.training import train_from_csv


class LocalFile:
    def __init__(self, path):
        self.path = Path(path)

    def save(self, destination):
        Path(destination).write_bytes(self.path.read_bytes())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train FraudShield AI from a CSV dataset.")
    parser.add_argument("csv", help="Path to the dataset CSV file")
    args = parser.parse_args()
    result = train_from_csv(LocalFile(args.csv))
    print(f"Training complete. Best model: {result['best_model']}")
