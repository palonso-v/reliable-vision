import argparse
from pathlib import Path

from reliablevision.config import Config
from reliablevision.training import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and evaluate ReliableVision PatchCore")
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.toml"))
    args = parser.parse_args()
    print(f"Artifacts: {run(Config.load(args.config))}")


if __name__ == "__main__":
    main()
