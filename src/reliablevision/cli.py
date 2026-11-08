import argparse
from dataclasses import replace
from pathlib import Path

from reliablevision.config import Config
from reliablevision.training import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and evaluate ReliableVision PatchCore")
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.toml"))
    parser.add_argument("--output", type=Path, help="Fresh workspace output directory")
    parser.add_argument("--tracking-uri", help="Override local SQLite tracking URI")
    args = parser.parse_args()
    config = Config.load(args.config)
    if args.output is not None:
        config = replace(config, output=str(args.output))
    if args.tracking_uri is not None:
        config = replace(config, tracking_uri=args.tracking_uri)
    print(f"Artifacts: {run(config)}")


if __name__ == "__main__":
    main()
