"""Audit original image content and existing seeded MVTec partitions without fitting."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from reliablevision.config import Config
from reliablevision.data import splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/mvtec.toml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/bottle-audit.json"))
    args = parser.parse_args()
    config = Config.load(args.config)
    if config.mode != "mvtec":
        parser.error("Audit requires real MVTec mode")
    if not args.output.resolve().is_relative_to(Path.cwd().resolve()):
        parser.error("Output must stay inside workspace")
    train, validation, test = splits(config)
    root = Path(config.data_root) / config.category
    groups = defaultdict(list)
    sizes = Counter()
    manifest = {}
    for name, samples in (("train", train), ("validation", validation), ("test", test)):
        manifest[name] = []
        for sample in samples:
            path = root / sample.identity
            with Image.open(path) as source:
                sizes[str(source.size)] += 1
                pixels = source.convert("RGB")
                digest = hashlib.sha256(str(pixels.size).encode() + pixels.tobytes()).hexdigest()
                dimensions = source.size
            groups[digest].append({"split": name, "id": sample.identity})
            manifest[name].append(
                {
                    "id": sample.identity,
                    "sha256": sample.digest,
                    "decoded_rgb_sha256": digest,
                    "label": sample.label,
                }
            )
            if sample.label:
                path = root / "ground_truth" / path.parent.name / f"{path.stem}_mask.png"
                with Image.open(path) as mask:
                    if mask.size != dimensions:
                        raise ValueError(f"Image/mask dimension mismatch: {path}")
    duplicates = [items for items in groups.values() if len(items) > 1]
    report = {
        "config": str(args.config),
        "counts": {k: len(v) for k, v in manifest.items()},
        "test_nominal": sum(s.label == 0 for s in test),
        "test_anomalous": sum(s.label == 1 for s in test),
        "original_image_dimensions": dict(sizes),
        "model_input": config.image_size,
        "encoded_duplicate_check": "passed by existing data.splits",
        "decoded_rgb_duplicates": duplicates,
        "manifest": manifest,
        "limitations": "No near-duplicate or acquisition-session independence guarantee",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "manifest"}, indent=2))
    if duplicates:
        raise SystemExit("Decoded duplicate image content found; investigate before fitting")


if __name__ == "__main__":
    main()
