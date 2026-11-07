import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from reliablevision.config import Config


@dataclass
class Sample:
    identity: str
    digest: str
    image: torch.Tensor
    mask: np.ndarray
    label: int


def load_sample(path: Path, category: Path, size: int) -> Sample:
    label = int(path.parent.name != "good")
    with Image.open(path) as source:
        pixels = np.array(source.convert("RGB").resize((size, size)), dtype=np.float32) / 255
    mask = np.zeros((size, size), dtype=bool)
    if label:
        mask_path = category / "ground_truth" / path.parent.name / f"{path.stem}_mask.png"
        with Image.open(mask_path) as source:
            mask = np.array(source.convert("L").resize((size, size), Image.Resampling.NEAREST)) > 0
        if not mask.any():
            raise ValueError(f"Empty defect mask: {mask_path}")
    return Sample(
        str(path.relative_to(category)),
        hashlib.sha256(path.read_bytes()).hexdigest(),
        torch.from_numpy(pixels).permute(2, 0, 1),
        mask,
        label,
    )


def synthetic(seed: int, size: int) -> tuple[list[Sample], list[Sample]]:
    rng = np.random.default_rng(seed)
    samples = []
    for i in range(24):
        pixels = rng.normal(0.4, 0.015, (3, size, size)).clip(0, 1).astype(np.float32)
        mask = np.zeros((size, size), dtype=bool)
        label = int(i >= 20)
        if label:
            mask[size // 4 : size // 2, size // 4 : size // 2] = True
            pixels[:, mask] = 0.95
        samples.append(
            Sample(
                f"synthetic/{i}",
                hashlib.sha256(pixels.tobytes()).hexdigest(),
                torch.from_numpy(pixels),
                mask,
                label,
            )
        )
    return samples[:16], samples[16:]


def splits(config: Config) -> tuple[list[Sample], list[Sample], list[Sample]]:
    if config.mode == "synthetic_smoke":
        nominal, test = synthetic(config.seed, config.image_size)
    else:
        category = Path(config.data_root) / config.category
        nominal = [
            load_sample(p, category, config.image_size)
            for p in sorted((category / "train" / "good").glob("*.png"))
        ]
        test = [
            load_sample(p, category, config.image_size)
            for p in sorted((category / "test").glob("*/*.png"))
        ]
    if len(nominal) < 3 or not test:
        raise ValueError("Need >=3 nominal images and nonempty test split; check dataset root")
    order = np.random.default_rng(config.seed).permutation(len(nominal))
    nval = max(1, min(len(nominal) - 1, round(len(nominal) * config.validation_fraction)))
    validation = [nominal[i] for i in order[:nval]]
    train = [nominal[i] for i in order[nval:]]
    # Reject identical content even when filenames differ, within and across partitions.
    hashes = [sample.digest for sample in train + validation + test]
    if len(hashes) != len(set(hashes)):
        raise ValueError("Duplicate image content: possible train/test leakage")
    return train, validation, test
