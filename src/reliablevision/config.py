import re
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    mode: str = "synthetic_smoke"
    data_root: str = "data/mvtec_ad"
    category: str = "synthetic"
    image_size: int = 32
    seed: int = 42
    validation_fraction: float = 0.25
    threshold_quantile: float = 0.95
    coreset_ratio: float = 0.1
    max_patches: int = 2048
    max_memory: int = 128
    distance_chunk: int = 256
    output: str = "artifacts/smoke"
    tracking_uri: str = "sqlite:///artifacts/mlflow.db"

    def __post_init__(self) -> None:
        if self.mode not in {"mvtec", "synthetic_smoke"}:
            raise ValueError("Unknown mode")
        if not 0 < self.validation_fraction < 1 or not 0 < self.threshold_quantile < 1:
            raise ValueError("Fractions must be between zero and one")
        if not 0 < self.coreset_ratio <= 1:
            raise ValueError("Invalid coreset ratio")
        if min(self.image_size, self.max_patches, self.max_memory, self.distance_chunk) < 1:
            raise ValueError("Sizes must be positive")
        if self.image_size < 16 or not re.fullmatch(r"[A-Za-z0-9_-]+", self.category):
            raise ValueError("Invalid image size or category")
        root = Path.cwd().resolve()
        for value in (self.data_root, self.output):
            if not Path(value).resolve().is_relative_to(root):
                raise ValueError("Paths must stay in workspace")
        if not self.tracking_uri.startswith("sqlite:///"):
            raise ValueError("Milestone 1 supports local SQLite tracking only")
        database = Path(self.tracking_uri.removeprefix("sqlite:///"))
        if not database.resolve().is_relative_to(root):
            raise ValueError("Tracking database must stay in workspace")

    @classmethod
    def load(cls, path: Path) -> "Config":
        with path.open("rb") as stream:
            return cls(**tomllib.load(stream))
