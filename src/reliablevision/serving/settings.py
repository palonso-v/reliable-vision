import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    artifact: str = "artifacts/smoke/model.pt"
    torch_threads: int = 2
    max_upload_bytes: int = 5 * 1024 * 1024
    max_image_pixels: int = 16_000_000

    def __post_init__(self) -> None:
        if not Path(self.artifact).resolve().is_relative_to(Path.cwd().resolve()):
            raise ValueError("Artifact must be inside the workspace")
        if not 1 <= self.torch_threads <= 32:
            raise ValueError("torch_threads must be between 1 and 32")
        if not 1 <= self.max_upload_bytes <= 32 * 1024 * 1024:
            raise ValueError("Invalid upload limit")
        if not 1 <= self.max_image_pixels <= 16_000_000:
            raise ValueError("Invalid pixel limit")

    @classmethod
    def from_env(cls) -> "Settings":
        path = Path(os.environ.get("RV_SERVE_CONFIG", "configs/serve.toml"))
        with path.open("rb") as stream:
            config = tomllib.load(stream)
        if "RV_MODEL_PATH" in os.environ:
            config["artifact"] = os.environ["RV_MODEL_PATH"]
        return cls(**config)
