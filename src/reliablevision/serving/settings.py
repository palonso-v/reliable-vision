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
    monitoring: bool = False
    reference: str = "artifacts/smoke/drift-reference.json"
    database: str = "artifacts/monitoring/monitor.sqlite"
    drift_window: int = 20
    drift_threshold: float = 0.3
    retention_days: int = 30

    def __post_init__(self) -> None:
        if not Path(self.artifact).resolve().is_relative_to(Path.cwd().resolve()):
            raise ValueError("Artifact must be inside the workspace")
        if not 1 <= self.torch_threads <= 32:
            raise ValueError("torch_threads must be between 1 and 32")
        if not 1 <= self.max_upload_bytes <= 32 * 1024 * 1024:
            raise ValueError("Invalid upload limit")
        if not 1 <= self.max_image_pixels <= 16_000_000:
            raise ValueError("Invalid pixel limit")
        if not 8 <= self.drift_window <= 10000 or not 0 < self.drift_threshold <= 1:
            raise ValueError("Invalid drift window or threshold")
        if not 1 <= self.retention_days <= 365:
            raise ValueError("Invalid retention")
        for value in (self.reference, self.database):
            if not Path(value).resolve().is_relative_to(Path.cwd().resolve()):
                raise ValueError("Monitoring paths must stay in workspace")

    @classmethod
    def from_env(cls) -> "Settings":
        path = Path(os.environ.get("RV_SERVE_CONFIG", "configs/serve.toml"))
        with path.open("rb") as stream:
            config = tomllib.load(stream)
        if "RV_MODEL_PATH" in os.environ:
            config["artifact"] = os.environ["RV_MODEL_PATH"]
        for key, name in (("reference", "RV_REFERENCE_PATH"), ("database", "RV_DATABASE")):
            if name in os.environ:
                config[key] = os.environ[name]
        if "RV_MONITORING" in os.environ:
            if os.environ["RV_MONITORING"] not in {"0", "1"}:
                raise ValueError("RV_MONITORING must be 0 or 1")
            config["monitoring"] = os.environ["RV_MONITORING"] == "1"
        return cls(**config)
