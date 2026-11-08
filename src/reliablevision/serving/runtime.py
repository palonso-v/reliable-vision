import hashlib
import io
import json
import math
import threading
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError

from reliablevision.model import PatchCore
from reliablevision.monitoring import input_features
from reliablevision.preprocessing import image_tensor
from reliablevision.serving.settings import Settings


class APIError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        self.status = status
        self.code = code
        self.message = message


class Runtime:
    """One immutable CPU model; reject overlapping predictions instead of queuing."""

    def __init__(self, settings: Settings) -> None:
        path = Path(settings.artifact)
        torch.set_num_threads(settings.torch_threads)
        self.model = PatchCore.load(path, "cpu")
        metadata = self.model.metadata
        config = metadata.get("config", {})
        size = config.get("image_size")
        if type(size) is not int or not 16 <= size <= 1024:
            raise ValueError("Artifact needs a valid embedded image_size")
        if not math.isfinite(self.model.threshold) or self.model.threshold < 0:
            raise ValueError("Artifact needs a finite nonnegative threshold")
        if self.model.chunk <= 0 or not torch.isfinite(self.model.memory).all():
            raise ValueError("Invalid model memory or distance chunk")
        mode = config.get("mode")
        if mode not in {"synthetic_smoke", "mvtec"}:
            raise ValueError("Artifact needs a recognized data mode")
        if self.model.smoke != (mode == "synthetic_smoke"):
            raise ValueError("Artifact mode and feature extractor disagree")
        self.settings = settings
        self.size = size
        self.lock = threading.Lock()
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        registry_path = path.parent / "registry.json"
        # Registry files are optional provenance; content hash is the immutable identity.
        registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}
        self.info: dict[str, Any] = {
            "model_version": digest,
            "artifact_schema": 1,
            "algorithm": "PatchCore",
            "category": config.get("category", "unknown"),
            "data_mode": mode,
            "synthetic": self.model.smoke,
            "device": "cpu",
            "image_size": size,
            "threshold": self.model.threshold,
            "score_semantics": "raw nearest-neighbor distance; not a probability",
            "registry_name": registry.get("name"),
            "registry_version": registry.get("version"),
            "torch_threads": settings.torch_threads,
            "max_concurrent_predictions": 1,
        }
        # Warm up before readiness; also verifies bank dimensions and model outputs.
        self.infer(torch.zeros(1, 3, size, size))

    def infer(self, batch: torch.Tensor) -> tuple[float, list[list[float]], float]:
        started = perf_counter()
        scores, maps = self.model(batch)
        elapsed = perf_counter() - started
        if not torch.isfinite(scores).all() or not torch.isfinite(maps).all():
            raise RuntimeError("Nonfinite model output")
        return float(scores[0]), maps[0].tolist(), elapsed

    def predict(self, payload: bytes, media_type: str) -> dict[str, Any]:
        if not self.lock.acquire(blocking=False):
            raise APIError(503, "busy", "Prediction capacity is occupied; retry later")
        try:
            try:
                with Image.open(io.BytesIO(payload)) as image:
                    expected = {"image/png": "PNG", "image/jpeg": "JPEG"}[media_type]
                    if image.format != expected:
                        raise APIError(
                            415, "image_type_mismatch", "Image format and header disagree"
                        )
                    if image.width * image.height > self.settings.max_image_pixels:
                        raise APIError(413, "image_too_large", "Decoded image exceeds pixel limit")
                    if getattr(image, "n_frames", 1) != 1:
                        raise APIError(
                            422, "animated_image", "Only single-frame images are supported"
                        )
                    image.load()
                    tensor = image_tensor(image, self.size)
            except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as error:
                raise APIError(422, "invalid_image", "Cannot decode image") from error
            except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
                raise APIError(
                    413, "image_too_large", "Decoded image exceeds safe limits"
                ) from error
            score, anomaly_map, seconds = self.infer(tensor[None])
            return {
                "_input_features": input_features(tensor),
                "score": score,
                "is_anomaly": score > self.model.threshold,
                "threshold": self.model.threshold,
                "anomaly_map": anomaly_map,
                "map_shape": [self.size, self.size],
                "map_coordinates": "resized_model_input",
                "inference_seconds": seconds,
                **self.info,
            }
        finally:
            self.lock.release()
