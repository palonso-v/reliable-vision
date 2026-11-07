import numpy as np
import torch
from PIL import Image, ImageFilter


def corrupt(image: torch.Tensor, kind: str, severity: int, seed: int) -> torch.Tensor:
    if severity not in (1, 2, 3):
        raise ValueError("Severity must be 1, 2 or 3")
    pixels = image.permute(1, 2, 0).cpu().numpy()
    if kind == "noise":
        pixels = pixels + np.random.default_rng(seed).normal(0, 0.03 * severity, pixels.shape)
    elif kind == "blur":
        source = Image.fromarray((pixels * 255).astype(np.uint8))
        pixels = np.asarray(source.filter(ImageFilter.GaussianBlur(severity))) / 255
    elif kind == "illumination":
        pixels = pixels * (1 + 0.2 * severity)
    else:
        raise ValueError(f"Unknown corruption: {kind}")
    return torch.from_numpy(np.clip(pixels, 0, 1).astype(np.float32)).permute(2, 0, 1)
