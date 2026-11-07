"""Shared image preprocessing contract for training and serving."""

import numpy as np
import torch
from PIL import Image


def image_tensor(source: Image.Image, size: int) -> torch.Tensor:
    pixels = (
        np.array(
            source.convert("RGB").resize((size, size), Image.Resampling.BICUBIC), dtype=np.float32
        )
        / 255
    )
    return torch.from_numpy(pixels).permute(2, 0, 1)
