from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class Features(nn.Module):
    def __init__(self, smoke: bool, pretrained: bool = True) -> None:
        super().__init__()
        self.smoke = smoke
        if not smoke:
            from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
            from torchvision.models.feature_extraction import create_feature_extractor

            backbone = wide_resnet50_2(
                weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1 if pretrained else None
            )
            self.extractor = create_feature_extractor(
                backbone, return_nodes={"layer2": "a", "layer3": "b"}
            )
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])
        self.requires_grad_(False)
        self.eval()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.smoke:
            # Deliberately weight-free fixture; not an ImageNet PatchCore benchmark.
            return F.avg_pool2d(images, 3, stride=2, padding=1)
        features = self.extractor((images - self.mean) / self.std)
        a = F.avg_pool2d(features["a"], 3, stride=1, padding=1)
        b = F.avg_pool2d(features["b"], 3, stride=1, padding=1)
        return torch.cat(
            [a, F.interpolate(b, a.shape[-2:], mode="bilinear", align_corners=False)], dim=1
        )


def greedy_coreset(patches: torch.Tensor, count: int) -> torch.Tensor:
    count = min(count, len(patches))
    selected = torch.zeros(len(patches), dtype=torch.bool, device=patches.device)
    distances = torch.full((len(patches),), float("inf"), device=patches.device)
    indices = []
    index = 0
    for _ in range(count):
        indices.append(index)
        selected[index] = True
        distances = torch.minimum(distances, (patches - patches[index]).square().sum(1))
        distances[selected] = -1
        index = int(distances.argmax())
    return patches[indices].clone()


class PatchCore(nn.Module):
    def __init__(self, smoke: bool, chunk: int = 1024, pretrained: bool = True) -> None:
        super().__init__()
        self.features = Features(smoke, pretrained)
        self.register_buffer("memory", torch.empty(0, 0))
        self.chunk = chunk
        self.smoke = smoke
        self.threshold = float("nan")
        self.metadata: dict[str, Any] = {}

    @torch.inference_mode()
    def fit(
        self, images: list[torch.Tensor], ratio: float, max_patches: int, max_memory: int, seed: int
    ) -> None:
        device = next(self.features.buffers()).device
        # Bounded priority reservoir: each patch receives an independent random priority.
        generator = torch.Generator().manual_seed(seed)
        pool = torch.empty(0, 0, device=device)
        priorities = torch.empty(0)
        for image in images:
            features = self.features(image[None].to(device))
            patches = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
            pool = patches if pool.numel() == 0 else torch.cat([pool, patches])
            priorities = torch.cat([priorities, torch.rand(len(patches), generator=generator)])
            keep = priorities.topk(min(max_patches, len(priorities))).indices
            priorities = priorities[keep]
            pool = pool[keep.to(device)]
        if not pool.numel():
            raise ValueError("Empty training data")
        self.memory = greedy_coreset(pool, max(1, min(max_memory, int(len(pool) * ratio))))

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.memory.numel():
            raise RuntimeError("Model has no memory bank")
        features = self.features(images)
        batch, channels, height, width = features.shape
        patches = features.permute(0, 2, 3, 1).reshape(-1, channels)
        distances = torch.cat(
            [torch.cdist(p, self.memory).min(1).values for p in patches.split(self.chunk)]
        )
        maps = distances.reshape(batch, 1, height, width)
        scores = maps.flatten(1).max(1).values
        maps = F.interpolate(maps, images.shape[-2:], mode="bilinear", align_corners=False)
        return scores, maps[:, 0]

    def save(self, path: Path, metadata: dict[str, Any]) -> None:
        torch.save(
            {
                "schema": 1,
                "smoke": self.smoke,
                "chunk": self.chunk,
                "threshold": self.threshold,
                "metadata": metadata,
                "state": {k: v.cpu() for k, v in self.state_dict().items()},
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "PatchCore":
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        if artifact["schema"] != 1:
            raise ValueError("Unsupported artifact schema")
        model = cls(artifact["smoke"], artifact["chunk"], pretrained=False)
        model.memory = artifact["state"]["memory"]
        model.load_state_dict(artifact["state"])
        model.threshold = artifact["threshold"]
        model.metadata = artifact["metadata"]
        return model.to(device).eval()
