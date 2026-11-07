from dataclasses import replace

import numpy as np
import pytest
import torch
from PIL import Image

from reliablevision.config import Config
from reliablevision.corruptions import corrupt
from reliablevision.data import splits
from reliablevision.evaluation import aupro, metrics
from reliablevision.model import PatchCore, greedy_coreset
from reliablevision.training import run


def test_splits_are_reproducible_and_disjoint():
    first = splits(Config())
    second = splits(Config())
    assert [[s.digest for s in part] for part in first] == [
        [s.digest for s in part] for part in second
    ]
    assert all(s.label == 0 for s in first[0] + first[1])
    assert not ({s.digest for s in first[0]} & {s.digest for s in first[1] + first[2]})


def test_duplicate_content_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for folder, count in (("train/good", 3), ("test/good", 1)):
        directory = tmp_path / "data" / "bottle" / folder
        directory.mkdir(parents=True)
        for i in range(count):
            Image.new("RGB", (32, 32), "white").save(directory / f"{i}.png")
    with pytest.raises(ValueError, match="Duplicate"):
        splits(Config(mode="mvtec", data_root="data", category="bottle"))


def test_coreset_and_reload(tmp_path):
    patches = torch.tensor([[0.0], [1.0], [10.0]])
    assert torch.equal(greedy_coreset(patches, 2), patches[[0, 2]])
    model = PatchCore(True)
    image = torch.rand(3, 32, 32)
    model.fit([image], 0.1, 100, 10, 42)
    model.threshold = 0.5
    path = tmp_path / "model.pt"
    model.save(path, {})
    loaded = PatchCore.load(path)
    for a, b in zip(model(image[None]), loaded(image[None]), strict=True):
        torch.testing.assert_close(a, b)
    assert loaded.threshold == 0.5


def test_metrics_perfect_and_reversed():
    masks = np.zeros((2, 8, 8), dtype=bool)
    masks[1, 2:4, 2:4] = True
    values = metrics(np.array([0, 1]), np.array([0, 1]), masks, masks.astype(float))
    assert all(v == pytest.approx(1) for v in values.values())
    assert aupro(masks, (~masks).astype(float)) == pytest.approx(0)
    assert aupro(masks, np.ones(masks.shape)) == pytest.approx(0.15)
    assert aupro(np.zeros_like(masks), masks.astype(float)) is None


def test_aupro_weights_regions_equally():
    masks = np.zeros((2, 8, 8), dtype=bool)
    masks[0, 1, 1] = True
    masks[1, 2:4, 2:4] = True
    scores = np.zeros(masks.shape)
    scores[0, 1, 1] = 1
    assert aupro(masks, scores) == pytest.approx(0.575)


def test_mvtec_layout_and_mask_loading(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for folder, count in (("train/good", 4), ("test/good", 1), ("test/crack", 1)):
        directory = tmp_path / "data" / "bottle" / folder
        directory.mkdir(parents=True)
        for i in range(count):
            value = i + (20 if folder.startswith("test") else 0)
            value += 10 if "crack" in folder else 0
            Image.new("RGB", (32, 32), (value, value, value)).save(directory / f"{i}.png")
    mask_dir = tmp_path / "data" / "bottle" / "ground_truth" / "crack"
    mask_dir.mkdir(parents=True)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:16, 8:16] = 255
    Image.fromarray(mask).save(mask_dir / "0_mask.png")
    train, validation, test = splits(Config(mode="mvtec", data_root="data", category="bottle"))
    assert len(train) == 3 and len(validation) == 1 and len(test) == 2
    assert sum(s.label for s in test) == 1
    assert sum(s.mask.sum() for s in test) == 64


def test_corruptions_are_seeded_and_bounded():
    image = torch.full((3, 32, 32), 0.4)
    for kind in ("noise", "blur", "illumination"):
        result = corrupt(image, kind, 2, 42)
        assert torch.equal(result, corrupt(image, kind, 2, 42))
        assert result.shape == image.shape
        assert 0 <= result.min() <= result.max() <= 1


def test_smoke_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = replace(Config(), tracking_uri=f"sqlite:///{tmp_path.as_posix()}/tracking.db")
    output = run(config)
    assert (output / "model.pt").is_file()
    assert (output / "metrics.json").read_text().count("image_auroc") == 10
    assert (output / "run_id.txt").read_text()
    assert (output / "registry.json").is_file()
    with pytest.raises(ValueError, match="empty"):
        run(config)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"validation_fraction": 0},
        {"coreset_ratio": 2},
        {"category": "../escape"},
        {"category": ".."},
        {"image_size": 0},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)
