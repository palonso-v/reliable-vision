"""Regression guard against using validation/test tensors to build the memory bank."""

from dataclasses import replace

import numpy as np

from reliablevision import training
from reliablevision.config import Config
from reliablevision.data import splits
from reliablevision.model import PatchCore


def test_fit_and_threshold_partitions(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "torch"))
    config = replace(Config(), tracking_uri=f"sqlite:///{tmp_path.as_posix()}/tracking.db")
    train, validation, test = splits(config)
    monkeypatch.setattr(training, "splits", lambda _: (train, validation, test))
    original_fit = PatchCore.fit
    fitted = []
    validation_scores = []

    def record_fit(self, images, *args):
        fitted.extend(images)
        original_fit(self, images, *args)
        validation_scores.extend(float(self(s.image[None])[0][0]) for s in validation)

    monkeypatch.setattr(PatchCore, "fit", record_fit)
    output = training.run(config)
    assert [id(image) for image in fitted] == [id(sample.image) for sample in train]
    assert not ({id(image) for image in fitted} & {id(s.image) for s in validation + test})
    loaded = PatchCore.load(output / "model.pt")
    assert loaded.threshold == float(
        np.quantile(validation_scores, config.threshold_quantile, method="higher")
    )
