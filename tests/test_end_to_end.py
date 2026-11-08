"""Dataset-free integration of actual training, artifact, reference, serving and review."""

import io
import json
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from reliablevision.config import Config
from reliablevision.serving.app import create_app
from reliablevision.serving.settings import Settings
from reliablevision.training import run
from scripts.build_drift_reference import build_reference


def test_training_to_monitored_prediction_and_review(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "torch"))
    config = replace(Config(), tracking_uri=f"sqlite:///{tmp_path.as_posix()}/tracking.db")
    output = run(config)
    assert len(json.loads((output / "metrics.json").read_text())) == 10
    reference = build_reference(output / "model.pt")
    settings = Settings(
        artifact=str(output / "model.pt"),
        monitoring=True,
        reference=str(reference),
        database=str(tmp_path / "review.sqlite"),
    )
    image = io.BytesIO()
    Image.new("RGB", (48, 40), (204, 204, 204)).save(image, "PNG")
    with TestClient(create_app(settings)) as client:
        identity = client.get("/model-info").json()["model_version"]
        for _ in range(settings.drift_window):
            response = client.post(
                "/predict",
                content=image.getvalue(),
                headers={"Content-Type": "image/png", "X-RV-Traffic": "simulated"},
            )
            assert response.status_code == 200
        result = response.json()
        assert result["model_version"] == identity
        assert result["synthetic"] is True
        assert result["monitoring"]["input_drift"] is True
        assert result["monitoring"]["distances"]["brightness_mean"] == pytest.approx(1)
        review_id = result["monitoring"]["review_id"]
        assert (
            client.patch(
                f"/reviews/{review_id}", json={"decision": "insufficient_evidence"}
            ).status_code
            == 200
        )
        assert 'traffic="simulated"' in client.get("/metrics").text
    with TestClient(create_app(settings)) as client:
        records = client.get("/reviews?status=resolved").json()["items"]
        assert records[0]["id"] == review_id
        assert records[0]["model_version"] == identity
        assert records[0]["decision"] == "insufficient_evidence"
        assert 'rv_input_drift_flag{traffic="simulated"} 1.0' in client.get("/metrics").text
    with pytest.raises(FileExistsError):
        build_reference(output / "model.pt")
    reference.unlink()
    manifest = output / "splits.json"
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="manifest hash"):
        build_reference(output / "model.pt")
