import hashlib
import io
import json
import sqlite3
import uuid
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from reliablevision.model import PatchCore
from reliablevision.monitoring import Monitor, input_features, reference_document
from reliablevision.serving.app import create_app
from reliablevision.serving.settings import Settings


@pytest.fixture
def bundle(tmp_path):
    model = PatchCore(True)
    torch.set_num_threads(2)
    model.fit([torch.full((3, 32, 32), 0.4)], 0.1, 100, 10, 42)
    model.threshold = 0.2
    artifact = tmp_path / "model.pt"
    model.save(
        artifact, {"config": {"mode": "synthetic_smoke", "category": "synthetic", "image_size": 32}}
    )
    version = hashlib.sha256(artifact.read_bytes()).hexdigest()
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps(reference_document([[0.4, 0.01]] * 20, version, {})))
    return Settings(
        artifact=str(artifact),
        monitoring=True,
        reference=str(reference),
        database=str(tmp_path / "monitor.sqlite"),
    ), version


def test_known_total_variation_and_independent_traffic(bundle):
    settings, version = bundle
    monitor = Monitor(
        Path(settings.database),
        Path(settings.reference),
        version,
        8,
    )
    for _ in range(8):
        outcome = monitor.observe([0.4, 0.01], 0.1, 0.2, uuid.uuid4().hex, "simulated")
    assert outcome["input_drift"] is False
    assert outcome["distances"] == {"brightness_mean": 0, "brightness_std": 0}
    for _ in range(7):
        outcome = monitor.observe([0.8, 0.01], 0.1, 0.2, uuid.uuid4().hex, "simulated")
    assert outcome["input_drift"] is None
    assert monitor.pending["unspecified"] == []
    outcome = monitor.observe([0.8, 0.01], 0.1, 0.2, "final", "simulated")
    assert outcome["input_drift"] is True
    assert outcome["distances"]["brightness_mean"] == pytest.approx(1)
    assert outcome["review_id"]


def payload() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (32, 32), (230, 230, 230)).save(output, "PNG")
    return output.getvalue()


def test_api_review_persistence_privacy_and_metrics(bundle):
    settings, version = bundle
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/predict",
            content=payload(),
            headers={"Content-Type": "image/png", "X-RV-Traffic": "simulated"},
        )
        assert response.status_code == 200
        item = client.get("/reviews").json()["items"][0]
        assert item["model_version"] == version and item["traffic"] == "simulated"
        assert item["request_id"] == response.json()["request_id"]
        assert not {"image", "anomaly_map", "ip", "filename", "notes"} & set(item)
        metrics = client.get("/metrics").text
        assert "rv_model_info" in metrics and version in metrics
        assert (
            client.post(
                "/predict", content=b"broken", headers={"Content-Type": "image/png"}
            ).status_code
            == 422
        )
        assert 'status="422",traffic="unspecified"} 1.0' in client.get("/metrics").text
    with TestClient(create_app(settings)) as client:
        assert client.get("/reviews").json()["items"][0]["id"] == item["id"]
        url = "/reviews/" + item["id"]
        assert (
            client.patch(url, json={"decision": "false_alarm", "notes": "private"}).status_code
            == 422
        )
        assert client.patch(url, json={"decision": "insufficient_evidence"}).status_code == 200
        assert client.patch(url, json={"decision": "false_alarm"}).status_code == 409
        resolved = client.get("/reviews?status=resolved").json()["items"][0]
        assert resolved["decision"] == "insufficient_evidence" and resolved["resolved_at"]
        assert client.get("/reviews?limit=10000").status_code == 422
        assert (
            client.patch(
                "/reviews/00000000000000000000000000000000", json={"decision": "false_alarm"}
            ).status_code
            == 404
        )


def test_storage_failure_is_closed_and_does_not_advance_window(bundle, monkeypatch):
    settings, _ = bundle
    with TestClient(create_app(settings)) as client:
        monitor = client.app.state.monitor

        def fail():
            raise sqlite3.OperationalError("sensitive path")

        monkeypatch.setattr(monitor, "connect", fail)
        response = client.post("/predict", content=payload(), headers={"Content-Type": "image/png"})
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "monitoring_unavailable"
        assert "sensitive" not in response.text
        assert not monitor.pending
        assert "rv_monitoring_failures_total 1.0" in client.get("/metrics").text


def test_reference_mismatch_fails_startup(bundle):
    settings, _ = bundle

    path = Path(settings.reference)
    document = json.loads(path.read_text())
    document["model_version"] = "wrong"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="mismatch"), TestClient(create_app(settings)):
        pass


def test_features_and_retention(bundle):

    settings, version = bundle
    assert input_features(torch.full((3, 8, 8), 0.5)) == pytest.approx([0.5, 0])
    monitor = Monitor(Path(settings.database), Path(settings.reference), version)
    monitor.observe([0.4, 0.01], 1, 0.2, "request", "unspecified")
    with monitor.connect() as connection:
        connection.execute("UPDATE reviews SET created_at='2000-01-01T00:00:00+00:00'")
    assert monitor.list_reviews("pending", 50) == []


def test_completed_window_transaction_rollback_and_restart(bundle):
    settings, version = bundle
    monitor = Monitor(Path(settings.database), Path(settings.reference), version, 8)
    monitor.observe([0.4, 0.01], 1, 0.2, "existing", "simulated")
    for index in range(7):
        monitor.observe([0.4, 0.01], 0.1, 0.2, f"base{index}", "simulated")
    previous = monitor.latest_windows()[0]["id"]
    for index in range(7):
        monitor.observe([0.8, 0.01], 0.1, 0.2, f"shift{index}", "simulated")
    with pytest.raises(sqlite3.IntegrityError):
        monitor.observe([0.8, 0.01], 1, 0.2, "existing", "simulated")
    assert len(monitor.pending["simulated"]) == 7
    assert monitor.latest_windows()[0]["id"] == previous
    outcome = monitor.observe([0.8, 0.01], 1, 0.2, "fresh", "simulated")
    restarted = Monitor(Path(settings.database), Path(settings.reference), version, 8)
    assert restarted.latest_windows()[0]["id"] == outcome["window_id"]
    assert restarted.latest_windows()[0]["flagged"] == 1
    assert not restarted.pending


@pytest.mark.parametrize("values", [[[float("nan"), 0]] * 8, [[2, 0]] * 8, [[0, 0]] * 2])
def test_invalid_reference_features(values):
    with pytest.raises(ValueError):
        reference_document(values, "version", {})
