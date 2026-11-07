import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from reliablevision.model import PatchCore
from reliablevision.preprocessing import image_tensor
from reliablevision.serving.app import create_app
from reliablevision.serving.runtime import APIError, Runtime
from reliablevision.serving.settings import Settings


@pytest.fixture
def artifact(tmp_path):
    path = tmp_path / "model.pt"
    torch.set_num_threads(2)
    model = PatchCore(True, chunk=64)
    model.fit([torch.full((3, 32, 32), 0.4)], 0.1, 128, 16, 42)
    model.threshold = 0.1
    model.save(
        path, {"config": {"image_size": 32, "mode": "synthetic_smoke", "category": "synthetic"}}
    )
    (tmp_path / "registry.json").write_text(json.dumps({"name": "test", "version": 7}))
    return path


@pytest.fixture
def client(artifact):
    with TestClient(create_app(Settings(artifact=str(artifact)))) as test_client:
        yield test_client


def png(size=(40, 24), mode="RGB") -> bytes:
    output = io.BytesIO()
    Image.new(mode, size, 128).save(output, "PNG")
    return output.getvalue()


def test_prediction_matches_original_artifact(client, artifact):
    payload = png()
    response = client.post("/predict", content=payload, headers={"Content-Type": "image/png"})
    assert response.status_code == 200
    prediction = response.json()
    model = PatchCore.load(artifact)
    with Image.open(io.BytesIO(payload)) as source:
        score, maps = model(image_tensor(source, 32)[None])
    assert prediction["score"] == pytest.approx(float(score[0]))
    np.testing.assert_allclose(prediction["anomaly_map"], maps[0].numpy())
    assert prediction["is_anomaly"] == (prediction["score"] > model.threshold)
    assert prediction["registry_version"] == 7
    assert prediction["synthetic"] is True
    assert prediction["request_id"] == response.headers["x-request-id"]
    assert len(prediction["model_version"]) == 64
    assert prediction["inference_seconds"] >= 0
    assert prediction["map_shape"] == [32, 32]
    assert "probability" not in prediction


def test_health_info_and_metrics(client):
    assert client.get("/health").json()["status"] == "ready"
    info = client.get("/model-info").json()
    assert info["device"] == "cpu" and info["max_concurrent_predictions"] == 1
    client.post("/predict", content=png(), headers={"Content-Type": "image/png"})
    client.post("/predict", content=b"invalid", headers={"Content-Type": "image/png"})
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    for name in (
        "rv_http_requests_total",
        "rv_http_request_seconds",
        "rv_inference_seconds",
        "rv_anomaly_score",
        "rv_predictions_active",
    ):
        assert name in response.text
    assert 'status="422"' in response.text


@pytest.mark.parametrize(
    ("payload", "media", "status", "code"),
    [
        (b"", "image/png", 422, "empty_image"),
        (b"broken", "image/png", 422, "invalid_image"),
        (b"arbitrary", "text/plain", 415, "unsupported_media_type"),
        (png(), "image/jpeg", 415, "image_type_mismatch"),
    ],
)
def test_validation(client, payload, media, status, code):
    response = client.post("/predict", content=payload, headers={"Content-Type": media})
    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert response.json()["request_id"] == response.headers["x-request-id"]


def test_upload_limits_and_chunked_bodies(artifact):
    settings = Settings(artifact=str(artifact), max_upload_bytes=128, max_image_pixels=100)
    with TestClient(create_app(settings)) as client:
        headers = {"Content-Type": "image/png"}
        assert client.post("/predict", content=b"x" * 129, headers=headers).status_code == 413
        # Streaming generator has no Content-Length: still enforce actual received bytes.
        assert (
            client.post("/predict", content=iter([b"x" * 64] * 3), headers=headers).status_code
            == 413
        )
        response = client.post("/predict", content=png(), headers=headers)
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "image_too_large"
        response = client.post(
            "/predict", content=b"x", headers={**headers, "Content-Length": "invalid"}
        )
        assert response.status_code == 400


@pytest.mark.parametrize("mode", ["L", "RGBA"])
def test_supported_color_conversion(client, mode):
    assert (
        client.post(
            "/predict", content=png(mode=mode), headers={"Content-Type": "image/png"}
        ).status_code
        == 200
    )


def test_unexpected_error_is_sanitized_and_capacity_released(client, monkeypatch):
    runtime = client.app.state.runtime
    original = runtime.infer

    def fail(batch):
        raise RuntimeError("private filesystem path and secret")

    monkeypatch.setattr(runtime, "infer", fail)
    response = client.post("/predict", content=png(), headers={"Content-Type": "image/png"})
    assert response.status_code == 500
    assert "secret" not in response.text
    assert response.json()["error"]["code"] == "internal_error"
    monkeypatch.setattr(runtime, "infer", original)
    assert (
        client.post("/predict", content=png(), headers={"Content-Type": "image/png"}).status_code
        == 200
    )


def test_concurrency_rejection_keeps_health_responsive(client, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    runtime = client.app.state.runtime
    original = runtime.predict

    def delayed(payload, media_type):
        entered.set()
        assert release.wait(10)
        return original(payload, media_type)

    monkeypatch.setattr(runtime, "predict", delayed)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(
            client.post, "/predict", content=png(), headers={"Content-Type": "image/png"}
        )
        try:
            assert entered.wait(10)
            response = client.post("/predict", content=png(), headers={"Content-Type": "image/png"})
            assert response.status_code == 503
            assert response.headers["retry-after"] == "1"
            assert client.get("/health").status_code == 200
        finally:
            release.set()
        assert pending.result().status_code == 200


def test_startup_rejects_missing_or_invalid_artifacts(artifact):
    settings = Settings(artifact=str(artifact))
    with (
        pytest.raises(FileNotFoundError),
        TestClient(create_app(replace(settings, artifact=str(artifact.parent / "missing.pt")))),
    ):
        pass
    checkpoint = torch.load(artifact, weights_only=True)
    checkpoint["threshold"] = float("nan")
    torch.save(checkpoint, artifact)
    with pytest.raises(ValueError, match="threshold"), TestClient(create_app(settings)):
        pass


def test_runtime_lock(artifact):
    runtime = Runtime(Settings(artifact=str(artifact)))
    runtime.lock.acquire()
    try:
        with pytest.raises(APIError, match="occupied"):
            runtime.predict(png(), "image/png")
    finally:
        runtime.lock.release()


def test_unknown_route_uses_structured_error(client):
    assert client.get("/does-not-exist").json()["error"]["code"] == "http_error"


def test_jpeg_and_multiframe_validation(client):
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), (100, 100, 100)).save(buffer, "JPEG")
    headers = {"Content-Type": "image/jpeg"}
    assert client.post("/predict", content=buffer.getvalue(), headers=headers).status_code == 200
    assert (
        client.post("/predict", content=buffer.getvalue()[:100], headers=headers).status_code == 422
    )
    frames = [Image.new("RGB", (32, 32), color) for color in ("red", "blue")]
    buffer = io.BytesIO()
    frames[0].save(buffer, "PNG", save_all=True, append_images=frames[1:], duration=100)
    response = client.post(
        "/predict", content=buffer.getvalue(), headers={"Content-Type": "image/png"}
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "animated_image"


def test_json_log_format():
    import logging

    from reliablevision.serving.app import JsonFormatter

    record = logging.LogRecord("test", logging.INFO, "", 0, "request_completed", (), None)
    record.fields = {"request_id": "abc", "status": 200}
    parsed = json.loads(JsonFormatter().format(record))
    assert parsed["request_id"] == "abc" and parsed["event"] == "request_completed"
    assert parsed["timestamp"].endswith("+00:00")
