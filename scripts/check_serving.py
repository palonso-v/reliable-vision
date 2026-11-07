"""Check a running service against an existing local PatchCore artifact."""

import argparse
import hashlib
import io
import json
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from reliablevision.model import PatchCore
from reliablevision.preprocessing import image_tensor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/smoke/model.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/docker-check.json"))
    args = parser.parse_args()
    if any(
        not p.resolve().is_relative_to(Path.cwd().resolve()) for p in (args.artifact, args.output)
    ):
        parser.error("Artifact and output must stay inside workspace")
    torch.set_num_threads(2)
    model = PatchCore.load(args.artifact)
    size = model.metadata["config"]["image_size"]
    with args.artifact.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    with urllib.request.urlopen(f"{args.url}/health", timeout=30) as response:
        assert json.load(response)["model_version"] == digest
    with urllib.request.urlopen(f"{args.url}/model-info", timeout=30) as response:
        info = json.load(response)
        assert info["model_version"] == digest and info["device"] == "cpu"
    pixels = np.random.default_rng(42).integers(80, 120, (37, 45, 3), dtype=np.uint8)
    source = Image.fromarray(pixels)
    payload = io.BytesIO()
    source.save(payload, "PNG")
    request = urllib.request.Request(
        f"{args.url}/predict", data=payload.getvalue(), headers={"Content-Type": "image/png"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.load(response)
        assert result["request_id"] == response.headers["x-request-id"]
    score, maps = model(image_tensor(source, size)[None])
    np.testing.assert_allclose(result["score"], float(score[0]), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(result["anomaly_map"], maps[0].numpy(), rtol=1e-4, atol=1e-5)
    assert result["model_version"] == digest
    assert result["is_anomaly"] == (result["score"] > model.threshold)
    bad_request = urllib.request.Request(
        f"{args.url}/predict", data=b"broken", headers={"Content-Type": "image/png"}
    )
    try:
        urllib.request.urlopen(bad_request, timeout=30)
    except urllib.error.HTTPError as error:
        assert error.code == 422 and json.load(error)["error"]["code"] == "invalid_image"
    else:
        raise AssertionError("Corrupt image was accepted")
    with urllib.request.urlopen(f"{args.url}/metrics", timeout=30) as response:
        assert "rv_inference_seconds_count" in response.read().decode()
    report = {
        "model_version": digest,
        "model_info": info,
        "status": "passed",
        "checks": [
            "health",
            "model-info",
            "prediction score/map parity",
            "structured invalid-image error",
            "metrics",
        ],
        "local_score": float(score[0]),
        "remote_score": result["score"],
        "max_map_absolute_difference": float(
            np.max(np.abs(np.array(result["anomaly_map"]) - maps[0].numpy()))
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Serving parity checks passed; evidence: {args.output}")


if __name__ == "__main__":
    main()
