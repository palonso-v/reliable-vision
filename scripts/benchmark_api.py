"""Sequential HTTP latency measurement using explicitly synthetic image input."""

import argparse
import io
import json
import platform
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("artifacts/http-latency.json"))
    args = parser.parse_args()
    if args.requests < 1 or args.warmup < 0:
        parser.error("Invalid request count")
    if not args.output.resolve().is_relative_to(Path.cwd().resolve()):
        parser.error("Output must stay in workspace")
    with urllib.request.urlopen(f"{args.url}/model-info", timeout=30) as response:
        info = json.load(response)
    size = info["image_size"]
    pixels = np.random.default_rng(42).integers(80, 120, (size, size, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, "PNG")
    payload = buffer.getvalue()
    http_seconds, inference_seconds = [], []
    for index in range(args.warmup + args.requests):
        request = urllib.request.Request(
            f"{args.url}/predict", data=payload, headers={"Content-Type": "image/png"}
        )
        started = perf_counter()
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.load(response)
        elapsed = perf_counter() - started
        if result["model_version"] != info["model_version"]:
            raise RuntimeError("Model changed during benchmark")
        if index >= args.warmup:
            http_seconds.append(elapsed)
            inference_seconds.append(result["inference_seconds"])

    def summary(values: list[float]) -> dict[str, float]:
        return {
            "p50_ms": float(np.percentile(values, 50) * 1000),
            "p95_ms": float(np.percentile(values, 95) * 1000),
            "mean_ms": float(np.mean(values) * 1000),
        }

    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "traffic": "synthetic benchmark; sequential local HTTP requests; not production",
        "model_info": info,
        "client_platform": platform.platform(),
        "requests": args.requests,
        "warmup": args.warmup,
        "concurrency": 1,
        "image_shape": [size, size, 3],
        "payload_bytes": len(payload),
        "http_latency": summary(http_seconds),
        "forward_latency": summary(inference_seconds),
        "observed_sequential_requests_per_second": len(http_seconds) / sum(http_seconds),
        "http_samples_seconds": http_seconds,
        "forward_samples_seconds": inference_seconds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "http": report["http_latency"],
                "forward": report["forward_latency"],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
