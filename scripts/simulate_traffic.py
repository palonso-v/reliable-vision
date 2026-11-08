"""Explicit synthetic monitoring demonstration. No real production data or dataset download."""

import argparse
import io
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--per-phase", type=int, default=40)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--output", type=Path, default=Path("artifacts/simulated-traffic.json"))
    args = parser.parse_args()
    if args.per_phase < 20 or args.interval < 0:
        parser.error("Need at least 20 samples per phase and a nonnegative interval")
    if not args.output.resolve().is_relative_to(Path.cwd().resolve()):
        parser.error("Output must remain in workspace")
    with urllib.request.urlopen(args.url + "/model-info", timeout=30) as response:
        info = json.load(response)
    if not info["synthetic"]:
        parser.error(
            "Demo requires the synthetic smoke artifact to avoid claiming realistic traffic"
        )
    rng = np.random.default_rng(42)
    events = []
    for phase, mean in (("baseline", 0.4), ("brightness_shift", 0.8)):
        for index in range(args.per_phase):
            size = info["image_size"]
            pixels = rng.normal(mean, 0.015, (size, size, 3)).clip(0, 1)
            payload = io.BytesIO()
            Image.fromarray((pixels * 255).round().astype(np.uint8)).save(payload, "PNG")
            request = urllib.request.Request(
                args.url + "/predict",
                data=payload.getvalue(),
                headers={"Content-Type": "image/png", "X-RV-Traffic": "simulated"},
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.load(response)
            events.append(
                {
                    "phase": phase,
                    "index": index,
                    "score": result["score"],
                    "model_version": result["model_version"],
                    "monitoring": result["monitoring"],
                }
            )
            time.sleep(args.interval)
    request = urllib.request.Request(
        args.url + "/predict",
        data=b"invalid synthetic image",
        headers={"Content-Type": "image/png", "X-RV-Traffic": "simulated"},
    )
    try:
        urllib.request.urlopen(request, timeout=30)
    except urllib.error.HTTPError as error:
        if error.code != 422:
            raise
    else:
        raise RuntimeError("Malformed input was not rejected")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"traffic": "simulated", "seed": 42, "events": events, "invalid_image_status": 422},
            indent=2,
        )
    )
    flagged = sum(bool(e["monitoring"].get("input_drift")) for e in events)
    print(f"SIMULATED traffic complete: {len(events)} predictions; flagged windows: {flagged}")
    if not flagged:
        raise SystemExit("No drift observed: check reference, window size and sample count")


if __name__ == "__main__":
    main()
