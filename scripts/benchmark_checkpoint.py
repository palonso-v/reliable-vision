"""Measure CPU inference on real images, including native PyTorch process memory."""

import argparse
import hashlib
import json
import threading
from pathlib import Path
from time import perf_counter

import numpy as np
import psutil
import torch
from PIL import Image

from reliablevision.model import PatchCore
from reliablevision.preprocessing import image_tensor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/bottle/model.pt"))
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--requests", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("artifacts/bottle-latency-224.json"))
    args = parser.parse_args()
    if min(args.size, args.requests) < 1 or args.warmup < 0:
        parser.error("Invalid sizes or counts")
    if not args.output.resolve().is_relative_to(Path.cwd().resolve()):
        parser.error("Output must stay inside workspace")
    torch.set_num_threads(2)
    process = psutil.Process()
    baseline = process.memory_info().rss
    peak = [baseline]
    stop = threading.Event()

    def monitor() -> None:
        while not stop.is_set():
            peak[0] = max(peak[0], process.memory_info().rss)
            stop.wait(0.02)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        start_load = perf_counter()
        model = PatchCore.load(args.artifact, "cpu")
        load_seconds = perf_counter() - start_load
        after_load = process.memory_info().rss
        config = model.metadata["config"]
        root = Path(config["data_root"]) / config["category"]
        files = sorted((root / "test").glob("*/*.png"))
        chosen = [files[int(i)] for i in np.linspace(0, len(files) - 1, args.requests)]
        records = []
        for index in range(args.warmup + args.requests):
            path = chosen[(index - args.warmup) % len(chosen)]
            started = perf_counter()
            with Image.open(path) as source:
                original_size = source.size
                batch = image_tensor(source, args.size)[None]
            before_forward = perf_counter()
            score, anomaly_map = model(batch)
            forward_seconds = perf_counter() - before_forward
            total_seconds = perf_counter() - started
            if not torch.isfinite(score).all() or not torch.isfinite(anomaly_map).all():
                raise RuntimeError("Nonfinite prediction")
            if index >= args.warmup:
                records.append(
                    {
                        "image": str(path.relative_to(root)),
                        "source_size": list(original_size),
                        "forward_seconds": forward_seconds,
                        "decode_resize_forward_seconds": total_seconds,
                    }
                )
        memory = process.memory_info()
    finally:
        stop.set()
        thread.join()

    def summary(key: str) -> dict[str, float]:
        values = [record[key] for record in records]
        return {
            "p50_ms": float(np.percentile(values, 50) * 1000),
            "p95_ms": float(np.percentile(values, 95) * 1000),
            "mean_ms": float(np.mean(values) * 1000),
        }

    with args.artifact.open("rb") as stream:
        artifact_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {
        "data": "real MVTec AD bottle test images",
        "device": "cpu",
        "torch_threads": 2,
        "concurrency": 1,
        "input_size": args.size,
        "artifact_training_input_size": config["image_size"],
        "native_resolution": all(
            record["source_size"] == [args.size, args.size] for record in records
        ),
        "artifact_sha256": artifact_hash,
        "quality_caveat": "Changing inference resolution does not revalidate model quality",
        "requests": args.requests,
        "warmup": args.warmup,
        "load_seconds": load_seconds,
        "rss_before_load_bytes": baseline,
        "rss_after_load_bytes": after_load,
        "rss_end_bytes": memory.rss,
        "sampled_peak_rss_bytes": peak[0],
        "os_lifetime_peak_rss_bytes": getattr(memory, "peak_wset", None),
        "memory_sampling_seconds": 0.02,
        "memory_scope": "Whole process including imports, model load, warmup and inference",
        "forward": summary("forward_seconds"),
        "decode_resize_forward": summary("decode_resize_forward_seconds"),
        "samples": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2))


if __name__ == "__main__":
    main()
