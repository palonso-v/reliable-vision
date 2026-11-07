"""Copy numeric checkpoint evidence and provenance to docs, excluding dataset pixels/weights."""

import hashlib
import json
import platform
from pathlib import Path

import psutil


def read(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    root = Path("artifacts/bottle")
    audit = read("artifacts/bottle-audit.json")
    audit.pop("manifest")
    profile = read("artifacts/bottle-profile.json")
    profile.pop("samples")
    analysis = read("artifacts/bottle/analysis.json")
    report = {
        "scope": "Real MVTec AD bottle; one fixed CPU experiment; no production evidence",
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_processors": psutil.cpu_count(),
            "psutil": psutil.__version__,
        },
        "metrics_resolution": [224, 224],
        "source_dimensions": [900, 900],
        "audit": audit,
        "pipeline_profile": profile,
        "analysis": analysis,
        "metrics": read("artifacts/bottle/metrics.json"),
        "benchmark_224": read("artifacts/bottle-latency-224.json"),
        "benchmark_900": read("artifacts/bottle-latency-900.json"),
        "metadata": read("artifacts/bottle/metadata.json"),
        "registry": read("artifacts/bottle/registry.json"),
        "run_id": (root / "run_id.txt").read_text(),
        "artifact_sha256": digest(root / "model.pt"),
        "artifact_bytes": (root / "model.pt").stat().st_size,
        "pretrained_weights_sha256": digest(
            Path("artifacts/torch/hub/checkpoints/wide_resnet50_2-95faca4d.pth")
        ),
        "source_sha256": {
            str(p): digest(p)
            for p in [
                Path("src/reliablevision") / name
                for name in (
                    "training.py",
                    "model.py",
                    "data.py",
                    "preprocessing.py",
                    "evaluation.py",
                )
            ]
        },
        "raw_resource_profile_sha256": digest(Path("artifacts/bottle-profile.json")),
        "audit_sha256": digest(Path("artifacts/bottle-audit.json")),
        "figures_location": "artifacts/bottle/localization; not committed (dataset derivatives)",
    }
    output = Path("docs/benchmarks/mvtec-bottle-checkpoint.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Numeric checkpoint evidence exported: {output}")


if __name__ == "__main__":
    main()
