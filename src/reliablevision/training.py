import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import mlflow
import numpy as np
import torch
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from reliablevision.config import Config
from reliablevision.corruptions import corrupt
from reliablevision.data import Sample, splits
from reliablevision.evaluation import metrics
from reliablevision.model import PatchCore


def evaluate(
    model: PatchCore,
    samples: list[Sample],
    device: str,
    corruption: str | None = None,
    severity: int = 1,
    seed: int = 42,
) -> dict[str, float | None]:
    scores, maps = [], []
    for index, sample in enumerate(samples):
        image = (
            sample.image
            if corruption is None
            else corrupt(sample.image, corruption, severity, seed + index)
        )
        score, anomaly_map = model(image[None].to(device))
        scores.append(float(score[0]))
        maps.append(anomaly_map[0].cpu().numpy())
    return metrics(
        np.array([s.label for s in samples]),
        np.array(scores),
        np.stack([s.mask for s in samples]),
        np.stack(maps),
    )


def run(config: Config) -> Path:
    os.environ.setdefault("TORCH_HOME", str(Path("artifacts/torch").resolve()))
    if not Path(os.environ["TORCH_HOME"]).resolve().is_relative_to(Path.cwd().resolve()):
        raise ValueError("TORCH_HOME must stay in workspace")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    device = "cuda" if config.mode == "mvtec" and torch.cuda.is_available() else "cpu"
    output = Path(config.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be empty: preserve previous run artifacts")
    output.mkdir(parents=True, exist_ok=True)
    train, validation, test = splits(config)
    manifest = {
        name: [
            {
                "id": s.identity,
                "sha256": s.digest,
                "label": s.label,
                "resized_mask_sha256": hashlib.sha256(s.mask.tobytes()).hexdigest(),
            }
            for s in samples
        ]
        for name, samples in (("train", train), ("validation", validation), ("test", test))
    }
    manifest_path = output / "splits.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    model = PatchCore(config.mode == "synthetic_smoke", config.distance_chunk).to(device)
    model.fit(
        [s.image for s in train],
        config.coreset_ratio,
        config.max_patches,
        config.max_memory,
        config.seed,
    )
    validation_scores = [float(model(s.image[None].to(device))[0][0]) for s in validation]
    model.threshold = float(
        np.quantile(validation_scores, config.threshold_quantile, method="higher")
    )
    metadata = {
        "config": asdict(config),
        "device": device,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "split_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "score_semantics": "raw nearest-neighbor distance; not a probability",
        "threshold_source": "held-out nominal validation",
        "preprocessing": "RGB square resize (PIL bicubic); ImageNet mean/std in full mode",
        "weights": "none" if model.smoke else "Wide_ResNet50_2_Weights.IMAGENET1K_V1",
        "implementation": "PatchCore baseline; max patch score; no paper reweighting",
    }
    model.save(output / "model.pt", metadata)
    reloaded = PatchCore.load(output / "model.pt", device)
    results = {"clean": evaluate(reloaded, test, device)}
    for kind in ("noise", "blur", "illumination"):
        for severity in (1, 2, 3):
            results[f"{kind}_{severity}"] = evaluate(
                reloaded, test, device, kind, severity, config.seed
            )
    (output / "metrics.json").write_text(json.dumps(results, indent=2, allow_nan=False))
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    frozen = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (output / "requirements.freeze.txt").write_text(frozen)
    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment("reliablevision")
    with mlflow.start_run(tags={"data_mode": config.mode, "category": config.category}) as run_info:
        mlflow.log_params(asdict(config))
        mlflow.log_param("threshold", model.threshold)
        for condition, values in results.items():
            mlflow.log_metrics({f"{condition}.{k}": v for k, v in values.items() if v is not None})
        mlflow.log_artifacts(str(output), artifact_path="model_bundle")
        (output / "run_id.txt").write_text(run_info.info.run_id)
        client = MlflowClient()
        name = f"reliablevision-{config.category}-{config.mode}"
        try:
            client.create_registered_model(name)
        except MlflowException as error:
            if error.error_code != "RESOURCE_ALREADY_EXISTS":
                raise
        version = client.create_model_version(
            name,
            f"runs:/{run_info.info.run_id}/model_bundle",
            run_id=run_info.info.run_id,
            tags={"data_mode": config.mode, "artifact_schema": "1"},
        )
        (output / "registry.json").write_text(
            json.dumps({"name": name, "version": version.version})
        )
    return output
