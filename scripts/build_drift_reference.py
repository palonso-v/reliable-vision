"""Build immutable aggregate input reference from the artifact's nominal training IDs only."""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

from reliablevision.data import synthetic
from reliablevision.model import PatchCore
from reliablevision.monitoring import input_features, reference_document
from reliablevision.preprocessing import image_tensor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/smoke/model.pt"))
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    if not artifact.is_relative_to(Path.cwd().resolve()):
        parser.error("Artifact must stay in workspace")
    output = artifact.parent / "drift-reference.json"
    if output.exists():
        parser.error("Reference already exists; do not overwrite an approved reference")
    model = PatchCore.load(artifact)
    config = model.metadata["config"]
    manifest_path = artifact.parent / "splits.json"
    entries = json.loads(manifest_path.read_text())["train"]
    values = []
    if config["mode"] == "synthetic_smoke":
        nominal, _ = synthetic(config["seed"], config["image_size"])
        available = {s.identity: s for s in nominal}
        for entry in entries:
            sample = available[entry["id"]]
            if sample.digest != entry["sha256"] or sample.label != 0:
                raise ValueError("Synthetic training provenance mismatch")
            values.append(input_features(sample.image))
    else:
        category = (Path(config["data_root"]) / config["category"]).resolve()
        for entry in entries:
            relative = Path(entry["id"].replace("\\", "/"))
            path = (category / relative).resolve()
            if relative.parts[:2] != ("train", "good") or not path.is_relative_to(category):
                raise ValueError("Reference source must be nominal training data")
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError("Training image hash mismatch")
            with Image.open(path) as image:
                values.append(input_features(image_tensor(image, config["image_size"])))
    with artifact.open("rb") as stream:
        model_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    reference = reference_document(
        values,
        model_hash,
        {
            "source": "artifact training partition only",
            "data_mode": config["mode"],
            "split_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        },
    )
    output.write_text(json.dumps(reference, indent=2), encoding="utf-8")
    print(f"Fixed reference: {output}; nominal training samples: {len(values)}")


if __name__ == "__main__":
    main()
