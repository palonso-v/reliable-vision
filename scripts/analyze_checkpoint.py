"""Verify validation-only threshold and produce real-data localization examples.

Images and derived figures remain local under ignored artifacts (MVTec license).
Heatmap display scale is fixed by validation scores, not test labels or per-map maxima.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from reliablevision.config import Config
from reliablevision.data import splits
from reliablevision.model import PatchCore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/mvtec.toml"))
    args = parser.parse_args()
    config = Config.load(args.config)
    torch.set_num_threads(2)
    output = Path(config.output)
    model = PatchCore.load(output / "model.pt")
    train, validation, test = splits(config)
    stored_manifest = json.loads((output / "splits.json").read_text())
    for name, samples in (("train", train), ("validation", validation), ("test", test)):
        assert [(s.identity, s.digest) for s in samples] == [
            (entry["id"], entry["sha256"]) for entry in stored_manifest[name]
        ]
    scores = [float(model(s.image[None])[0][0]) for s in validation]
    expected = float(np.quantile(scores, config.threshold_quantile, method="higher"))
    assert expected == model.threshold
    # All validation scores are nominal. No test scores/labels set the scale.
    display_max = max(max(scores), 1e-12)
    rows = []
    first_by_type = {}
    worst_errors = {}
    for sample in test:
        score, maps = model(sample.image[None])
        value = float(score[0])
        predicted = int(value > model.threshold)
        row = {
            "id": sample.identity,
            "label": sample.label,
            "score": value,
            "predicted_anomaly": bool(predicted),
        }
        rows.append(row)
        group = Path(sample.identity).parent.name
        first_by_type.setdefault(group, (sample, maps[0].numpy(), row))
        if predicted != sample.label:
            key = "false_positive" if predicted else "false_negative"
            # Explicit error diagnostics, not representative performance selection.
            margin = abs(value - model.threshold)
            if key not in worst_errors or margin > worst_errors[key][0]:
                worst_errors[key] = (margin, sample, maps[0].numpy(), row)
    examples = [(f"first_{name}", *items) for name, items in sorted(first_by_type.items())]
    examples += [(name, *items[1:]) for name, items in worst_errors.items()]
    figure_dir = output / "localization"
    figure_dir.mkdir(exist_ok=True)
    examples_metadata = []
    for name, sample, anomaly_map, row in examples:
        with Image.open(Path(config.data_root) / config.category / sample.identity) as source:
            image = source.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
        intensity = np.clip(anomaly_map / display_max, 0, 1)
        rgb = np.stack([intensity, np.zeros_like(intensity), 1 - intensity], axis=-1)
        heat = Image.fromarray((rgb * 255).astype(np.uint8)).resize((224, 224))
        overlay = Image.blend(image, heat, 0.45)
        mask = Image.fromarray(sample.mask.astype(np.uint8) * 255).convert("RGB")
        mask = mask.resize((224, 224), Image.Resampling.NEAREST)
        canvas = Image.new("RGB", (896, 268), "white")
        draw = ImageDraw.Draw(canvas)
        for index, (panel, title) in enumerate(
            zip(
                (image, mask, heat, overlay),
                ("Image (resized)", "Ground truth", "Raw map / validation max", "Overlay"),
                strict=True,
            )
        ):
            canvas.paste(panel, (index * 224, 24))
            draw.text((index * 224 + 4, 4), title, fill="black")
        draw.text(
            (4, 250),
            f"{sample.identity} | score={row['score']:.3f} "
            f"threshold={model.threshold:.3f} | {name}",
            fill="black",
        )
        canvas.save(figure_dir / f"{name}.png")
        examples_metadata.append({"selection": name, **row, "figure": f"{name}.png"})
    tp = sum(row["label"] == 1 and row["predicted_anomaly"] for row in rows)
    fp = sum(row["label"] == 0 and row["predicted_anomaly"] for row in rows)
    report = {
        "threshold": model.threshold,
        "recomputed_validation_threshold": expected,
        "threshold_matches": True,
        "validation_count": len(scores),
        "validation_scores": scores,
        "heatmap_display_max": display_max,
        "heatmap_display_source": "maximum nominal validation image score",
        "confusion": {
            "tp": tp,
            "fn": sum(r["label"] == 1 for r in rows) - tp,
            "fp": fp,
            "tn": sum(r["label"] == 0 for r in rows) - fp,
        },
        "test_predictions": rows,
        "examples": examples_metadata,
        "memory_bank_shape": list(model.memory.shape),
        "memory_bank_bytes": model.memory.numel() * model.memory.element_size(),
        "split_manifest_sha256": hashlib.sha256((output / "splits.json").read_bytes()).hexdigest(),
    }
    (output / "analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in {"test_predictions", "validation_scores"}},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
