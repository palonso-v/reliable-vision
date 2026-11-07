# Milestone 1 technical report

## Scope

Model and evaluation pipeline only. FastAPI, Docker, Prometheus, Grafana, simulated
traffic, drift and human review belong to the next approved milestones.

The initial implementation includes a typed installable package, two TOML profiles,
MVTec AD loading, deterministic nominal holdout, content-based duplicate rejection,
frozen pretrained patch features, bounded patch reservoir, greedy coreset,
chunked nearest-neighbor scoring and localization. Artifacts save the actual
extractor state as well as the memory bank; reload does not download weights.

Four ranking/localization metrics are evaluated for clean test images and nine
fixed corruption conditions. Undefined metrics are null rather than fabricated.
MLflow stores configuration, metrics and model bundles in local SQLite tracking
and registers a model version separately for synthetic and MVTec runs.

## Engineering decisions

- Reserve nominal training images for validation before memory-bank fitting;
  use only this holdout for the image-score threshold. No defect examples are
  necessary for threshold fitting. Test masks participate only in evaluation.
- Hash image bytes to detect exact duplicates, and resized masks to record the
  actual evaluation targets. Filenames alone do not establish split independence.
- Bound candidate patches with seeded priority sampling, then apply greedy
  k-center. Chunk query distances instead of allocating the entire query-bank matrix.
- Use a custom versioned torch artifact with weights-only deserialization.
  Persist preprocessing configuration and raw-distance semantics. Treat artifacts
  as trusted local inputs; weights-only loading is not a general untrusted-file boundary.
- Keep a weight-free synthetic CPU profile so pipeline development does not require
  licensed data, internet access at test time, or a GPU.
- Reject nonempty output directories to preserve previous evidence.

## Limitations and remaining evidence

No MVTec dataset is bundled or available in this workspace. Full pretrained MVTec
experiments and model-quality claims therefore remain unverified. The synthetic
profile uses pooled RGB features and is a software fixture, not a trained industrial
detector. No production deployment or operational validation has taken place.

The baseline uses maximum patch distance without paper image-score reweighting,
Gaussian map smoothing or embedding projection. It must not be compared to published
PatchCore results as an exact reproduction. The bounded candidate pool can lose
rare normal patches. Exact coreset selection can still be slow at full scale.
The loader currently holds resized category images and evaluation maps in host
memory; larger datasets need streaming and disk-backed metric accumulation.

Exact-content checks do not catch near duplicates, common acquisition sessions or
semantic leakage. A small nominal validation quantile has uncertain false-positive
behavior. Corruptions are project-defined stresses with brighter illumination only.
Dataset licensing limits commercial use under the standard license.

## Learning and interview preparation

See [LEARNING.md](../LEARNING.md) for the concepts and five technical interview
questions associated with this milestone.

## Validation evidence — 7 November 2026

- Windows, Python 3.13.15, PyTorch 2.15.0, torchvision 0.29.2; exact installed
  dependency versions are recorded in `requirements-lock.txt`.
- `python -m pytest -q`: **13 passed**, one upstream SQLAlchemy/MLflow
  `noload` deprecation warning, 32.37 seconds. Tests include the complete synthetic
  run, SQLite tracking and registry creation, checkpoint reload, metric fixtures,
  corruption determinism, split reproducibility, duplicate rejection, MVTec mask
  loading and invalid configuration.
- `ruff check .`, `ruff format --check .`, Python compilation and `pip check`: passed.
- Standalone `python -m reliablevision.cli --config configs/smoke.toml`: passed.
  It produced ten evaluation conditions and MLflow run
  `0e83db63483d4a648b039ac789139778`, registered as
  `reliablevision-synthetic-synthetic_smoke`, version 1. Generated evidence remains
  local under ignored `artifacts/` and `mlruns/`.
- Full backbone feature-path check with random weights and a 32x32 input:
  expected `(1, 1536, 4, 4)` features confirmed. This validates architecture wiring,
  not pretrained model quality. No pretrained weights were downloaded.
- `torch.cuda.is_available()` returned false. No GPU experiment ran.

Initial checks exposed a dependency-install file lock and a missing parent for
pytest scratch storage. Installation was completed sequentially and the scratch
path was corrected before the successful validation above. The remaining upstream
deprecation warning does not fail tests; future dependency upgrades need validation.

Milestone 2 remains paused pending user approval.
