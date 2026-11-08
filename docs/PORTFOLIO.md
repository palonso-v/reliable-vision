# Portfolio presentation

## CV bullets

- Built ReliableVision, an end-to-end visual anomaly detection portfolio system
  using a PyTorch PatchCore baseline, FastAPI, MLflow, Docker Compose,
  Prometheus/Grafana and persistent human review; implemented leakage controls,
  artifact identity, input validation, drift checks and automated integration tests.
- Evaluated one MVTec AD category (bottle, 83 test images): image AUROC/AP 1.0000,
  pixel AUROC 0.9849 and AUPRO 0.9337. Reported noise-induced localization degradation
  and CPU resource costs instead of generalizing to the full benchmark or production.

## Concise LinkedIn/project description

ReliableVision is a locally validated ML engineering portfolio project for
industrial visual anomaly detection. It connects nominal-image PatchCore fitting,
leakage-aware evaluation and MLflow versioning to a CPU FastAPI service, monitoring
dashboard, fixed-reference input drift and a persistent review workflow. The real
MVTec bottle experiment achieved image AUROC 1.0000 and pixel AUROC 0.9849, while
severe noise reduced AUPRO from 0.9337 to 0.7458. The project includes reproducible
CPU smoke tests, Docker packaging and a configured GitHub Actions workflow.
PatchCore is an existing method; the contribution is the engineering around it.
One-category/local validation is not a claim of production readiness.

## Technical interview opening (about 90 seconds)

“I built ReliableVision to demonstrate the work between an anomaly-detection
experiment and a maintainable service. I used PatchCore as a baseline: frozen
ImageNet features and a compact nominal memory bank, with nearest-neighbor image
scores and localization maps. I separated training, nominal validation and final
test data, saved content-hash manifests, and fitted the image threshold only on
nominal validation.

“The FastAPI service loads the same artifact and shared preprocessing used during
evaluation. I added bounded image decoding, model-version traceability, local
MLflow tracking, Prometheus/Grafana monitoring and a SQLite review queue. Input
drift uses a fixed histogram reference, so traffic cannot silently redefine normal.

“The bottle result was strong, but the robustness and resource measurements matter
just as much: severe noise weakened localization, full-model CPU forward p95 was
about 472 ms, and fitting/evaluation peaked near 2.58 GiB. The monitoring demo also
showed score flags without brightness drift. I treat those as limitations to
investigate, not evidence of production reliability. My contribution is the
reproducible workflow, safeguards and operational integration, not a new algorithm.”

## Evidence worth showing

| Topic | Repository evidence |
| --- | --- |
| Real ML evaluation | `docs/BOTTLE_VALIDATION.md`, recorded numeric evidence and local licensed figures. |
| Leakage and reproducibility | `data.py`, `training.py`, split manifests, threshold-boundary tests and reference-manifest checks. |
| Serving parity | `model.py`, shared preprocessing, API tests and `check_serving.py`. |
| Operational tradeoffs | One-worker admission, fail-closed SQLite persistence, traffic labels and privacy-minimized schemas. |
| Delivery and integration | Hash-locked image build, Compose provisioning, full synthetic integration test and CI definition. |

Do not present synthetic smoke scores as MVTec results, one-category AUROC as a
15-category aggregate, raw distances as probabilities, or a configured CI workflow
as a successful hosted run. No uptime, business-impact, novelty or production
deployment claim is supported by this repository.
