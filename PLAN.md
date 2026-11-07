# Implementation plan

Status: Milestone 1 committed; Milestone 2 implemented and validated (see
`docs/MILESTONE_2.md`); Milestones 3 and 4 require further user approval.

## Milestone 1 — model and evaluation
Build an installable typed Python package, TOML configuration, MVTec loader with content-addressed split manifests, frozen pretrained Wide ResNet features, local patch aggregation, deterministic greedy coreset and nearest-neighbor PatchCore scoring. Separate nominal training/validation before fitting. Save a portable versioned model artifact and threshold selected from nominal validation only. Implement image AUROC/AP, pixel AUROC and connected-component AUPRO (FPR <= 0.30), plus deterministic noise, blur and illumination evaluation. Track configurations, metrics and artifacts in local MLflow. Provide offline synthetic CPU smoke mode and leakage/metric/artifact tests.

Acceptance: lint and tests pass; synthetic end-to-end CLI creates reloadable artifacts, split manifest, clean/corrupted metrics and MLflow run. Document implementation deviations and unavailable full-data evidence. Commit, report and pause.

## Milestone 2 — serving and packaging (requires approval)
Add FastAPI prediction, health, model metadata and metrics endpoints; bounded image decoding, schema validation, structured logs and errors. Load artifacts once at startup; distinguish scores from probabilities. Add CPU Docker image, Compose service and API integration tests. Acceptance: endpoint and malformed-input tests, artifact compatibility checks and Docker build where available; commit and pause.

## Milestone 3 — monitoring and simulation (requires approval)
Add Prometheus latency, throughput, error and score metrics; provision Grafana dashboards and Compose monitoring. Build explicitly simulated normal/shift traffic. Persist drift windows and human-review queue with audit records; keep alerts separate from automatic model updates. Acceptance: seeded shift tests, review lifecycle tests and local monitoring checks; commit and pause.

## Milestone 4 — integration and report (requires approval)
Complete end-to-end tests, GitHub Actions lint/test/Docker checks, operating guides, architecture diagram, reproducibility and security documentation. Produce final report separating synthetic checks, MVTec evidence and operational evidence. Acceptance: all available checks pass, documented limitations, clean commit and final handoff.

## Evaluation protocol
One category per model. Seeded holdout from train/good; no official test image participates in fitting, thresholding or model selection. Hash image contents to detect duplicate leakage. Threshold is nominal validation quantile, without a probability interpretation or guaranteed FPR. Fix hyperparameters before final evaluation. Corrupt test RGB only and preserve masks. Full runs require user-provided licensed MVTec AD data; no benchmark claims from synthetic mode.
