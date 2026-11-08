# ReliableVision completion report

**8 November 2026 — all four milestones implemented and locally validated.**
The repository is ready for technical portfolio review and local reproduction.
No production deployment, hosted CI success, public publication or business-impact
claim is implied. PatchCore is the baseline method; the contribution is the
engineering workflow and evidence around it.

## Delivered system

- Separate training/evaluation/serving, shared preprocessing and a versioned
  PatchCore artifact loaded by the CPU API.
- Seeded nominal splits, content manifests, training-only memory construction,
  nominal-validation-only thresholds and raw-distance semantics.
- Clean/corrupted image AUROC/AP, pixel AUROC and normalized AUPRO reporting.
- Local MLflow experiment tracking and custom-bundle model registry versions.
- FastAPI inference, readiness, identity, metrics and persistent review endpoints;
  bounded inputs, sanitized errors and structured application logs.
- Prometheus/Grafana, fixed-reference input drift, independent traffic classes,
  transactional review persistence and privacy-minimized storage.
- CPU smoke workflow, hash-pinned serving dependencies, digest-pinned images,
  scoped non-root Compose state initialization, tests and GitHub Actions.
- A polished README/architecture, reproduction guide, historical reports,
  CV/LinkedIn/interview summary and **16 final answered interview questions**.

## Verified final checks

| Check | Observed outcome |
| --- | --- |
| Windows full test suite | **40 passed**, two upstream warnings, **48.48 s**. |
| Separate Linux CPU container suite | **40 passed**, three warnings, **94.76 s**. |
| Constrained Linux development installation | Passed on the pinned CPU runtime; dependency consistency passed. |
| Fresh smoke training/evaluation | Passed on Windows and Linux with separate output/tracking paths; ten conditions and registered bundles created. |
| Fixed-reference construction | Passed on fresh smoke; real bottle reference verified **167 recorded training images** and embedded manifest hash. |
| Final Compose build/start | Passed using `reliablevision:local`, state initialization and fresh smoke/review directories. |
| Actual smoke HTTP parity | Score equal; maximum map difference **1.862645149230957e-09**. |
| Simulation/monitoring | **80 simulated predictions**, two shifted-window flags; actual scrape up=1 and firing input-drift alert. |
| Grafana | Eight provisioned panels; all **nine PromQL queries** accepted. |
| Review persistence | A resolved decision with the fresh model/reference identity survived an actual API restart. |
| Real dataset re-audit | 167/42/83 partitions, all sources 900x900, no encoded or decoded exact duplicates found. |
| Real model HTTP check | Same-Linux-runtime map parity exact; Windows/Linux score equal, small map difference investigated below. |
| Quality checks | Ruff lint/format, compilation, host/container dependency checks and Git whitespace checks passed. |
| Cleanup | Temporary services/network stopped; workspace evidence/state retained and ignored by Git. |

Windows warnings are Starlette/httpx TestClient deprecation and MLflow/SQLAlchemy
`noload` deprecation. Linux also warned about pytest cache permissions on a reused
Windows bind mount; the documented container command now uses an artifact-local
cache path. Training temporarily inside the minimal serving image reported no Git
executable for optional MLflow Git tags. It still saved explicit split/config/
environment artifacts. Neither full test suite had failures. These are local
observations, not hosted CI logs.

The fresh smoke model hash was
`c8ba678384aaa9d288ae652f53d6747a73c108a5d6ff36b03a2cdee591ec5430`;
reference hash was
`285be0c69a46aadfd45f1b760a8623e6b65cfe53ecf3f18ee1b18105c23e3326`.
There were 61 pending records before the demo resolution, including the preceding
serving check and simulation; these are not accuracy labels. Generated final
evidence stays in ignored `artifacts/final-*` and `artifacts/monitoring-check.json`.

## Real-image numerical discrepancy and resolution

The initial Windows-to-Linux real-image map check **failed** its strict comparison:
18 of 50,176 pixels exceeded `rtol=1e-4, atol=1e-5`. Maximum difference was
**3.2551586627960205e-05**. This was a verification failure, not an image decision
error or a failed unit-test suite. It was investigated rather than hidden.

For `test/broken_small/000.png` (900x900), preprocessing tensors were byte-identical:
SHA-256 `2654322001c831e208b1f75db205499117801dddc584e834842a410317256b49`.
Same-Linux-runtime direct inference/API comparison passed the original strict
tolerance with **zero map difference**. Both Windows and Linux image scores were
exactly **2.4554145336151123**, with unchanged threshold **1.9235504865646362**.
This isolates a cross-runtime numerical difference rather than a separate input/
scoring pipeline, consistent with float32 CPU-kernel behavior.

The checker retains its strict default and now accepts an explicit, recorded
`--map-atol`. Cross-platform verification passed at **5e-5** absolute tolerance
with unchanged relative tolerance. This is a measured bound for this comparison,
not a universal numerical guarantee or detector/calibration threshold. No weights,
algorithm, data partition or decision threshold were changed. Investigate larger
discrepancies separately.

## Real evaluation versus synthetic demonstrations

The original frozen bottle experiment used **167 nominal train / 42 nominal
validation / 83 final test** images, one seed, 900x900 sources and 224x224 model
inputs/maps/masks. It is not the full MVTec benchmark.

| Real bottle condition | Image AUROC | Image AP | Pixel AUROC | AUPRO <= 0.30 |
| --- | ---: | ---: | ---: | ---: |
| Clean | 1.0000 | 1.0000 | 0.9849 | 0.9337 |
| Severe noise | 0.9254 | 0.9794 | 0.9316 | 0.7458 |
| Severe blur | 0.9810 | 0.9940 | 0.9525 | 0.8478 |

CPU forward p50/p95 at 224x224 was **419/472 ms**, **659 MiB** process peak across
load/warmup/inference. Native 900x900 resource-only inference was **6.30/6.42 s**,
**835 MiB**, using the same 224-trained bank. Complete original fit/evaluation/
tracking took **858.93 s**, with **2.58 GiB** process-tree peak. These historical
measurements were not redefined as M4 performance benchmarks. The frozen model was
loaded and checked, not retrained/tuned against final-test outcomes in this milestone.

The synthetic smoke model uses weight-free pooled RGB features at 32x32. Its
scores/alerts demonstrate software contracts only. Baseline score flags occurred
without brightness drift in the monitoring demo; neither that observation nor two
shift alerts measures industrial accuracy. Full original measurements, definitions
and limitations are in [BOTTLE_VALIDATION.md](BOTTLE_VALIDATION.md).

## Reproduce and inspect

1. Follow [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for Windows or Linux dependencies.
   Use Python 3.13.15 for the validated environment.
2. Run tests/lint, train a fresh smoke bundle and build its reference. Outputs
   are not overwritten; use a new `--output`/`--tracking-uri` or config for repeats.
3. Build Compose, initialize scoped state, then run serving, simulation,
   Prometheus/Grafana and restart-review checks from the guide.
4. Obtain MVTec AD officially under CC BY-NC-SA 4.0 to reproduce the fixed bottle
   audit/profile/analysis commands. No dataset download is automated.
5. Keep images, figures, weights, databases and generated runs local. Read
   [PORTFOLIO.md](PORTFOLIO.md) and [LEARNING.md](../LEARNING.md) for presentation
   and answered interview preparation.

GitHub Actions specifies corresponding CPU smoke/Compose checks with immutable
action commits and restricted credentials. It **has not been executed on GitHub
during this work**; no push or remote deployment occurred. The Linux validation
used a clean container for dependencies with workspace source mounted; it is not
evidence of an untouched remote runner or a full Python/platform matrix.

## Remaining limitations and recommended next steps

1. **Broader ML evidence:** validate representative acquisition sessions and other
   categories in a deliberate future evaluation. Preserve an independent final
   test and report uncertainty; do not tune against the recorded bottle test.
2. **Method fidelity:** assess paper reweighting, embeddings/preprocessing,
   smoothing and coreset changes as controlled experiments. The current baseline
   remains frozen and is not an exact paper reproduction.
3. **Operational evidence:** measure realistic full-model HTTP/load/memory,
   failure recovery and deployment outcomes before defining SLOs.
4. **Security/governance:** add authentication/TLS, authenticated reviewer identity,
   external-image governance and retention/backup controls for exposed use.
   Current safeguards are not a security certification. Choose a code publication/
   license policy before public release; dataset terms remain independent.
5. **Monitoring validity:** calibrate windows/thresholds on suitable stationary
   validation traffic; obtain representative labels to establish degradation.
   Review-only labels are biased. No automatic retraining or externally delivered
   alert notifications are implemented.
6. **Scale when justified:** one worker and SQLite are deliberate local limits.
   Distributed coordination would be a separate requirement, not a reason to add
   an orchestration platform to this portfolio now.

All requested milestones are handed off. These limitations are future work, not
unfinished claims of production deployment or generalized benchmark performance.
