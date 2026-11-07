# Milestone 2 technical report

Implemented and validated on 7 November 2026. Milestone 3 has not begun.

## Delivered behavior

FastAPI loads the existing Milestone 1 `model.pt` with `PatchCore.load` during
lifespan startup, validates the artifact and runs a CPU warmup. The loader now
exposes its saved metadata without changing schema 1 or the scoring algorithm.
Dataset loading and serving share one explicit RGB/bicubic/float32 preprocessing
function. No parallel inference algorithm, model fitting or pretrained-weight
download was added to the service.

`/predict`, `/health`, `/model-info` and `/metrics` are implemented. Prediction
responses carry raw distance, threshold decision, resized localization map,
forward latency, request ID, artifact SHA-256, category/data mode and optional
MLflow registry name/version. The synthetic flag remains visible. Error responses
are structured and sanitized; application logs are JSON with UTC timestamps and
bounded route labels. Byte, decoded-pixel, MIME/format and single-frame validation
protect the image decoding path.

Prometheus exposition covers request count/status, application handling latency,
model-forward latency, active predictions and raw-score distributions. Scraping,
dashboards, drift, human review and production-stream simulation remain Milestone 3.

The Docker image pins the official Python 3.13.15 Linux amd64 manifest by digest
and installs a complete hash-locked CPU serving dependency set. No training/MLflow
runtime, dataset or checkpoint is included. Compose mounts a chosen model bundle
read-only, binds localhost port 8000, limits resources, runs non-root, drops
capabilities and enables a healthcheck and read-only root filesystem.

## Engineering decisions and limits

- The exact model hash identifies deployed bytes. Optional registry sidecar values
  add provenance but do not replace that identity or prove registry membership.
- One process and one admitted prediction at a time avoid unconstrained CPU
  oversubscription. Decoding and inference run in a worker thread; overlapping
  requests receive 503/Retry-After. No unbounded application queue is created.
- PyTorch intra-op thread count is process-global and set to two by default.
  More workers duplicate memory and require a different metrics aggregation policy.
  There is no batching, hot reload, online fitting or calibration.
- Raw PNG/JPEG request bodies allow streaming byte limits without multipart spool
  files. Pixel limits apply before full decode. Upstream connection/body timeouts,
  authentication, TLS and per-client rate limits are still needed for exposed use.
- Linux CPU wheels have their own version/hash lock. The Windows development lock
  is separate. The pinned base and complete wheel hashes fix runtime inputs;
  byte-identical OCI output across build timestamps/toolchains is not claimed.
- JSON maps use resized model coordinates and can be expensive at higher
  resolutions. Forward timing excludes decoding, map conversion and serialization;
  HTTP measurements include those costs and client response parsing.

See [SERVING.md](SERVING.md) for configuration, API semantics, reproducible lock
generation, startup/shutdown commands and measurement procedures.

## Validation evidence

- Host: Windows, Python 3.13.15, CPU. Final full suite: **29 passed**, two upstream
  deprecation warnings, **32.95 seconds**. The warnings are Starlette's deprecated
  httpx TestClient adapter and MLflow/SQLAlchemy's `noload` loader strategy.
- Tests use small synthetic checkpoints only. They cover direct score/map parity,
  readiness/info, metrics, PNG/JPEG decoding, grayscale/alpha conversion, malformed
  and animated images, oversized and chunked bodies, structured errors, recovery
  after internal failures, concurrent rejection and health responsiveness,
  startup failures, direct runtime locking, and JSON log structure. All Milestone 1
  regression tests also passed.
- Ruff lint/format checks, Python compilation, host `pip check`, Docker build-time
  `pip check` and running-container `pip check`: passed.
- `docker compose config --quiet`, `docker compose build api` and
  `docker compose up -d --wait`: passed. The service became healthy with the actual
  Milestone 1 smoke bundle mounted read-only. Runtime UID was **10001**;
  PyTorch was **2.15.0+cpu**, with **no CUDA build**.
- `scripts/check_serving.py`: passed over HTTP against the container. All four
  endpoints responded as expected. Direct host score and remote score both were
  **0.031989410519599915**; maximum map absolute difference was
  **1.862645149230957e-09**. Artifact identity was
  `90f13d8ae82061cc866b53f859dda0571c4ee11568f4a9184f222553da58252d`,
  with original smoke registry version **1**.
- The temporary service and Compose network were stopped/removed after validation.
  No deployment is left running. Docker engine and build cache remain available.

The existing checkpoint was used directly; no MVTec dataset or GPU was required.
Evidence copies are in [benchmarks/milestone2-parity.json](benchmarks/milestone2-parity.json)
and [benchmarks/milestone2-smoke.json](benchmarks/milestone2-smoke.json).

## Actual local latency measurement

Docker Desktop/WSL2 Linux amd64 on an **AMD Ryzen 5 7530U**. Compose quota: two CPUs,
2 GiB memory; PyTorch intra-op threads: two; one Uvicorn worker. This is a **32x32
synthetic smoke model**, with a **2,444-byte synthetic PNG**, not the pretrained
Wide ResNet model. The client ran on the Windows host over localhost.

The reported run used **10 warmups and 100 measured sequential requests**, at
concurrency one. Full HTTP timing includes image submission, map response transfer
and client JSON parsing. Forward timing is reported by the server around model
forward only.

| Scope | p50 | p95 | Mean |
| --- | ---: | ---: | ---: |
| HTTP round trip | 11.32 ms | 30.92 ms | 14.04 ms |
| Model forward | 0.87 ms | 1.41 ms | 0.96 ms |

Observed sequential rate was **71.24 requests/s**. This is not maximum throughput
or an SLO. A preliminary run overlapped runtime introspection (HTTP p50/p95
10.51/28.85 ms); it was repeated after that work finished. Both raw local reports
are retained under `artifacts/`, and the latter run's raw samples are committed in
the benchmark evidence file. Host scheduling and virtualization remain uncontrolled.

No full pretrained-checkpoint serving benchmark, MVTec model-quality evaluation,
sustained load test or production operational evidence exists. The tests and local
container run do not establish production readiness. Full-model resource sizing
must be measured independently.

## Learning and review

[LEARNING.md](../LEARNING.md) includes the milestone concepts and five technical
interview questions. Milestone 2 is ready for review; Milestone 3 remains paused
pending explicit user approval.
