# Monitoring, drift and review

## Local startup and simulation

The existing serving extras and hash-locked Docker CPU dependencies are sufficient;
no dataset, pretrained-weight or new Python package download is required for this
demo. A smoke bundle must already exist from Milestone 1. Build its fixed reference
once before starting the now monitoring-enabled Compose stack:

```powershell
.\.venv\Scripts\python scripts/build_drift_reference.py --artifact artifacts/smoke/model.pt
docker compose config --quiet
docker compose up --build -d --wait
.\.venv\Scripts\python scripts/simulate_traffic.py
.\.venv\Scripts\python scripts/check_monitoring.py
```

The reference builder refuses to overwrite an existing file. If the artifact and
its approved reference already exist, skip that command. API startup fails when
monitoring is enabled but its reference is missing, malformed or belongs to another
model hash. There is no automatic fitting, reference updating or synthetic fallback.

Local endpoints:

- API: `http://127.0.0.1:8000/docs`; metrics: `/metrics`; review queue: `/reviews`.
- Prometheus: `http://127.0.0.1:9090`; firing alerts: `/alerts`.
- Grafana: `http://127.0.0.1:3000/d/reliablevision`. Local anonymous **Viewer** access
  is intentional; no login or external account is required.

The simulator is restricted to the smoke model. It sends 40 seeded baseline and
40 brighter synthetic images, then one malformed request to exercise errors.
Every request carries `X-RV-Traffic: simulated`; the response and persisted records
repeat that label. Other traffic is called **unspecified**, never presumed real
production. This header is caller-supplied metadata, not authenticated provenance.
Traffic classes have separate buffers and metrics, with a bounded set of values.
No real production stream is connected. Increasing `--per-phase` may be necessary
if you configure a larger drift window.

`check_monitoring.py` verifies an actual successful scrape, a firing simulated
drift alert, the provisioned eight-panel dashboard, a queue record and an explicit
`insufficient_evidence` review decision. To verify persistence:

```powershell
docker compose restart api
docker compose up -d --wait
.\.venv\Scripts\python scripts/check_monitoring.py --after-restart
docker compose down
```

Verification artifacts are under ignored `artifacts/`. Compose shutdown preserves
the database and monitoring histories in workspace bind mounts under
`artifacts/monitoring`, `artifacts/prometheus` and `artifacts/grafana`.
Linux hosts need those bind directories writable by the container service UIDs
(API 10001; inspect pinned Prometheus/Grafana images for their UIDs). Docker Desktop
was used for validation. Do not grant general filesystem access to fix a permission
error; scope ownership/permissions to these generated state directories.

## Fixed-reference statistical method

Measurements come from the same RGB [0,1], resized tensor used for prediction.
Luminance is `0.2126 R + 0.7152 G + 0.0722 B`; the two monitored features are its
spatial mean and population standard deviation. The bank and model scores do not
fit these measurements. The reference builder reads only the artifact's **nominal
training IDs** from `splits.json`, verifies their hashes and uses the shared
preprocessing. It neither uses validation/test labels nor fits on incoming traffic.
Synthetic references use only the recorded nominal synthetic training IDs.

For each feature, ten fixed equal-width bins cover [0,1]. The reference stores
aggregate bin counts, sample count, feature/edge definitions, training-manifest
hash and model artifact hash. It stores no images or per-reference-image feature
values. At least eight reference samples are required; this low minimum enables
smoke tests and is not a sample-size recommendation for deployment.

For each independent traffic class, collect a **non-overlapping window of 20**
successful predictions by default. Let `p` and `q` be reference and window bin
frequencies. Compute total variation distance:

```text
TV(p, q) = 0.5 * sum(abs(p_bin - q_bin))
```

TV ranges from zero (identical histograms) to one (disjoint bins). A window is
flagged when **either** feature has TV **strictly above 0.30**. These fixed defaults
were specified before running simulation. This is an interpretable empirical
effect-size statistic, **not a hypothesis-test p-value**, confidence level or
calibrated false-alarm probability. Finite samples, bin boundaries, correlated
images and repeated windows can produce false alarms. Changes within a bin or
changes that preserve these two summaries can be missed. Select operating
thresholds/windows using separate stationary validation traffic and a stated
false-alarm budget before any real deployment; do not tune on final-test defects.

The reference file SHA-256 is its version. It is fixed in memory for a process and
never updated from traffic. An intentional new model or reference requires an
offline review and service restart. Model hash mismatch is rejected. Existing
completed windows and reviews retain both model and reference hashes.

Incomplete windows live only in bounded RAM and reset on restart. Completed
windows are persisted transactionally. Last-window distance/flag/freshness gauges
are restored on startup for the same model/reference; counters and latency/score
histograms reset with the process. Alerts require a window timestamp no older than
five minutes. A stale flag can remain visible on the dashboard; inspect freshness
instead of interpreting silence or a stale flag as current evidence.

## Metrics and dashboard

| Metric | Meaning |
| --- | --- |
| `rv_http_requests_total{route,method,status,traffic}` | Request count; rates give throughput. |
| `rv_http_request_seconds` | Application handling through response creation; excludes socket transmission. |
| `rv_http_errors_total{route,status,traffic}` | Explicit HTTP error count, including malformed inputs and overload. |
| `rv_model_info{model_version,data_mode}` | Current artifact identity, value one. |
| `rv_inference_seconds` | Model-forward time only. |
| `rv_anomaly_score` | Raw distance cumulative histogram; not calibrated probabilities. |
| `rv_predictions_active` | Admitted prediction requests. |
| `rv_input_drift_tv{feature,traffic}` | Last completed window's per-feature TV. |
| `rv_input_drift_flag{traffic}` | Last completed window exceeded the threshold. |
| `rv_input_drift_last_window_timestamp_seconds` | Freshness of that window. |
| `rv_drift_windows_total{traffic,flagged}` | Completed windows in this process. |
| `rv_monitoring_failures_total` | SQLite monitoring/review failures. |

All request-controlled dimensions are bounded. IDs, image contents and filenames
are not labels. A single loaded model contributes one identity series; the info
gauge describes current deployment, not retrospective attribution of a rate window
spanning a model change. Use saved versioned records for that purpose.

Prediction request counts, errors, handling/forward timings and raw-score histograms
are labeled by traffic. Dashboard curves keep simulated and unspecified requests
separate. Non-prediction endpoints use `control` for their HTTP traffic label.

Provisioned panels cover throughput, p95 handling/forward latency, errors, artifact
identity, drift distance, drift flags, score bucket rates and monitoring failures.
Histogram quantiles are bucket approximations, especially unreliable with sparse
traffic. Prometheus scrapes/evaluates every five seconds. Rules demonstrate fresh
input drift and an error-rate condition. Firing alerts are visible locally in
Prometheus; **no Alertmanager, email or external notification is configured**.
The `for: 0s` drift alert is for a local demonstration, not a production alert policy.

## Human review lifecycle

A prediction creates one pending record when its raw score exceeds its saved
validation threshold, or when it closes a flagged drift window. Drift-based review
selects the **closing prediction as a representative**; it does not retrospectively
flag every member or assert that this particular image is defective.

```powershell
curl.exe 'http://127.0.0.1:8000/reviews?status=pending&limit=50'
```

Each record stores a random review ID, server-generated request ID, model hash,
reference hash, traffic label, raw score/threshold, reason codes, optional drift
window ID, timestamps and pending/resolved decision. List queries are bounded to
100 items and sorted oldest-first. Use `status=resolved` for completed records.

Submit `PATCH /reviews/{id}` with JSON:

```json
{"decision": "insufficient_evidence"}
```

Allowed decisions are `confirmed_anomaly`, `false_alarm`, `insufficient_evidence`.
The first resolution is immutable; another returns 409. Missing IDs return 404,
malformed IDs/decisions or extra fields return 422. There are no free-text notes
or personal reviewer names. This is a persistent local decision workflow, **not
an authenticated multi-user audit system**. A reviewer needs trustworthy external
context or a separately governed source-image system to confirm defects. Without
it, choose insufficient evidence; the service cannot recreate an unstored image.
No decision automatically changes thresholds, refits a model or updates reference.

Window and review inserts commit in one SQLite transaction. Buffers advance only
after commit; a failed insert leaves the prior window intact. When enabled
monitoring cannot persist a prediction, the service returns sanitized **503
monitoring_unavailable**, increments its failure counter and returns no successful
prediction response. This fail-closed policy trades inference availability for
review durability. A retry has a new request ID; exactly-once client delivery is
not guaranteed. SQLite is for one local worker, not high-throughput distributed use.

## Privacy, retention and operational limits

No raw customer images, encoded bodies, anomaly maps, per-request input feature
vectors, filenames, IP addresses, user agents or free-form content are written by
the monitoring layer. Bodies/maps exist transiently for inference and are returned
to the caller; callers/proxies have their own retention policies. Application logs
omit them. SQLite keeps only numeric/categorical metadata and aggregate window
statistics. Such metadata is minimized, not a guarantee of anonymization.

Review and drift rows expire after 30 days by default, pruned at startup and on
observation/list activity. With no activity there is no scheduled deletion. SQLite
free pages/backups may retain older bytes; pruning is not secure erasure. Prometheus
retention is seven days. Grafana's local database and operator-access logs have
their own lifecycle. Bound mount access and backups to your chosen policy.

Settings in `configs/serve.toml` retain monitoring-off defaults for direct local
startup unless explicitly enabled. Compose sets `RV_MONITORING=1`,
`RV_REFERENCE_PATH=/app/models/drift-reference.json`, `RV_DATABASE=/app/state/monitor.sqlite`.
The corresponding Settings fields `drift_window`, `drift_threshold` and
`retention_days` are configurable through a chosen `RV_SERVE_CONFIG` TOML.

All published ports bind loopback. Review APIs have no authentication or TLS;
Grafana is intentionally local anonymous Viewer. Do not expose this setup as a
production control plane. Run one API worker; the existing admission lock serializes
predictions, and each worker would otherwise maintain its own window and metrics.
Storage overhead and fail-closed persistence add latency; no new production SLO or
sustained-load claim is made. New images use fixed digests; optional Grafana plugin
preinstallation and update checks are disabled. No paid or external cloud service
is needed.

**Input drift is not model-performance degradation.** A benign lighting shift may
change these measurements without harming ranking. New defect types may hurt
performance while leaving brightness summaries unchanged. Model scores may change
without input TV firing. Only suitable, representative labeled outcomes and
delayed performance evaluation can establish degradation. Review-queue labels are
selection-biased; do not compute overall accuracy from flagged samples alone.

References: [Prometheus alerting rules](https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/),
[Grafana provisioning](https://grafana.com/docs/grafana/latest/administration/provisioning/).
