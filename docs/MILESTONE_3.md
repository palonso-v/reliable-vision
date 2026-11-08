# Milestone 3 technical report

Implemented and validated locally on **8 November 2026**. Milestone 4 has not begun.

## Outcome and design

The existing CPU PatchCore API now has explicit error counters and a loaded-model
identity gauge alongside request/forward latency, request count, active requests
and raw-score histograms. A per-application registry prevents cross-test leakage;
bounded labels exclude request IDs and image data. Compose provisions a five-second
Prometheus scrape, two alert rules and an eight-panel Grafana dashboard. Prometheus
and Grafana image tags are pinned to verified digests. All ports bind loopback.

Input drift uses interpretable luminance mean and standard deviation from the
existing resized input tensor. Fixed ten-bin reference histograms are built offline
using only the model artifact's recorded nominal training IDs and verified content
hashes. Reference/model mismatch prevents monitoring-enabled startup. The default
statistic is per-feature **total variation distance**, evaluated on non-overlapping
20-image windows; either feature above 0.30 flags a window. This is an effect-size
heuristic, not a calibrated significance test or proof of degraded accuracy.

Simulated and unspecified traffic have separate bounded buffers. Completed windows
persist their reference/model identity, threshold, sample count and distances.
Last-window gauges and freshness restore after restart; incomplete buffers reset.
No reference update, retraining or threshold change occurs from live observations.

Anomaly-score flags and drift-window closing predictions create pending SQLite
review records. Records retain model/reference hashes, random request/review IDs,
score/threshold, categorical reasons and timestamps. `/reviews` lists pending or
resolved records; `PATCH /reviews/{id}` records one immutable categorical decision.
Missing/invalid IDs and repeat resolutions have structured errors. No names or
free-text review notes are accepted. Human decisions do not trigger model changes.

Window/review inserts share a transaction. A failure rolls back both and leaves
the input window unchanged. With enabled monitoring, persistence failures return
503 instead of returning a prediction whose review could not be recorded.

No image bytes, source filenames, anomaly maps, per-request input measurements,
IP addresses or user-agent values are stored by the monitoring layer. Persistent
data is restricted to numeric/categorical metadata and aggregate window statistics.
Thirty-day row pruning is activity-based; Prometheus retains seven days. The
monitoring database and infrastructure data remain under ignored workspace artifacts.

## Actual verification

- Full test suite: **39 passed**, two existing upstream deprecation warnings,
  **39.79 seconds** in final validation. Coverage includes TV distance fixtures, independent traffic
  windows, reference/model mismatch, review lifecycle and restart persistence,
  privacy fields, retention, storage errors and transaction rollback. Milestone 1,
  serving and training/threshold boundary regressions also passed.
- Docker CPU API build and `docker compose up -d --wait`: passed. API,
  Prometheus 3.10.0 and Grafana 12.3.1 started locally.
- `promtool check config` validated the configuration and both alert rules.
- The real HTTP simulation sent **80 explicitly simulated predictions** plus one
  intentionally malformed image, which returned **422**. The fixed reference had
  **12 nominal synthetic training samples**, not MVTec or production samples.
- Two baseline windows had brightness TV **0.0666667** and **0.0833333**, below
  the 0.30 threshold. Two shifted windows had brightness TV **1.0** and were
  flagged. Brightness standard-deviation TV was **0** in all four windows.
- The actual Prometheus scrape target returned up=1 and
  **ReliableVisionInputDrift fired with traffic=simulated**. No external alert
  delivery channel was configured or exercised.
- Grafana's HTTP API returned the provisioned **eight-panel dashboard**. This is
  provisioning/API evidence; a visual browser rendering check was not performed.
  All **nine panel PromQL queries** were accepted by the actual Prometheus API.
  Request, latency, error and score curves are separated by bounded traffic labels;
  control-plane requests have a distinct `control` label.
- The queue contained **60 pending review records** before the demo resolution:
  20 baseline-phase score flags and 40 shifted-phase score flags. An explicit
  `insufficient_evidence` decision was submitted. That decision and its exact model
  identity survived a real API process restart.
- Running-container dependency verification passed. Application lint, formatting,
  compilation and host dependency checks are part of final validation.
  All of those checks passed. A final 40-prediction simulation recheck after the
  per-traffic metrics update produced one shifted-window flag; the nine dashboard
  queries and existing persisted review were reverified. The temporary Compose
  services were stopped after verification; workspace-backed state was preserved.

The demo's model hash was
`90f13d8ae82061cc866b53f859dda0571c4ee11568f4a9184f222553da58252d`;
reference hash was
`fcb6636cb1e0c9cdccd4c4bf08433d80431b254212db1f813ee04eabd15448b3`.
Verification output and the generated SQLite database are retained locally under
`artifacts/`, not committed as generated runs. No MVTec images, weights or customer
data are included in this milestone's monitoring demo.

## Unexpected observations and limitations

The smoke detector flagged half of the nominal-baseline simulation images for
review even though neither baseline drift window fired. This is a toy-model score
observation, **not an industrial false-positive estimate**. The tiny bank/validation
holdout and PNG quantization versus float synthetic training are possible causes;
no causal ablation was run. The threshold was not changed to improve the demo.
This directly illustrates why unchanged brightness summaries do not establish
unchanged score behavior, accuracy or confidence calibration.

Grafana initially auto-downloaded optional plugins. Automatic preinstallation is
now disabled in Compose, along with update checks and analytics. Previously cached
optional plugins can remain in the ignored local Grafana state; the dashboard
uses built-in panels and requires none of them. No Python runtime dependency was
added, so the serving wheel lock remains unchanged.

The baseline/shift source is synthetic and both monitoring thresholds and window
sizes are demonstration defaults, not deployment-calibrated operating settings.
Summary brightness features miss many semantic/local defect shifts. Only suitable,
representative labeled outcomes can demonstrate actual performance degradation.
Flagged-review labels alone are selection-biased and do not support overall accuracy.

SQLite, in-memory partial windows, synchronous fail-closed persistence and one API
worker limit scale and availability. Review decisions are durable but do not carry
authenticated reviewer identity. With no retained images, a reviewer must use
separately governed external context or explicitly mark insufficient evidence.
Activity-based expiry is not scheduled/secure erasure. No authentication, TLS,
external notifications, production deployment or sustained operational evidence
was added. Local anonymous Grafana Viewer access is for loopback use only.

## How to reproduce and learn

See [MONITORING.md](MONITORING.md) for startup, fixed-reference construction, the
statistical definition, review endpoints, privacy/retention details and simulation.
[LEARNING.md](../LEARNING.md) contains milestone concepts and five interview questions.
Milestone 3 is ready for review; Milestone 4 requires user approval.
