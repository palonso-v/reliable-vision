# Milestone 4 — final integration and portfolio handoff

Completed locally on **8 November 2026**, without adding a new model, dataset,
orchestration platform or cloud service.

## Changes

- Added a complete synthetic integration test: fitting/evaluation → saved artifact
  → training-only fixed reference → FastAPI prediction → drift window → review
  resolution → process-lifecycle reload and persisted decision/flag.
- Hardened reference construction by checking the manifest against the hash
  embedded in the fitted artifact before reading training IDs. Modified manifests
  and accidental reference overwrites are rejected. Data paths stay in workspace.
- Added fresh-output/tracking overrides to the existing CLI so smoke runs can be
  repeated without erasing earlier evidence. No scoring, coreset, threshold or
  real-data hyperparameter change was made.
- Added a scoped Docker-state initializer for non-root bind-mount ownership and
  an optional isolated review-state directory. This fixes fresh Linux setup;
  it is not an extra runtime service or repository-wide permission relaxation.
- Added bounded provisioning-readiness retries to the monitoring verification
  client and a real-PNG option to the serving parity checker.
- Added pinned GitHub Actions checks for lint, tests, CPU smoke, Docker build,
  actual HTTP/alert/dashboard checks and review restart persistence. Workflow
  credentials are not persisted and permissions are read-only. No hosted run is
  claimed; publishing/pushing was not requested.
- Rewrote the README, added Windows/Linux reproduction instructions, expected
  artifacts, hardware guidance, a CV/LinkedIn/interview summary and the final report.
  Updated the learning guide with complete-system decisions and **16 answered
  interview questions**, including the five milestone themes of contribution,
  artifact parity, review durability, reproducibility and deployment readiness.

## Validation and review

The complete Windows suite passed **40 tests**, with the two known upstream
deprecation warnings, in **48.48 seconds**. A separate Linux CPU container installed
the constrained development stack and passed **40 tests in 94.76 seconds**. That
Linux run additionally reported a pytest cache permission warning on the reused
Windows bind mount; the documented container command now uses an artifact-local
cache path. Minimal-image training also warned that Git was unavailable for
MLflow's optional automatic Git tags. Those warnings were not model/test failures.

Fresh synthetic training/evaluation and fixed-reference construction succeeded on
Windows and Linux with separate output/tracking paths. The final Windows smoke
bundle was then used in the actual Compose stack, rather than substituting the
old model. API parity, drift alert, dashboard queries and persistent review were
checked over HTTP. Detailed results and remaining limits are in
[FINAL_REPORT.md](FINAL_REPORT.md); real bottle measurements remain in their
separate report and were not replaced by smoke results.

Final code review focused on input bounds, sanitized error paths, trusted-artifact
assumptions, data/reference lineage, bounded metric dimensions, privacy-minimized
records, non-root filesystem access and one-worker concurrency. This is a focused
engineering review, not an independent penetration test or a claim of security
certification. Existing authentication/TLS, scaling, retention and statistical
limitations remain explicitly documented.

## Handoff

The real-image Windows/Linux parity command initially failed a tight map tolerance
at 18 of 50,176 pixels (maximum difference 3.255e-5). Preprocessing was byte-identical
and same-Linux-runtime maps matched exactly. An explicitly recorded comparison
tolerance of 5e-5 passed without any model/threshold change; the checker retains
its strict default. The failure and diagnostics are fully reported in the final
report rather than presented as bitwise cross-platform reproducibility.

- [README](../README.md): project architecture, evidence and quick start.
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md): exact environment/workflow commands.
- [FINAL_REPORT.md](FINAL_REPORT.md): verification ledger and next steps.
- [PORTFOLIO.md](PORTFOLIO.md): concise presentation without novelty/production claims.
- [LEARNING.md](../LEARNING.md): concepts, tradeoffs and answered interview preparation.

All four milestones are complete. No deployment, remote publication or additional
feature work is implied by this handoff.
