# Reproducibility and operating instructions

All commands run from the repository root. Keep data, weights, environments,
generated runs and service state in the workspace's ignored paths. Use only trusted
model artifacts. No commands here download MVTec automatically.

## Requirements and dependency scopes

- Validated interpreter: **Python 3.13.15** on Windows and Linux amd64. Source declares
  Python 3.11+; the locks are not a claim that all Python/platform combinations passed.
- CPU smoke mode is 32x32, weight-free and dataset-free. Allow roughly 2 GiB free
  RAM for tests; the complete Docker monitoring stack needs additional RAM/disk.
- Real bottle experiment: source 900x900, model input 224x224. Observed pipeline
  peak was 2.58 GiB and elapsed time about 14.3 minutes on Ryzen 5 7530U, two
  PyTorch threads. Allow at least 4 GiB available RAM, more for larger categories;
  hardware, contention and package versions affect time/memory. This is guidance,
  not a hard resource guarantee. GPU fitting/evaluation is selected when CUDA exists.
- Docker with Linux containers is required for the service stack. Image/runtime
  dependency downloads require internet initially. Model loading/smoke tests do
  not fetch weights or data. Docker Desktop's VM also needs adequate resources.
- `requirements-lock.txt`: exact validated Windows development package versions;
  used as version constraints for the checked Linux development installation.
- `requirements/serve.lock`: complete hash-pinned **Linux amd64 / CPython 3.13**
  serving wheels, including CPU-only torch/torchvision. It is not a Windows lock.
- `pyproject.toml`: supported version ranges and optional `serve`, `dev`, `validation`
  dependency groups. Other environments must resolve and test their own versions.

The source environment, wheel hashes and pinned image digest fix important inputs,
but do not promise bit-identical models/OCI images across devices and toolchains.

## Windows PowerShell

Explicit interpreter paths avoid activation/execution-policy changes:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-lock.txt
.\.venv\Scripts\python -m pip install --no-build-isolation --no-deps -e .
.\.venv\Scripts\python -m pip check
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\ruff check .
.\.venv\Scripts\ruff format --check .
.\.venv\Scripts\python -m reliablevision.cli --config configs/smoke.toml
.\.venv\Scripts\python scripts/build_drift_reference.py --artifact artifacts/smoke/model.pt
```

The lock already includes serving/test/validation dependencies. For a deliberate
unlocked upgrade use `pip install -e '.[serve,dev,validation]'`, then validate and
refresh locks instead of claiming the old dependency environment was reproduced.

## Linux amd64 / Python 3.13

```bash
python3.13 -m venv .venv
. .venv/bin/activate
python -m pip install --require-hashes -r requirements/serve.lock
python -m pip install --no-build-isolation -c requirements-lock.txt -e '.[serve,dev,validation]'
python -m pip check
python -m pytest -q
ruff check .
ruff format --check .
python -m reliablevision.cli --config configs/smoke.toml
python scripts/build_drift_reference.py --artifact artifacts/smoke/model.pt
```

Install the CPU runtime first to avoid default Linux CUDA wheels. Serving-lock
hashes prevent silently changing those wheels. Development constraints fix
versions but do not hash-pin the additional training/dev Linux wheels.

## Outputs and repeat runs

`configs/smoke.toml` defaults to `artifacts/smoke` and local SQLite MLflow tracking.
The training command rejects a nonempty output directory. Preserve earlier evidence
and choose a new directory; do not delete/overwrite an existing trained model:

```powershell
.\.venv\Scripts\python -m reliablevision.cli --config configs/smoke.toml --output artifacts/another-smoke --tracking-uri sqlite:///artifacts/another-mlflow.db
.\.venv\Scripts\python scripts/build_drift_reference.py --artifact artifacts/another-smoke/model.pt
```

Use `.venv/bin/python` on Linux. The overrides enter the saved resolved configuration;
they do not change the seed/algorithm. Use a fresh tracking DB when switching
platforms: MLflow artifact URIs can contain absolute host paths. Do not share one
SQLite tracking DB between Windows and Linux environments.

| Artifact | Purpose |
| --- | --- |
| `model.pt` | Schema 1, frozen feature weights, memory bank, validation threshold, embedded resolved metadata. |
| `splits.json` | Partition identities, labels and content/mask hashes for leakage/provenance checks. |
| `metrics.json` | Clean and nine corruption conditions; null for undefined metrics. |
| `metadata.json` / `requirements.freeze.txt` | Device/configuration/version provenance and installed environment. |
| `run_id.txt` / `registry.json` | Local MLflow run and registered custom-bundle version. |
| `drift-reference.json` | Immutable aggregate training-input histograms tied to model and split-manifest hashes. |

MLflow's local database/artifact store keeps run metrics and versioned bundles.
These are custom model bundles, not pyfunc models; optional local sidecars are
created after artifact logging. Serve the local completed bundle directory, not
an assumed interchangeable registry download. Verify sibling-file provenance if
retrieving artifacts elsewhere. The SHA-256 of `model.pt` is the API model version.

## Compose from a fresh checkout

Create the smoke bundle/reference first. The initializer only prepares workspace
state directories and gives the non-root container UIDs access on Linux; it does
not train, modify model artifacts or loosen permissions across the workspace.

PowerShell:

```powershell
docker compose config --quiet
docker compose build api
docker run --rm --user 0 --mount "type=bind,source=${PWD},target=/workspace" --entrypoint python reliablevision:local /workspace/scripts/prepare_state.py
docker compose up -d --wait --wait-timeout 120
.\.venv\Scripts\python scripts/check_serving.py
.\.venv\Scripts\python scripts/simulate_traffic.py
.\.venv\Scripts\python scripts/check_monitoring.py
.\.venv\Scripts\python scripts/check_monitoring.py --queries-only
docker compose restart api
docker compose up -d --wait --wait-timeout 90
.\.venv\Scripts\python scripts/check_monitoring.py --after-restart
docker compose down
```

On Linux, the same one-line Docker commands work; use the activated environment's
`python` for scripts. This includes parity, actual scrape/alert, dashboard-query
and review-restart checks. It is a local integration demonstration, not sustained
operational evidence. `down` preserves workspace state; expiry does not securely
erase SQLite free pages/backups. Stop the stack after use.

To serve another bundle or isolate another review run, set `RV_MODEL_DIR` and
`RV_STATE_DIR` before **each** Compose invocation. For example in PowerShell:

```powershell
$env:RV_MODEL_DIR = './artifacts/another-smoke'
$env:RV_STATE_DIR = './artifacts/another-review-state'
docker run --rm --user 0 --env RV_STATE_DIR=artifacts/another-review-state --mount "type=bind,source=${PWD},target=/workspace" --entrypoint python reliablevision:local /workspace/scripts/prepare_state.py
docker compose up -d --wait
```

The reference must belong to the selected model; no overwrite or live refitting is
performed. Set the same environment variables in later shells or Compose may
recreate the default deployment. Prometheus/Grafana histories can contain older
instances: use model identity and time range to interpret them.

## Real MVTec AD bottle

1. Obtain original MVTec AD via the [official source](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
   and its form. Review CC BY-NC-SA 4.0, including noncommercial use restrictions.
   Download/extract manually under `data/`; no dataset is downloaded by these tools.
2. Preserve this structure, including every provided test image and mask:

   ```text
   data/mvtec_ad/bottle/train/good/*.png
   data/mvtec_ad/bottle/test/good/*.png
   data/mvtec_ad/bottle/test/<defect>/*.png
   data/mvtec_ad/bottle/ground_truth/<defect>/*_mask.png
   ```

3. Use `configs/mvtec.toml` unchanged for the recorded operating point. The first
   full run fetches official Wide ResNet50-2 V1 weights if not cached, so review
   that download before starting. CLI default `TORCH_HOME` is `artifacts/torch`;
   its actual torchvision checkpoint cache is `artifacts/torch/hub/checkpoints`.
   Keep it inside the workspace. Artifact loading never downloads weights.
4. With a fresh `artifacts/bottle` output directory, run:

   ```powershell
   .\.venv\Scripts\python scripts/audit_mvtec.py
   .\.venv\Scripts\python scripts/profile_run.py
   .\.venv\Scripts\python scripts/analyze_checkpoint.py
   .\.venv\Scripts\python scripts/benchmark_checkpoint.py
   .\.venv\Scripts\python scripts/benchmark_checkpoint.py --size 900 --requests 10 --warmup 2 --output artifacts/bottle-latency-900.json
   .\.venv\Scripts\python scripts/build_drift_reference.py --artifact artifacts/bottle/model.pt
   ```

   The profiler invokes the existing training CLI and saves the complete run log
   and native-process memory sampling. Analysis recomputes the validation threshold,
   checks saved split identities and writes localization panels. Benchmarks run
   sequentially, without other model/test jobs, and distinguish source dimensions
   from tensor dimensions. Native 900 inference is resource-only with the same bank.

5. For repeat full runs, copy the TOML to a new workspace config with a fresh
   output and tracking URI. Pass that config to audit/profile/analysis; pass its
   `model.pt` to benchmark/reference tools. Keep hyperparameters fixed rather than
   tuning against the final test. Source/device/numerical differences can affect
   measured outputs; publish the actual results, including failures.

See [BOTTLE_VALIDATION.md](BOTTLE_VALIDATION.md) for all original measurements,
corruption definitions, local figure paths, split/model hashes and methodological
differences. Numeric evidence is committed; raw images, figures and weights are not.

## CI and troubleshooting

For a real PNG check against a running bottle service:

```powershell
.\.venv\Scripts\python scripts/check_serving.py --artifact artifacts/bottle/model.pt --image data/mvtec_ad/bottle/test/broken_small/000.png --map-atol 0.00005 --output artifacts/bottle-http-check.json
```

That explicit map tolerance records the Windows/Linux float32 portability case in
[FINAL_REPORT.md](FINAL_REPORT.md); it is not a detector threshold or a guarantee
for other images/platforms. The checker retains its stricter default. Same-runtime
Linux verification of this image had zero map difference, and preprocessing tensors
matched byte-for-byte across platforms.

`.github/workflows/ci.yml` pins official action commits, Python 3.13.15 and a read-only
token scope. It installs CPU wheels and constrained development dependencies, runs
lint/tests, trains a synthetic model, builds/starts Compose and exercises actual
HTTP inference, alerts and review persistence. It does not download MVTec, fetch
pretrained weights, upload artifacts or deploy. A workflow definition is not proof
of a successful hosted run; see the final report for evidence actually observed.

- Nonempty outputs/reference: choose a fresh bundle; never erase prior evidence to rerun.
- Missing dataset: stop and obtain it officially; do not substitute synthetic results.
- Missing Docker engine: start Docker Desktop/Linux engine, then `docker info`.
- State permission errors: use the scoped initializer; do not chmod the whole repository.
- Missing review/metrics: direct Uvicorn defaults monitoring off; Compose enables it.
- Alert absent: send enough samples, allow a scrape/evaluation cycle, check freshness.
- Client tests warn about Starlette/httpx or SQLAlchemy: known upstream warnings,
  tracked in final evidence; changing those dependencies requires retesting.
- Linux-on-Windows pytest cache permissions: use a fresh artifact-local `cache_dir`
  as in `scripts/check_clean_linux.sh`; this does not affect prediction behavior.
- Minimal serving image has no Git executable: a temporary training check inside
  it can emit MLflow source-Git warnings. Runs still save explicit manifests/configs;
  use a development environment with Git for automatic source revision tags.
