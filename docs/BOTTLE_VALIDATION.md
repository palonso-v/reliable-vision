# Real-data MVTec AD bottle validation

Checkpoint completed on **7 November 2026**. Milestone 3 has not begun.

## Results

The unchanged real-data pipeline completed successfully. On the full clean test
split, using 224x224 maps and resized masks:

| Metric | Measured value |
| --- | ---: |
| Image AUROC | 1.000000 |
| Image AP | 1.000000 (stored float: 0.9999999999999998) |
| Pixel AUROC | 0.984932 |
| AUPRO, normalized FPR <= 0.30 | 0.933681 |

The saved threshold **1.9235504865646362** exactly matched recomputation from the
42 nominal validation scores. At this fixed threshold, TP=63, TN=20, FP=0, FN=0.
No threshold was selected on final-test labels. The quantile uses `method=higher`;
with 42 scores at quantile 0.99 it selects the maximum validation score. This does
not establish a 1% false-positive guarantee, particularly from such a small holdout.
No image classification errors were observed on this category's clean test split;
that is not a claim of error-free production behavior.

| Condition | Image AUROC | Image AP | Pixel AUROC | AUPRO <= 0.30 |
| --- | ---: | ---: | ---: | ---: |
| Clean | 1.000000 | 1.000000 | 0.984932 | 0.933681 |
| Noise 1 | 0.980159 | 0.994641 | 0.961797 | 0.810597 |
| Noise 2 | 0.960317 | 0.989171 | 0.944948 | 0.771248 |
| Noise 3 | 0.925397 | 0.979439 | 0.931557 | 0.745788 |
| Blur 1 | 1.000000 | 1.000000 | 0.983069 | 0.939003 |
| Blur 2 | 0.997619 | 0.999244 | 0.969459 | 0.903733 |
| Blur 3 | 0.980952 | 0.993988 | 0.952516 | 0.847774 |
| Illumination 1 | 1.000000 | 1.000000 | 0.984809 | 0.932610 |
| Illumination 2 | 1.000000 | 1.000000 | 0.984697 | 0.932009 |
| Illumination 3 | 1.000000 | 1.000000 | 0.984496 | 0.932025 |

Noise sigma is 0.03/0.06/0.09 in [0,1] RGB; blur radius is 1/2/3; illumination gain
is 1.2/1.4/1.6 with clipping. These project-defined corruptions are applied **after
resizing**, use unchanged masks and do not refit the bank or threshold. Noise is
the strongest stress in these tests. Mild blur slightly improved AUPRO; the
observation is not evidence for a repeatable preprocessing benefit and was not used
to tune the model. Stable brightness-gain AUROC does not cover dim light or all
illumination changes. The high AP under severe noise must be interpreted alongside
the high anomaly prevalence and degraded AUROC/AUPRO.

Local figures are `artifacts/bottle/localization/first_broken_large.png`,
`first_broken_small.png`, `first_contamination.png` and `first_good.png`.
All four panels were visually inspected. The small-break hotspot is near its mask;
large-break and contamination maps are broader than the annotated defects.
Normal-image maps also have nonzero distances. Colors are not defect probabilities,
and values above the fixed display maximum saturate. Perfect image classification
does not imply sharp localization or a calibrated pixel mask.

## Actual CPU resource measurements

AMD Ryzen 5 7530U (6 cores / 12 logical processors), Windows/Python 3.13.15,
PyTorch runtime 2.15.0+cpu, two intra-op threads,
concurrency one. Real images are 900x900. Each benchmark ran in a fresh process
after the fit/analysis work finished and before tests, with no other project model
jobs running. Filesystem cache and general host activity were not controlled.

| Measurement | Evaluated input 224x224 | Native input 900x900 (resource-only) |
| --- | ---: | ---: |
| Warmups / measured images | 5 / 30 | 2 / 10 |
| Forward p50 / p95 | 419.34 / 471.66 ms | 6298.80 / 6420.97 ms |
| Decode+resize+forward p50 / p95 | 466.71 / 522.79 ms | 6348.33 / 6469.48 ms |
| Forward mean | 411.89 ms | 6287.07 ms |
| Process peak RSS across load/warmup/inference | 658.59 MiB | 834.81 MiB |
| Checkpoint load time | 6.43 s | 6.83 s |

The full fit/evaluation/tracking CLI took **858.93 seconds** and sampled peak
process-tree RSS was **2,774,130,688 bytes (2.58 GiB)**, at 250 ms sampling.
Maximum individual Windows lifetime peak working set observed was 2,772,185,088
bytes. This is not isolated fit time; it includes all ten evaluation conditions,
data loading, artifact serialization and MLflow tracking.

The bank contains **1,000 x 1,536 float32 values: 6,144,000 bytes**. Its 1% ratio
is applied after the 100,000-candidate cap, so the final bank contains about
**0.764%** of the 167 x 28 x 28 = 130,928 original training patch vectors. Calling
it the paper's PatchCore-1% would be misleading. The candidate pool and temporary
full-space greedy distance computations explain why construction needs much more
memory than the final bank. The checkpoint file is 105,822,047 bytes.

Native-resolution timing uses the same 224-trained bank and is **not** accompanied
by a new quality claim. The small native sample makes tail estimates uncertain.
The weight-free Milestone 2 smoke timings do not describe this full model.
These are direct checkpoint measurements, not full-model HTTP benchmarks or
production saturation tests. Container resource requirements were not revalidated
under these real-model operating points.

## Reproducibility and test evidence

- Local MLflow run: `0629b18a0ddb4e74a8d2b999f21741b6`.
- Registry: `reliablevision-bottle-mvtec`, version **1**.
- Model SHA-256: `557f63f2f06455ee481a9e2f82995cb63ae70ed997eb5dc0462c576db26a52db`.
- Split-manifest SHA-256: `bbc622ee1b31eaa8cede9ca6f48da5afbdefe82222ab1625f8fba4a726592284`.
- Numeric evidence, raw latency samples, configuration, validation/test scores,
  hashes and profiling summaries: [mvtec-bottle-checkpoint.json](benchmarks/mvtec-bottle-checkpoint.json).
- Source/weights hashes and per-run environment freeze are recorded; no official
  source-archive checksum was verified. Dataset pixels, figures and model weights
  remain local and ignored by Git. Local hashes establish identity for this run,
  not independent proof of original archive authenticity.
- Full regression suite: **30 passed, two upstream deprecation warnings, 37.13 s**.
  Warnings are Starlette/httpx TestClient deprecation and MLflow/SQLAlchemy `noload`.
  No test failed. The earlier pre-fit cache/network failure is documented below.
- `pip check` passed. Lint, formatting and compilation checks accompany the final
  change. No algorithm/configuration change was made in response to test metrics.

This checkpoint satisfies a first real-category validation and reveals meaningful
noise/localization and CPU-cost limitations. It does not establish generalization
across categories, paper-level reproduction, calibrated probabilities, production
readiness or operational reliability. Milestone 3 remains unstarted for review.

## Fixed protocol

Original MVTec AD bottle images were supplied locally by the user. The category
contains 209 nominal training images and 83 test images (20 good, 63 defective),
with 63 defect masks. All 292 source images are 900x900. The unchanged
`configs/mvtec.toml` uses seed 42, two PyTorch CPU threads, a 224x224 bicubic RGB
input, a 20% nominal holdout and validation quantile 0.99. CUDA was unavailable.

The resulting partitions are **167 train, 42 nominal validation, 83 final test**.
No final-test data was used to select hyperparameters, thresholds or candidate
patches. The existing pipeline performs clean evaluation and nine predefined
corruption conditions without refitting. This checkpoint is one category, one
seed and one operating point; it is not the paper's category-averaged benchmark.

## Data and threshold audit

`scripts/audit_mvtec.py` ran the existing split loader and also hashed original
decoded RGB pixels and checked image/mask dimensions. No duplicate encoded image
files or identical decoded RGB images were found within or across partitions.
Near duplicates and correlated acquisition sessions are not ruled out.

Source inspection shows `training.run` passes only training tensors to
`PatchCore.fit`; reservoir and coreset construction operate on that list. The
threshold is an upper quantile of held-out nominal validation image scores.
Final-test labels/masks are used only for evaluation. A regression test captures
the exact tensors passed to fitting and checks threshold equality against the
nominal validation quantile. Artifact analysis separately rechecks the saved
split identities/content hashes and recomputes the threshold from the actual
saved model, without consulting test labels.

## Comparison with the original method

Reference: [Roth et al., PatchCore, paper v2, sections 3.1–3.3 and 4.1](https://arxiv.org/html/2106.08265v2).

| Component | This baseline and its relevant differences |
| --- | --- |
| Frozen features | ImageNet Wide ResNet50-2 V1; layer2/layer3, local 3x3 pooling and bilinear alignment. |
| Image preprocessing | Direct 900→224 square resize; the paper resizes to 256 then center-crops to 224. |
| Feature embedding | Concatenated 1536-dimensional local features; no reference implementation's adaptive dimension-mapping stages. |
| Coreset | Seeded 100,000-patch reservoir before exact full-space greedy selection; no paper random projection for coreset acceleration. |
| Image score | Maximum nearest-neighbor patch distance, **without equation 7 neighborhood reweighting**. |
| Localization | Bilinear map upsampling; no paper Gaussian smoothing (sigma 4). |
| Nearest neighbors | Chunked `torch.cdist` Euclidean distances; not FAISS. |
| Protocol | Nominal validation holdout and memory caps; no test-optimized operating threshold or all-category aggregate. |

The paper's reweighting uses distances to neighbors of the selected memory
feature to scale the maximum patch distance. Its omission can affect image score
rankings and AUROC/AP without changing the raw patch maps. This checkpoint does
not silently introduce that modification after final-test evaluation.

For implementation details, also consult the authors' [feature embedding and
scoring code](https://github.com/amazon-science/patchcore-inspection/blob/main/src/patchcore/patchcore.py)
and [embedding/rescaling utilities](https://github.com/amazon-science/patchcore-inspection/blob/main/src/patchcore/common.py).
The paper specification and code should not be assumed identical in every detail.

## Measurement and localization scope

Ranking/localization metrics apply to 224x224 model maps and nearest-neighbor
resized masks. AUPRO is normalized over background FPR 0–0.30, using 8-connected
defect regions. AP depends on the test anomaly prevalence (63/83); it is not a
calibrated probability metric. No test threshold is optimized.

Localization panels contain resized source image, ground truth, raw-distance
heatmap and overlay. Display maximum comes only from nominal validation scores,
and is shared across panels; values beyond it clip visually. This is a display
scale, not probability calibration or pixel threshold fitting. Representative
examples are the first sorted test filename in each defect/good group. The largest
margin false-positive/false-negative examples, if any, are explicitly error
diagnostics. Images and derived figures remain under ignored `artifacts/` in
accordance with dataset handling rules.

Benchmarks use real 900x900 files: file decoding + preprocessing + model forward
is timed separately from forward only. A 224x224 benchmark matches the evaluated
operating point. A separate native 900x900 inference benchmark keeps the **same
224-trained memory bank**: it measures resource scaling, not native-resolution
model quality or a validated deployment setting. Neither includes API JSON/map
serialization or network transfer. Sequential warm-file measurements are not
saturation throughput or an SLO.

Memory is process RSS including native PyTorch allocations, sampled during model
load, warmup and inference, with Windows lifetime peak working set also recorded.
Training CLI profiling includes its launcher/interpreter process tree and the
entire fit/evaluation/tracking run, not just bank construction. Host scheduling,
filesystem cache and allocator reuse remain uncontrolled. Baseline imports and
model-load transients contribute to peaks; bank tensor bytes alone are not RSS.

## Failure history

The first attempt failed before fitting: pretrained weights were cached in
`artifacts/torch/checkpoints`, but the CLI uses `TORCH_HOME/hub/checkpoints`.
Its attempted network fallback was blocked by the sandbox. The failed split
manifest, profile and log were preserved as `artifacts/bottle-initial-failed-*`.
The existing downloaded weights were copied to the correct cache and the same
configuration was rerun. No test-based hyperparameter change was made.

The first resource wrapper also measured only the tiny Windows virtualenv
launcher. It was corrected to sample the process tree, including the real Python
interpreter. The failed attempt's memory samples are not valid training-memory
evidence. Corrected run and independent inference profiles are the reported source.

## Reproduction

Use the licensed local dataset and a fresh output directory. Install the existing
environment and optional `pip install -e '.[validation]'` for pinned psutil.
Pretrained weights must be cached or their official download permitted.

```powershell
.\.venv\Scripts\python scripts/audit_mvtec.py
.\.venv\Scripts\python scripts/profile_run.py
.\.venv\Scripts\python scripts/analyze_checkpoint.py
.\.venv\Scripts\python scripts/benchmark_checkpoint.py
.\.venv\Scripts\python scripts/benchmark_checkpoint.py --size 900 --requests 10 --warmup 2 --output artifacts/bottle-latency-900.json
.\.venv\Scripts\python scripts/export_checkpoint.py
.\.venv\Scripts\python -m pytest -q
```

The fit CLI rejects nonempty output folders. Preserve previous model and evidence
files and use a fresh TOML output path for another run. Downstream script defaults
point to `artifacts/bottle`; supply the corresponding artifact/config paths for
another experiment. Do not run heavy tests concurrently with latency measurements.
