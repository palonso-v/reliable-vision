# Learning guide

## Milestone 1

PatchCore is a nonparametric detector. A pretrained frozen CNN maps image patches
into a feature space. A compact memory bank represents nominal appearance, and
distance from this bank measures novelty. It does not learn defect classes or
estimate a probability of failure. A detector must be fitted separately for each
category or explicitly validated across categories.

Local pooling gives patches context. Combining intermediate layers captures
different scales. Greedy k-center sampling tries to cover the nominal feature
space by repeatedly choosing the point furthest from the current bank. Its cost
grows with candidate count, embedding dimension and bank size. Reservoir sampling
bounds memory but can miss rare normal appearances; batching distance queries
bounds temporary query-to-bank allocations.

Leakage can occur even with different filenames. Hashes detect exact duplicates,
but cannot rule out near duplicates or correlated captures. Split by acquisition
session when metadata is available. A normal validation quantile gives a usable
decision threshold without consulting defects in the final test set, but small
samples and distribution shifts make its deployed false-positive rate uncertain.

AUROC ranks positive versus negative examples; AP emphasizes positive retrieval
and depends on prevalence. Pixel AUROC can hide small-region failures in a large
background. PRO averages recall per connected defect region; integrating it only
over low false-positive rates highlights localization at practical background
error levels. Metric definitions, interpolation and connectivity must be reported.

Reproducibility needs more than a seed: data identities, preprocessing, exact
dependencies, pretrained weights, device, algorithm choices and artifact schema
all matter. Synthetic smoke tests establish that software paths work; only real
held-out data can provide evidence about model quality. Corruption testing probes
sensitivity but does not replace deployment monitoring.

### Five technical interview questions

1. Why use intermediate pretrained patch features instead of final classification logits?
2. How does greedy coreset selection trade memory and latency against nominal coverage?
3. How would you select a threshold without using final-test defects, and assess uncertainty?
4. When can pixel AUROC look strong while AUPRO or human inspection shows poor localization?
5. What must you save to reproduce a PatchCore result, and which leakage can hashes miss?

## Milestone 2

A serving service should consume the same trained model artifact and preprocessing
contract used during evaluation. Reimplementing scoring in an API creates two
algorithms that can silently diverge. Parity tests compare API scores and maps with
direct checkpoint inference. Artifact hashes identify exact bytes; registry version
numbers provide additional lifecycle context. Neither is a confidence estimate.

Startup and readiness are different from process liveness. Loading and validating
the model during lifespan lets the service fail before accepting predictions if
the checkpoint is incompatible. A warmup checks the actual model path rather than
merely confirming a file exists. Runtime weights and thresholds stay immutable.

CPU model calls block an event loop unless offloaded. Worker threads preserve
responsiveness, but unconstrained parallel model calls can oversubscribe native
PyTorch threads and multiply memory use. Explicit single-request admission creates
a measurable capacity policy. A 503 with retry guidance communicates overload
without an unbounded application queue. Scaling requires measurement and a plan
for model memory and per-process metrics.

Compressed byte limits do not bound decoded image memory. Check both bytes while
streaming and pixel dimensions before loading image data. Validate file signatures
against declared MIME types and handle malformed/truncated images. Structured
errors should let clients act without exposing internals. Bounded metric labels
prevent user-controlled identifiers from exhausting monitoring storage.

Reproducible containers require a fixed base digest, complete dependency versions
and hashes, a specified platform, and a controlled build context. A pinned tag alone
can move. CPU-only wheels avoid accidental CUDA dependency downloads. Models mounted
read-only stay separate from the service image, allowing artifact changes without
rebuilding code. Input reproducibility does not guarantee bit-identical image output.

Latency has several scopes: model forward, application handling and end-to-end
HTTP. Report the scope, model, image size, device, thread count, warmup, concurrency
and sample size. A synthetic smoke fixture and a full pretrained backbone have very
different costs. Local sequential measurements cannot establish production readiness.

### Five technical interview questions

1. How would you prove that API predictions match the evaluated model artifact?
2. Why does putting a PyTorch call inside `async def` not make CPU inference nonblocking?
3. How do worker count, PyTorch thread count and admission policy affect latency and memory?
4. Why must image services enforce both compressed-byte and decoded-pixel limits?
5. What do base-image digests and wheel hashes guarantee, and what must still be measured?

## Real-data checkpoint — initial availability check (historical)

The initial workspace inspection found no MVTec dataset. At that point there were no
real-data detection/localization metrics, examples or resource measurements to
interpret. Missing evidence is not a score of zero, and synthetic smoke results
cannot substitute for real-data validation. The user subsequently supplied bottle;
the actual results are explained below and in [the validation report](docs/BOTTLE_VALIDATION.md).

Source inspection confirms the intended training/validation/test separation, but
actual split independence must also be verified on the supplied files. Encoded-file
hashes catch exact byte copies; re-encoding can evade them. Pixel hashes, similarity
inspection or acquisition-session metadata can provide additional checks. Test
labels may be used to compute evaluation metrics, but never to fit the threshold
or choose a coreset after inspecting final-test performance.

The implemented maximum patch distance omits the original paper's image-score
reweighting. That can change image rankings and AUROC/AP even when patch maps stay
the same. A single category, modified preprocessing, a nominal validation holdout
and bounded memory also prevent direct comparison with a published multi-category
aggregate. Report the actual method and operating point, including unsuccessful
experiments, instead of assigning published results to this implementation.

Real-image benchmarking must state both original image dimensions and the tensor
resolution actually processed by the CNN. Decoding a large source image and then
resizing to 224x224 is different from native-resolution inference. Latency must
identify included work, warmups, thread count and concurrency. Process memory
includes native tensor allocations and model weights; Python-only allocation
tracking is insufficient. No such real-data measurements have run yet.

### Five technical interview questions

1. What evidence distinguishes a functioning synthetic smoke test from a validated real-data detector?
2. How would you verify training-only memory-bank construction and validation-only threshold fitting?
3. Why can identical-file hashes miss leakage, and how would you strengthen the audit?
4. How can omitting PatchCore image-score reweighting affect image AUROC without changing localization maps?
5. How would you benchmark latency and peak process memory when source images are resized before inference?

## Real-data checkpoint — bottle results

The fixed CPU run used 167 nominal training images, 42 nominal validation images
and the complete 83-image final test split. Original images are 900x900; evaluated
model maps and masks are 224x224. Encoded-file and original decoded-RGB hash checks
found no exact duplicates. Neither check proves absence of near-duplicate images
or correlated acquisition sessions.

Clean image AUROC and AP rounded to 1.0000, pixel AUROC was 0.9849 and AUPRO at
FPR <= 0.30 was 0.9337. This is a strong observed result for bottle at this one
seed and resolution, not evidence of equivalent behavior on all categories or
production traffic. AP's stored 0.9999999999999998 is floating-point rounding.
The threshold 1.9235504865646362 exactly matched recomputation on nominal validation
and produced TP=63, TN=20, FP=0, FN=0 on the clean test set. With only 20 good test
images, zero observed false positives does not establish a zero future error rate.
Raw distances and this threshold do not define probabilities.

Noise severity 3 reduced image AUROC to 0.9254 and AUPRO to 0.7458, while AP stayed
0.9794. Read these together: AP depends on the high test anomaly prevalence (63/83),
and apparently strong image-level retrieval can coexist with weaker localization.
Severe blur reduced AUPRO to 0.8478. Mild blur slightly improved AUPRO (0.9390 vs
0.9337); that observation alone does not establish a repeatable benefit or justify
tuning against this final test set. Brightness gains retained image AUROC=1.0,
but this limited brighter-only stress does not cover dim lighting or all exposure
changes. Corruptions were applied after resizing, not to original 900x900 images.

Visual inspection showed a small-defect hotspot near its mask and broader regions
for large-break/contamination examples. A high pixel AUROC and perfect image
classification do not imply sharp segmentation boundaries. The fixed display scale
is derived from nominal validation scores, so saturated heatmap colors must not
be interpreted as calibrated confidence or a fitted pixel decision mask.

The model bank has 1,000 vectors of dimension 1,536 (6,144,000 tensor bytes).
Because 1% selection is applied after the 100,000-candidate reservoir cap, this is
about 0.764% of the 130,928 original training patch vectors, not the paper's 1%
of all candidate patches. CPU full-space greedy construction was expensive: the
complete fit/evaluation/tracking CLI took 858.93 seconds and peaked near 2.58 GiB
sampled process-tree RSS. Training/fit time alone was not separately instrumented.
This illustrates why final bank size is not an estimate of peak construction memory.

The full pretrained detector also costs far more than a weight-free smoke fixture:
at 224x224, forward p50/p95 was 419/472 ms, while decoding real 900x900 source files,
resizing and forwarding took 467/523 ms. Peak process RSS across load/warmup/inference
was about 659 MiB. These are local sequential CPU measurements with two PyTorch
threads. Native-resolution inference is a separate resource experiment with the
same 224-trained bank; it does not establish quality at the new resolution.

The native 900x900 resource run used two warmups and ten measured images: forward
p50/p95 was 6.30/6.42 seconds, with peak process RSS about 835 MiB. It processed
roughly sixteen times as many input pixels as 224x224 and was much slower. Because
the bank remained fitted at 224x224, this is a resource-scaling result only; changing
inference resolution is a model operating-point change that needs fresh validation.

Image-score reweighting from paper equation 7, Gaussian map smoothing, projected
coreset acceleration and reference embedding/preprocessing details remain omitted
or different. Perfect category results cannot resolve those methodological differences.
No production-readiness or paper-level reproduction claim follows from this checkpoint.

### Five technical interview questions about this evaluation

1. Why can AUROC=1 and zero observed false positives on 20 good images still be insufficient deployment evidence?
2. How did the saved split manifest and validation-only threshold recomputation support the leakage audit, and what remains unproven?
3. Why did noise yield AP=0.9794 while AUROC and AUPRO degraded more substantially?
4. Why is a 6 MB memory bank compatible with a roughly 2.58 GiB fitting/evaluation process peak?
5. Why must native 900x900 inference with a 224-trained bank be reported separately from validated 224x224 quality and latency?

## Milestone 3 — monitoring, drift and human review

Service health and model quality require different evidence. Counters, latency
histograms and errors describe operations. A model-info metric identifies the
loaded artifact but does not supply accuracy labels. Bound metric labels: arbitrary
request IDs or filenames would create unbounded time-series cardinality and can
leak customer information. Prometheus histogram quantiles are bucket approximations.

This milestone monitors two input summaries, luminance mean and population standard
deviation. The reference is a fixed aggregate histogram from nominal training images,
versioned by file hash and tied to the exact model hash. Ten predefined bins allow
an interpretable total variation statistic, half the sum of absolute bin-frequency
differences. Zero means matching histograms; one means disjoint bins. The threshold
0.30 with windows of 20 is a demo effect-size heuristic, not a p-value or guaranteed
false-alarm rate. Binning, small samples and serial dependence matter.

Fixed references prevent silently learning a shifted stream as the new normal.
They also age; replacement needs deliberate offline validation rather than an
automatic rolling update. Simulated and unspecified traffic are separated so demo
events cannot silently contaminate another traffic window. Partial windows reset
on process restart, while completed windows and last-state metrics are recoverable.
Freshness is essential: an old drift flag is not current evidence.

The actual simulation produced baseline brightness TV distances 0.067/0.083 and
shifted-window distances 1.0/1.0. The simulated Prometheus drift alert fired. Yet
the model also flagged 20 of 40 baseline images for review without input drift.
That mismatch shows why these brightness statistics cannot certify score stability
or model quality. It is a toy-fixture observation, not an industrial error-rate
measurement. No threshold was retuned to hide it.

Input drift describes a change in measured input distribution. Performance
degradation requires labeled outcomes, representative sampling and an evaluation
protocol. A harmless lighting change can trigger drift; a new defect type can
harm performance without changing global brightness. Review queues are biased
toward flagged cases, so their labels alone cannot estimate overall accuracy.

Human review here means an explicit pending-to-resolved categorical decision with
model/reference identity and timestamps. It does not mean automatic retraining or
authenticated multi-user auditing. The schema deliberately omits images, filenames,
client identity and free text. Reviewers need external governed context to confirm
a defect; otherwise insufficient evidence is an honest outcome. Minimized metadata
can still have sensitivity and needs access control, retention and backup policies.

Persisting a drift window and review together avoids partial records. Advance the
buffer only after transaction commit. The chosen fail-closed behavior returns 503
if monitoring cannot persist a prediction. This improves review durability at the
cost of inference availability. It does not guarantee exactly-once client delivery.
SQLite suits this one-worker demo; distributed workers need coordinated windows,
storage and metrics aggregation. Activity-driven row pruning is not secure erasure.

### Five technical interview questions

1. Why can a drift alert fire without accuracy loss, or accuracy fall without an input-drift alert?
2. What do histogram TV distance, fixed bins and a fixed reference make interpretable, and which changes can they miss?
3. Why did the baseline simulation create score-based reviews despite no brightness-drift flags, and what evidence would diagnose it?
4. How do transactional review persistence and fail-closed inference trade durability against availability?
5. How would you estimate actual deployed performance without treating the selection-biased review queue as a representative labeled dataset?
