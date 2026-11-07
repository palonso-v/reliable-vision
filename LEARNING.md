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
