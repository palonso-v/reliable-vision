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
