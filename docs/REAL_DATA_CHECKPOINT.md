# Initial real-data availability checkpoint (historical)

Initially checked 7 November 2026 with no dataset present. The user subsequently
supplied bottle. See [the real-data validation report](BOTTLE_VALIDATION.md) for
the subsequent experiment and measurements. The initial check downloaded no dataset.

## Availability and evidence

The configured root `data/mvtec_ad` does not exist. A workspace search for PNG/JPEG
images and tar/zip archives found no candidate dataset. The search excluded the
virtual environment, Git internals, MLflow storage and generated pytest scratch
fixtures; those fixtures are not MVTec data. Other personal directories/drives
were not searched. An existing copy must be placed inside this workspace to use it.

No real-data experiment, localization example, latency/memory benchmark or new
test run was performed during this blocked checkpoint. Image AUROC, image AP,
pixel AUROC and AUPRO are **not measured**, rather than zero or inferred from
synthetic results. Earlier milestone tests do not establish real-data performance.

## Exact preparation steps

1. Open the [official original MVTec AD page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad).
   Use **Download dataset / Fill in the form** and follow MVTec's download
   instructions. Select original **MVTec AD**, not AD 2, LOCO or 3D-AD.
2. Review the stated **CC BY-NC-SA 4.0** terms. The standard dataset license does
   not permit commercial use. Obtain **bottle** with all its nominal training
   images, complete good/defective test split and defect masks. If the official
   download offers only the full archive, that is sufficient; no unofficial
   reformatted mirror is needed. No direct category URL or archive size has been
   assumed or verified here. The download form is not submitted by the agent.
3. From this project's PowerShell terminal, prepare local directories:

   ```powershell
   New-Item -ItemType Directory -Force .\data\downloads, .\data\mvtec_ad
   ```

4. Save the official tar.xz archive under this workspace as
   `data/downloads/mvtec-ad.tar.xz`. Inspect and extract it:

   ```powershell
   tar -tf .\data\downloads\mvtec-ad.tar.xz | Select-Object -First 20
   tar -xf .\data\downloads\mvtec-ad.tar.xz -C .\data\mvtec_ad
   ```

   If the archive contains an extra wrapper directory, arrange its `bottle`
   directory directly under `data/mvtec_ad`. If MVTec supplies another archive
   format, extract it there using the appropriate archive tool instead.
5. Check that these paths contain the actual PNG files, not another nested archive:

   ```text
   data/mvtec_ad/bottle/
     train/good/*.png
     test/good/*.png
     test/<defect_type>/*.png
     ground_truth/<defect_type>/*_mask.png
   ```

   Keep every provided training/test image and mask; do not manually select an
   easy subset or move test images into training. `data/` is already Git-ignored.
6. Tell the agent that `data/mvtec_ad/bottle` is ready. The agent will complete
   the checkpoint before Milestone 3. No manual model training is required.

The existing experiment entry point is:

```powershell
.\.venv\Scripts\python -m reliablevision.cli --config configs/mvtec.toml
```

This is a reference command, not evidence that a run occurred. It defaults to
category `bottle`, model input 224x224 and `artifacts/bottle`; the output must be
empty or changed to a fresh path. It chooses CPU if CUDA is unavailable. Full mode
also needs the torchvision Wide ResNet50-2 V1 pretrained weights and may download
them on first use into the workspace's `artifacts/torch`. Any approval needed for
that download will be handled explicitly before it occurs.

## Source audit completed without data

- `data.splits` starts with only `train/good`, then creates seeded, disjoint
  nominal training and validation partitions. Official `test/*` remains separate.
- `training.run` passes only the training partition's tensors to `model.fit`.
  Reservoir selection and greedy coreset construction operate on those tensors.
- Threshold fitting uses only model scores from nominal validation and a fixed
  quantile. Final-test labels/masks are consumed by evaluation, not thresholding.
- SHA-256 checks reject identical encoded image files within and across all
  partitions. **This has not been executed against actual MVTec data.** It does
  not identify re-encoded copies, near duplicates or acquisition-session leakage.

## Implementation comparison and pending evaluation

The baseline follows PatchCore's nominal memory-bank and locally aggregated
mid-level feature approach, but is not an exact paper reproduction. It uses a
frozen ImageNet Wide ResNet50-2, layer2/layer3 3x3 local pooling, spatial feature
alignment, greedy coreset selection and nearest-neighbor distances.

Its image score is **maximum raw patch distance**. The paper's additional
neighborhood-based image-score reweighting is **not implemented**. Candidate
patches are first bounded by a seeded reservoir; coreset selection uses exact
full-feature distances rather than a projected embedding. Localization uses
bilinear map resizing without Gaussian smoothing. Input preprocessing directly
resizes to a square; paper-specific preprocessing, embedding construction and
hyperparameters must not be assumed equivalent. The training holdout and explicit
memory caps also affect comparisons. No paper-level performance claim is valid.

Once data is available, finish the paper comparison against the primary paper,
run the fixed configuration, record data/split hashes and all four metrics, and
save representative good/defective image–mask–heatmap examples without fitting
visualization normalization on final-test labels. Record original image dimensions
and separately report preprocessing-inclusive latency and model-forward latency
at the configured 224x224 resolution. The existing synthetic HTTP benchmark does
not satisfy this measurement. Report process RSS/peak memory with the measurement
scope and tools stated; Python allocation tracking alone misses native PyTorch
allocations. Native-resolution inference, if measured, is a separate operating
point requiring an explicit compatible configuration and resource budget.

The loader currently keeps resized images in RAM and evaluation retains maps;
full-resolution evaluation can be expensive. Report resource failures rather than
silently reducing test data, changing hyperparameters after seeing results or
substituting synthetic inputs.

Sources: [official MVTec AD source and license](https://www.mvtec.com/research-teaching/datasets/mvtec-ad),
[PatchCore paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper).
