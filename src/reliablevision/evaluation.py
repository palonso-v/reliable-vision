import numpy as np
from scipy.ndimage import label
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def aupro(masks: np.ndarray, maps: np.ndarray, limit: float = 0.3) -> float | None:
    """Exact rank-sweep PRO, equal weight per 8-connected defect region.

    Weighted ROC yields the average region overlap as TPR. Interpolate at the
    specified FPR boundary; divide area by boundary for normalized AUPRO.
    """
    weights = np.zeros(masks.shape, dtype=float)
    regions = 0
    for index, mask in enumerate(masks):
        components, count = label(mask, structure=np.ones((3, 3)))
        for component in range(1, count + 1):
            region = components == component
            weights[index][region] = 1 / region.sum()
        regions += count
    if regions == 0 or masks.all():
        return None
    weights[~masks] = 1 / (~masks).sum()
    fpr, pro, _ = roc_curve(
        masks.ravel(), maps.ravel(), sample_weight=weights.ravel(), drop_intermediate=False
    )
    below = fpr < limit
    x = np.r_[fpr[below], limit]
    y = np.r_[pro[below], np.interp(limit, fpr, pro)]
    return float(np.trapezoid(y, x) / limit)


def metrics(
    labels: np.ndarray, scores: np.ndarray, masks: np.ndarray, maps: np.ndarray
) -> dict[str, float | None]:
    if masks.shape != maps.shape or not np.isfinite(maps).all():
        raise ValueError("Invalid anomaly maps")
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite scores")
    return {
        "image_auroc": float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None,
        "image_ap": float(average_precision_score(labels, scores)) if labels.any() else None,
        "pixel_auroc": float(roc_auc_score(masks.ravel(), maps.ravel()))
        if len(np.unique(masks)) == 2
        else None,
        "aupro_0.30": aupro(masks, maps),
    }
